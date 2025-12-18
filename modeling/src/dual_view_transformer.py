import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import TransformerEncoderLayer

# NOTE: Claude did the heavy lifting to implement
# Flash Attention using Torch's scaled_dot_product_attention
# and to extract the attention maps.

# NOTE: remember that ECG data are in shape (batch, time, channels) not (batch, channels, time).

# NOTE: Flash attention must be disabled when capturing attention weights.

# Gradient norm for the time and channel streams is accessible during training, and their pooled representations
# before cross-attention are also returned for auxiliary losses.

@dataclass
class ECGModelConfig:
    sequence_length: int = 2500
    num_channels: int = 12
    encoder_embed_dim: int = 512
    d_model: int = 512
    time_token_dim: int = 512
    channel_token_dim: int = 512
    time_heads: int = 8
    channel_heads: int = 4
    time_layers: int = 12
    channel_layers: int = 12
    ff_multiplier: int = 6
    dropout: float = 0.1
    temperature: float = 0.5
    projection_dim: int = 512
    time_conv_kernel_size: int = 7
    channel_conv_kernel_size: int = 5
    channel_conv_stride: int = 2
    channel_token_dropout: float = 0.0
    fusion_residual_dropout: float = 0.0
    dtype: torch.dtype = torch.bfloat16
    fusion_hidden_dim: Optional[int] = None
    fusion_dropout: float = 0.0
    use_flash_attention: bool = True


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int, dtype: torch.dtype) -> None:
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model, dtype=dtype)
        pe[0, :, 0::2] = torch.sin(position * div_term).to(dtype)
        pe[0, :, 1::2] = torch.cos(position * div_term).to(dtype)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TimeConvEmbedding(nn.Module):
    def __init__(self, config: ECGModelConfig) -> None:
        super().__init__()
        padding = config.time_conv_kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(
                config.num_channels,
                config.time_token_dim,
                kernel_size=config.time_conv_kernel_size,
                padding=padding,
                bias=False,
                dtype=config.dtype,
            ),
            nn.GELU(),
            nn.Conv1d(
                config.time_token_dim,
                config.time_token_dim,
                kernel_size=1,
                dtype=config.dtype,
            ),
            nn.GELU(),
        )
        self.proj = nn.Linear(
            config.time_token_dim, config.encoder_embed_dim, dtype=config.dtype
        )
        self.norm = nn.LayerNorm(config.encoder_embed_dim, dtype=config.dtype)
        self.dropout = nn.Dropout(config.dropout)
        self.dtype = config.dtype
        self.to(dtype=config.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != self.dtype:
            x = x.to(self.dtype)
        x = x.permute(0, 2, 1)  # (batch, channels, time)
        x = self.net(x)
        x = x.permute(0, 2, 1)
        x = self.proj(x)
        x = self.norm(x)
        return self.dropout(x)


class ChannelConvEmbedding(nn.Module):
    def __init__(self, config: ECGModelConfig) -> None:
        super().__init__()
        depthwise_kwargs = dict(
            kernel_size=config.channel_conv_kernel_size,
            stride=config.channel_conv_stride,
            padding=config.channel_conv_kernel_size // 2,
            groups=config.num_channels,
            bias=False,
            dtype=config.dtype,
        )
        self.net = nn.Sequential(
            nn.Conv1d(config.num_channels, config.num_channels, **depthwise_kwargs),
            nn.GELU(),
            nn.Conv1d(
                config.num_channels,
                config.num_channels,
                kernel_size=1,
                dtype=config.dtype,
            ),
            nn.GELU(),
            nn.Conv1d(config.num_channels, config.num_channels, **depthwise_kwargs),
            nn.GELU(),
            nn.Conv1d(
                config.num_channels,
                config.num_channels,
                kernel_size=1,
                dtype=config.dtype,
            ),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(config.channel_token_dim),
        )
        self.channel_proj = nn.Linear(
            config.channel_token_dim, config.encoder_embed_dim, dtype=config.dtype
        )
        self.dtype = config.dtype
        self.to(dtype=config.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != self.dtype:
            x = x.to(self.dtype)
        x = x.permute(0, 2, 1)
        x = self.net(x)
        x = self.channel_proj(x)
        return x


class TransformerEncoderLayerWithAttention(TransformerEncoderLayer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.capture_attention: bool = False
        self.last_attn_weights: Optional[torch.Tensor] = None

    def _sa_block(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        is_causal: bool = False,
    ) -> torch.Tensor:
        attn_kwargs = dict(
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=self.capture_attention,
            average_attn_weights=False,
        )
        try:
            attn_output, attn_weights = self.self_attn(
                x, x, x, **attn_kwargs, is_causal=is_causal
            )
        except TypeError:
            attn_output, attn_weights = self.self_attn(x, x, x, **attn_kwargs)

        if self.capture_attention:
            self.last_attn_weights = attn_weights
        else:
            self.last_attn_weights = None

        return self.dropout1(attn_output)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:  # type: ignore[override]
        x = src
        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal
            )
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(
                x
                + self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal)
            )
            x = self.norm2(x + self._ff_block(x))
        return x


class FlashTransformerEncoderLayer(nn.Module):
    """Transformer encoder layer using Flash Attention for memory efficiency."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.self_attn = FlashMultiheadAttention(d_model, nhead, dropout, dtype)
        self.norm1 = nn.LayerNorm(d_model, dtype=dtype)
        self.norm2 = nn.LayerNorm(d_model, dtype=dtype)
        self.dropout = nn.Dropout(dropout)

        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward, dtype=dtype),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model, dtype=dtype),
            nn.Dropout(dropout),
        )

        self.capture_attention: bool = False
        self.last_attn_weights: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm architecture
        normed = self.norm1(x)
        self.self_attn.capture_attention = self.capture_attention
        attn_out, _ = self.self_attn(
            normed, normed, normed, need_weights=self.capture_attention
        )
        self.last_attn_weights = self.self_attn.last_attn_weights
        x = x + self.dropout(attn_out)
        x = x + self.ff(self.norm2(x))
        return x


class TimeTransformer(nn.Module):
    def __init__(self, config: ECGModelConfig) -> None:
        super().__init__()
        self.use_flash = config.use_flash_attention

        if self.use_flash:
            self.layers = nn.ModuleList(
                [
                    FlashTransformerEncoderLayer(
                        d_model=config.encoder_embed_dim,
                        nhead=config.time_heads,
                        dim_feedforward=config.encoder_embed_dim * config.ff_multiplier,
                        dropout=config.dropout,
                        dtype=config.dtype,
                    )
                    for _ in range(config.time_layers)
                ]
            )
            self.layers[-1].capture_attention = True
        else:
            encoder_layer = TransformerEncoderLayerWithAttention(
                d_model=config.encoder_embed_dim,
                nhead=config.time_heads,
                dim_feedforward=config.encoder_embed_dim * config.ff_multiplier,
                dropout=config.dropout,
                batch_first=True,
                activation="gelu",
                norm_first=True,
                dtype=config.dtype,
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=config.time_layers
            )
            for idx, layer in enumerate(self.encoder.layers):
                layer.capture_attention = idx == (config.time_layers - 1)

        self.last_attn_weights: Optional[torch.Tensor] = None
        self.dtype = config.dtype
        self.to(dtype=config.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != self.dtype:
            x = x.to(self.dtype)

        if self.use_flash:
            for layer in self.layers:
                x = layer(x)
            self.last_attn_weights = self.layers[-1].last_attn_weights
        else:
            x = self.encoder(x)
            self.last_attn_weights = getattr(
                self.encoder.layers[-1], "last_attn_weights", None
            )

        return x


class ChannelTransformer(nn.Module):
    def __init__(self, config: ECGModelConfig) -> None:
        super().__init__()
        self.use_flash = config.use_flash_attention

        if self.use_flash:
            self.layers = nn.ModuleList(
                [
                    FlashTransformerEncoderLayer(
                        d_model=config.encoder_embed_dim,
                        nhead=config.channel_heads,
                        dim_feedforward=config.encoder_embed_dim * config.ff_multiplier,
                        dropout=config.dropout,
                        dtype=config.dtype,
                    )
                    for _ in range(config.channel_layers)
                ]
            )
            self.layers[-1].capture_attention = True
        else:
            encoder_layer = TransformerEncoderLayerWithAttention(
                d_model=config.encoder_embed_dim,
                nhead=config.channel_heads,
                dim_feedforward=config.encoder_embed_dim * config.ff_multiplier,
                dropout=config.dropout,
                batch_first=True,
                activation="gelu",
                norm_first=True,
                dtype=config.dtype,
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=config.channel_layers
            )
            for idx, layer in enumerate(self.encoder.layers):
                layer.capture_attention = idx == (config.channel_layers - 1)

        self.last_attn_weights: Optional[torch.Tensor] = None
        self.dtype = config.dtype
        self.to(dtype=config.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != self.dtype:
            x = x.to(self.dtype)

        if self.use_flash:
            for layer in self.layers:
                x = layer(x)
            self.last_attn_weights = self.layers[-1].last_attn_weights
        else:
            x = self.encoder(x)
            self.last_attn_weights = getattr(
                self.encoder.layers[-1], "last_attn_weights", None
            )

        return x


class FlashMultiheadAttention(nn.Module):
    """Memory-efficient attention using PyTorch's scaled_dot_product_attention (Flash Attention)."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.dtype = dtype

        self.q_proj = nn.Linear(embed_dim, embed_dim, dtype=dtype)
        self.k_proj = nn.Linear(embed_dim, embed_dim, dtype=dtype)
        self.v_proj = nn.Linear(embed_dim, embed_dim, dtype=dtype)
        self.out_proj = nn.Linear(embed_dim, embed_dim, dtype=dtype)

        self.last_attn_weights: Optional[torch.Tensor] = None
        self.capture_attention: bool = False
        self.use_flash: bool = True

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len, _ = query.shape
        kv_seq_len = key.shape[1]

        q = (
            self.q_proj(query)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(key)
            .view(batch_size, kv_seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(value)
            .view(batch_size, kv_seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        attn_weights = None
        # Use manual attention when flash is disabled or when capturing weights
        if not self.use_flash or (need_weights and self.capture_attention):
            scale = 1.0 / math.sqrt(self.head_dim)
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
            attn_weights = F.softmax(attn_weights, dim=-1)
            if self.training and self.dropout > 0:
                attn_weights = F.dropout(attn_weights, p=self.dropout)
            attn_output = torch.matmul(attn_weights, v)
            self.last_attn_weights = attn_weights
        else:
            # Use Flash Attention - O(N) memory instead of O(N^2)
            attn_output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
            )
            self.last_attn_weights = None

        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.embed_dim)
        )
        return self.out_proj(attn_output), attn_weights


class BidirectionalCrossAttention(nn.Module):
    def __init__(self, config: ECGModelConfig) -> None:
        super().__init__()
        self.use_flash = config.use_flash_attention

        if self.use_flash:
            self.time_to_channel = FlashMultiheadAttention(
                config.encoder_embed_dim,
                config.time_heads,
                config.dropout,
                config.dtype,
            )
            self.channel_to_time = FlashMultiheadAttention(
                config.encoder_embed_dim,
                config.channel_heads,
                config.dropout,
                config.dtype,
            )
        else:
            self.time_to_channel = nn.MultiheadAttention(
                config.encoder_embed_dim,
                config.time_heads,
                batch_first=True,
                dtype=config.dtype,
            )
            self.channel_to_time = nn.MultiheadAttention(
                config.encoder_embed_dim,
                config.channel_heads,
                batch_first=True,
                dtype=config.dtype,
            )

        self.time_norm = nn.LayerNorm(config.encoder_embed_dim, dtype=config.dtype)
        self.channel_norm = nn.LayerNorm(config.encoder_embed_dim, dtype=config.dtype)
        self.last_time_to_channel_weights: Optional[torch.Tensor] = None
        self.last_channel_to_time_weights: Optional[torch.Tensor] = None

    def forward(
        self, time_tokens: torch.Tensor, channel_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.use_flash:
            time_cross, _ = self.time_to_channel(
                time_tokens, channel_tokens, channel_tokens
            )
            channel_cross, _ = self.channel_to_time(
                channel_tokens, time_tokens, time_tokens
            )
            self.last_time_to_channel_weights = self.time_to_channel.last_attn_weights
            self.last_channel_to_time_weights = self.channel_to_time.last_attn_weights
        else:
            time_cross, self.last_time_to_channel_weights = self.time_to_channel(
                time_tokens,
                channel_tokens,
                channel_tokens,
                need_weights=True,
                average_attn_weights=False,
            )
            channel_cross, self.last_channel_to_time_weights = self.channel_to_time(
                channel_tokens,
                time_tokens,
                time_tokens,
                need_weights=True,
                average_attn_weights=False,
            )

        fused_time = self.time_norm(time_tokens + time_cross)
        fused_channel = self.channel_norm(channel_tokens + channel_cross)
        return fused_time, fused_channel


class FusionHead(nn.Module):
    def __init__(self, config: ECGModelConfig) -> None:
        super().__init__()
        fusion_input = config.encoder_embed_dim * 2
        hidden_dim = config.fusion_hidden_dim or config.d_model
        self.linear_in = nn.Linear(fusion_input, hidden_dim, dtype=config.dtype)
        self.norm_in = nn.LayerNorm(hidden_dim, dtype=config.dtype)
        self.linear_out = nn.Linear(hidden_dim, config.d_model, dtype=config.dtype)
        self.norm_out = nn.LayerNorm(config.d_model, dtype=config.dtype)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(config.fusion_dropout)
        self.residual_dropout = nn.Dropout(config.fusion_residual_dropout)

        # Gated residual path from early embeddings to improve gradient flow
        self.residual_proj = nn.Linear(fusion_input, config.d_model, dtype=config.dtype)
        self.residual_gate = nn.Parameter(torch.tensor(0.1, dtype=config.dtype))

    def forward(
        self,
        time_repr: torch.Tensor,
        channel_repr: torch.Tensor,
        time_residual: Optional[torch.Tensor] = None,
        channel_residual: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        fused = torch.cat([time_repr, channel_repr], dim=-1)
        hidden = self.activation(self.norm_in(self.linear_in(fused)))
        hidden = self.dropout(hidden)
        mapped = self.linear_out(hidden)
        output = self.activation(self.norm_out(mapped))

        if time_residual is not None and channel_residual is not None:
            early_fused = torch.cat([time_residual, channel_residual], dim=-1)
            residual = self.residual_proj(early_fused)
            residual = self.residual_dropout(residual)
            output = output + torch.sigmoid(self.residual_gate) * residual

        return output


class ProjectionHead(nn.Module):
    def __init__(self, config: ECGModelConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.d_model, config.d_model, dtype=config.dtype),
            nn.LayerNorm(config.d_model, dtype=config.dtype),
            nn.ReLU(),
            nn.Linear(config.d_model, config.projection_dim, dtype=config.dtype),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class ECGEncoder(nn.Module):
    def __init__(self, config: ECGModelConfig) -> None:
        super().__init__()
        self.time_conv_embedding = TimeConvEmbedding(config)
        self.time_positional_encoding = SinusoidalPositionalEncoding(
            config.encoder_embed_dim, config.sequence_length, config.dtype
        )
        self.time_transformer = TimeTransformer(config)
        self.channel_conv_embedding = ChannelConvEmbedding(config)
        self.channel_norm = nn.LayerNorm(config.encoder_embed_dim, dtype=config.dtype)
        self.channel_dropout = nn.Dropout(config.dropout)
        self.channel_token_dropout = config.channel_token_dropout
        self.channel_transformer = ChannelTransformer(config)
        self.cross_attention = BidirectionalCrossAttention(config)
        self.fusion = FusionHead(config)
        self.projection = ProjectionHead(config)
        self.dtype = config.dtype
        self.to(dtype=config.dtype)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.dtype != self.dtype:
            x = x.to(self.dtype)

        time_tokens = self.time_conv_embedding(x)
        time_early = time_tokens.mean(dim=1)  # Pool early time features for residual
        time_tokens = self.time_positional_encoding(time_tokens)
        time_tokens = self.time_transformer(time_tokens)

        channel_tokens = self.channel_conv_embedding(x)
        channel_early = channel_tokens.mean(
            dim=1
        )  # Pool early channel features for residual
        channel_tokens = self.channel_norm(channel_tokens)
        channel_tokens = self.channel_dropout(channel_tokens)
        if self.training and self.channel_token_dropout > 0.0:
            drop_mask = (
                torch.rand(
                    channel_tokens.size(0),
                    channel_tokens.size(1),
                    1,
                    device=channel_tokens.device,
                )
                < self.channel_token_dropout
            ).to(channel_tokens.dtype)
            channel_tokens = channel_tokens * (1.0 - drop_mask)
        channel_tokens = self.channel_transformer(channel_tokens)

        # Stream representations for auxiliary loss (before cross-attention)
        time_stream_repr = F.normalize(time_tokens.mean(dim=1), dim=-1)
        channel_stream_repr = F.normalize(channel_tokens.mean(dim=1), dim=-1)

        fused_time, fused_channel = self.cross_attention(time_tokens, channel_tokens)

        # Retain branch tensors for gradient monitoring
        if self.training:
            fused_time.retain_grad()
            fused_channel.retain_grad()
            self._last_fused_time = fused_time
            self._last_fused_channel = fused_channel
        else:
            self._last_fused_time = None
            self._last_fused_channel = None

        # Fixed residual skip around cross-attention to preserve self-attention representations
        fused_time = fused_time + time_tokens
        fused_channel = fused_channel + channel_tokens

        time_repr = fused_time.mean(dim=1)
        channel_repr = fused_channel.mean(dim=1)

        representation = self.fusion(
            time_repr,
            channel_repr,
            time_residual=time_early,
            channel_residual=channel_early,
        )
        projection = self.projection(representation)
        return representation, projection, time_stream_repr, channel_stream_repr

    def get_attention_maps(
        self, head_average: bool = False
    ) -> Dict[str, Optional[torch.Tensor]]:
        maps: Dict[str, Optional[torch.Tensor]] = {
            "time_self": self.time_transformer.last_attn_weights,
            "channel_self": self.channel_transformer.last_attn_weights,
            "time_to_channel": self.cross_attention.last_time_to_channel_weights,
            "channel_to_time": self.cross_attention.last_channel_to_time_weights,
        }
        if not head_average:
            return maps

        averaged: Dict[str, Optional[torch.Tensor]] = {}
        for name, tensor in maps.items():
            if tensor is None:
                averaged[name] = None
            else:
                averaged[name] = tensor.mean(dim=1)
        return averaged

    def set_flash_attention(self, enabled: bool) -> None:
        """Enable or disable Flash Attention at runtime for all attention layers."""
        for module in self.modules():
            if isinstance(module, FlashMultiheadAttention):
                module.use_flash = enabled


@torch.no_grad()
def collect_head_averaged_attention(
    encoder: ECGEncoder,
) -> Dict[str, torch.Tensor]:
    """Return head-averaged attention maps that were cached during the last forward pass."""
    maps = encoder.get_attention_maps(head_average=True)
    return {
        name: tensor.detach().clone()
        for name, tensor in maps.items()
        if tensor is not None
    }
