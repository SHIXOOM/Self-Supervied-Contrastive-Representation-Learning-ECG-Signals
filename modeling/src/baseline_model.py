from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SimpleECGConfig:
    """Configuration for simple ResNet-based ECG encoder."""

    sequence_length: int = 1000
    num_channels: int = 12
    base_filters: int = 120
    projection_dim: int = 128
    dropout: float = 0.1
    dtype: torch.dtype = torch.float32


class BasicBlock1D(nn.Module):
    """Basic residual block for 1D convolutions."""

    expansion = 1

    def __init__(
        self, in_channels, out_channels, stride=1, downsample=None, dropout=0.1
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=7,
            stride=stride,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=7, stride=1, padding=3, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class SimpleECGEncoder(nn.Module):
    """
    Simple ResNet-based encoder for ECG signals.

    Architecture inspired by:
    - He et al., "Deep Residual Learning for Image Recognition" (2015)
    - Rajpurkar et al., "Cardiologist-level arrhythmia detection" (2017)

    This is a much simpler baseline compared to the complex transformer architecture.
    """

    def __init__(self, config: SimpleECGConfig):
        super().__init__()
        self.config = config

        # Initial convolution to project channels
        self.conv1 = nn.Conv1d(
            config.num_channels,
            config.base_filters,
            kernel_size=15,
            stride=2,
            padding=7,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(config.base_filters)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Residual layers
        self.layer1 = self._make_layer(
            config.base_filters, config.base_filters, 2, stride=1
        )
        self.layer2 = self._make_layer(
            config.base_filters, config.base_filters * 2, 2, stride=2
        )
        self.layer3 = self._make_layer(
            config.base_filters * 2, config.base_filters * 4, 2, stride=2
        )
        self.layer4 = self._make_layer(
            config.base_filters * 4, config.base_filters * 8, 2, stride=2
        )

        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(config.base_filters * 8, config.base_filters * 4),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.base_filters * 4, config.projection_dim),
        )

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm1d(out_channels),
            )

        layers = []
        layers.append(
            BasicBlock1D(
                in_channels, out_channels, stride, downsample, self.config.dropout
            )
        )
        for _ in range(1, blocks):
            layers.append(
                BasicBlock1D(out_channels, out_channels, dropout=self.config.dropout)
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        # x shape: (batch, time, channels) -> need (batch, channels, time)
        if x.dim() == 3:
            x = x.permute(0, 2, 1)

        # Initial conv
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Residual blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global pooling
        x = self.avgpool(x)
        representation = x.squeeze(-1)  # (batch, base_filters * 8)

        # Projection for contrastive learning
        projection = self.projection_head(representation)
        projection = F.normalize(projection, dim=-1)

        return representation, projection
