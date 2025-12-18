# Self-Supervied Contrastive Representation Learning ECG Signals

## How to run (Bash for Linux & Mac)

```bash
cd Self-Supervied-Contrastive-Representation-Learning-ECG-Signals/data/raw
#!/bin/bash
curl -L -o ~/ptbxl-original-dataset.zip\
  https://www.kaggle.com/api/v1/datasets/download/rohitdwivedula/ptbxl-original-dataset
```

Unzip the dataset

```bash
unzip ptbxl-original-dataset.zip
```

Run the preprocessing file, note that both files will write the data to the same place.  
For creating non-overlapping segments:
```bash
python preprocessing/preprocess_ptbxl.py
```

For creating interleaving segments, run

```bash
python preprocessing/reprocessing.py
```


To start training on a single GPU:

```bash
cd ../ # Go back to the root dir, however you want to do it.

cd modeling
nohup python -m src.single_gpu_training --epochs 350 --batch_size 85 > train.log 2>&1 &
nohup tensorboard --logdir runs --port 7008 --host 0.0.0.0 > tensorboard.out 2>&1 & # Tensorboard
```

If you want to kill a training run early:

```bash
kill -9 $(pgrep -f single_gpu_training)
```

Please note that the project is made to run on Nvidia's GPUs and CUDA.