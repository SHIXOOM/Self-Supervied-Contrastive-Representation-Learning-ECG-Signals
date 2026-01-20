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


References:
```
1. Oh, J., Chung, H., Kwon, J. M., Hong, D. G., & Choi, E. (2022, April). Lead-agnostic self-supervised learning for local and global representations of electrocardiogram. In Conference on Health, Inference, and Learning (pp. 338-353). PMLR.
2. McKeen, K., Masood, S., Toma, A., Rubin, B., & Wang, B. (2025). Ecg-fm: An open electrocardiogram foundation model. JAMIA open, 8(5), ooaf122.
3. Kiyasseh, D., Zhu, T., & Clifton, D. A. (2021, July). Clocs: Contrastive learning of cardiac signals across space, time, and patients. In International Conference on Machine Learning (pp. 5606-5615). PMLR.
4. Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020, November). A simple framework for contrastive learning of visual representations. In International conference on machine learning (pp. 1597-1607). PmLR.
```
