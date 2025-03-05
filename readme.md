# repo title

> **paper title.**
> abstract.

This repository is anonymized for review.

## 💁 Usage
1. Contact Meta for the access to the OpenEDS2019 and OpenEDS2020 datasets.

2. Create conda environment with `conda env create -f environment.yml` and then activate the environment with `conda activate imb`.

3. For an interactive play with general neural style transfer and iris style transfer, try `nst.ipynb` and `iris_nst.ipynb`.

    For training iris classifiers, gaze estimators, and reproducing our experiment results, run `experiments.sh`.

## 🔧 Environment
Important libraries and their versions by **March 3rd, 2025**:

| Library | Version |
| --- | ----------- |
| Python | 3.12.7 by Anaconda|
| PyTorch | 2.6.0 for CUDA 12.6 |
| OpenCV-Python | 4.11.0.86 |
| Scikit-Learn | 1.6.1 |
| Scikit-Image | 0.25.1 |
| segmentation-models-pytorch | 0.4.0 |
| polarTransform | 2.0.0 |
| WandB | 0.19.5 |

Others:
- The program should be run a computer with at least 32GB RAM. If run on NVIDIA GPU, a minimum VRAM requirement is 32GB. We obtained our results on a cluster with AMD EPYC 7763 64-Core and 4x NVIDIA A100 80GB PCIe.

- We used [Weights & Bias](https://wandb.ai/site) for figures instead of tensorboard. Please install it (already included in `environment.yml`) and set up it (run `wandb login`) properly beforehand.

- The weights of pre-trained iris classifiers and gaze estimators are too large to be uploaded. Therefore the training of these models should be run first after this repo is cloned. The weight of pre-trained RITnet is small and already included in the repo. It can also be downloaded from https://bitbucket.org/eye-ush/ritnet/src/master/. The weight of pre-trained EfficientNet is also too large, but it can be downloaded here: https://github.com/zeenolife/openeds_2020.

## 🗺 Instructions on dataset
It should be noticed that the ground truth labels for the test set of the OpenEDS2019 and OpenEDS2020 datasets have already been released. Please place all image folders and label folders according to the arguments of the `load_data_openeds2019` and `load_data_openeds2020` functions in `data_preprocessing.py`.