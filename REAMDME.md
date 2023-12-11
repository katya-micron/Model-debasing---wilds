## Overview

This repository contains code and resources for bias mitigation in deep learning. The goal of this project is to provide a comprehensive guide to mitigate bias in machine learning models.

Introduction
Machine learning models learn from the training data. The modeling step can perpetuate existing bias in the data.
When a machine learning-based system is applied naively, without accounting for bias, it tends to excel in generating accurate predictions for the average population. However, this approach doesn’t necessarily encourage the model to learn and make accurate predictions for samples who belong to underrepresented groups due to sampling bias. In other words, the model may achieve an overall better performance, but it does so by prioritizing accuracy for well-represented groups at the expense of lower performance (resulting in systematic errors) for the underrepresented groups. In order to give all features the same importance, one needs to standardize the data but not only at the very beginning but throughout the modelling phase too. The model adoption of Group Normalization, Weight Standardization, suitable activation functions like SiLu, and LogitNorm play crucial roles in improving the performance, robustness, and security of neural network models across various applications and challenges in the field of deep learning.
For more information, please refer to our paper

This repo is a modification of the original WILDS repo [https://github.com/p-lambda/wilds/tree/main](https://github.com/google-research/big_transfer)
## Installation

We recommend using pip to install WILDS:
```bash
pip install wilds
```

If you have already installed it, please check that you have the latest version:
```bash
python -c "import wilds; print(wilds.__version__)"
# This should print "2.0.0". If it doesn't, update by running:
pip install -U wilds

```
Clone the current repository and  install from source:
```bash
cd wilds
pip install -e .
```
Our contributions are located in the examples/ folder. We have included Resnet with Group Normalization(GN) and weight standardization (ws), and SiLU activation (examples/models/).
Additionally, we have added an extra loss function that applies LogitNorm normalization  (in loss.py). We have also modified the detection fasterrcnn to use a ResNet backbone with GN and SiLU and multi-resizing.
For many of our demonstrations, we have used pre-trained models, such as the [BiT](https://github.com/google-research/big_transfer) pre-trained models, which can be obtained by following the guidelines from https://github.com/google-research/big_transfer.
 For example, ResNet-50x1 pre-trained on ImageNet-21k is available.

```bash
wget https://storage.googleapis.com/bit_models/BiT-M-R50x1.{npz|h5}

```
For Resnet50 with GN and SiLU, pretrained on ImageNet-1K please contact the authors of this repo.

## Using the example scripts

The models and dataset-specific hyperparameters are provided in examples/configs/datasets.py. If you’re using a pre-trained model, you’ll need to add the path to the pre-trained model.

To run an example, we recommend that you first download the specific dataset using the standalone wilds/download_datasets.py script. For example:
```bash

python wilds/download_datasets.py --root_dir data --dataset iwildcam
```
and then run the training:

```bash
python examples/run_expt.py --dataset iwildcam --algorithm ERM --root_dir data
```



