# Implementation of the paper "Improving Semantic Segmentation Performance on Minority Classes Using Instance-Wise Uncertainty"

This repository contains all necessary code to reproduce the experiments in "Improving Semantic Segmentation Performance on Minority Classes Using Instance-Wise Uncertainty". It extends the existing repository https://github.com/VainF/DeepLabV3Plus-Pytorch.

### Experiment 1

First, you must train the models to generate the uncertainty masks. Train a DeepLabV3+ model with the ResNet50, ResNet101 and MobileNet backbones on either the ACDC or CityScapes datasets. To achieve this, use the command:
```
python main.py --model deeplabv3plus_resnet101 --gpu_id 0 --crop_val --lr 0.0001 --crop_size 768 --batch_size 2 --val_batch_size 1 --output_stride 16 --dataset cityscapes --data_root ./datasets/data/cityscapes --total_itrs 50000 --random_seed -1
```
