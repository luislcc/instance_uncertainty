# Implementation of the paper "Improving Semantic Segmentation Performance on Minority Classes Using Instance-Wise Uncertainty"

This repository contains all necessary code to reproduce the experiments in "Improving Semantic Segmentation Performance on Minority Classes Using Instance-Wise Uncertainty". It extends the existing repository https://github.com/VainF/DeepLabV3Plus-Pytorch.

### Experiment 1

First, you must train the models to generate the uncertainty masks. Train a DeepLabV3+ model with the ResNet50, ResNet101 and MobileNet backbones on either the ACDC or CityScapes datasets. To achieve this, use the command:
```python
python main.py --model deeplabv3plus_resnet101 --gpu_id 0 --crop_val --lr 0.0001 --crop_size 768 --batch_size 2 --val_batch_size 1 --output_stride 16 --dataset cityscapes --data_root ./datasets/data/cityscapes --total_itrs 50000 --random_seed -1
```
After training the three models, use them as checkpoints and combine them in an ensemble to generate the uncertainty masks for the training set:
```
python main.py --model deep_ensemble --weight checkpoints/best_deeplabv3plus_resnet101_cityscapes_os16_seed7830581.pth --weight checkpoints/best_deeplabv3plus_mobilenet_cityscapes_os16_seed6009574.pth --weight checkpoints/best_deeplabv3plus_resnet50_cityscapes_os16_seed8344116.pth --gpu_id 0 --crop_val --lr 0.0001 --crop_size 768 --batch_size 1 --val_batch_size 1 --output_stride 16 --dataset cityscapes --data_root ./datasets/data/cityscapes --make_uncertainty_masks
```
Finally, you can train new models with instance-wise uncertainty masks by using the uncertay_cross_entropy loss type:
```
python main.py --model deeplabv3plus_resnet101 --gpu_id 0 --crop_val --loss_type uncertainty_cross_entropy --lr 0.0001 --crop_size 768 --batch_size 2 --val_batch_size 1 --output_stride 16 --dataset cityscapes --data_root ./datasets/data/cityscapes --total_itrs 50000 --random_seed -1
```
