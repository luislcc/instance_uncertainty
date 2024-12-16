#!/bin/bash


i=1

while [[ $i -le 7 ]] ; do
	echo "Model $i" 
	python  main.py --model deeplabv3plus_resnet101 --crop_val --dataset cityscapes --gpu_id 3  --lr 0.0001 --random_seed -1  --crop_size 768 --batch_size 2 --val_batch_size 1 --total_itrs 30000 --output_stride 16 --data_root ./datasets/data/cityscapes
	(( i+= 1 ))
done
