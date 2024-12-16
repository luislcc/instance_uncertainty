#!/bin/bash


i=1

while [[ $i -le 30 ]] ; do
	echo "Model $i" 
	python main.py --model deeplabv3plus_mobilenet --gpu_id 0 --crop_val --lr 0.001 --loss_type uncertainty_cross_entropy --crop_size 768 --batch_size 2 --val_batch_size 1 --output_stride 16 --dataset acdc --data_root ./datasets/data/ACDC --total_itrs 48000 --random_seed -1
	(( i+= 1 ))
done
