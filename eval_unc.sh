#!/bin/bash

regex_pattern=".*deeplabv3plus_resnet101.*acdc.*seed[0-9]+\.pth$"
directory="/data/lmma/exp_paper/"
i=1


find "$directory" -type f -regex "$regex_pattern" -print0 |
while IFS= read -r -d '' file; do
	echo "Testing with file: $file $i"
	command="python main.py --model deeplabv3plus_resnet101 --ckpt \"$file\" --gpu_id 1 --crop_val --lr 0.0001 --crop_size 768 --batch_size 2 --val_batch_size 1 --output_stride 16 --dataset cityscapes --data_root ./datasets/data/cityscapes --total_itrs 48000 --random_seed -1 --test_only"
	eval "$command"
	(( i+= 1 ))
done