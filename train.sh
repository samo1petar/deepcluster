#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python train.py \
    --data 'path/to/dataset/like/Flickr25K' \
    --arch 'vgg16' \
    --nmb_cluster 1000 \
    --cluster_alg 'KMeans' \
    --batch 16 \
    --resume 'path/to/checkpoint.pth.tar' \
    --exp 'path/to/dir/where/experiment/will/be/saved' \
    --lr 0.05 \
    --wd -5 \
    --workers 6 \
    --start_epoch 0 \
    --epochs 200 \
    --momentum 0.9 \
    --checkpoints 2500 \
    --reassign 1 \
    --sobel \
    --verbose
