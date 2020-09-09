#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python train.py \
    --data 'path/to/data/like/Flickr25K' \
    --arch 'vgg16' \
    --nmb_cluster 1000 \
    --cluster_alg 'KMeans' \
    --batch 16 \
    --resume 'path/to/checkpoint.pth.tar' \
    --exp 'dir/to/save/experiment' \
    --lr 0.0005 \
    --wd -5 \
    --workers 6 \
    --start_epoch 425 \
    --epochs 500 \
    --momentum 0.9 \
    --checkpoints 1500 \
    --reassign 1 \
    --sobel \
    --verbose \
    --fine_tune
