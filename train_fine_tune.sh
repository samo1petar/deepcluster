#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python train_fine_tune.py \
    --data '/home/david/Datasets/Flickr/Flickr25K' \
    --arch 'vgg16' \
    --nmb_cluster 1000 \
    --cluster_alg 'KMeans' \
    --batch 16 \
    --resume '/home/david/Projects/deepcluster/deepcluster_models/vgg16/checkpoint.pth.tar' \
    --exp '/home/david/Projects/deepcluster/experiments/exp_3' \
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
    --dropout 0.1
