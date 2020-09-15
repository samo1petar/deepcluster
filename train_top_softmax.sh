#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python train_fine_tune.py \
    --data '/home/david/Datasets/Flickr/Flickr25K' \
    --arch 'vgg16' \
    --nmb_cluster 1000 \
    --cluster_alg 'KMeans' \
    --batch 16 \
    --resume '/home/david/Projects/deepcluster/checkpoint.pth.tar' \
    --exp '/home/david/Projects/deepcluster/experiments/exp_4_softmax_drop_0.8' \
    --lr 0.0005 \
    --wd -5 \
    --workers 6 \
    --start_epoch 425 \
    --epochs 450 \
    --momentum 0.9 \
    --checkpoints 250000 \
    --reassign 1 \
    --sobel \
    --verbose \
    --dropout 0.1
