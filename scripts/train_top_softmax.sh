#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python train_top_softmax.py \
    --data '/path/to/data/like/Flickr25K' \
    --arch 'vgg16' \
    --batch 16 \
    --resume '/path/to/checkpoint.pth.tar' \
    --exp '/path/to/experiments/dir' \
    --lr 0.0005 \
    --wd -5 \
    --workers 6 \
    --start_epoch 425 \
    --epochs 450 \
    --checkpoints 250000 \
    --sobel \
    --verbose \
    --dropout 0.1
