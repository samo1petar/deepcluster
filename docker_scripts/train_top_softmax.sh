#!/bin/bash

docker run --runtime=nvidia --rm --name deepcluster_cont \
    --mount type=bind,source=$1,target='/home/data' \
    --mount type=bind,source=$2,target='/home/results' \
    deepcluster python train_top_softmax.py \
                        --data '/home/data' \
                        --arch 'vgg16' \
                        --batch 16 \
                        --resume '/home/deepcluster/checkpoint.pth.tar' \
                        --exp '/home/results' \
                        --lr 0.0005 \
                        --wd -5 \
                        --workers 6 \
                        --start_epoch 425 \
                        --epochs 450 \
                        --checkpoints 250000 \
                        --sobel \
                        --verbose \
                        --dropout 0.1
