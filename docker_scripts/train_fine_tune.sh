#!/bin/bash

docker run --runtime=nvidia --rm --shm-size 16G --name deepcluster_cont \
    --mount type=bind,source=$1,target='/home/data' \
    --mount type=bind,source=$2,target='/home/results' \
    deepcluster python train_fine_tune.py \
                        --data '/home/data' \
                        --arch 'vgg16' \
                        --nmb_cluster 1000 \
                        --cluster_alg 'KMeans' \
                        --batch 16 \
                        --resume '/home/deepcluster/checkpoint.pth.tar' \
                        --exp '/home/results' \
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
