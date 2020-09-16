#!/bin/bash

docker run --runtime=nvidia --rm --name deepcluster_cont \
    --mount type=bind,source=$1,target='/home/data' \
    --mount type=bind,source=$2,target='/home/results' \
    deepcluster python test.py \
                        --data '/home/data' \
                        --save '/home/results' \
                        --arch 'vgg16' \
                        --nmb_cluster 1000 \
                        --cluster_alg 'KMeans' \
                        --batch 32 \
                        --resume '/home/deepcluster/checkpoint.pth.tar' \
                        --sobel
