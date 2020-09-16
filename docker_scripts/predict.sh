#!/bin/bash

docker run --runtime=nvidia --rm --name deepcluster_cont \
    --mount type=bind,source=$1,target='/home/data' \
    --mount type=bind,source=$2,target='/home/results' \
    deepcluster python predict.py \
                        --data '/home/data' \
                        --cluster_index '/home/deepcluster/cluster_index' \
                        --checkpoint '/home/deepcluster/checkpoint.pth.tar' \
                        --classes 'available_classes.json' \
                        --save '/home/results' \
                        --arch 'vgg16' \
                        --cluster_alg 'KMeans' \
                        --batch 32 \
                        --top_n 1 \
                        --sobel
