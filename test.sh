#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python test.py \
  --data 'path/to/dataset/like/Flickr25K' \
  --arch 'vgg16' \
  --nmb_cluster 1000 \
  --cluster_alg 'KMeans' \
  --batch 32 \
  --resume 'path/to/checkpoint.pth.tar' \
  --sobel
