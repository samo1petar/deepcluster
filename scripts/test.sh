#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python test.py \
  --data '/path/to/data/like/Flickr25K' \
  --save '/path/to/save/dir' \
  --arch 'vgg16' \
  --nmb_cluster 1000 \
  --cluster_alg 'KMeans' \
  --batch 32 \
  --resume '/path/to/deepcluster/checkpoint.pth.tar' \
  --sobel
