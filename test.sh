#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python test.py \
  --data '/home/data' \
  --arch 'vgg16' \
  --nmb_cluster 1000 \
  --cluster_alg 'KMeans' \
  --batch 32 \
  --resume '/home/deepcluster/checkpoint.pth.tar' \
  --sobel
