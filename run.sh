#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python main.py \
  --data '/home/david/Datasets/Flickr/Flickr25K' \
  --arch 'vgg16' \
  --nmb_cluster 2000 \
  --cluster_alg 'KMeans' \
  --batch 32 \
  --resume '/home/david/Projects/deepcluster/deepcluster_models/vgg16/checkpoint.pth.tar' \
  --sobel
