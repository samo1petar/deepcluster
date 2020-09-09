#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python test.py \
  --data '/home/david/Datasets/Flickr/Flickr25K' \
  --arch 'vgg16' \
  --nmb_cluster 1000 \
  --cluster_alg 'KMeans' \
  --batch 32 \
  --resume '/home/david/Projects/deepcluster/experiments/exp_6/checkpoints/checkpoint_443.0.pth.tar' \
  --sobel


#  --resume '/home/david/Projects/deepcluster/deepcluster_models/vgg16/checkpoint.pth.tar'
# '/home/david/Projects/deepcluster/experiments/exp_1/checkpoint.pth.tar'

# 1. /home/david/Projects/deepcluster/experiments/exp_3/checkpoints/checkpoint_28.0.pth.tar
# 2. /home/david/Projects/deepcluster/experiments/exp_3/checkpoint.pth.tar