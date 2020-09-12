#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python predict.py \
  --data_centroids '/home/david/Datasets/Flickr/Flickr25K' \
  --data_predict '/home/david/Datasets/Flickr/Flickr15K/images' \
  --arch 'vgg16' \
  --nmb_cluster 1000 \
  --cluster_alg 'KMeans' \
  --batch 32 \
  --resume '/home/david/Projects/deepcluster/checkpoint.pth.tar' \
  --sobel
