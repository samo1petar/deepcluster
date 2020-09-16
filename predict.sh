#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python predict.py \
  --data_predict '/home/david/Datasets/Flickr/test' \
  --cluster_index '/home/david/Projects/deepcluster/cluster_index' \
  --checkpoint '/home/david/Projects/deepcluster/checkpoint.pth.tar' \
  --classes 'available_classes.json' \
  --save_dir '/home/david/Datasets/predictions' \
  --arch 'vgg16' \
  --cluster_alg 'KMeans' \
  --batch 32 \
  --top_n 1 \
  --sobel
