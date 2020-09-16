#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python predict.py \
  --data '/path/to/data/like/Flickr25K' \
  --cluster_index '/path/to/cluster_index' \
  --checkpoint '/path/to/checkpoint.pth.tar' \
  --classes '/path/to/available_classes.json' \
  --save '/path/to/save_dir' \
  --arch 'vgg16' \
  --cluster_alg 'KMeans' \
  --batch 32 \
  --top_n 1 \
  --sobel
