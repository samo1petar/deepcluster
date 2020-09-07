import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import lib.clustering
from lib.data import transform_to_dict
from lib.utils import accuracy
from lib.utils import visualize

import models
from models.utils import compute_features, restore

from IPython import embed


resume = '/home/david/Projects/deepcluster/deepcluster_models/vgg16/checkpoint.pth.tar'

model = models.__dict__['vgg16'](sobel=True)
model.features = torch.nn.DataParallel(model.features)
for param in model.features.parameters():
    param.requires_grad = False

model.cuda()

# # remove head
model.top_layer = None
model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])

restore(model, resume)

# preprocessing of data
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

tra = [transforms.Resize(256),
       transforms.CenterCrop(224),
       transforms.ToTensor(),
       normalize]



dataset = datasets.ImageFolder('/home/david/Datasets/Flickr/Flickr25K', transform=transforms.Compose(tra))

dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size=32,
                                         num_workers=1,
                                         pin_memory=True)

features = compute_features(dataloader, model, len(dataset))

# ['Kmeans', 'PIC']
nmb_cluster = 250
cluster_alg = clustering.__dict__['Kmeans'](nmb_cluster)

loss = cluster_alg.cluster(features, verbose=True)
train_dataset = clustering.cluster_assign(cluster_alg.images_lists, dataset.imgs)

data = transform_to_dict(train_dataset.imgs)

accuracy(data)

visualize(data)

embed()
