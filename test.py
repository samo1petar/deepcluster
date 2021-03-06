import argparse
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import clustering
from clustering.utils import cluster_assign
from lib.data import transform_to_dict
from lib.utils import accuracy, check_classes, visualize
import models
from models.utils import compute_features, restore


def parse_args():

    parser = argparse.ArgumentParser(description='PyTorch Implementation of DeepCluster in Python3')

    parser.add_argument('--data', type=str, help='Path to dataset.')
    parser.add_argument('--save', type=str, default='', help='Path to dir where data will be saved.')
    parser.add_argument('--arch', default='vgg16', type=str, choices=['alexnet', 'vgg16'], help='CNN architecture')
    parser.add_argument('--sobel', action='store_true', help='Sobel filtering')
    parser.add_argument('--nmb_cluster', '--k', type=int, default=1000, help='number of cluster for k-means (default: 1000)')
    parser.add_argument('--cluster_alg', default='KMeans', type=str, choices=['KMeans', 'PIC'], help='clustering algorithm (default: KMeans)')
    parser.add_argument('--batch', default=256, type=int, help='mini-batch size (default: 256)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to checkpoint (default: None)')

    return parser.parse_args()


def main(args):

    assert args.arch == 'vgg16', 'Only vgg16 architecture is supported for now.'
    assert args.cluster_alg == 'KMeans', 'Only KMeans clustering algorithm is supported for now.'

    if not args.sobel:
        print ('Warning: All development is done with sobel filter active. Running without sobel filter is unchecked teritory.')

    model = models.__dict__[args.arch](sobel=args.sobel)
    model.features = torch.nn.DataParallel(model.features)

    for param in model.features.parameters():
        param.requires_grad = False

    model.cuda()

    # remove head
    model.top_layer = None
    model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])

    restore(model, args.resume)

    # preprocessing of data
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    tra = [transforms.Resize(256),
           transforms.CenterCrop(224),
           transforms.ToTensor(),
           normalize]

    dataset = datasets.ImageFolder(args.data, transform=transforms.Compose(tra))

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=args.batch,
                                             num_workers=1,
                                             pin_memory=True)

    features = compute_features(dataloader, model, len(dataset), args.batch)

    algs = {
        'KMeans': clustering.KMeans,
        'PIC': clustering.PIC,
    }
    cluster_alg = algs[args.cluster_alg](args.nmb_cluster)

    loss = cluster_alg.cluster(features, verbose=True)
    train_dataset = cluster_assign(cluster_alg.images_lists, dataset.imgs)

    data = transform_to_dict(train_dataset.imgs)

    accuracy(data)

    if args.save:
        print ('Saving images to ', args.save, end='\r')
        save_dir = visualize(data, path=args.save)
        check_classes(save_dir, train_dataset.imgs)
        print ('Saving images to ', args.save, 'done')


if __name__ == '__main__':
    args = parse_args()
    main(args)
