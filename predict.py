import argparse
from IPython import embed
import json
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import faiss

import clustering
from clustering.utils import cluster_assign, match_predictions_and_cls
from lib.data import transform_to_dict
from lib.utils import accuracy, check_classes, visualize, get_available_classes, clean_predictions
import models
from models.utils import compute_features, restore


def parse_args():

    parser = argparse.ArgumentParser(description='PyTorch Implementation of DeepCluster in Python3')

    parser.add_argument('--data_centroids', type=str, help='Path to dataset from which centroids will be taken.')
    parser.add_argument('--data_predict', type=str, help='Path to predict')
    parser.add_argument('--arch', default='vgg16', type=str, choices=['alexnet', 'vgg16'], help='CNN architecture')
    parser.add_argument('--sobel', action='store_true', help='Sobel filtering')
    parser.add_argument('--nmb_cluster', '--k', type=int, default=10000, help='number of cluster for k-means (default: 10000)')
    parser.add_argument('--cluster_alg', default='KMeans', type=str, choices=['KMeans', 'PIC'], help='clustering algorithm (default: Kmeans)')
    parser.add_argument('--batch', default=256, type=int, help='mini-batch size (default: 256)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to checkpoint (default: None)')

    return parser.parse_args()


def main(args):

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

    '''
    dataset = datasets.ImageFolder(args.data_centroids, transform=transforms.Compose(tra))

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

    faiss.write_index(faiss.index_gpu_to_cpu(cluster_alg.index), 'cluster_index')

    embed()

    train_dataset = cluster_assign(cluster_alg.images_lists, dataset.imgs)

    data = transform_to_dict(train_dataset.imgs)
    
    # save_data(data)
    available_classes = get_available_classes(data, thr_perc=0.3)
    with open('available_classes.json', 'w') as f:
        json.dump(available_classes, f, indent=4)

    accuracy(data)

    exit()

    '''

    # save_dir = visualize(data)
    # check_classes(save_dir, train_dataset.imgs)

    index = faiss.read_index('cluster_index')

    with open('available_classes.json', 'r') as f:
        available_classes = json.load(f)

    dataset_predict = datasets.ImageFolder(args.data_predict, transform=transforms.Compose(tra))
    dataloader_predict = torch.utils.data.DataLoader(
        dataset_predict,
        batch_size=args.batch,
        num_workers=1,
        pin_memory=True,
    )
    features_predict = compute_features(dataloader_predict, model, len(dataset_predict), args.batch)

    embed()

    D, I = index.search(features_predict, 10)

    I = clean_predictions(I, available_classes)

    predictions = match_predictions_and_cls(I, dataset_predict.imgs, available_classes)

    all_classes = set(available_classes.keys())

    correct = 0
    wrong = 0
    for x in predictions:
        if predictions[x]['real_cls'] in all_classes:
            if predictions[x]['real_cls'] in predictions[x]['cls_str'][:1]:
                correct += 1
            else:
                wrong += 1
    print ('Accuracy is: ', correct / (correct + wrong))

    for x in predictions:
        print(predictions[x]['real_cls'], predictions[x]['cls_str'])

    embed()


if __name__ == '__main__':
    args = parse_args()
    main(args)
