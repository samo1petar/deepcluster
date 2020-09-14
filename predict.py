import argparse
import json
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import faiss

from clustering.utils import match_predictions_and_cls
from lib.utils import clean_predictions, save_predictions_imgs, save_predictions_json
import models
from models.utils import compute_features, restore


def parse_args():

    parser = argparse.ArgumentParser(description='PyTorch Implementation of DeepCluster in Python3')

    parser.add_argument('--data_predict', type=str, help='Path to directory to predict')
    parser.add_argument('--cluster_index', type=str, default='cluster_index', help='path to clustering index file (default: cluster_index)')
    parser.add_argument('--checkpoint', default='', type=str, metavar='PATH', help='path to checkpoint (default: None)')
    parser.add_argument('--classes', type=str, default='classes.json', help='path to json file with classes description (default: classes.json)')
    parser.add_argument('--save_dir', type=str, default='', help='path to dir to save results (default: <don\'t save>)')
    parser.add_argument('--arch', default='vgg16', type=str, choices=['alexnet', 'vgg16'], help='CNN architecture. alexnet or vgg16 (default: vgg16)')
    parser.add_argument('--sobel', action='store_true', help='Sobel filtering')
    parser.add_argument('--cluster_alg', default='KMeans', type=str, choices=['KMeans', 'PIC'], help='clustering algorithm (default: Kmeans)')
    parser.add_argument('--batch', default=256, type=int, help='mini-batch size (default: 256)')
    parser.add_argument('--top_n', type=int, default=1, help='top N for accuracy (default: 1)')

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

    restore(model, args.checkpoint)

    # preprocessing of data
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    tra = [transforms.Resize(256),
           transforms.CenterCrop(224),
           transforms.ToTensor(),
           normalize]

    index = faiss.read_index(args.cluster_index)

    with open('available_classes.json', 'r') as f:
        available_classes = json.load(f)

    dataset_predict = datasets.ImageFolder(args.data_predict, transform=transforms.Compose(tra))
    dataloader_predict = torch.utils.data.DataLoader(
        dataset_predict,
        batch_size=args.batch,
        num_workers=1,
        pin_memory=True,
    )
    print ('Computing features...')
    features_predict = compute_features(dataloader_predict, model, len(dataset_predict), args.batch)

    print ('Classifying features...')
    D, I = index.search(features_predict, 10)

    I = clean_predictions(I, available_classes)

    predictions = match_predictions_and_cls(I, dataset_predict.imgs, available_classes)

    for x in predictions:
        print(predictions[x]['real_cls'], predictions[x]['cls_str'], x.rsplit('/', 2)[1] + '_' + x.rsplit('/', 2)[2])

    if args.save_dir:
        print ('Saving images and predictions json')
        save_predictions_imgs(predictions, args.save_dir)
        save_predictions_json(predictions, args.save_dir)

    correct = 0
    wrong = 0

    for x in predictions:
        if predictions[x]['real_cls'] in available_classes.values():
            if predictions[x]['real_cls'] in predictions[x]['cls_str'][:args.top_n]:
                correct += 1
            else:
                wrong += 1
    print ('Accuracy for known classes is: ', correct / (correct + wrong))


if __name__ == '__main__':
    args = parse_args()
    main(args)
