# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import os
import time
from sklearn.metrics.cluster import normalized_mutual_info_score
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import clustering
from clustering.utils import cluster_assign, arrange_clustering
import models
from models.utils import compute_features, restore
from lib.utils import AverageMeter, Logger, UnifLabelSampler


lr=0.05
wd=-5
k=2000
workers=6
batch=16
start_epoch = 0
epochs = 600
checkpoints = 25000
exp = '/home/david/Projects/deepcluster/experiments/exp_1'
arch = 'vgg16'
verbose = True
momentum = 0.9
resume = '/home/david/Projects/deepcluster/deepcluster_models/vgg16/checkpoint.pth.tar'
data = '/home/david/Datasets/Flickr/Flickr25K'
reassign = 1


def main():
    # # fix random seeds
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed_all(args.seed)
    # np.random.seed(args.seed)

    # CNN
    if verbose:
        print('Architecture: {}'.format(arch))
    model = models.__dict__[arch](sobel=True)
    fd = int(model.top_layer.weight.size()[1])
    model.top_layer = None
    model.features = torch.nn.DataParallel(model.features)
    model.cuda()
    cudnn.benchmark = True

    # create optimizer
    optimizer = torch.optim.SGD(
        filter(lambda x: x.requires_grad, model.parameters()),
        lr=lr,
        momentum=momentum,
        weight_decay=10**wd,
    )

    # define loss function
    criterion = nn.CrossEntropyLoss().cuda()

    restore(model, resume)

    # creating checkpoint repo
    exp_check = os.path.join(exp, 'checkpoints')
    if not os.path.isdir(exp_check):
        os.makedirs(exp_check)

    # creating cluster assignments log
    cluster_log = Logger(os.path.join(exp, 'clusters'))

    # preprocessing of data
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    tra = [transforms.Resize(256),
           transforms.CenterCrop(224),
           transforms.ToTensor(),
           normalize]

    # load the data
    end = time.time()
    dataset = datasets.ImageFolder(data, transform=transforms.Compose(tra))
    if verbose:
        print('Load dataset: {0:.2f} s'.format(time.time() - end))

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch,
                                             num_workers=workers,
                                             pin_memory=True)

    algs = {
        'KMeans': clustering.KMeans,
        'PIC': clustering.PIC,
    }
    cluster_alg = algs['KMeans'](500)

    # training convnet with cluster_alg
    for epoch in range(start_epoch, epochs):
        end = time.time()

        # remove head
        model.top_layer = None
        model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])

        # get the features for the whole dataset
        features = compute_features(dataloader, model, len(dataset), batch)

        # cluster the features
        if verbose:
            print('Cluster the features')
        clustering_loss = cluster_alg.cluster(features, verbose=verbose)

        # assign pseudo-labels
        if verbose:
            print('Assign pseudo labels')
        train_dataset = cluster_assign(cluster_alg.images_lists,
                                                  dataset.imgs)

        # uniformly sample per target
        sampler = UnifLabelSampler(int(reassign * len(train_dataset)),
                                   cluster_alg.images_lists)

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch,
            num_workers=workers,
            sampler=sampler,
            pin_memory=True,
        )

        # set last fully connected layer
        mlp = list(model.classifier.children())
        mlp.append(nn.ReLU(inplace=True).cuda())
        model.classifier = nn.Sequential(*mlp)
        model.top_layer = nn.Linear(fd, len(cluster_alg.images_lists))
        model.top_layer.weight.data.normal_(0, 0.01)
        model.top_layer.bias.data.zero_()
        model.top_layer.cuda()

        # train network with clusters as pseudo-labels
        end = time.time()
        loss = train(train_dataloader, model, criterion, optimizer, epoch)

        # print log
        if verbose:
            print('###### Epoch [{0}] ###### \n'
                  'Time: {1:.3f} s\n'
                  'Clustering loss: {2:.3f} \n'
                  'ConvNet loss: {3:.3f}'
                  .format(epoch, time.time() - end, clustering_loss, loss))
            try:
                nmi = normalized_mutual_info_score(
                    arrange_clustering(cluster_alg.images_lists),
                    arrange_clustering(cluster_log.data[-1])
                )
                print('NMI against previous assignment: {0:.3f}'.format(nmi))
            except IndexError:
                pass
            print('####################### \n')
        # save running checkpoint
        torch.save({'epoch': epoch + 1,
                    'arch': arch,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict()},
                   os.path.join(exp, 'checkpoint.pth.tar'))

        # save cluster assignments
        cluster_log.log(cluster_alg.images_lists)


def train(loader, model, crit, opt, epoch):
    """Training of the CNN.
        Args:
            loader (torch.utils.data.DataLoader): Data loader
            model (nn.Module): CNN
            crit (torch.nn): loss
            opt (torch.optim.SGD): optimizer for every parameters with True
                                   requires_grad in model except top layer
            epoch (int)
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    data_time = AverageMeter()
    forward_time = AverageMeter()
    backward_time = AverageMeter()

    # switch to train mode
    model.train()

    # create an optimizer for the last fc layer
    optimizer_tl = torch.optim.SGD(
        model.top_layer.parameters(),
        lr=lr,
        weight_decay=10**wd,
    )

    end = time.time()
    for i, (input_tensor, target) in enumerate(loader):
        data_time.update(time.time() - end)

        # save checkpoint
        n = len(loader) * epoch + i
        if n % checkpoints == 0:
            path = os.path.join(
                exp,
                'checkpoints',
                'checkpoint_' + str(n / checkpoints) + '.pth.tar',
            )
            if verbose:
                print('Save checkpoint at: {0}'.format(path))
            torch.save({
                'epoch': epoch + 1,
                'arch': arch,
                'state_dict': model.state_dict(),
                'optimizer' : opt.state_dict()
            }, path)

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input_tensor.cuda())
        target_var = torch.autograd.Variable(target)

        output = model(input_var)
        loss = crit(output, target_var)

        # record loss
        losses.update(loss.data.item(), input_tensor.shape[0])

        # compute gradient and do SGD step
        opt.zero_grad()
        optimizer_tl.zero_grad()
        loss.backward()
        opt.step()
        optimizer_tl.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if verbose and (i % 200) == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data: {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss: {loss.val:.4f} ({loss.avg:.4f})'
                  .format(epoch, i, len(loader), batch_time=batch_time,
                          data_time=data_time, loss=losses))

    return losses.avg

main()