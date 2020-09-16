import argparse
import os
from random import shuffle
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from clustering.utils import cluster_assign_with_original_labels
import models
from IPython import embed
from models.utils import compute_tensor_features, restore
from lib.utils import AverageMeter, save_losses


def parse_args():

    parser = argparse.ArgumentParser(description='PyTorch Implementation of DeepCluster in Python3')

    parser.add_argument('--data', type=str, help='Path to dataset.')
    parser.add_argument('--arch', type=str, default='vgg16', choices=['alexnet', 'vgg16'], help='CNN architecture')
    parser.add_argument('--sobel', action='store_true', help='Sobel filtering')
    parser.add_argument('--batch', type=int, default=256, help='mini-batch size (default: 256)')
    parser.add_argument('--resume', type=str, default='', metavar='PATH', help='path to checkpoint (default: None)')
    parser.add_argument('--experiment', '--exp', type=str, metavar='PATH', help='path to dir where train will be saved')
    parser.add_argument('--learning_rate', '--lr', type=float, default=0.05, help='learning rate')
    parser.add_argument('--weight_decay', '--wd', type=float, default=-5, help='weight decay')
    parser.add_argument('--workers', type=int, default=6, help='number of data loading workers (default: 4)')
    parser.add_argument('--start_epoch', type=int, default=0, help='manual epoch number (useful on restarts) (default: 0)')
    parser.add_argument('--epochs', type=int, default=200, help='number of total epochs to run (default: 200)')
    parser.add_argument('--checkpoints', type=int, default=250000, help='how many iterations between two checkpoints (default: 25000)')
    parser.add_argument('--verbose', action='store_true', help='chatty')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout percentage in Dropout layers (default: 0.5')
    parser.add_argument('--seed', type=int, default=None, help='random seed (default: None)')


    return parser.parse_args()


def main(args):

    # fix random seeds
    if args.seed:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)

    # CNN
    if args.verbose:
        print('Architecture: {}'.format(args.arch))
    model = models.__dict__[args.arch](sobel=args.sobel)
    fd = int(model.top_layer.weight.size()[1])
    model.top_layer = None
    model.features = torch.nn.DataParallel(model.features)
    model.cuda()
    cudnn.benchmark = True

    # define loss function
    criterion = nn.CrossEntropyLoss().cuda()

    restore(model, args.resume)

    # creating checkpoint repo
    exp_check = os.path.join(args.experiment, 'checkpoints')
    if not os.path.isdir(exp_check):
        os.makedirs(exp_check)

    # preprocessing of data
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    tra = [transforms.Resize(256),
           transforms.CenterCrop(224),
           transforms.ToTensor(),
           normalize]

    # load the data
    end = time.time()
    dataset = datasets.ImageFolder(args.data, transform=transforms.Compose(tra))
    if args.verbose:
        print('Load dataset: {0:.2f} s'.format(time.time() - end))

    imgs = dataset.imgs
    shuffle(imgs)
    train_imgs = imgs[:int(0.8 * len(imgs))]
    test_imgs = imgs[int(0.8 * len(imgs)):]

    train_dataset = cluster_assign_with_original_labels(train_imgs)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch,
        num_workers=args.workers,
        pin_memory=True,
        shuffle=True,
    )

    test_dataset = cluster_assign_with_original_labels(test_imgs)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch,
        num_workers=args.workers,
        pin_memory=True,
        shuffle=True,
    )

    train_features, train_targets = compute_tensor_features(train_dataloader, model, args.batch)
    test_features, test_targets = compute_tensor_features(test_dataloader, model, args.batch)

    top_layer = nn.Sequential(nn.Linear(4096, 4096), nn.Dropout(args.dropout), nn.Linear(4096, 251)) #, nn.Softmax(dim=1))
    top_layer[0].weight.data.normal_(0, 0.01)
    top_layer[0].bias.data.zero_()
    top_layer[2].weight.data.normal_(0, 0.01)
    top_layer[2].bias.data.zero_()
    top_layer.cuda()

    # create an optimizer for the top layer
    optimizer = torch.optim.SGD(
        top_layer.parameters(),
        lr=args.learning_rate,
        weight_decay=10**args.weight_decay,
    )

    train_losses = []
    test_losses = []

    train_losses.append(test(train_features, train_targets, top_layer, criterion, args.start_epoch))
    test_losses.append(test(test_features, test_targets, top_layer, criterion, args.start_epoch))

    for epoch in range(args.start_epoch, args.epochs):

        train_loss = train(train_features, train_targets, top_layer, criterion, optimizer, epoch)
        test_loss = test(test_features, test_targets, top_layer, criterion, epoch, 'Test')
        test(train_features, train_targets, top_layer, criterion, epoch, 'Train')

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        if args.verbose:
            print('###### Epoch [{}] ###### \n'
                  'Train Loss: {:.3f} s\n'
                  'Test Loss: {:.3f}'
                  .format(epoch, train_loss, test_loss))
            print('####################### \n')

        torch.save({'epoch': epoch + 1,
                    'arch': 'top_layer',
                    'state_dict': top_layer.state_dict()},
                   os.path.join(args.experiment, 'checkpoints', 'checkpoint_' + str(epoch) + '.pth.tar'))

    save_losses(train_losses, test_losses, os.path.join(args.experiment, 'loss.json'))


def test(features, targets, model, crit, epoch, text=''):
    losses = AverageMeter()
    model.eval()

    correct = 0
    wrong = 0

    with torch.no_grad():
        size = features.shape[0]
        for i, (input_tensor, target) in enumerate(zip(features, targets)):

            target = target.cuda()
            input_var = torch.autograd.Variable(input_tensor.cuda())
            target_var = torch.autograd.Variable(target)

            output = model(input_var)
            loss = crit(output, target_var)

            losses.update(loss.data.item(), input_tensor.shape[0])

            output_np = output.data.cpu().numpy()
            prediction = np.argmax(output_np, axis=1)
            target_np = target.cpu().numpy()

            correct += np.where(prediction == target_np)[0].shape[0]
            wrong += np.where(prediction != target_np)[0].shape[0]

    print('Epoch: [{0}][{1}/{2}]\t'
          'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
          '{3} Acc: {4}'
          .format(epoch, i, size, text, correct / (correct + wrong), loss=losses))

    return losses.avg


def train(features, targets, model, crit, opt, epoch):
    """Training of the CNN.
        Args:
            features (torch.tensor): [N, batch, cls_num]
            targets (torch.tensor): [N, batch]
            model (nn.Module): CNN
            crit (torch.nn): loss
            opt (torch.optim.SGD): optimizer for every parameters with True
                                   requires_grad in model except top layer
            epoch (int)
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    data_time = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    size = features.shape[0]
    for i, (input_tensor, target) in enumerate(zip(features, targets)):
        data_time.update(time.time() - end)

        # save checkpoint
        n = size * epoch + i
        if n % args.checkpoints == 0:
            path = os.path.join(
                args.experiment,
                'checkpoints',
                'checkpoint_' + str(n / args.checkpoints) + '.pth.tar',
            )
            if args.verbose:
                print('Save checkpoint at: {0}'.format(path))
            torch.save({
                'epoch': epoch + 1,
                'arch': 'top_layer',
                'state_dict': model.state_dict(),
                'optimizer' : opt.state_dict()
            }, path)

        target = target.cuda()
        input_var = torch.autograd.Variable(input_tensor.cuda())
        target_var = torch.autograd.Variable(target)

        output = model(input_var)
        loss = crit(output, target_var)

        # record loss
        losses.update(loss.data.item(), input_tensor.shape[0])

        # compute gradient and do SGD step
        opt.zero_grad()
        loss.backward()
        opt.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.verbose and (i % 200) == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data: {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss: {loss.val:.4f} ({loss.avg:.4f})'
                  .format(epoch, i, size, batch_time=batch_time,
                          data_time=data_time, loss=losses))

    return losses.avg


if __name__ == '__main__':
    args = parse_args()
    main(args)