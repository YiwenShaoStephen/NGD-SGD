'''
Training script for CIFAR-10/100
'''
from __future__ import print_function

import argparse
import os
import shutil
import time
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models.cifar as models
import torch.optim.lr_scheduler as lr_scheduler
from ngd import NGD

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
# Datasets
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--optimizer', type=str, default='sgd',
                    help='optimizer type')
parser.add_argument('--scheduler', type=str, default='step',
                    help='Learning rate scheduler')
parser.add_argument('--milestones', type=int, nargs='+', default=[150, 225],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '--pf', default=10, type=int,
                    help='print frequency')
parser.add_argument('--beta1', default=0.9, type=float,
                    help='adam beta1')
parser.add_argument('--beta2', default=0.999, type=float, help='adam beta2')
# Checkpoints
parser.add_argument('--exp', default='exp/resnet20', type=str, metavar='PATH',
                    help='path to save checkpoint and log (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: resnet18)')
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--block-name', type=str, default='BasicBlock',
                    help='the building block for Resnet and Preresnet: BasicBlock, Bottleneck (default: Basicblock for cifar10/cifar100)')
parser.add_argument('--cardinality', type=int, default=8,
                    help='Model cardinality (group).')
parser.add_argument('--widen-factor', type=int, default=4,
                    help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--growthRate', type=int, default=12,
                    help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=2,
                    help='Compression Rate (theta) for DenseNet.')
# Miscs
parser.add_argument('--manualSeed', type=int, default=0, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--update-period', type=int, default=4)

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Validate dataset
assert args.dataset == 'cifar10' or args.dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100.'

# Use CUDA
use_cuda = torch.cuda.is_available()

# Random seed
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy


def main():
    global best_acc
    writer = SummaryWriter(args.exp)
    print('Saving model and logs to {}'.format(args.exp))
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    # Data
    print('==> Preparing dataset %s' % args.dataset)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    if args.dataset == 'cifar10':
        dataloader = datasets.CIFAR10
        num_classes = 10
    else:
        dataloader = datasets.CIFAR100
        num_classes = 100

    trainset = dataloader(root='./data', train=True,
                          download=True, transform=transform_train)
    trainloader = data.DataLoader(
        trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)

    testset = dataloader(root='./data', train=False,
                         download=False, transform=transform_test)
    testloader = data.DataLoader(
        testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    # Model
    print("==> creating model '{}'".format(args.arch))
    if args.arch.startswith('resnext'):
        model = models.__dict__[args.arch](
            cardinality=args.cardinality,
            num_classes=num_classes,
            depth=args.depth,
            widen_factor=args.widen_factor,
            dropRate=args.drop,
        )
    elif args.arch.startswith('densenet'):
        model = models.__dict__[args.arch](
            num_classes=num_classes,
            depth=args.depth,
            growthRate=args.growthRate,
            compressionRate=args.compressionRate,
            dropRate=args.drop,
        )
    elif args.arch.startswith('wrn'):
        model = models.__dict__[args.arch](
            num_classes=num_classes,
            depth=args.depth,
            widen_factor=args.widen_factor,
            dropRate=args.drop,
        )
    elif args.arch.endswith('resnet'):
        model = models.__dict__[args.arch](
            block_name=args.block_name,
            num_classes=num_classes,
            depth=args.depth,
        )
    else:
        model = models.__dict__[args.arch](num_classes=num_classes)

    #model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel()
                                           for p in model.parameters()) / 1000000.0))
    criterion = nn.CrossEntropyLoss()
    if args.optimizer == 'ngd':
        optimizer = NGD(model.parameters(), lr=args.lr,
                        momentum=args.momentum, weight_decay=args.weight_decay,
                        update_period=args.update_period)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
                              momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr,
                               betas=(args.beta1, args.beta2), eps=1e-08, weight_decay=args.weight_decay,
                               amsgrad=False)
    elif args.optimizer == 'amsgrad':
        optimizer = optim.Adam(model.parameters(), lr=args.lr,
                               betas=(args.beta1, args.beta2), eps=1e-08, weight_decay=args.weight_decay,
                               amsgrad=True)
    elif args.optimizer == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=args.lr, lr_decay=0, weight_decay=args.weight_decay,
                                  initial_accumulator_value=0)

    # Resume
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(
            args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    # learning rate scheduler
    if args.scheduler == 'step':
        scheduler = lr_scheduler.MultiStepLR(
            optimizer, milestones=args.milestones, gamma=args.gamma, last_epoch=start_epoch - 1)
    elif args.scheduler == 'exp':
        gamma = args.gamma ** (1.0 / args.epochs)  # final_lr = init_lr * gamma
        scheduler = lr_scheduler.ExponentialLR(
            optimizer, gamma=gamma, last_epoch=start_epoch - 1)

    # Train and val
    for epoch in range(start_epoch, args.epochs):
        scheduler.step()

        train_loss, train_acc = train(
            trainloader, model, criterion, optimizer, writer, epoch, use_cuda)
        test_loss, test_acc = test(
            testloader, model, criterion, writer, epoch, use_cuda)

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'acc': test_acc,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best, exp=args.exp)

    print('Best acc:')
    print(best_acc)


def train(trainloader, model, criterion, optimizer, writer, epoch, use_cuda, norm_order=2):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()

    lr = optimizer.param_groups[0]['lr']
    writer.add_scalar('lr', lr, epoch)
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        prec1 = accuracy(outputs.data, targets.data, topk=(1,))[0]
        losses.update(loss.detach().item(), inputs.size(0))
        top1.update(prec1.detach().item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print progress
        if batch_idx % args.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec {prec.val:.4f} ({prec.avg:.4f})\t'.format(
                      epoch, batch_idx, len(trainloader), batch_time=batch_time,
                      loss=losses, prec=top1))
    # log to TensorBoard
    writer.add_scalar('train_loss', losses.avg, epoch)
    writer.add_scalar('train_prec', top1.avg, epoch)

    return (losses.avg, top1.avg)


def test(testloader, model, criterion, writer, epoch, use_cuda, norm_order=2):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1 = accuracy(outputs.data, targets.data, topk=(1,))[0]
        losses.update(loss.detach().item(), inputs.size(0))
        top1.update(prec1.detach().item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print progress
        if batch_idx % args.print_freq == 0:
            print('Validation: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec {prec.val:.4f} ({prec.avg:.4f})\t'.format(
                      epoch, batch_idx, len(testloader), batch_time=batch_time,
                      loss=losses, prec=top1))
    # log to TensorBoard
    writer.add_scalar('valid_loss', losses.avg, epoch)
    writer.add_scalar('valid_prec', top1.avg, epoch)

    return (losses.avg, top1.avg)


class AverageMeter(object):
    """Computes and stores the average and current value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def save_checkpoint(state, is_best, exp='exp', filename='checkpoint.pth.tar'):
    filepath = os.path.join(exp, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(
            exp, 'model_best.pth.tar'))


if __name__ == '__main__':
    main()
