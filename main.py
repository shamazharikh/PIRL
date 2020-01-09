import argparse
import os
import sys
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.models as models

import datasets
from transforms import JigSaw, Rotate
from Models.PIRL import PIRLModel, PIRLLoss
from utils import AverageMeter
from MemoryBank.LinearAverage import LinearAverage



model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PIRL CIFAR100 Training')

parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--download', action='store_true', 
                    help='Flag to download data')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--save-freq', '-s', default=1, type=int,
                    metavar='N', help='model saving frequency (default: every epoch)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--recompute', action='store_true', 
                    help='Recompute memory bank for validation')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--low-dim', default=128, type=int,
                    metavar='D', help='feature dimension')
parser.add_argument('--nce-t', default=0.07, type=float, 
                    metavar='T', help='temperature parameter for softmax')
parser.add_argument('--nce-m', default=0.5, type=float,
                    help='momentum for non-parametric updates')
parser.add_argument('--loss_lambda', default=0.1, type=float,
                    help='weight of NCE for transformed input')
parser.add_argument('--iter_size', default=1, type=int,
                    help='caffe style iter size')

def train(epoch, model, memorybank, criterion, trainloader, optimizer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()   

    # switch to train mode
    model.train()

    end = time.time()
    optimizer.zero_grad()
    for i, (image, transformed_image, index) in enumerate(trainloader):
        data_time.update(time.time() - end)
        index = index.cuda(non_blocking=True)

        # compute output
        image_features, transformed_image_features = model(image, transformed_image)
        transformed_output, output, _ = memorybank(image_features, transformed_image_features, index)
        loss = (criterion(output, index) + criterion(transformed_output, index)) / args.iter_size

        loss.backward()
        # measure accuracy and record loss
        losses.update(loss.item() * args.iter_size, image.size(0))
        if (i+1) % args.iter_size == 0:
            # compute gradient and do SGD step
            optimizer.step()
            optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   epoch, i, len(trainloader), batch_time=batch_time,
                   data_time=data_time, loss=losses))

def validate(epoch, model, memorybank, criterion, trainloader, valloader, recompute_memory=0):
    model.eval()
    net_time = AverageMeter()
    cls_time = AverageMeter()
    losses = AverageMeter()
    correct = 0.
    total = 0
    testsize = valloader.dataset.__len__()
    trainFeatures = memorybank.memory.t()
    if recompute_memory:
        transform_backup = trainloader.dataset.transform
        trainloader.dataset.transform = valloader.dataset.transform
        temploader = torch.utils.data.DataLoader(trainloader.dataset, batch_size=100, shuffle=False, num_workers=1)
        for batch_idx, (images, _, index) in enumerate(temploader):
            index = index.cuda(non_blocking=True)
            batchSize = images.size(0)
            features = model(images)
            trainFeatures[:, batch_idx*batchSize:batch_idx*batchSize+batchSize] = features.data.t()
        trainloader.dataset.transform = transform_backup
    
    end = time.time()
    with torch.no_grad():
        for batch_idx, (images, transformed_images, indexes) in enumerate(valloader):
            indexes = indexes.cuda(non_blocking=True)
            batchSize = images.size(0)
            targets = torch.zeros(batchSize, dtype=torch.long)
            features, transformed_features = model(images, transformed_images)
            transformed_output, _, output_similarity = memorybank(features, transformed_features)
            similarity_vectors = torch.cat([output_similarity, transformed_output], dim=-1)
            val_loss = criterion(similarity_vectors, targets)
            losses.update(val_loss.item(), batchSize)
            net_time.update(time.time() - end)
            end = time.time()

            total += targets.size(0)
            correct += similarity_vectors.argmax(dim=-1).eq(targets.data).sum().item()
            
            cls_time.update(time.time() - end)
            end = time.time()

            print('Test [{}/{}]\t'
                  'Net Time {net_time.val:.3f} ({net_time.avg:.3f})\t'
                  'Cls Time {cls_time.val:.3f} ({cls_time.avg:.3f})\t'
                  'Top1: {:.2f}'.format(
                  total, testsize, correct*100./total, net_time=net_time, cls_time=cls_time))

    return correct/total

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 100 epochs"""
    lr = args.lr
    if epoch < 120:
        lr = args.lr
    elif epoch >= 120 and epoch < 160:
        lr = args.lr * 0.1
    else:
        lr = args.lr * 0.01
    #lr = args.lr * (0.1 ** (epoch // 100))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

best_prec1 = 0

def main():
    global args, best_prec1
    args = parser.parse_args()

    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = PIRLModel(args.arch)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = PIRLModel(args.arch, pretrained=False)
    
    if not args.distributed:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)

    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[1.0, 1.0, 1.0])

    train_dataset = datasets.MNISTInstance(
        root=args.data,
        download=True,
        transform=transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2,1.)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            Rotate(return_image=False),
            normalize,
            JigSaw((3, 3))
        ]))
    
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_dataset = datasets.MNISTInstance(
        root=args.data, 
        download=True,
        transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            JigSaw((3, 3)),
            normalize,
        ]))
    valloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )

    ndata = train_dataset.__len__()
    memorybank = LinearAverage(args.low_dim, ndata, args.nce_t, args.nce_m).cuda()
    criterion = PIRLLoss(loss_lambda=args.loss_lambda).cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            # best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            memorybank = checkpoint['memorybank']
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # if args.evaluate:
    #     kNN(0, model, lemniscate, train_loader, val_loader, 200, args.nce_t)
    #     return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(epoch, model, memorybank, criterion, trainloader, optimizer)

        # evaluate on validation set
        _ = validate(epoch, model, memorybank, criterion, trainloader, valloader, recompute_memory=args.recompute)

        # remember best prec@1 and save checkpoint
        # is_best = accuracy > best_accuracy
        # best_prec1 = max(accuracy, best_accuracy)
        if epoch % args.save_freq == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'memorybank': memorybank,
                'optimizer' : optimizer.state_dict(),
            }, epoch+1)
            
    # # evaluate KNN after last epoch
    # kNN(0, model, lemniscate, train_loader, val_loader, 200, args.nce_t)

if __name__ == '__main__':
    main()