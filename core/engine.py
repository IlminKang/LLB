import os
import math
import sys
import time
import torch
import torch.distributed as dist
import utils
from enum import Enum


def train_epoch(model, optimizer, data_loader, device, epoch, print_freq=10, scaler=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':3.3f')
    ce_loss = AverageMeter('CE', ':3.3f')
    ent_loss = AverageMeter('ENT', ':3.3f')
    top1 = AverageMeter('Acc@1', ':3.2f')
    progress = ProgressMeter(
        len(data_loader),
        [batch_time, data_time, losses, top1, ce_loss, ent_loss, ft_ratio],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    end = time.time()
    for i, (data) in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images, cls = data
        b = images.shape[0]

        images = images.to(device)
        cls = cls.to(device)

        # compute output
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            outputs, loss, nv_loss, entropy = model(images, labels=cls)

        acc1, acc5 = accuracy(outputs, cls, topk=(1,5))

        # record
        losses.update(loss.item(), b)
        top1.update(acc1[0], b)
        ce_loss.update(nv_loss, b)
        ent_loss.update(entropy, b)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            progress.display(i + 1)

    progress.display_summary()

    return losses.avg, top1.avg, ce_loss.avg, ent_loss.avg


def eval_epoch(model, data_loader, device, epoch, print_freq=10):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':3.6f')
    ce_loss = AverageMeter('CE', ':3.3f')
    ent_loss = AverageMeter('ENT', ':3.3f')
    top1 = AverageMeter('Acc@1', ':3.2f')
    progress = ProgressMeter(
        len(data_loader),
        [batch_time, losses, top1, ce_loss, ent_loss, ft_ratio],
        prefix="Test: [{}]".format(epoch))

    # switch to train mode
    model.eval()
    end = time.time()
    with torch.no_grad():
        for i, (data) in enumerate(data_loader):
            images, cls = data
            b = images.shape[0]

            images = images.to(device)
            cls = cls.to(device)

            outputs, loss, nv_loss, entropy = model(images, labels=cls)

            acc1, acc5 = accuracy(outputs, cls, topk=(1, 5))

            # record
            losses.update(loss.item(), b)
            top1.update(acc1[0], b)
            ce_loss.update(nv_loss, b)
            ent_loss.update(entropy, b)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                progress.display(i + 1)

    losses.all_reduce()
    top1.all_reduce()
    ce_loss.all_reduce()
    ent_loss.all_reduce()

    progress.display_summary()

    return losses.avg, top1.avg, ce_loss.avg, ent_loss.avg

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
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

    def all_reduce(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def display_summary(self):
        entries = [">>>"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        if batch_size == 0:
            batch_size = 0.00001

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res