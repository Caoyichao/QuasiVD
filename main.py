# -*- coding: utf-8 -*-

import shutil
import os
import argparse
import glob
import random
import time
import torch

from model import Model
from get_mobilev3 import FastDet
from ctdet import CTDetDataset
from loss import CtdetLoss, CtclsLoss


parser = argparse.ArgumentParser()
parser.add_argument('--arch', default='mobilenet', help='FFNet | mobilenet')
parser.add_argument('--num_class', type=int, default=1)
parser.add_argument('--root', default='E:/python/Two-frame-detection/Data')
parser.add_argument('--MODEL_PATH', default='data')
parser.add_argument('--log_path', default='log')
parser.add_argument('--BATCH', type=int, default=16)
parser.add_argument('--EPOCHS', type=int, default=50)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--seed', type=int, default=317)
parser.add_argument('--down_ratio', type=int, default=4)
parser.add_argument('--device', default=torch.device('cuda'))
parser.add_argument('--resume', action='store_true')
parser.add_argument('--hm_weight', type=float, default=1, help='loss weight for keypoint heatmaps.')
parser.add_argument('--wh_weight', type=float, default=0.1, help='loss weight for bounding box size.')
parser.add_argument('--off_weight', type=float, default=1, help='loss weight for keypoint local offsets.')
parser.add_argument('--mask_weight', type=float, default=0.1)
parser.add_argument('--print_freq', type=int, default=50)
parser.add_argument('--flip', type=float, default=0.5, help='probability of applying flip augmentation.')

args = parser.parse_args()


min_loss = float('inf')
def main():
    '''
    项目的超参
    '''
    global min_loss, args

    # CUDA set
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    # torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # 获取全量数据
    train_samples, val_samples = split_train_val(args.root)
    train_len = len(train_samples)
    val_len = len(val_samples)
    print('train len is %d, val len is %d'%(train_len, val_len))

    # Create data
    args.train = parse_label(train_samples)
    args.val = parse_label(val_samples)


    train_loader = torch.utils.data.DataLoader(
        CTDetDataset(args, 'train'),
        batch_size=args.BATCH,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = torch.utils.data.DataLoader(
        CTDetDataset(args, 'val'),
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Build network
    print('Creating model: {}'.format(args.arch))
    if args.arch == 'FFNet':
        model = Model(args.num_class)
    elif args.arch == 'mobilenet':
        checkpoint_path = 'data/mbv3_large.old.pth.tar'
        model = FastDet(args.num_class, out_channels=80, output_shape=512 // 4, pretrained=checkpoint_path)
    else:
        raise ('No Model!!')
    # torch.cuda.set_device(1)
    model = torch.nn.DataParallel(model, device_ids=[0]).cuda()
    # model = model.cuda()
    
    input = torch.zeros([1,3,512,512])
    input2 = torch.zeros([1,1,128,128])
    output, mask = model(input,input2)

    # 判断路径是否存在
    if not os.path.exists(args.MODEL_PATH):
        os.makedirs(args.MODEL_PATH)


    start_epoch = 0
    if args.resume:
        print('==> Load model from checkpoint..')
        checkpoint = torch.load(os.path.join(args.MODEL_PATH, '{}_checkpoint.pth.tar'.format(args.arch)))
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
        args.lr = 0.001
        start_epoch = 31


    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    # optimizer = torch.optim.AdamW(model.parameters(), args.lr)
    criterion = CtdetLoss(args)
    # criterion = CtclsLoss(args)

    time_stample = time.strftime("%Y-%m-%d-%H", time.localtime())
    log_name = '{}/{}_{}.txt'.format(args.log_path, args.arch, time_stample)
    log = open(log_name, 'a')
    for epoch in range(start_epoch, args.EPOCHS):
        adjust_learning_rate(optimizer, epoch, args)

        train(train_loader, model, criterion, optimizer, epoch, args, log)

        # evaluate on validation set
        loss = validate(val_loader, model, criterion, args, log)
        is_min = loss < min_loss
        if is_min:
            min_loss = loss
        print('\nCurrent loss: %.5f, Minimum loss: %.5f\n' % (loss, min_loss))
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'min_loss': min_loss,
        }, is_min)
    log.close()


def train(train_loader, model, criterion, optimizer, epoch, args, log):
    losses = AverageMeter()
    true_top1 = AverageMeter()
    # switch to train mode
    model.train()
    for i, batch in enumerate(train_loader):
        for k in batch:
            batch[k] = batch[k].to(device=args.device, non_blocking=True)

        output, mask = model(batch['input1'], batch['input2'])
        # true_prec1 = true_accuracy(output.data, batch['cls'])[0]
        # true_top1.update(true_prec1.item(), batch['input'].size(0))
        loss, loss_info = criterion(output, mask, batch)
        loss = loss.mean()
        losses.update(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % args.print_freq == 0:
            output = ('epoch: [{0}][{1}]/[{2}]\t lr: {lr:.5f}\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                # 'Top1 {true_top1.val:.4f} ({true_top1.avg:.4f})'
                .format(
                epoch, i, len(train_loader), loss=losses, lr=optimizer.param_groups[-1]['lr'], true_top1=true_top1
            ))
            log.write(output + '\n')
            print(output)
        elif i == len(train_loader)-1:
            output = ('epoch: [{0}][{1}]/[{2}]\t lr: {lr:.5f}\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      # 'Top1 {true_top1.val:.4f} ({true_top1.avg:.4f})'
                      .format(
                epoch, i, len(train_loader), loss=losses, lr=optimizer.param_groups[-1]['lr'], true_top1=true_top1
            ))
            log.write(output + '\n')
            print(output)


def validate(val_loader, model, criterion, args, log):
    losses = AverageMeter()
    true_top1 = AverageMeter()
    # switch to val mode
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            for k in batch:
                batch[k] = batch[k].to(device=args.device, non_blocking=True)
            output, mask = model(batch['input1'], batch['input2'])
            # true_prec1 = true_accuracy(output.data, batch['cls'])[0]
            # true_top1.update(true_prec1.item(), batch['input'].size(0))
            loss, loss_info = criterion(output, mask, batch)
            loss = loss.mean()
            losses.update(loss)

            if i % args.print_freq == 0:
                output = ('Test: [{0}]/[{1}]\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          # 'Top1 {true_top1.val:.4f} ({true_top1.avg:.4f})'
                          .format(
                    i, len(val_loader), loss=losses, true_top1=true_top1
                ))
                log.write(output + '\n')
                print(output)
            elif i == len(val_loader)-1:
                output = ('Test: [{0}]/[{1}]\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          # 'Top1 {true_top1.val:.4f} ({true_top1.avg:.4f})'
                          .format(
                    i, len(val_loader), loss=losses, true_top1=true_top1
                ))
                log.write(output + '\n')
                print(output)
    return losses.avg


def adjust_learning_rate(optimizer, epoch, args):
    if epoch == 100:
        lr = 0.0003
    else:
        lr = args.lr * (0.1 ** (epoch // 30))
    optimizer.param_groups[0]['lr'] = lr


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, '%s/%s_checkpoint.pth.tar' % (args.MODEL_PATH, args.arch))
    if is_best:
        print('save the best model!!!')
        shutil.copyfile('%s/%s_checkpoint.pth.tar' % (args.MODEL_PATH, args.arch),'%s/%s_best.pth.tar' % (args.MODEL_PATH, args.arch))


def true_accuracy(output, target, topk=(1,)):
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


class AverageMeter(object):
    """Computes and stores the average and current value"""
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


def parse_label(samples):
    res = []
    for sub_folder in samples:
        sub_dict = {}
        images = sorted(glob.glob(sub_folder + '/*.jpg'))
        sub_dict['image_path'] = images
        labels = sorted(glob.glob(sub_folder + '/*.txt'))
        sub_labs = []
        for lab in labels:
            sub_lab = []
            with open(lab, 'r') as f:
                items = f.readlines()
            for item in items:
                info = item.strip().split(' ')
                info = [int(i) for i in info]
                sub_lab.append(info)
            sub_labs.append(sub_lab)
        sub_dict['label'] = sub_labs
        res.append(sub_dict)
    return res


def split_train_val(root):
    random.seed(42)
    neg_folder = 'Neg'
    pos_folder = 'Pos'
    ratio = 0.95
    neg_samples = [os.path.join(root, neg_folder, f) for f in os.listdir(os.path.join(root, neg_folder))]
    pos_samples = [os.path.join(root, pos_folder, f) for f in os.listdir(os.path.join(root, pos_folder))]
    samples = neg_samples + pos_samples
    random.shuffle(samples)
    number = len(samples)
    node = int(number * ratio)
    train_samples = samples[:node]
    val_samples = samples[node:]
    return train_samples, val_samples


if __name__ == '__main__':
    main()

