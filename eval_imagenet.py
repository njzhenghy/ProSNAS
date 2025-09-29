import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from models.snndarts_retrain.LEAStereo import LEAStereo, LEAStereo_ImageNet
import fitlog
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
import random

parser = argparse.ArgumentParser("imagenet")
parser.add_argument('--data', type=str, default='../data/Imagenet', help='location of the data corpus')
# parser.add_argument('--data', type=str, default='../data/tiny-imagenet-200', help='location of the data corpus')

parser.add_argument('--batch_size', type=int, default=160, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=10, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--experiment_description', type=str, help='description of experiment')
parser.add_argument('--fea_num_layers', type=int, default=10)
parser.add_argument('--fea_filter_multiplier', type=int, default=36)
parser.add_argument('--fea_block_multiplier', type=int, default=3)
parser.add_argument('--fea_step', type=int, default=3)
parser.add_argument('--net_arch_fea', default=None, type=str)
parser.add_argument('--cell_arch_fea', default=None, type=str)
parser.add_argument('--use_DGS', default=False, type=bool)
# parser.add_argument('--gpu', type=int, default=0, help='gpu device id')


args = parser.parse_args()


CIFAR_CLASSES=1000

class CrossEntropyLabelSmooth(nn.Module):

  def __init__(self, num_classes, epsilon):
    super(CrossEntropyLabelSmooth, self).__init__()
    self.num_classes = num_classes
    self.epsilon = epsilon
    self.logsoftmax = nn.LogSoftmax(dim=1)

  def forward(self, inputs, targets):
    log_probs = self.logsoftmax(inputs)
    targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
    targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
    loss = (-targets * log_probs).mean(0).sum()
    return loss



def main():

    # torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    cudnn.enabled=True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    leastereo = LEAStereo_ImageNet(init_channels=3, CIFAR_CLASSES=CIFAR_CLASSES, args=args)
    model = leastereo
    model = model.cuda()
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load("/home/wanli/Code/ProSNAS_new/eval-EXP-20240424-201838/weights.pt"))


    criterion = CrossEntropyLabelSmooth(CIFAR_CLASSES, 0.1)
    # criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
        )

    traindir = os.path.join(args.data, 'train')
    validdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_data = dset.ImageFolder(
        traindir,
        transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.2),
        transforms.ToTensor(),
        normalize,
        ]))
    valid_data = dset.ImageFolder(
        validdir,
        transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
        ]))

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=32)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=32)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, 0.97)
    best_acc = 0
    valid_acc, valid_obj = infer(valid_queue, model, criterion)
    print('valid_acc %f'% (valid_acc))

def convert_str2index(this_str, is_b=False, is_wight=False, is_cell=False):
    if is_wight:
        this_str = this_str.split('.')[:-1] + ['conv1','weight']
    elif is_b:
        this_str = this_str.split('.')[:-1] + ['snn_optimal','b']
    elif is_cell:
        this_str = this_str.split('.')[:4]
    else:
        this_str = this_str.split('.')
    new_index = []
    for i, value in enumerate(this_str):
        if value.isnumeric():
            new_index.append('[%s]'%value)
        else:
            if i == 0:
                new_index.append(value)
            else:
                new_index.append('.'+value)
    return ''.join(new_index)
    

def clamp(this, v_low=1, v_high=6):
    if this <= v_low:
        return v_low
    elif this >= v_high:
        return v_high
    else:
        return this

tem_b_all = list()

def train(train_queue, model, criterion, optimizer):
    param = {'mode':'optimal'}
    
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.train()

    for step, (input, target) in enumerate(train_queue):
        input = Variable(input).cuda()
        target = Variable(target).cuda()

        optimizer.zero_grad()
        logits, logits_aux = model(input, param)
        loss = criterion(logits, target)
        if args.auxiliary:
            loss_aux = criterion(logits_aux, target)
            loss += args.auxiliary_weight*loss_aux
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        
        optimizer.step()
        with torch.no_grad():
            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)


        if step % args.report_freq == 0:
            logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    param = {'mode':'optimal'}
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    for step, (input, target) in enumerate(valid_queue):
        input = Variable(input, volatile=True).cuda()
        target = Variable(target, volatile=True).cuda()

        with torch.no_grad():
            logits, _ = model(input, param)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            print('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


if __name__ == '__main__':
    main() 

