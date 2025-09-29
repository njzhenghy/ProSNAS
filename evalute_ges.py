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
from models.snndarts_retrain.LEAStereo import LEAStereo2
import fitlog
import torch.nn.functional as F
from models.snndarts_search.build_model import AutoStereo

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data/dvsges', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=50, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
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
parser.add_argument('--fea_num_layers', type=int, default=8)
parser.add_argument('--fea_filter_multiplier', type=int, default=48)
parser.add_argument('--fea_block_multiplier', type=int, default=3)
parser.add_argument('--fea_step', type=int, default=3)
parser.add_argument('--net_arch_fea', default=None, type=str)
parser.add_argument('--cell_arch_fea', default=None, type=str)
parser.add_argument('--use_DGS', default=False, type=bool)

parser.add_argument('--resume', type=str, default='./search-EXP-20231109-200215/epoch_3.pt',
                        help='put the path to resuming file if needed')

parser.add_argument('--CIFAR10', action='store_true', default=False, help='CIFAR 10')
parser.add_argument('--CIFAR100', action='store_true', default=False, help='CIFAR 100')
parser.add_argument('--DVSCIFAR10', action='store_true', default=False, help='DVS CIFAR 100')


args = parser.parse_args()


from spikingjelly.datasets.dvs128_gesture import DVS128Gesture


CIFAR_CLASSES = 10
T = 8

def main():

    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled=True
    torch.cuda.manual_seed(args.seed)
    
    model = LEAStereo2(2, 11, args=args)
    model = model.cuda()
    model = torch.nn.DataParallel(model)
    utils.load(model, args.resume)
    # model = torch.nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    train_data = DVS128Gesture(root=args.data, train=True, data_type='frame', frames_number=T, split_by='number')
    test_set = DVS128Gesture(root=args.data, train=False, data_type='frame', frames_number=T, split_by='number')

    train_queue = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=12)
    valid_queue = torch.utils.data.DataLoader(
            test_set, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=12)

    valid_acc, valid_obj = infer(valid_queue, model, criterion)
    print('valid_acc %f'% (valid_acc))



def infer(valid_queue, model, criterion):
    param = {'mode':'optimal'}
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()
    A_s = 0
    T_s = 0
    for step, (input, target) in enumerate(valid_queue):
        with torch.no_grad():
            input = Variable(input, volatile=True).cuda()
            target = Variable(target, volatile=True).cuda()

            logits, _ = model(input, T,  param)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            print('valid %03d %e %f %f'%( step, objs.avg, top1.avg, top5.avg))

    return top1.avg, objs.avg


if __name__ == '__main__':
  main() 

