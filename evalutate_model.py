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
import random
from torch.autograd import Variable
from models.snndarts_retrain.LEAStereo import LEAStereo
import fitlog
import torch.nn.functional as F
from models.snndarts_search.build_model import AutoStereo

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
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
parser.add_argument('--fea_filter_multiplier', type=int, default=36)
parser.add_argument('--fea_block_multiplier', type=int, default=3)
parser.add_argument('--fea_step', type=int, default=3)
parser.add_argument('--net_arch_fea', default=None, type=str)
parser.add_argument('--cell_arch_fea', default=None, type=str)
parser.add_argument('--use_DGS', default=False, type=bool)
parser.add_argument('--gpu', type=int, default=1, help='gpu device id')

parser.add_argument('--CIFAR10', action='store_true', default=False, help='CIFAR 10')
parser.add_argument('--CIFAR100', action='store_true', default=False, help='CIFAR 100')
parser.add_argument('--DVSCIFAR10', action='store_true', default=False, help='DVS CIFAR 100')

parser.add_argument('--resume', type=str, default='./search-EXP-20231109-184830/epoch_3.pt',
                        help='put the path to resuming file if needed')




args = parser.parse_args()



if args.CIFAR10 or args.DVSCIFAR10:
    CIFAR_CLASSES = 10
elif args.CIFAR100:
    CIFAR_CLASSES = 100
else:
    raise NotImplementedError


def main():

    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    cudnn.enabled=True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # model = torch.load(args.resume)
    # model = model.cuda()
    leastereo = LEAStereo(init_channels=3, CIFAR_CLASSES=CIFAR_CLASSES, args=args)
    model = leastereo
    model = model.cuda()
    model.load_state_dict(torch.load("/home/wanli/Code/ProSNAS_new/eval-EXP-20240511-214649/weights.pt"))


    criterion = nn.CrossEntropyLoss()

    if args.CIFAR10:
        train_transform, valid_transform = utils._data_transforms_cifar10(args)
        train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
        valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)
        train_queue = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)
        valid_queue = torch.utils.data.DataLoader(
            valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)
    elif args.CIFAR100:
        train_transform, valid_transform = utils._data_transforms_cifar100(args)
        train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
        valid_data = dset.CIFAR100(root=args.data, train=False, download=True, transform=valid_transform)
        train_queue = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)
        valid_queue = torch.utils.data.DataLoader(
            valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)
    elif args.DVSCIFAR10:
        train_data, valid_data = utils.build_dvscifar(path=args.data)
        train_queue = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size,
            pin_memory=True, num_workers=2)   # 9000

        valid_queue = torch.utils.data.DataLoader(
            valid_data, batch_size=20,
            pin_memory=True, num_workers=2)

    valid_acc, valid_obj = infer(valid_queue, model, criterion)
    print('valid_acc %f'% (valid_acc))



def infer(valid_queue, model, criterion):
    param = {'mode':'optimal'}
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()
    param['snn_output'] = 'mem'
    A_s = 0
    T_s = 0
    A_spike_lists = [0 for i in range(9)]
    T_spike_lists = [0 for i in range(9)]
    for step, (input, target) in enumerate(valid_queue):
        input = Variable(input, volatile=True).cuda()
        target = Variable(target, volatile=True).cuda()

        logits, _, A_spikes, T_spikes, A_spike_list, T_spike_list = model(input, param)
        loss = criterion(logits, target)
        A_s += A_spikes
        T_s += T_spikes
        for idx,(A,T) in enumerate(zip(A_spike_list, T_spike_list)):
            A_spike_lists[idx] = A_spike_lists[idx]+A
            T_spike_lists[idx] = T_spike_lists[idx]+T
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            print('valid %03d %e %f %f'%( step, objs.avg, top1.avg, top5.avg))
    print('activate rate %f', A_s/T_s)
    for idx,(A,T) in enumerate(zip(A_spike_lists, T_spike_lists)):
        r = A/T
        print('layer %d activate rate %f'%(idx, r))        
    return top1.avg, objs.avg


if __name__ == '__main__':
  main() 

