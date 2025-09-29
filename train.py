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
from models.snndarts_retrain.LEAStereo import LEAStereo, LEAStereo_ImageNet
import fitlog
import torch.nn.functional as F

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
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--time_stamp', type=int, default=6, help='time_stamp of SNN')
parser.add_argument('--TinyImageNet200', action='store_true', default=False, help='TinyImageNet 100')
parser.add_argument('--CIFAR10', action='store_true', default=False, help='CIFAR 10')
parser.add_argument('--CIFAR100', action='store_true', default=False, help='CIFAR 100')
parser.add_argument('--DVSCIFAR10', action='store_true', default=False, help='DVS CIFAR 100')
parser.add_argument('--dataset', type=str, default='CIFAR10', help='Name of dataset')
parser.add_argument('--author', type=str, default='Ours', help='Author')
args = parser.parse_args()

args.save = 'eval_{}_{}-{}-{}'.format(args.dataset, args.author, args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

fitlog_debug = True 
if fitlog_debug:
    fitlog.debug()
else:
    fitlog.commit(__file__,fit_msg=args.experiment_description)
    log_path = "logs"
    if not os.path.isdir(log_path):
        os.mkdir(log_path)
    fitlog.set_log_dir(log_path)
    fitlog.create_log_folder()
    fitlog.add_hyper(args)
    # opt.fitlog_path = os.path.join(log_path,fitlog.get_log_folder())


if args.CIFAR10 or args.DVSCIFAR10:
    CIFAR_CLASSES = 10
elif args.CIFAR100:
    CIFAR_CLASSES = 100
elif args.TinyImageNet200:
    CIFAR_CLASSES = 200
else:
    raise NotImplementedError


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

    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    cudnn.enabled=True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
        
    logging.info("args = %s", args)
    leastereo = LEAStereo(init_channels=3, CIFAR_CLASSES=CIFAR_CLASSES, args=args)
    model = leastereo
    model = model.cuda()

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
    criterion = CrossEntropyLabelSmooth(CIFAR_CLASSES, 0.1)
    # criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
        )
    # optimizer = torch.optim.Adam(
    #     model.parameters(),
    #     args.learning_rate,
    #     weight_decay=args.weight_decay
    #     )

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
 
    if args.TinyImageNet200:
        train_data, valid_data = utils.build_tiny200(args.data)
        train_queue = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size,
            pin_memory=True, num_workers=2)   # 9000

        valid_queue = torch.utils.data.DataLoader(
            valid_data, batch_size=20,
            pin_memory=True, num_workers=2)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
    best_acc = 0
    time_stamp = args.time_stamp
    for epoch in range(args.epochs):
        logging.info('epoch %d lr %e', epoch, scheduler.get_last_lr()[0])
        # model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

        train_acc, train_obj = train(train_queue, model, criterion, optimizer)
        logging.info('train_acc %f', train_acc)
        fitlog.add_metric(train_acc,epoch,'train_top1')
        fitlog.add_metric(train_obj,epoch,'train_loss')

        valid_acc, valid_obj = infer(valid_queue, model, criterion)
        logging.info('valid_acc %f', valid_acc)
        fitlog.add_metric(valid_acc,epoch,'valid_top1')
        fitlog.add_metric(valid_obj,epoch,'valid_loss')
        
        if valid_acc >= best_acc:
            best_acc = valid_acc
            utils.save(model, os.path.join(args.save, 'weights.pt'))
        
        scheduler.step()
fitlog.finish()


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
        # logits, logits_aux, a, b, _, _ = model(input, args.time_stamp, param)
        logits, logits_aux = model(input,  param)
        loss = criterion(logits, target)
        if args.auxiliary:
            loss_aux = criterion(logits_aux, target)
            loss += args.auxiliary_weight*loss_aux
        loss.backward()

        
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        
        optimizer.step()


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

        # logits, _, a, b, _, _ = model(input, args.time_stamp, param)
        logits, logits_aux = model(input,  param)
        loss = criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


if __name__ == '__main__':
  main() 

