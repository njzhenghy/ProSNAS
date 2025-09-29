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
from models.snndarts_retrain.LEAStereo import LEAStereo, LEAStereo2
import fitlog
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data/dvsges', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=10, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.05, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=100, help='report frequency')
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

args = parser.parse_args()
args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
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
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture

T = 1
CIFAR_CLASSES = 10

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
    
        
    logging.info("args = %s", args)
    leastereo = LEAStereo(3, 10, args=args)
    model = leastereo
    model = model.cuda()
    model = torch.nn.DataParallel(model)
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
    criterion = nn.CrossEntropyLoss()
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

    transform = transforms.Compose([
        # transforms.Resize((224, 224)),  # 调整图像大小
        transforms.ToTensor(),          # 转换为Tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化
    ])

    # 加载 SVHN 数据集
    train_dataset = datasets.SVHN(root='../data', split='train', download=True, transform=transform)
    test_dataset = datasets.SVHN(root='../data', split='test', download=True, transform=transform)

    train_queue = DataLoader(dataset=train_dataset, batch_size=256, shuffle=True, num_workers=4)
    valid_queue = DataLoader(dataset=test_dataset, batch_size=256, shuffle=False, num_workers=4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
    best_acc = 0
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
            # utils.save(model, os.path.join(args.save, 'weights.pt'))
            torch.save(model, 'spiking_svhn_searched.pth')
        
        scheduler.step()

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
        logits, logits_aux, a, b, _, _  = model(input, T, param)
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
    A_s = 0
    T_s = 0
    for step, (input, target) in enumerate(valid_queue):
        with torch.no_grad():
            input = Variable(input).cuda()
            target = Variable(target).cuda()

            logits, _, A_spikes, T_spikes, _, _ = model(input, T, param)
            loss = criterion(logits, target)
            A_s += A_spikes
            T_s += T_spikes
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

