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
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
# from architect import Architect
from models.snndarts_search.build_model import AutoStereo, AutoStereo2
import fitlog
from MyOptimizer.projstorm import ProjSTORM
from MyOptimizer.storm import STORM
from MyOptimizer.adastorm import AdaStorm
from MyOptimizer.projadastorm import ProAdaStorm
import random
import math
from utils import calculate_weight, calculate_weight2
from torch.utils.data import random_split
from torch.utils.data import DataLoader, Subset, TensorDataset

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data/dvsges', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
# parser.add_argument('--batch_size', type=int, default=256, help='batch size')
# parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=450, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=10, help='report frequency')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
# parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--init_channels', type=int, default=3, help='num of init channels')
# parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
# parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--CIFAR10', action='store_true', default=False, help='CIFAR 10')
parser.add_argument('--CIFAR100', action='store_true', default=False, help='CIFAR 100')
parser.add_argument('--DVSCIFAR10', action='store_true', default=False, help='DVS CIFAR 100')
parser.add_argument('--arch_learning_rate', type=float, default=0.1, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--experiment_description', type=str, help='description of experiment')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')


######### LEStereo params ##################
parser.add_argument('--fea_num_layers', type=int, default=8)
parser.add_argument('--fea_filter_multiplier', type=int, default=8)
parser.add_argument('--fea_block_multiplier', type=int, default=3)
parser.add_argument('--fea_step', type=int, default=3)
parser.add_argument('--net_arch_fea', default=None, type=str)
parser.add_argument('--cell_arch_fea', default=None, type=str)
parser.add_argument('--drop_rate', default=0.5, type=float)
parser.add_argument('--fitlog_path',type=str,default='debug')

######### search params ##################

args = parser.parse_args()

args.save = './results/vr3/vr2-{}-search-{}-{}'.format(args.seed,args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

from spikingjelly.datasets.dvs128_gesture import DVS128Gesture


def load_model_parameters(model, parameters_old):
    start = 0
    for param in model.feature.parameters():
        offset = len(torch.reshape(param, [-1]))
        param.data = torch.reshape(parameters_old[start:start + offset], param.shape)
        start = start + offset
    return model

T = 8

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
       
    logging.info("args = %s", args)

    criterion = nn.CrossEntropyLoss()
    model = AutoStereo2(2, 11, args = args)
    model = model.cuda()
    # model = torch.nn.DataParallel(model)

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer =AdaStorm(
        model.feature.parameters(),
        args.learning_rate,
        betas=(args.momentum, 0.999))
    
    # optimizer =STORM(
    #     model.feature.parameters(),
    #     args.learning_rate,
    #     momentum=args.momentum)


    train_data = DVS128Gesture(root=args.data, train=True, data_type='frame', frames_number=T, split_by='number')
    test_set = DVS128Gesture(root=args.data, train=False, data_type='frame', frames_number=T, split_by='number')

    train_data, valid_data = random_split(train_data, [0.6, 0.4])
    num_train = len(train_data)
    iter_num = math.ceil(num_train/args.batch_size)
    num_val = len(valid_data)
    val_batch = int(num_val/iter_num)
    logging.info(num_train)
    logging.info(num_val)
    # train_queue = torch.utils.data.DataLoader(
    #         train_data, batch_size=args.batch_size, 
    #         sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
    #         pin_memory=True, num_workers=16)   # 25000

    # valid_queue = torch.utils.data.DataLoader(
    #     train_data, batch_size=val_batch,
    #     sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
    #     pin_memory=True, num_workers=16)
    train_queue = DataLoader(train_data, batch_size=args.batch_size, 
                            shuffle=True, pin_memory=True, num_workers=16)
    valid_queue = DataLoader(valid_data, batch_size=val_batch,  
                            shuffle=True, pin_memory=True, num_workers=16)

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, float(args.epochs), eta_min=args.learning_rate_min)
    logging.info('pre training')
    
    logging.info('searching')
    model.feature.sample()
    
    input_tr, target_tr = next(iter(train_queue))
    input_tr = Variable(input_tr, requires_grad=False).to(torch.cuda.current_device())
    target_tr = Variable(target_tr, requires_grad=False).to(torch.cuda.current_device())
    input_search, target_search = next(iter(valid_queue))
    input_search = Variable(input_search, requires_grad=False).to(torch.cuda.current_device())
    target_search = Variable(target_search, requires_grad=False).to(torch.cuda.current_device())

    logits_tr = model(input_tr, T)
    loss_tr = criterion(logits_tr, target_tr)#+args.weight_decay*torch.norm(params)**2
    optimizer.zero_grad()
    loss_tr.backward()
    for param in model.feature.parameters():
        if param.grad is not None:
            param.grad.add_(param.data, alpha=args.weight_decay)
    
    
    tau = 0.1
    arch_param=[]
    alphas=torch.clone(model.feature.alphas.data)
    alphas.requires_grad=True
    gammas=torch.clone(model.feature.gammas.data)
    gammas.requires_grad=True
    normalized_alphas=torch.clone(model.feature.normalized_alphas)
    normalized_gammas=torch.clone(model.feature.normalized_gammas)
    arch_param.append(alphas)
    arch_param.append(gammas)
    architect_optimizer = ProAdaStorm(arch_param,
                                lr=args.arch_learning_rate, 
                                betas=(args.momentum, 0.999))
    
    # architect_optimizer = ProjSTORM(arch_param,
    #                             lr=args.arch_learning_rate, 
    #                             momentum=args.momentum)
    
    with torch.no_grad():
        X = normalized_alphas
        A = arch_param[0]
        X_tmp = (X - 0.5) * 2
        C = 1 / (torch.sum(X * A, dim=-1)+1e-8)
        nabla_log_pXA = torch.mul(X_tmp, C.reshape((A.shape[0], 1)))    
        
        Y = normalized_gammas
        B = arch_param[1]
        Y_tmp = (Y - 0.5) * 2
        C = 1 / (torch.sum(Y * B, dim=-1)+1e-8)
        nabla_log_pYB = torch.mul(Y_tmp, C.reshape((B.shape[0],  1))) 
        with torch.no_grad():
            logits_search = model(input_search, T)
            loss_val = criterion(logits_search, target_search)
    arch_param[0].grad = nabla_log_pXA* loss_val 
    arch_param[1].grad = nabla_log_pYB* loss_val 
    d_arp_list = []
    for p in arch_param:
        d_arp_list.append(p.grad)
    with torch.no_grad():
        torch._foreach_add_(d_arp_list, arch_param, alpha=args.arch_weight_decay)
    k=0
    d_p_list_old=None
    d_arp_list_old=None
    pi = 1
    for epoch in range(args.epochs):
        # lr = scheduler.get_last_lr()[0]
        logging.info('epoch %d ', epoch)

        # training
        objs = utils.AvgrageMeter()
        top1 = utils.AvgrageMeter()
        top5 = utils.AvgrageMeter()

        objs_val = utils.AvgrageMeter()
        top1_val = utils.AvgrageMeter()
        top5_val = utils.AvgrageMeter()
        model.train()

        alphas_old1 = torch.clone(arch_param[0])
        gammas_old1 = torch.clone(arch_param[1])  
        normalized_alphas_old1=torch.clone(model.feature.normalized_alphas)
        normalized_gammas_old1=torch.clone(model.feature.normalized_gammas)

        logging.info(model.feature.alphas)
        logging.info(model.feature.gammas)
        logging.info(model.feature.normalized_alphas)
        logging.info(model.feature.normalized_gammas)

        for step, (input_tr, target_tr) in enumerate(train_queue):
            n = input_tr.size(0)
            etak = 1/ np.power(10000 + k, 1 / 3)
            etak = np.clip(etak, a_min=0, a_max=1)
            # etak = 1
            k+=1
            parameters_old = torch.clone(
                torch.unsqueeze(torch.cat([torch.reshape(param, [-1]) for param in model.feature.parameters()]), 1))
            alphas_old = torch.clone(arch_param[0])
            gammas_old = torch.clone(arch_param[1])  
            normalized_alphas_old=torch.clone(model.feature.normalized_alphas)
            normalized_gammas_old=torch.clone(model.feature.normalized_gammas)
            
            optimizer.step(etak,d_p_list_old)
            architect_optimizer.step(etak, pi, d_arp_list_old)
            optimizer.zero_grad()
            architect_optimizer.zero_grad()

            parameters_new = torch.clone(
                torch.unsqueeze(torch.cat([torch.reshape(param, [-1]) for param in model.feature.parameters()]), 1))
            alphas_new = torch.clone(arch_param[0])
            gammas_new = torch.clone(arch_param[1]) 
            model.feature.alphas.data = alphas_new
            model.feature.gammas.data = gammas_new 

            input_tr = Variable(input_tr, requires_grad=False).to(torch.cuda.current_device())
            target_tr = Variable(target_tr, requires_grad=False).to(torch.cuda.current_device())
            input_search, target_search = next(iter(valid_queue))
            input_search = Variable(input_search, requires_grad=False).to(torch.cuda.current_device())
            target_search = Variable(target_search, requires_grad=False).to(torch.cuda.current_device())
            model.feature.sample()
            normalized_alphas_new=torch.clone(model.feature.normalized_alphas)
            normalized_gammas_new=torch.clone(model.feature.normalized_gammas)
            
            model = load_model_parameters(model, parameters_old)
            
            arch_param[0].data = alphas_old
            arch_param[1].data = gammas_old 
            with torch.no_grad():
                X = normalized_alphas_new
                A = arch_param[0]
                X_tmp = (X - 0.5) * 2
                C = 1 / (torch.sum(X * A, dim=-1)+1e-8)
                nabla_log_pXA = torch.mul(X_tmp, C.reshape((A.shape[0], 1)))    
                
                Y = normalized_gammas_new
                B = arch_param[1]
                Y_tmp = (Y - 0.5) * 2
                C = 1 / (torch.sum(Y * B, dim=-1)+1e-8)
                nabla_log_pYB = torch.mul(Y_tmp, C.reshape((B.shape[0],  1))) 
                with torch.no_grad():
                    logits_search = model(input_search, T)
                    loss_val = criterion(logits_search, target_search)
            arch_param[0].grad = nabla_log_pXA* loss_val + args.arch_weight_decay*arch_param[0].data
            arch_param[1].grad = nabla_log_pYB* loss_val + args.arch_weight_decay*arch_param[1].data
            d_arp_list_old = []
            for p in arch_param:
                d_arp_list_old.append(torch.clone(p.grad))
                
            # model.feature.alphas.data = alphas_old
            # model.feature.gammas.data = gammas_old
            model.feature.normalized_alphas.data = normalized_alphas_old
            model.feature.normalized_gammas.data = normalized_gammas_old
            
            logits_tr = model(input_tr, T)
            loss_tr = criterion(logits_tr, target_tr)#+args.weight_decay*torch.norm(params)**2 
            optimizer.zero_grad()
            architect_optimizer.zero_grad()
            loss_tr.backward()
            d_p_list_old = []
            for param in model.feature.parameters():
                if param.grad is not None:
                    param.grad.add_(param.data, alpha=args.weight_decay) 
                    d_p_list_old.append(torch.clone(param.grad))
                    
            weight_old = calculate_weight2(alphas_old,gammas_old,
                            normalized_alphas_new,
                            normalized_gammas_new, tau=tau)

            model = load_model_parameters(model, parameters_new)
            model.feature.alphas.data = alphas_new
            model.feature.gammas.data = gammas_new
            model.feature.normalized_alphas.data = normalized_alphas_new
            model.feature.normalized_gammas.data = normalized_gammas_new
            arch_param[0].data = alphas_new
            arch_param[1].data = gammas_new

            logits_tr = model(input_tr, T)
            params = torch.unsqueeze(torch.cat([torch.reshape(param, [-1]) for param in model.feature.parameters()]), 1)
            loss_tr = criterion(logits_tr, target_tr)#+args.weight_decay*torch.norm(params)**2
            optimizer.zero_grad()
            architect_optimizer.zero_grad()
            loss_tr.backward()
            for param in model.feature.parameters():
                if param.grad is not None:
                    param.grad.add_(param.data, alpha=args.weight_decay) 
            prec1, prec5 = utils.accuracy(logits_tr, target_tr, topk=(1, 5))
            objs.update(loss_tr.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)
      
            with torch.no_grad():
                X = normalized_alphas_new
                A = arch_param[0]
                X_tmp = (X - 0.5) * 2
                C = 1 / (torch.sum(X * A, dim=-1)+1e-8)
                nabla_log_pXA = torch.mul(X_tmp, C.reshape((A.shape[0], 1)))    
                
                Y = normalized_gammas_new
                B = arch_param[1]
                Y_tmp = (Y - 0.5) * 2
                C = 1 / (torch.sum(Y * B, dim=-1)+1e-8)
                nabla_log_pYB = torch.mul(Y_tmp, C.reshape((B.shape[0],  1))) 
                with torch.no_grad():
                    logits_search = model(input_search, T)
                    loss_val = criterion(logits_search, target_search)
            arch_param[0].grad = nabla_log_pXA* loss_val 
            arch_param[1].grad = nabla_log_pYB* loss_val 
            d_arp_list = []
            for p in arch_param:
                d_arp_list.append(p.grad)
            with torch.no_grad():
                torch._foreach_add_(d_arp_list, arch_param, alpha=args.arch_weight_decay)
            
            nv = input_search.size(0)
            prec1, prec5 = utils.accuracy(logits_search, target_search, topk=(1, 5))
            objs_val.update(loss_val.item(), nv)
            top1_val.update(prec1.item(), nv)
            top5_val.update(prec5.item(), nv)
            weight = calculate_weight2(alphas_new,gammas_new,
                            normalized_alphas_new,
                            normalized_gammas_new, tau=tau)
            pi = weight_old/weight    
            
            
            if step % args.report_freq == 0:
                up_norm_A = torch.norm(alphas_new-alphas_old,'fro')
                up_norm_C = torch.norm(gammas_new-gammas_old,'fro')
                up_norm_X = torch.norm(normalized_alphas_new-normalized_alphas_old,'fro')
                up_norm_Z = torch.norm(normalized_gammas_new-normalized_gammas_old,'fro')
                # logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
                # logging.info('train %03d %e %f %f val %e %f %f ', step, objs.avg, top1.avg, top5.avg, objs_val.avg, top1_val.avg, top5_val.avg)
                logging.info('train %03d %e %f %f val %e %f %f norm %f %f %f %f ', step, objs.avg, top1.avg, top5.avg, objs_val.avg, top1_val.avg, top5_val.avg,
                             up_norm_A, up_norm_C, up_norm_X, up_norm_Z)
                
            
            # if fitlog_debug:
            #     break
        
        logging.info('train_acc %f', top1.avg)
        # fitlog.add_metric(top1.avg,epoch,'train_top1')
        # fitlog.add_metric(objs.avg,epoch,'train_loss')

        # validation
        valid_acc, valid_obj = infer(valid_queue, model, criterion)
        logging.info('valid_acc %f', valid_acc)
        # utils.load(model, args.resume)
        # torch.save(model, os.path.join(args.save, 'epoch_%s.pt'%epoch))
        utils.save(model, os.path.join(args.save, 'epoch_%s.pt'%epoch))

    # fitlog.finish()



def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            input = Variable(input).to(torch.cuda.current_device())
            target = Variable(target).to(torch.cuda.current_device())

            logits = model(input, T)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            if step % args.report_freq == 0:
                logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
            # if fitlog_debug:
            #     break
    return top1.avg, objs.avg


if __name__ == '__main__':
    main()
