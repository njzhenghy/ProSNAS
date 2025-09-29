import os
import numpy as np
import torch
import shutil
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder, MNIST
import random
import torch.nn as nn
import torchvision.datasets as datasets
def calculate_weight2(alphas,gammas,
                    normalized_alphas,
                    normalized_gammas, tau=0.1):
    
    weight = torch.prod(alphas ** normalized_alphas)*torch.prod(gammas ** normalized_gammas)
    return weight+1e-8

def calculate_weight(alphas,betas,gammas,
                    normalized_alphas,normalized_betas,
                    normalized_gammas, tau=0.1):
    
    k =alphas.shape[1]
    fir=-k*torch.log(torch.sum(alphas/((normalized_alphas+1e-8)**tau), dim=-1))
    sec= torch.sum(torch.log(alphas/((normalized_alphas+1e-8)**(tau+1))), dim=-1)
    loss=torch.sum(fir+sec)

    k =gammas.shape[1]
    fir=-k*torch.log(torch.sum(gammas/((normalized_gammas+1e-8)**tau), dim=-1))
    sec= torch.sum(torch.log(gammas/((normalized_gammas+1e-8)**(tau+1))), dim=-1)
    loss += torch.sum(fir+sec)

    for layer in range(len(betas)):     
        if layer == 0:
            k = normalized_betas[layer].shape[1]
            fir=-k*torch.log(torch.sum(betas[layer][0]/((normalized_betas[layer][0]+1e-8)**tau), dim=-1))
            sec= torch.sum(torch.log(betas[layer][0]/((normalized_betas[layer][0]+1e-8)**(tau+1))), dim=-1)
            loss+=torch.sum(fir+sec)

        elif layer == 1:
            k = normalized_betas[layer].shape[1]
            fir=-k*torch.log(torch.sum(betas[layer][0:2]/((normalized_betas[layer][0:2]+1e-8)**tau), dim=-1))
            sec= torch.sum(torch.log(betas[layer][0:2]/((normalized_betas[layer][0:2]+1e-8)**(tau+1))), dim=-1)
            loss+=torch.sum(fir+sec)

        else:
            k = normalized_betas[layer].shape[1]
            fir=-k*torch.log(torch.sum(betas[layer][0:3]/((normalized_betas[layer][0:3]+1e-8)**tau), dim=-1))
            sec= torch.sum(torch.log(betas[layer][0:3]/((normalized_betas[layer][0:3]+1e-8)**(tau+1))), dim=-1)
            loss+=torch.sum(fir+sec)
    return loss


def get_grads2(model, params_with_grad, d_p_list, input_tr, target_tr, input_search, target_search, criterion, args, tau=0.1):
    logits_tr = model(input_tr)
    loss_tr = criterion(logits_tr, target_tr)
    # a = torch.autograd.grad(loss_tr, model.module.feature.parameters())
    loss_tr.backward() 
    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    for param in model.module.feature.parameters():
        if param.grad is not None:
            d_p_list.append(param.grad.add_(param, alpha=args.weight_decay))
            params_with_grad.append(param)  

    with torch.no_grad():
        logits_search = model(input_search)
        loss_val = criterion(logits_search, target_search)
    return  (logits_tr, loss_tr), (logits_search, loss_val)


def get_grads(model, params_with_grad, d_p_list, input_tr, target_tr, input_search, target_search, criterion, args, tau=0.1):
    logits_tr = model(input_tr)
    loss_tr = criterion(logits_tr, target_tr)
    loss_tr.backward() 
    for param in model.module.feature.parameters():
        if param.grad is not None:
            d_p_list.append(param.grad+args.weight_decay*param)
            params_with_grad.append(param)  

    alphas=torch.clone(model.module.feature.alphas.data)
    alphas.requires_grad=True
    gammas=torch.clone(model.module.feature.gammas.data)
    gammas.requires_grad=True
    betas=torch.clone(model.module.feature.betas.data)
    betas.requires_grad=True
    normalized_alphas=torch.clone(model.module.feature.normalized_alphas)
    normalized_gammas=torch.clone(model.module.feature.normalized_gammas)
    normalized_betas=torch.clone(model.module.feature.normalized_betas)
    
    k =alphas.shape[1]
    fir=-k*torch.log(torch.sum(alphas/((normalized_alphas+1e-8)**tau), dim=-1))
    sec= torch.sum(torch.log(alphas/((normalized_alphas+1e-8)**(tau+1))), dim=-1)
    loss=torch.sum(fir+sec)
    nabla_log_pXA = torch.autograd.grad(loss,alphas)[0]

    k =gammas.shape[1]
    fir=-k*torch.log(torch.sum(gammas/((normalized_gammas+1e-8)**tau), dim=-1))
    sec= torch.sum(torch.log(gammas/((normalized_gammas+1e-8)**(tau+1))), dim=-1)
    loss=torch.sum(fir+sec)
    nabla_log_pYB = torch.autograd.grad(loss,gammas)[0]

    loss = 0
    for layer in range(len(betas)):     
        if layer == 0:
            k = normalized_betas[layer].shape[1]
            fir=-k*torch.log(torch.sum(betas[layer][0]/((normalized_betas[layer][0]+1e-8)**tau), dim=-1))
            sec= torch.sum(torch.log(betas[layer][0]/((normalized_betas[layer][0]+1e-8)**(tau+1))), dim=-1)
            loss+=torch.sum(fir+sec)

        elif layer == 1:
            k = normalized_betas[layer].shape[1]
            fir=-k*torch.log(torch.sum(betas[layer][0:2]/((normalized_betas[layer][0:2]+1e-8)**tau), dim=-1))
            sec= torch.sum(torch.log(betas[layer][0:2]/((normalized_betas[layer][0:2]+1e-8)**(tau+1))), dim=-1)
            loss+=torch.sum(fir+sec)

        else:
            k = normalized_betas[layer].shape[1]
            fir=-k*torch.log(torch.sum(betas[layer][0:3]/((normalized_betas[layer][0:3]+1e-8)**tau), dim=-1))
            sec= torch.sum(torch.log(betas[layer][0:3]/((normalized_betas[layer][0:3]+1e-8)**(tau+1))), dim=-1)
            loss+=torch.sum(fir+sec)
    nabla_log_pZC = torch.autograd.grad(loss, betas)[0]

    with torch.no_grad():
        logits_search = model(input_search)
        loss_val = criterion(logits_search, target_search)
        nabla_loss_val_A = nabla_log_pXA * loss_val + args.arch_weight_decay * alphas
        nabla_loss_val_B = nabla_log_pYB * loss_val + args.arch_weight_decay * gammas
        nabla_loss_val_C = nabla_log_pZC * loss_val + args.arch_weight_decay * betas
    return (nabla_loss_val_A, nabla_loss_val_B, nabla_loss_val_C), (logits_tr, loss_tr), (logits_search, loss_val)

class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    # correct_k = correct[:k].view(-1).float().sum(0)
    correct_k = correct[:k].reshape(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar10(args):
  CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
  CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])
  if args.cutout:
    train_transform.transforms.append(Cutout(args.cutout_length))

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
  return train_transform, valid_transform


def _data_transforms_cifar100(args):
  CIFAR_MEAN = [0.5071, 0.4867, 0.4408]
  CIFAR_STD = [0.2675, 0.2565, 0.2761]

  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])
  if args.cutout:
    train_transform.transforms.append(Cutout(args.cutout_length))

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
  return train_transform, valid_transform

def build_tiny200(args):
    num_label = 200
    normalize = transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821))
    transform_train = transforms.Compose(
        [transforms.RandomResizedCrop(32), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
         normalize, ])
    transform_test = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), normalize, ])
    trainset = datasets.ImageFolder(root='../data/tiny-imagenet-200/train/', transform=transform_train)
    testset =  datasets.ImageFolder(root='../data/tiny-imagenet-200/test/', transform=transform_test)
    # train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True)
    # test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, pin_memory=True)
    return trainset, testset

class DVSCifar10(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.resize = transforms.Resize(size=(48, 48))  # 48 48
        self.tensorx = transforms.ToTensor()
        self.imgx = transforms.ToPILImage()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        data, target = torch.load(self.root + '/{}.pt'.format(index))
        new_data = []
        for t in range(data.size(0)):
            # new_data.append(self.tensorx(self.resize(self.imgx(data[..., t]))))
            new_data.append(self.tensorx(self.resize(self.imgx(data[t]))))
        data = torch.stack(new_data, dim=0)
        if self.transform is not None:
            flip = random.random() > 0.5
            if flip:
                data = torch.flip(data, dims=(3,))
            off1 = random.randint(-5, 5)
            off2 = random.randint(-5, 5)
            data = torch.roll(data, shifts=(off1, off2), dims=(2, 3))

        if self.target_transform is not None:
            target = self.target_transform(target)
        return data, target.long().squeeze(-1)

    def __len__(self):
        return len(os.listdir(self.root))


def build_dvscifar(path):
    train_path = path + '/train'
    val_path = path + '/test'
    train_dataset = DVSCifar10(root=train_path, transform=False)
    val_dataset = DVSCifar10(root=val_path)

    return train_dataset, val_dataset


def build_imagenet():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    root = '/data_smr/dataset/ImageNet'
    train_root = os.path.join(root,'train')
    val_root = os.path.join(root,'val')
    train_dataset = ImageFolder(
        train_root,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    )
    val_dataset = ImageFolder(
        val_root,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    )
    return train_dataset, val_dataset



def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


def save_checkpoint(state, is_best, save):
  filename = os.path.join(save, 'checkpoint.pth.tar')
  torch.save(state, filename)
  if is_best:
    best_filename = os.path.join(save, 'model_best.pth.tar')
    shutil.copyfile(filename, best_filename)


def save(model, model_path):
  torch.save(model.state_dict(), model_path)


def load(model, model_path):
  model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1.-drop_prob
    mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
    x.div_(keep_prob)
    x.mul_(mask)
  return x


def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.mkdir(path)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    os.mkdir(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)

