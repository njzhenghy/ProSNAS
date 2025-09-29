import torch.nn as nn
import torch.nn.functional as F
import models.snndarts_search.cell_level_search_2d as cell_level_search
from models.snndarts_search.genotypes_2d import PRIMITIVES, PRIMITIVES_SPIKE
from models.snndarts_search.operations_2d import *
from models.snndarts_search.decoding_formulas import Decoder
import pdb
from torch.distributions.one_hot_categorical import OneHotCategorical
from projection import projectionA, projectionB

Tensor = torch.Tensor
def my_gumbel_softmax(logits: Tensor, tau: float = 1, hard: bool = False, eps: float = 1e-10, dim: int = -1) -> Tensor:
    u = torch.rand_like(logits)
    g = -torch.log(-torch.log(u))
    gumbels=(torch.log(logits)+g)/tau
    y_soft = gumbels.softmax(dim)
    return y_soft

class DispEntropy(nn.Module):
    def __init__(self, maxdisp):
        super(DispEntropy, self).__init__()
        self.softmax = nn.Softmin(dim=1)
        self.maxdisp = maxdisp

    def forward(self, x):
        x = F.interpolate(x, [self.maxdisp, x.size()[3]*3, x.size()[4]*3], mode='trilinear', align_corners=False)
        x = torch.squeeze(x, 1)
        e = torch.sum(-F.softmax(x,dim=1) * F.log_softmax(x,dim=1),1)
        m = 1.0- torch.isnan(e).type(torch.cuda.FloatTensor)
        x = e*m
        x = self.softmax(x)
        return x

class DisparityRegression(nn.Module):
    def __init__(self, maxdisp):
        super(DisparityRegression, self).__init__()
        self.maxdisp = maxdisp

    def forward(self, x):
        assert(x.is_contiguous() == True)
        with torch.cuda.device_of(x):
            disp = torch.reshape(torch.arange(0, self.maxdisp, device=torch.cuda.current_device(), dtype=torch.float32),[1,self.maxdisp,1,1])
            disp = disp.repeat(x.size()[0], 1, x.size()[2], x.size()[3])
            out = torch.sum(x * disp, 1)
        return out

class Disp(nn.Module):
    def __init__(self, maxdisp=192):
        super(Disp, self).__init__()
        self.maxdisp = maxdisp
        self.softmax = nn.Softmin(dim=1)
        self.disparity = DisparityRegression(maxdisp=self.maxdisp)

    def forward(self, x):
        x = F.interpolate(x, [self.maxdisp, 260, 346], mode='trilinear', align_corners=False)
        x = torch.squeeze(x, 1)
        x = self.softmax(x)      
        x = self.disparity(x)
        return x


class AutoFeature(nn.Module):
    def __init__(self, frame_rate, CIFAR_CLASSES, args ,p=0.0):
        super(AutoFeature, self).__init__()
        cell=cell_level_search.Cell
        self.cells = nn.ModuleList()
        self.p = p
        self._num_layers = args.fea_num_layers
        self._step = args.fea_step
        self._block_multiplier = args.fea_block_multiplier
        self._filter_multiplier = args.fea_filter_multiplier
        self._initialize_alphas_betas()
        self.args = args
        f_initial = int(self._filter_multiplier)
        half_f_initial = int(f_initial / 2)
        self._num_end = f_initial * self._block_multiplier

        # self.stem0 = ConvBR(frame_rate, f_initial * self._block_multiplier, kernel_size=3, stride=1, padding=1)
        self.stem0 = SNN_2d(frame_rate, f_initial * self._block_multiplier, kernel_size=3, stride=1, padding=1)

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        # self.classifier = SNN_2d_fc(192, 10)
        self.classifier = nn.Linear(192, CIFAR_CLASSES)

        '''
            cell(step, block, prev_prev, prev_down, prev_same, prev_up, filter_multiplier)

            prev_prev, prev_down etc depend on tiers. If cell is in the first tier, then it won`t have prev_down.
            If cell is in the second tier, prev_down should be filter_multiplier *2, if third, then *4.(filter_multiplier is an absolute number.)
        '''

        
        self.cell1 = cell(self._step, self._block_multiplier, -1,
                        None, f_initial, None,
                        self._filter_multiplier,self.p)
        self.cell2 = cell(self._step, self._block_multiplier, f_initial,
                        None, self._filter_multiplier, None,
                        self._filter_multiplier,self.p)

        self.cell3 = cell(self._step, self._block_multiplier, -1,
                        self._filter_multiplier, None, None,
                        self._filter_multiplier * 2,self.p)
        self.cell4 = cell(self._step, self._block_multiplier, -1,
                        None, self._filter_multiplier * 2, None,
                        self._filter_multiplier * 2,self.p)
        self.cell5 = cell(self._step, self._block_multiplier, self._filter_multiplier * 2,
                        None, self._filter_multiplier * 2, None,
                        self._filter_multiplier * 2,self.p)

        self.cell6 = cell(self._step, self._block_multiplier, -1,
                        self._filter_multiplier * 2, None, None,
                        self._filter_multiplier * 4,self.p)
        self.cell7 = cell(self._step, self._block_multiplier, -1,
                        None, self._filter_multiplier * 4, None,
                        self._filter_multiplier * 4,self.p)

        self.cell8 = cell(self._step, self._block_multiplier, self._filter_multiplier * 4,
                        None, self._filter_multiplier * 4, None,
                        self._filter_multiplier * 4,self.p)


        self.down_sample1 =  ConvBR(self._num_end , self._num_end*2,  3, 2, 1)
        self.down_sample2 =  ConvBR(self._num_end*2 , self._num_end*4,  3, 2, 1)
        self.down_sample3 =  ConvBR(self._num_end*4 , self._num_end*8,  3, 2, 1)
        self.down_sample4 =  ConvBR(self._num_end*8 , self._num_end*8,  1, 1, 0, bn=False, relu=False) 

    def sample(self,tau=0.1):

        sampler_alphas = OneHotCategorical(probs=self.alphas)
        sampler_gammas = OneHotCategorical(probs=self.gammas)
        
        self.normalized_alphas = sampler_alphas.sample().cuda()
        self.normalized_gammas = sampler_gammas.sample().cuda()
    
        

    def forward(self, x, param):
        self.level_3 = []
        self.level_6 = []
        self.level_12 = []
        self.level_24 = []
        # stem0 = self.stem0(x)
        stem0, spike_num = self.stem0(x, param)
        count = 0

        res1, = self.cell1(None, None, stem0, None, self.normalized_alphas, self.normalized_gammas, param)

        res2, = self.cell2(stem0,
                                    None,
                                    res1,
                                    None,
                                    self.normalized_alphas, self.normalized_gammas, param)


        res3, = self.cell3(None,
                                    res2,
                                    None,
                                    None,
                                    self.normalized_alphas, self.normalized_gammas, param)
        


        res4,  = self.cell4(None,
                            None,
                            res3,
                            None,
                            self.normalized_alphas, self.normalized_gammas, param)
        res5, = self.cell5(res3,
                            None,
                            res4,
                            None,
                            self.normalized_alphas, self.normalized_gammas, param)
        res6, = self.cell6(None,
                            res5,
                            None,
                            None,
                            self.normalized_alphas, self.normalized_gammas, param)
        res7, = self.cell7(None,
                            None,
                            res6,
                            None,
                            self.normalized_alphas, self.normalized_gammas, param)
        res8, = self.cell8(res6,
                            None,
                            res7,
                            None,
                            self.normalized_alphas, self.normalized_gammas, param)
        
        # downsample
        result_12 = self.down_sample4(self.down_sample3(res8))
        sum_feature_map = result_12
        
        pooling_out = self.global_pooling(sum_feature_map)
        logits_buf = self.classifier(pooling_out.view(pooling_out.size(0),-1))
        # logits_buf = self.classifier(pooling_out.view(pooling_out.size(0),-1),param)
        return logits_buf


    def _initialize_alphas_betas(self):
        k = sum(1 for i in range(self._step) for n in range(2 + i))
        num_ops = len(PRIMITIVES)
        num_spike = len(PRIMITIVES_SPIKE)

        # alphas = torch.ones(k, num_ops).cuda()/num_ops
        # gammas = torch.ones(k, num_spike).cuda()/num_ops

        alphas = torch.randn(k, num_ops).cuda()
        gammas = torch.randn(k, num_spike).cuda()

        self.alphas = projectionA(alphas)
        self.gammas = projectionA(gammas)
        

        self._arch_parameters = [
            self.alphas,
            self.gammas,
            # self.betas,
        ]
        
    def arch_parameters(self):
        return self._arch_parameters

    # def weight_parameters(self):
    #     return [param for name, param in self.named_parameters() if name not in self._arch_param_names]

    def update_p(self):
        for cell in self.cells:
            cell.p = self.p
            cell.update_p()
    
    def genotype(self):
        decoder = Decoder(self.alphas_cell, self._block_multiplier, self._step)
        return decoder.genotype_decode()

class AutoFeature2(nn.Module):
    def __init__(self, frame_rate, CIFAR_CLASSES, args ,p=0.0):
        super(AutoFeature2, self).__init__()
        cell=cell_level_search.Cell
        self.cells = nn.ModuleList()
        self.p = p
        self._num_layers = args.fea_num_layers
        self._step = args.fea_step
        self._block_multiplier = args.fea_block_multiplier
        self._filter_multiplier = args.fea_filter_multiplier
        self._initialize_alphas_betas()
        self.args = args
        f_initial = int(self._filter_multiplier)
        half_f_initial = int(f_initial / 2)
        self._num_end = f_initial * self._block_multiplier

        # self.stem0 = ConvBR(frame_rate, f_initial * self._block_multiplier, kernel_size=3, stride=1, padding=1)
        self.stem0 = SNN_2d(frame_rate, f_initial * self._block_multiplier, kernel_size=3, stride=1, padding=1)

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        # self.classifier = SNN_2d_fc(192, 10)
        self.classifier = nn.Linear(192, CIFAR_CLASSES)

        '''
            cell(step, block, prev_prev, prev_down, prev_same, prev_up, filter_multiplier)

            prev_prev, prev_down etc depend on tiers. If cell is in the first tier, then it won`t have prev_down.
            If cell is in the second tier, prev_down should be filter_multiplier *2, if third, then *4.(filter_multiplier is an absolute number.)
        '''

        
        self.cell1 = cell(self._step, self._block_multiplier, -1,
                        None, f_initial, None,
                        self._filter_multiplier,self.p)
        self.cell2 = cell(self._step, self._block_multiplier, f_initial,
                        None, self._filter_multiplier, None,
                        self._filter_multiplier,self.p)

        self.cell3 = cell(self._step, self._block_multiplier, -1,
                        self._filter_multiplier, None, None,
                        self._filter_multiplier * 2,self.p)
        self.cell4 = cell(self._step, self._block_multiplier, -1,
                        None, self._filter_multiplier * 2, None,
                        self._filter_multiplier * 2,self.p)
        self.cell5 = cell(self._step, self._block_multiplier, self._filter_multiplier * 2,
                        None, self._filter_multiplier * 2, None,
                        self._filter_multiplier * 2,self.p)

        self.cell6 = cell(self._step, self._block_multiplier, -1,
                        self._filter_multiplier * 2, None, None,
                        self._filter_multiplier * 4,self.p)
        self.cell7 = cell(self._step, self._block_multiplier, -1,
                        None, self._filter_multiplier * 4, None,
                        self._filter_multiplier * 4,self.p)

        self.cell8 = cell(self._step, self._block_multiplier, self._filter_multiplier * 4,
                        None, self._filter_multiplier * 4, None,
                        self._filter_multiplier * 4,self.p)


        self.down_sample1 =  ConvBR(self._num_end , self._num_end*2,  3, 2, 1)
        self.down_sample2 =  ConvBR(self._num_end*2 , self._num_end*4,  3, 2, 1)
        self.down_sample3 =  ConvBR(self._num_end*4 , self._num_end*8,  3, 2, 1)
        self.down_sample4 =  ConvBR(self._num_end*8 , self._num_end*8,  1, 1, 0, bn=False, relu=False) 

    def sample(self,tau=0.1):

        sampler_alphas = OneHotCategorical(probs=self.alphas)
        sampler_gammas = OneHotCategorical(probs=self.gammas)
        
        self.normalized_alphas = sampler_alphas.sample().cuda()
        self.normalized_gammas = sampler_gammas.sample().cuda()
    
        

    def forward(self, x, param):
        self.level_3 = []
        self.level_6 = []
        self.level_12 = []
        self.level_24 = []
        # stem0 = self.stem0(x)
        stem0, spike_num = self.stem0(x, param)
        count = 0

        res1, = self.cell1(None, None, stem0, None, self.normalized_alphas, self.normalized_gammas, param)

        res2, = self.cell2(stem0,
                                    None,
                                    res1,
                                    None,
                                    self.normalized_alphas, self.normalized_gammas, param)


        res3, = self.cell3(None,
                                    res2,
                                    None,
                                    None,
                                    self.normalized_alphas, self.normalized_gammas, param)
        


        res4,  = self.cell4(None,
                            None,
                            res3,
                            None,
                            self.normalized_alphas, self.normalized_gammas, param)
        res5, = self.cell5(res3,
                            None,
                            res4,
                            None,
                            self.normalized_alphas, self.normalized_gammas, param)
        res6, = self.cell6(None,
                            res5,
                            None,
                            None,
                            self.normalized_alphas, self.normalized_gammas, param)
        res7, = self.cell7(None,
                            None,
                            res6,
                            None,
                            self.normalized_alphas, self.normalized_gammas, param)
        res8, = self.cell8(res6,
                            None,
                            res7,
                            None,
                            self.normalized_alphas, self.normalized_gammas, param)
        
        # downsample
        result_12 = self.down_sample4(self.down_sample3(res8))
        sum_feature_map = result_12
        
        pooling_out = self.global_pooling(sum_feature_map)
        logits_buf = self.classifier(pooling_out.view(pooling_out.size(0),-1))
        # logits_buf = self.classifier(pooling_out.view(pooling_out.size(0),-1),param)
        return logits_buf


    def _initialize_alphas_betas(self):
        k = sum(1 for i in range(self._step) for n in range(2 + i))
        num_ops = len(PRIMITIVES)
        num_spike = len(PRIMITIVES_SPIKE)

        # alphas = torch.ones(k, num_ops).cuda()/num_ops
        # gammas = torch.ones(k, num_spike).cuda()/num_ops

        alphas = torch.randn(k, num_ops).cuda()
        gammas = torch.randn(k, num_spike).cuda()

        self.alphas = projectionA(alphas)
        self.gammas = projectionA(gammas)
        

        self._arch_parameters = [
            self.alphas,
            self.gammas,
            # self.betas,
        ]
        
    def arch_parameters(self):
        return self._arch_parameters

    # def weight_parameters(self):
    #     return [param for name, param in self.named_parameters() if name not in self._arch_param_names]

    def update_p(self):
        for cell in self.cells:
            cell.p = self.p
            cell.update_p()
    
    def genotype(self):
        decoder = Decoder(self.alphas_cell, self._block_multiplier, self._step)
        return decoder.genotype_decode()

