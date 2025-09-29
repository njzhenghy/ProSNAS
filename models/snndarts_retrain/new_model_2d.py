import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.snndarts_search.genotypes_2d import PRIMITIVES
from models.snndarts_search.genotypes_2d import Genotype
from models.snndarts_search.operations_2d import *
import torch.nn.functional as F
import numpy as np
import pdb
from models.snndarts_search.neuron import ComplementaryLIFNeuron

decay = 0.2


OPS = {
    'skip_connect': lambda Cin,Cout, stride, signal, s: (CNN(Cin, Cout, kernel_size=3,  stride=stride, padding=1, s=s) if signal == 1 else Identity(Cin, Cout, signal, s=s)),
    'cnn_3x3': lambda Cin,Cout, stride, signal, s: CNN(Cin, Cout, kernel_size=3, stride=stride, padding=1, s=s),
    # 'cnn_5x5': lambda Cin,Cout, stride, signal: CNN(Cin, Cout, kernel_size=3, stride=stride, padding=2),
    # 'cnn_7x7': lambda Cin,Cout, stride, signal: CNN(Cin, Cout, kernel_size=3, stride=stride, padding=2)
}
OPS_clif = {
    'skip_connect': lambda Cin,Cout, stride, signal, s: (CNN_clif(Cin, Cout, kernel_size=3,  stride=stride, padding=1, s=s) if signal == 1 else Identity(Cin, Cout, signal, s=s)),
    'cnn_3x3': lambda Cin,Cout, stride, signal, s: CNN_clif(Cin, Cout, kernel_size=3, stride=stride, padding=1, s=s),
    # 'cnn_5x5': lambda Cin,Cout, stride, signal: CNN(Cin, Cout, kernel_size=3, stride=stride, padding=2),
    # 'cnn_7x7': lambda Cin,Cout, stride, signal: CNN(Cin, Cout, kernel_size=3, stride=stride, padding=2)
}
OPS_Spike = {
    'spike3': 3,
    'spike5': 5,
    # 'spike7': 7,
}
class Identity(nn.Module):
    def __init__(self, C_in, C_out, signal, s):
        super(Identity, self).__init__()
        self.signal = signal

    def forward(self, x, n_gammas=None, param=None):
        return x


class CNN_clif(nn.Module):

    def __init__(self, input_c, output_c, kernel_size=3, stride=1, padding=1, s='spike3'):
        super(CNN_clif, self).__init__()
        
        self.conv1 = nn.Conv2d(input_c, output_c, kernel_size=kernel_size, stride=stride, padding=padding)
        self.act_fun = ActFun_changeable().apply
        if s == 'spike3':
            self.b = 3
        else:
            self.b = 5
        self.mem = None
        #self.alpha_diffb = nn.Parameter(1e-3*torch.ones(3).cuda(),requires_grad=True)
        self.bn = nn.BatchNorm2d(output_c)
        self._initialize_weights()
        self.clif = ComplementaryLIFNeuron()

    def forward(self, input, param): #20
        if not self.bn.training:
            conv_bn = fuse_conv_bn_eval(self.conv1,self.bn)
            mem_this = conv_bn(input)
        else:
            mem_this = self.bn(self.conv1(input))

        device = input.device
        # if param['is_first']:
        #     self.mem = torch.zeros_like(self.conv1(input), device=device)

        # self.mem =  self.mem + mem_this
        # spike = self.act_fun(self.mem, self.b) 
        # self.mem = self.mem *decay *  (1. - spike) 
        spike = self.clif(mem_this)
        return spike
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()



class CNN(nn.Module):

    def __init__(self, input_c, output_c, kernel_size=3, stride=1, padding=1, s='spike3'):
        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv2d(input_c, output_c, kernel_size=kernel_size, stride=stride, padding=padding)
        self.act_fun = ActFun_changeable().apply
        if s == 'spike3':
            self.b = 3
        else:
            self.b = 5
        self.mem = None
        #self.alpha_diffb = nn.Parameter(1e-3*torch.ones(3).cuda(),requires_grad=True)
        self.bn = nn.BatchNorm2d(output_c)
        self._initialize_weights()

    def forward(self, input, param): #20
        if not self.bn.training:
            conv_bn = fuse_conv_bn_eval(self.conv1,self.bn)
            mem_this = conv_bn(input)
        else:
            mem_this = self.bn(self.conv1(input))

        device = input.device
        if param['is_first']:
            self.mem = torch.zeros_like(self.conv1(input), device=device)

        self.mem =  self.mem + mem_this
        spike = self.act_fun(self.mem, self.b) 
        self.mem = self.mem *decay *  (1. - spike) 
        return spike
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

class MixedOp(nn.Module):
    def __init__(self):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()


    def forward(self, x, weights, left_or_right):
        opt_outs = []
        for i in range(3):
            opt_out = self._ops[i](x, left_or_right)
            opt_out = weights[i] * opt_out
            opt_outs.append(opt_out)
        return sum(opt_outs)  

class Cell(nn.Module):
    def __init__(self, steps, block_multiplier, prev_prev_fmultiplier,
                 prev_filter_multiplier, cell_arch, spike_arch, network_arch,
                 filter_multiplier, downup_sample,args=None):
        super(Cell, self).__init__()
        self.cell_arch = cell_arch
        self.spike_arch = spike_arch
        self.C_in = block_multiplier * filter_multiplier
        self.C_out = filter_multiplier
        self.C_prev = int(block_multiplier * prev_filter_multiplier)
        self.C_prev_prev = int(block_multiplier * prev_prev_fmultiplier)
        self.downup_sample = downup_sample
        self._steps = steps
        self.block_multiplier = block_multiplier
        self._ops = nn.ModuleList()
        if downup_sample == -1:
            self.scale = 0.5
        elif downup_sample == 1:
            self.scale = 2

        self.cell_arch = torch.sort(self.cell_arch,dim=0)[0].to(torch.uint8)
        for x, s in zip(self.cell_arch, self.spike_arch):
            primitive = PRIMITIVES[x[1]]
            spike = PRIMITIVES_SPIKE[s[1]]
            if x[0] in [0,2,5]:
                op = OPS[primitive](self.C_prev_prev, self.C_out, stride=1, signal=1, s=spike)
            elif x[0] in [1,3,6]:
                op = OPS[primitive](self.C_prev, self.C_out, stride=1, signal=1, s=spike)
            else:
                op = OPS[primitive](self.C_out, self.C_out, stride=1, signal=1, s=spike)

            self._ops.append(op)

        self.mem = None
        self.act_fun = ActFun_changeable().apply

    def scale_dimension(self, dim, scale):
        return (int((float(dim) - 1.0) * scale + 1.0) if dim % 2 == 1 else int((float(dim) * scale)))

    def forward(self, prev_prev_input, prev_input, param):
        s0 = prev_prev_input
        s1 = prev_input
        if self.downup_sample != 0:
            feature_size_h = self.scale_dimension(s1.shape[2], self.scale)
            feature_size_w = self.scale_dimension(s1.shape[3], self.scale)
            s1 = F.interpolate(s1, [feature_size_h, feature_size_w], mode='nearest')
        if (s0.shape[2] != s1.shape[2]) or (s0.shape[3] != s1.shape[3]):
            s0 = F.interpolate(s0, (s1.shape[2], s1.shape[3]),
                                            mode='nearest')

        device = prev_input.device

        states = [s0, s1]
        offset = 0
        ops_index = 0
        act_spike = 0
        total_spike = 0
        for i in range(self._steps):
            new_states = []
            for j, h in enumerate(states):
                branch_index = offset + j
                if branch_index in self.cell_arch[:, 0]:
                    if prev_prev_input is None and j == 0:
                        ops_index += 1
                        continue
                    if isinstance(self._ops[ops_index],Identity):
                        new_state = self._ops[ops_index](h, param)
                    else:
                        param['mixed_at_mem'] = True
                        new_state = self._ops[ops_index](h, param)
                        param['mixed_at_mem'] = False
                        with torch.no_grad():
                            act_spike += torch.sum(new_state>0.5)
                            total_spike += torch.sum(torch.ones_like(new_state))
                    if param['is_first']:
                        self.mem = [torch.zeros_like(new_state,device=device)]*self._steps
                    

                    new_states.append(new_state)
                    ops_index += 1

            s = sum(new_states)
            spike = s
            # if param['mem_output']:
            #     spike = s
            # else:
            #     self.mem[i] = self.mem[i] + s
            #     spike = self.act_fun(self.mem[i],3)
            #     self.mem[i] = self.mem[i] * decay * (1. - spike) 

            offset += len(states)
            states.append(spike)

        concat_feature = torch.cat(states[-self.block_multiplier:], dim=1) 
        return (prev_input, concat_feature), (act_spike, total_spike)


class Cell_clif(nn.Module):
    def __init__(self, steps, block_multiplier, prev_prev_fmultiplier,
                 prev_filter_multiplier, cell_arch, spike_arch, network_arch,
                 filter_multiplier, downup_sample,args=None):
        super(Cell_clif, self).__init__()
        self.cell_arch = cell_arch
        self.spike_arch = spike_arch
        self.C_in = block_multiplier * filter_multiplier
        self.C_out = filter_multiplier
        self.C_prev = int(block_multiplier * prev_filter_multiplier)
        self.C_prev_prev = int(block_multiplier * prev_prev_fmultiplier)
        self.downup_sample = downup_sample
        self._steps = steps
        self.block_multiplier = block_multiplier
        self._ops = nn.ModuleList()
        if downup_sample == -1:
            self.scale = 0.5
        elif downup_sample == 1:
            self.scale = 2

        self.cell_arch = torch.sort(self.cell_arch,dim=0)[0].to(torch.uint8)
        for x, s in zip(self.cell_arch, self.spike_arch):
            primitive = PRIMITIVES[x[1]]
            spike = PRIMITIVES_SPIKE[s[1]]
            if x[0] in [0,2,5]:
                op = OPS_clif[primitive](self.C_prev_prev, self.C_out, stride=1, signal=1, s=spike)
            elif x[0] in [1,3,6]:
                op = OPS_clif[primitive](self.C_prev, self.C_out, stride=1, signal=1, s=spike)
            else:
                op = OPS_clif[primitive](self.C_out, self.C_out, stride=1, signal=1, s=spike)

            self._ops.append(op)

        self.mem = None
        self.act_fun = ActFun_changeable().apply

    def scale_dimension(self, dim, scale):
        return (int((float(dim) - 1.0) * scale + 1.0) if dim % 2 == 1 else int((float(dim) * scale)))

    def forward(self, prev_prev_input, prev_input, param):
        s0 = prev_prev_input
        s1 = prev_input
        if self.downup_sample != 0:
            feature_size_h = self.scale_dimension(s1.shape[2], self.scale)
            feature_size_w = self.scale_dimension(s1.shape[3], self.scale)
            s1 = F.interpolate(s1, [feature_size_h, feature_size_w], mode='nearest')
        if (s0.shape[2] != s1.shape[2]) or (s0.shape[3] != s1.shape[3]):
            s0 = F.interpolate(s0, (s1.shape[2], s1.shape[3]),
                                            mode='nearest')

        device = prev_input.device

        states = [s0, s1]
        offset = 0
        ops_index = 0
        act_spike = 0
        total_spike = 0
        for i in range(self._steps):
            new_states = []
            for j, h in enumerate(states):
                branch_index = offset + j
                if branch_index in self.cell_arch[:, 0]:
                    if prev_prev_input is None and j == 0:
                        ops_index += 1
                        continue
                    if isinstance(self._ops[ops_index],Identity):
                        new_state = self._ops[ops_index](h, param)
                    else:
                        param['mixed_at_mem'] = True
                        new_state = self._ops[ops_index](h, param)
                        param['mixed_at_mem'] = False
                        with torch.no_grad():
                            act_spike += torch.sum(new_state>0.5)
                            total_spike += torch.sum(torch.ones_like(new_state))
                    if param['is_first']:
                        self.mem = [torch.zeros_like(new_state,device=device)]*self._steps
                    

                    new_states.append(new_state)
                    ops_index += 1

            s = sum(new_states)
            spike = s

            offset += len(states)
            states.append(spike)

        concat_feature = torch.cat(states[-self.block_multiplier:], dim=1) 
        return (prev_input, concat_feature), (act_spike, total_spike)



def check_spike(input):
    input = input.cpu().detach().clone().reshape(-1)
    all_01 = torch.sum(input == 0) + torch.sum(input == 1)
    print(all_01 == input.shape[0])

class newFeature_ImageNet(nn.Module):
    def __init__(self, frame_rate, CIFAR_CLASSES, network_arch, cell_arch, spike_arch, cell=Cell, args=None):
        super(newFeature_ImageNet, self).__init__()
        self.args = args
        self.cells = nn.ModuleList()
        self.network_arch = torch.from_numpy(network_arch)
        self.cell_arch = torch.from_numpy(cell_arch)
        self.spike_arch = torch.from_numpy(spike_arch)
        self._step = args.fea_step
        self._num_layers = args.fea_num_layers
        self._block_multiplier = args.fea_block_multiplier
        self._filter_multiplier = args.fea_filter_multiplier

        f_initial = int(self._filter_multiplier)
        half_f_initial = int(f_initial/2)
        self._num_end = self._filter_multiplier*self._block_multiplier

        self.stem0 = SNN_2d_Super(frame_rate, int(f_initial * self._block_multiplier/2), kernel_size=3, stride=2, padding=1,b=3) 
        self.stem1 = SNN_2d_Super(int(f_initial * self._block_multiplier/2), f_initial * self._block_multiplier, kernel_size=3, stride=2, padding=1,b=3)# DGS
        self.auxiliary_head = AuxiliaryHeadCIFAR2(864, CIFAR_CLASSES)

        filter_param_dict = {0: 1, 1: 2, 2: 4, 3: 8}
        self.global_pooling = SNN_Adaptivepooling(1)
        self.classifier = SNN_2d_fc(864, CIFAR_CLASSES)
        for i in range(self._num_layers):
            level_option = torch.sum(self.network_arch[i], dim=1)
            prev_level_option = torch.sum(self.network_arch[i - 1], dim=1)
            prev_prev_level_option = torch.sum(self.network_arch[i - 2], dim=1)
            level = torch.argmax(level_option).item()
            prev_level = torch.argmax(prev_level_option).item()
            prev_prev_level = torch.argmax(prev_prev_level_option).item()

            if i == 0:
                downup_sample = - torch.argmax(torch.sum(self.network_arch[0], dim=1))
                _cell = cell(self._step, self._block_multiplier, int(self._filter_multiplier/2),
                             self._filter_multiplier,
                             self.cell_arch, self.spike_arch, self.network_arch[i],
                             int(self._filter_multiplier * filter_param_dict[level]),
                             downup_sample, self.args)

            else:
                three_branch_options = torch.sum(self.network_arch[i], dim=0)
                downup_sample = torch.argmax(three_branch_options).item() - 1
                if i == 1:
                    _cell = cell(self._step, self._block_multiplier,
                                 self._filter_multiplier,
                                 int(self._filter_multiplier * filter_param_dict[prev_level]),
                                 self.cell_arch, self.spike_arch, self.network_arch[i],
                                 int(self._filter_multiplier * filter_param_dict[level]),
                                 downup_sample, self.args)

                else:
                    _cell = cell(self._step, self._block_multiplier,
                                 int(self._filter_multiplier * filter_param_dict[prev_prev_level]),
                                 int(self._filter_multiplier * filter_param_dict[prev_level]),
                                 self.cell_arch, self.spike_arch, self.network_arch[i],
                                 int(self._filter_multiplier * filter_param_dict[level]), downup_sample, self.args)

            self.cells += [_cell]




    def forward(self, x, param):
        stem0 = self.stem0(x, param) 
        stem1 = self.stem1(stem0, param) 
        out = (stem0,stem1)
        
        for i in range(self._num_layers):
            param['mem_output'] = False
            out = self.cells[i](out[0], out[1], param)
            if i == 7:
                if self.training:
                    logits_aux = self.auxiliary_head(out[-1], param)

        last_output = out[-1]
        # if self.training:
        #     return last_output, logits_aux
        # else:
        #     return last_output, None
        pooling_out = self.global_pooling(last_output) 
        logits_buf = self.classifier(pooling_out.view(pooling_out.size(0),-1), param) 
        if self.training:
            return logits_buf, logits_aux
        else:
            return logits_buf, None
            
    def get_params(self):
        bn_params = []
        non_bn_params = []
        for name, param in self.named_parameters():
            if 'bn' in name or 'downsample.1' in name:
                bn_params.append(param)
            else:
                bn_params.append(param)
        return bn_params, non_bn_params
    
    
class AuxiliaryHeadCIFAR2(nn.Module):

  def __init__(self, C, num_classes):
    """assuming input size 8x8"""
    super(AuxiliaryHeadCIFAR2, self).__init__()
    
    self.pooling = SNN_Avgpooling(5, stride=2, padding=0)
    # self.conv0 = SNN_2d(C, 576, kernel_size=3, stride=2, padding=2, b=3)
    self.conv1 = SNN_2d(C, 128, 1, padding=0, b=3)
    self.conv2 = SNN_2d(128, 768, 2, padding=0, b=3)
    self.classifier = SNN_2d_fc(768, num_classes)

  def forward(self, x, param):
    # x = self.conv0(x, param)
    x = self.pooling(x, param)
    spike1 = self.conv1(x, param)
    spike2 = self.conv2(spike1, param)
    result = self.classifier(spike2.view(spike2.size(0),-1),param)
    return result

class newFeature(nn.Module):
    def __init__(self, frame_rate, CIFAR_CLASSES, network_arch, cell_arch, spike_arch, cell=Cell, args=None):
        super(newFeature, self).__init__()
        self.args = args
        self.cells = nn.ModuleList()
        self.network_arch = torch.from_numpy(network_arch)
        self.cell_arch = torch.from_numpy(cell_arch)
        self.spike_arch = torch.from_numpy(spike_arch)
        self._step = args.fea_step
        self._num_layers = args.fea_num_layers
        self._block_multiplier = args.fea_block_multiplier
        self._filter_multiplier = args.fea_filter_multiplier

        f_initial = int(self._filter_multiplier)
        half_f_initial = int(f_initial/2)
        self._num_end = self._filter_multiplier*self._block_multiplier

        self.stem0 = SNN_2d_Super(frame_rate, f_initial * self._block_multiplier, kernel_size=3, stride=1, padding=1,b=3) 
        self.stem1 = SNN_2d_Super(f_initial * self._block_multiplier, f_initial * self._block_multiplier, kernel_size=3, stride=1, padding=1,b=3)# DGS
        self.auxiliary_head = AuxiliaryHeadCIFAR(432, CIFAR_CLASSES)
        # self.auxiliary_head = AuxiliaryHeadCIFAR(456, CIFAR_CLASSES)
        filter_param_dict = {0: 1, 1: 2, 2: 4, 3: 8}
        self.global_pooling = SNN_Adaptivepooling(1)
        self.classifier = SNN_2d_fc(432, CIFAR_CLASSES)
        # self.classifier = SNN_2d_fc(456, CIFAR_CLASSES)
        for i in range(self._num_layers):
            level_option = torch.sum(self.network_arch[i], dim=1)
            prev_level_option = torch.sum(self.network_arch[i - 1], dim=1)
            prev_prev_level_option = torch.sum(self.network_arch[i - 2], dim=1)
            level = torch.argmax(level_option).item()
            prev_level = torch.argmax(prev_level_option).item()
            prev_prev_level = torch.argmax(prev_prev_level_option).item()

            if i == 0:
                downup_sample = - torch.argmax(torch.sum(self.network_arch[0], dim=1))
                _cell = cell(self._step, self._block_multiplier, self._filter_multiplier,
                             self._filter_multiplier,
                             self.cell_arch, self.spike_arch, self.network_arch[i],
                             int(self._filter_multiplier * filter_param_dict[level]),
                             downup_sample, self.args)

            else:
                three_branch_options = torch.sum(self.network_arch[i], dim=0)
                downup_sample = torch.argmax(three_branch_options).item() - 1
                if i == 1:
                    _cell = cell(self._step, self._block_multiplier,
                                 self._filter_multiplier,
                                 int(self._filter_multiplier * filter_param_dict[prev_level]),
                                 self.cell_arch, self.spike_arch, self.network_arch[i],
                                 int(self._filter_multiplier * filter_param_dict[level]),
                                 downup_sample, self.args)

                else:
                    _cell = cell(self._step, self._block_multiplier,
                                 int(self._filter_multiplier * filter_param_dict[prev_prev_level]),
                                 int(self._filter_multiplier * filter_param_dict[prev_level]),
                                 self.cell_arch, self.spike_arch, self.network_arch[i],
                                 int(self._filter_multiplier * filter_param_dict[level]), downup_sample, self.args)

            self.cells += [_cell]




    def forward(self, x, param):
        A_spike = 0
        T_spike = 0
        stem0,spike_num = self.stem0(x, param) 
        A_spike+=spike_num[0]
        T_spike+=spike_num[1]
        stem1 = stem0 #self.stem1(stem0, param) 
        out = (stem0,stem1)
        
        A_spike_list= []
        T_spike_list= []
        A_spike_list.append(spike_num[0])
        T_spike_list.append(spike_num[1])
        for i in range(self._num_layers):
            param['mem_output'] = False
            out, spike_num  = self.cells[i](out[0], out[1], param)
            if i == 2*8//3:
                if self.training:
                    logits_aux = self.auxiliary_head(out[-1], param)
            A_spike+=spike_num[0]
            T_spike+=spike_num[1]
            A_spike_list.append(spike_num[0])
            T_spike_list.append(spike_num[1])
        last_output = out[-1]
        # if self.training:
        #     return last_output, logits_aux
        # else:
        #     return last_output, None
        pooling_out = self.global_pooling(last_output) 
        logits_buf = self.classifier(pooling_out.view(pooling_out.size(0),-1), param) 
        if self.training:
            return logits_buf, logits_aux, A_spike, T_spike, A_spike_list, T_spike_list
        else:
            return logits_buf, None, A_spike, T_spike, A_spike_list, T_spike_list
            
    def get_params(self):
        bn_params = []
        non_bn_params = []
        for name, param in self.named_parameters():
            if 'bn' in name or 'downsample.1' in name:
                bn_params.append(param)
            else:
                bn_params.append(param)
        return bn_params, non_bn_params


class newFeature_clif(nn.Module):
    def __init__(self, frame_rate, CIFAR_CLASSES, network_arch, cell_arch, spike_arch, cell=Cell_clif, args=None):
        super(newFeature_clif, self).__init__()
        self.args = args
        self.cells = nn.ModuleList()
        self.network_arch = torch.from_numpy(network_arch)
        self.cell_arch = torch.from_numpy(cell_arch)
        self.spike_arch = torch.from_numpy(spike_arch)
        self._step = args.fea_step
        self._num_layers = args.fea_num_layers
        self._block_multiplier = args.fea_block_multiplier
        self._filter_multiplier = args.fea_filter_multiplier

        f_initial = int(self._filter_multiplier)
        half_f_initial = int(f_initial/2)
        self._num_end = self._filter_multiplier*self._block_multiplier

        self.stem0 = SNN_2d_Super_clif(frame_rate, f_initial * self._block_multiplier, kernel_size=3, stride=1, padding=1,b=3) 
        self.stem1 = SNN_2d_Super_clif(f_initial * self._block_multiplier, f_initial * self._block_multiplier, kernel_size=3, stride=1, padding=1,b=3)# DGS
        self.auxiliary_head = AuxiliaryHeadCIFAR_clif(432, CIFAR_CLASSES)
        # self.auxiliary_head = AuxiliaryHeadCIFAR(456, CIFAR_CLASSES)
        filter_param_dict = {0: 1, 1: 2, 2: 4, 3: 8}
        self.global_pooling = SNN_Adaptivepooling(1)
        self.classifier = SNN_2d_fc(432, CIFAR_CLASSES)
        # self.classifier = SNN_2d_fc(456, CIFAR_CLASSES)
        for i in range(self._num_layers):
            level_option = torch.sum(self.network_arch[i], dim=1)
            prev_level_option = torch.sum(self.network_arch[i - 1], dim=1)
            prev_prev_level_option = torch.sum(self.network_arch[i - 2], dim=1)
            level = torch.argmax(level_option).item()
            prev_level = torch.argmax(prev_level_option).item()
            prev_prev_level = torch.argmax(prev_prev_level_option).item()

            if i == 0:
                downup_sample = - torch.argmax(torch.sum(self.network_arch[0], dim=1))
                _cell = cell(self._step, self._block_multiplier, self._filter_multiplier,
                             self._filter_multiplier,
                             self.cell_arch, self.spike_arch, self.network_arch[i],
                             int(self._filter_multiplier * filter_param_dict[level]),
                             downup_sample, self.args)

            else:
                three_branch_options = torch.sum(self.network_arch[i], dim=0)
                downup_sample = torch.argmax(three_branch_options).item() - 1
                if i == 1:
                    _cell = cell(self._step, self._block_multiplier,
                                 self._filter_multiplier,
                                 int(self._filter_multiplier * filter_param_dict[prev_level]),
                                 self.cell_arch, self.spike_arch, self.network_arch[i],
                                 int(self._filter_multiplier * filter_param_dict[level]),
                                 downup_sample, self.args)

                else:
                    _cell = cell(self._step, self._block_multiplier,
                                 int(self._filter_multiplier * filter_param_dict[prev_prev_level]),
                                 int(self._filter_multiplier * filter_param_dict[prev_level]),
                                 self.cell_arch, self.spike_arch, self.network_arch[i],
                                 int(self._filter_multiplier * filter_param_dict[level]), downup_sample, self.args)

            self.cells += [_cell]




    def forward(self, x, param):
        A_spike = 0
        T_spike = 0
        stem0,spike_num = self.stem0(x, param) 
        A_spike+=spike_num[0]
        T_spike+=spike_num[1]
        stem1 = stem0 #self.stem1(stem0, param) 
        out = (stem0,stem1)
        
        A_spike_list= []
        T_spike_list= []
        A_spike_list.append(spike_num[0])
        T_spike_list.append(spike_num[1])
        for i in range(self._num_layers):
            param['mem_output'] = False
            out, spike_num  = self.cells[i](out[0], out[1], param)
            if i == 2*8//3:
                if self.training:
                    logits_aux = self.auxiliary_head(out[-1], param)
            A_spike+=spike_num[0]
            T_spike+=spike_num[1]
            A_spike_list.append(spike_num[0])
            T_spike_list.append(spike_num[1])
        last_output = out[-1]
        # if self.training:
        #     return last_output, logits_aux
        # else:
        #     return last_output, None
        pooling_out = self.global_pooling(last_output) 
        logits_buf = self.classifier(pooling_out.view(pooling_out.size(0),-1), param) 
        if self.training:
            return logits_buf, logits_aux, A_spike, T_spike, A_spike_list, T_spike_list
        else:
            return logits_buf, None, A_spike, T_spike, A_spike_list, T_spike_list
            
    def get_params(self):
        bn_params = []
        non_bn_params = []
        for name, param in self.named_parameters():
            if 'bn' in name or 'downsample.1' in name:
                bn_params.append(param)
            else:
                bn_params.append(param)
        return bn_params, non_bn_params


class newFeature2(nn.Module):
    def __init__(self, frame_rate, CIFAR_CLASSES, network_arch, cell_arch, spike_arch, cell=Cell, args=None):
        super(newFeature2, self).__init__()
        self.args = args
        self.cells = nn.ModuleList()
        self.network_arch = torch.from_numpy(network_arch)
        self.cell_arch = torch.from_numpy(cell_arch)
        self.spike_arch = torch.from_numpy(spike_arch)
        self._step = args.fea_step
        self._num_layers = args.fea_num_layers
        self._block_multiplier = args.fea_block_multiplier
        self._filter_multiplier = args.fea_filter_multiplier

        f_initial = int(self._filter_multiplier)
        half_f_initial = int(f_initial/2)
        self._num_end = self._filter_multiplier*self._block_multiplier

        self.stem0 = SNN_2d_Super(frame_rate, f_initial * self._block_multiplier, kernel_size=3, stride=1, padding=1,b=3) 
        self.stem1 = SNN_2d_Super(f_initial * self._block_multiplier, f_initial * self._block_multiplier, kernel_size=3, stride=1, padding=1,b=3)# DGS
        self.auxiliary_head = AuxiliaryHeadCIFAR2(432, CIFAR_CLASSES)
        # self.auxiliary_head = AuxiliaryHeadCIFAR(456, CIFAR_CLASSES)
        filter_param_dict = {0: 1, 1: 2, 2: 4, 3: 8}
        self.global_pooling = SNN_Adaptivepooling(1)
        self.classifier = SNN_2d_fc(432, CIFAR_CLASSES)
        # self.classifier = SNN_2d_fc(456, CIFAR_CLASSES)
        for i in range(self._num_layers):
            level_option = torch.sum(self.network_arch[i], dim=1)
            prev_level_option = torch.sum(self.network_arch[i - 1], dim=1)
            prev_prev_level_option = torch.sum(self.network_arch[i - 2], dim=1)
            level = torch.argmax(level_option).item()
            prev_level = torch.argmax(prev_level_option).item()
            prev_prev_level = torch.argmax(prev_prev_level_option).item()

            if i == 0:
                downup_sample = - torch.argmax(torch.sum(self.network_arch[0], dim=1))
                _cell = cell(self._step, self._block_multiplier, self._filter_multiplier,
                             self._filter_multiplier,
                             self.cell_arch, self.spike_arch, self.network_arch[i],
                             int(self._filter_multiplier * filter_param_dict[level]),
                             downup_sample, self.args)

            else:
                three_branch_options = torch.sum(self.network_arch[i], dim=0)
                downup_sample = torch.argmax(three_branch_options).item() - 1
                if i == 1:
                    _cell = cell(self._step, self._block_multiplier,
                                 self._filter_multiplier,
                                 int(self._filter_multiplier * filter_param_dict[prev_level]),
                                 self.cell_arch, self.spike_arch, self.network_arch[i],
                                 int(self._filter_multiplier * filter_param_dict[level]),
                                 downup_sample, self.args)

                else:
                    _cell = cell(self._step, self._block_multiplier,
                                 int(self._filter_multiplier * filter_param_dict[prev_prev_level]),
                                 int(self._filter_multiplier * filter_param_dict[prev_level]),
                                 self.cell_arch, self.spike_arch, self.network_arch[i],
                                 int(self._filter_multiplier * filter_param_dict[level]), downup_sample, self.args)

            self.cells += [_cell]




    def forward(self, x, param):
        A_spike = 0
        T_spike = 0
        stem0 = self.stem0(x, param) 
        stem1 = stem0 #self.stem1(stem0, param) 
        out = (stem0,stem1)
        
        for i in range(self._num_layers):
            param['mem_output'] = False
            out, spike_num = self.cells[i](out[0], out[1], param)
            if i == 2*8//3:
                if self.training:
                    logits_aux = self.auxiliary_head(out[-1], param)
            A_spike+=spike_num[0]
            T_spike+=spike_num[1]
        last_output = out[-1]
        # if self.training:
        #     return last_output, logits_aux
        # else:
        #     return last_output, None
        pooling_out = self.global_pooling(last_output) 
        logits_buf = self.classifier(pooling_out.view(pooling_out.size(0),-1), param) 
        if self.training:
            return logits_buf, logits_aux, A_spike, T_spike
        else:
            return logits_buf, None, A_spike, T_spike
            
    def get_params(self):
        bn_params = []
        non_bn_params = []
        for name, param in self.named_parameters():
            if 'bn' in name or 'downsample.1' in name:
                bn_params.append(param)
            else:
                bn_params.append(param)
        return bn_params, non_bn_params

class AuxiliaryHeadCIFAR2(nn.Module):

  def __init__(self, C, num_classes):
    """assuming input size 8x8"""
    super(AuxiliaryHeadCIFAR2, self).__init__()
    
    self.pooling = SNN_Avgpooling(5, stride=3, padding=0)

    self.conv1 = SNN_2d(C, 128, 1, padding=0, b=3)
    self.conv2 = SNN_2d(128, 768, 2, padding=0, b=3)
    self.classifier = SNN_2d_fc(768*9*9, num_classes)

  def forward(self, x, param):
    x = self.pooling(x, param)
    spike1 = self.conv1(x, param)
    spike2 = self.conv2(spike1, param)
    result = self.classifier(spike2.view(spike2.size(0),-1),param)
    return result

class AuxiliaryHeadCIFAR(nn.Module):

  def __init__(self, C, num_classes):
    """assuming input size 8x8"""
    super(AuxiliaryHeadCIFAR, self).__init__()
    
    self.pooling = SNN_Avgpooling(5, stride=3, padding=0)

    self.conv1 = SNN_2d(C, 128, 1, padding=0, b=3)
    self.conv2 = SNN_2d(128, 768, 2, padding=0, b=3)
    self.classifier = SNN_2d_fc(768, num_classes)

  def forward(self, x, param):
    x = self.pooling(x, param)
    spike1, _ = self.conv1(x, param)
    spike2, _ = self.conv2(spike1, param)
    result = self.classifier(spike2.view(spike2.size(0),-1),param)
    return result


class AuxiliaryHeadCIFAR_clif(nn.Module):

  def __init__(self, C, num_classes):
    """assuming input size 8x8"""
    super(AuxiliaryHeadCIFAR_clif, self).__init__()
    
    self.pooling = SNN_Avgpooling(5, stride=3, padding=0)

    self.conv1 = SNN_2d_clif(C, 128, 1, padding=0, b=3)
    self.conv2 = SNN_2d_clif(128, 768, 2, padding=0, b=3)
    self.classifier = SNN_2d_fc(768, num_classes)

  def forward(self, x, param):
    x = self.pooling(x, param)
    spike1, _ = self.conv1(x, param)
    spike2, _ = self.conv2(spike1, param)
    result = self.classifier(spike2.view(spike2.size(0),-1),param)
    return result

class newFeature3(nn.Module):
    def __init__(self, frame_rate, CIFAR_CLASSES, network_arch, cell_arch, spike_arch, cell=Cell, args=None):
        super(newFeature3, self).__init__()
        self.args = args
        self.cells = nn.ModuleList()
        self.network_arch = torch.from_numpy(network_arch)
        self.cell_arch = torch.from_numpy(cell_arch)
        self.spike_arch = torch.from_numpy(spike_arch)
        self._step = args.fea_step
        self._num_layers = args.fea_num_layers
        self._block_multiplier = args.fea_block_multiplier
        self._filter_multiplier = args.fea_filter_multiplier

        f_initial = int(self._filter_multiplier)
        half_f_initial = int(f_initial/2)
        self._num_end = self._filter_multiplier*self._block_multiplier

        self.stem0 = SNN_2d_Super(frame_rate, f_initial * self._block_multiplier, kernel_size=3, stride=1, padding=1,b=3) 
        self.stem1 = SNN_2d_Super(f_initial * self._block_multiplier, f_initial * self._block_multiplier, kernel_size=3, stride=1, padding=1,b=3)# DGS
        self.auxiliary_head = AuxiliaryHeadCIFAR3(432, CIFAR_CLASSES)
        # self.auxiliary_head = AuxiliaryHeadCIFAR(456, CIFAR_CLASSES)
        filter_param_dict = {0: 1, 1: 2, 2: 4, 3: 8}
        self.global_pooling = SNN_Adaptivepooling(1)
        self.classifier = SNN_2d_fc(432, CIFAR_CLASSES)
        # self.classifier = SNN_2d_fc(456, CIFAR_CLASSES)
        for i in range(self._num_layers):
            level_option = torch.sum(self.network_arch[i], dim=1)
            prev_level_option = torch.sum(self.network_arch[i - 1], dim=1)
            prev_prev_level_option = torch.sum(self.network_arch[i - 2], dim=1)
            level = torch.argmax(level_option).item()
            prev_level = torch.argmax(prev_level_option).item()
            prev_prev_level = torch.argmax(prev_prev_level_option).item()

            if i == 0:
                downup_sample = - torch.argmax(torch.sum(self.network_arch[0], dim=1))
                _cell = cell(self._step, self._block_multiplier, self._filter_multiplier,
                             self._filter_multiplier,
                             self.cell_arch, self.spike_arch, self.network_arch[i],
                             int(self._filter_multiplier * filter_param_dict[level]),
                             downup_sample, self.args)

            else:
                three_branch_options = torch.sum(self.network_arch[i], dim=0)
                downup_sample = torch.argmax(three_branch_options).item() - 1
                if i == 1:
                    _cell = cell(self._step, self._block_multiplier,
                                 self._filter_multiplier,
                                 int(self._filter_multiplier * filter_param_dict[prev_level]),
                                 self.cell_arch, self.spike_arch, self.network_arch[i],
                                 int(self._filter_multiplier * filter_param_dict[level]),
                                 downup_sample, self.args)

                else:
                    _cell = cell(self._step, self._block_multiplier,
                                 int(self._filter_multiplier * filter_param_dict[prev_prev_level]),
                                 int(self._filter_multiplier * filter_param_dict[prev_level]),
                                 self.cell_arch, self.spike_arch, self.network_arch[i],
                                 int(self._filter_multiplier * filter_param_dict[level]), downup_sample, self.args)

            self.cells += [_cell]




    def forward(self, x, param):
        stem0 = self.stem0(x, param) 
        stem1 = stem0 #self.stem1(stem0, param) 
        out = (stem0,stem1)
        
        for i in range(self._num_layers):
            param['mem_output'] = False
            out = self.cells[i](out[0], out[1], param)
            if i == 2*8//3:
                if self.training:
                    logits_aux = self.auxiliary_head(out[-1], param)

        last_output = out[-1]
        # if self.training:
        #     return last_output, logits_aux
        # else:
        #     return last_output, None
        pooling_out = self.global_pooling(last_output) 
        logits_buf = self.classifier(pooling_out.view(pooling_out.size(0),-1), param) 
        if self.training:
            return logits_buf, logits_aux
        else:
            return logits_buf, None
            
    def get_params(self):
        bn_params = []
        non_bn_params = []
        for name, param in self.named_parameters():
            if 'bn' in name or 'downsample.1' in name:
                bn_params.append(param)
            else:
                bn_params.append(param)
        return bn_params, non_bn_params

class AuxiliaryHeadCIFAR3(nn.Module):

  def __init__(self, C, num_classes):
    """assuming input size 8x8"""
    super(AuxiliaryHeadCIFAR3, self).__init__()
    
    self.pooling = SNN_Avgpooling(5, stride=3, padding=0)

    self.conv1 = SNN_2d(C, 128, 1, padding=0, b=3)
    self.conv2 = SNN_2d(128, 768, 2, padding=0, b=3)
    self.classifier = SNN_2d_fc(768*13*18, num_classes)

  def forward(self, x, param):
    x = self.pooling(x, param)
    spike1 = self.conv1(x, param)
    spike2 = self.conv2(spike1, param)
    result = self.classifier(spike2.view(spike2.size(0),-1),param)
    return result