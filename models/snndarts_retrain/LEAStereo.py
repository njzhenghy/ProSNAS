import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from models.snndarts_search.SNN import *
from models.snndarts_search.decoding_formulas import network_layer_to_space
from models.snndarts_retrain.new_model_2d import newFeature, newFeature_ImageNet, newFeature2, newFeature3, newFeature_clif
import time

class LEAStereo(nn.Module):
    def __init__(self, init_channels=3, CIFAR_CLASSES=10, args=None):
        super(LEAStereo, self).__init__()
        p=0.0
        # network_path_fea = [1, 2, 3, 3, 3, 3, 3, 3]
        network_path_fea = [0, 0, 1, 1, 1, 2, 2, 2]
        network_path_fea = np.array(network_path_fea)
        network_arch_fea = network_layer_to_space(network_path_fea)

        # change your architecture here!
        cell_arch_fea = [[0, 0],
       [1, 1],
       [2, 1],
       [3, 0],
       [4, 0],
       [5, 0],
       [6, 0],
       [7, 0],
       [8, 1]]
        cell_spike_fea = [[0, 0],
       [1, 0],
       [2, 0],
       [3, 1],
       [4, 0],
       [5, 1],
       [6, 0],
       [7, 1],
       [8, 1]]
        cell_arch_fea = np.array(cell_arch_fea)
        cell_spike_fea = np.array(cell_spike_fea)

        self.feature = newFeature(init_channels, CIFAR_CLASSES, network_arch_fea, cell_arch_fea, cell_spike_fea, args=args)
        # self.global_pooling = SNN_Adaptivepooling(1)
        # self.classifier = SNN_2d_fc(576, 10)

    def forward(self, input, timestamp=4, param=None): 
        # param['snn_output'] = 'mem'
        
        logits = None
        logits_aux_list = []
        # timestamp = 6
        A_s = 0
        T_s = 0
        A_spike_lists = [0 for i in range(9)]
        T_spike_lists = [0 for i in range(9)]
        for i in range(timestamp):
            param['mixed_at_mem'] = False
            if i == 0:
                param['is_first'] = True
            else:
                param['is_first'] = False
            
            feature_out, logits_aux, A_spikes, T_spikes, A_spike_list, T_spike_list = self.feature(input, param) 
            for idx,(A,T) in enumerate(zip(A_spike_list, T_spike_list)):
                A_spike_lists[idx] = A_spike_lists[idx]+A
                T_spike_lists[idx] = T_spike_lists[idx]+T
            logits_aux_list.append(logits_aux)
            A_s += A_spikes
            T_s += T_spikes
            if logits is None:
                logits = []
            logits.append(feature_out)

        test = torch.stack(logits)
        logits = torch.sum(test,dim=0) / timestamp

        if self.training:
            test2 = torch.stack(logits_aux_list)
            logits_aux_final = torch.mean(test2, dim=0)
            return logits, logits_aux_final, A_s, T_s, A_spike_lists, T_spike_lists
        else:
            return logits, None, A_s, T_s, A_spike_lists, T_spike_lists
        # return logits, None


class LEAStereo_clif(nn.Module):
    def __init__(self, init_channels=3, CIFAR_CLASSES=10, args=None):
        super(LEAStereo_clif, self).__init__()
        p=0.0
        # network_path_fea = [1, 2, 3, 3, 3, 3, 3, 3]
        network_path_fea = [0, 0, 1, 1, 1, 2, 2, 2]
        network_path_fea = np.array(network_path_fea)
        network_arch_fea = network_layer_to_space(network_path_fea)

        # change your architecture here!
        cell_arch_fea = [[0, 0],
       [1, 1],
       [2, 1],
       [3, 0],
       [4, 0],
       [5, 0],
       [6, 0],
       [7, 0],
       [8, 1]]
        cell_spike_fea = [[0, 0],
       [1, 0],
       [2, 0],
       [3, 1],
       [4, 0],
       [5, 1],
       [6, 0],
       [7, 1],
       [8, 1]]
        cell_arch_fea = np.array(cell_arch_fea)
        cell_spike_fea = np.array(cell_spike_fea)

        self.feature = newFeature_clif(init_channels, CIFAR_CLASSES, network_arch_fea, cell_arch_fea, cell_spike_fea, args=args)
        # self.global_pooling = SNN_Adaptivepooling(1)
        # self.classifier = SNN_2d_fc(576, 10)

    def forward(self, input, timestamp=4, param=None): 
        # param['snn_output'] = 'mem'
        
        logits = None
        logits_aux_list = []
        # timestamp = 6
        A_s = 0
        T_s = 0
        A_spike_lists = [0 for i in range(9)]
        T_spike_lists = [0 for i in range(9)]
        for i in range(timestamp):
            param['mixed_at_mem'] = False
            if i == 0:
                param['is_first'] = True
            else:
                param['is_first'] = False
            
            feature_out, logits_aux, A_spikes, T_spikes, A_spike_list, T_spike_list = self.feature(input, param) 
            for idx,(A,T) in enumerate(zip(A_spike_list, T_spike_list)):
                A_spike_lists[idx] = A_spike_lists[idx]+A
                T_spike_lists[idx] = T_spike_lists[idx]+T
            logits_aux_list.append(logits_aux)
            A_s += A_spikes
            T_s += T_spikes
            if logits is None:
                logits = []
            logits.append(feature_out)

        test = torch.stack(logits)
        logits = torch.sum(test,dim=0) / timestamp

        if self.training:
            test2 = torch.stack(logits_aux_list)
            logits_aux_final = torch.mean(test2, dim=0)
            return logits, logits_aux_final, A_s, T_s, A_spike_lists, T_spike_lists
        else:
            return logits, None, A_s, T_s, A_spike_lists, T_spike_lists
        # return logits, None

class LEAStereo2(nn.Module):
    def __init__(self, init_channels=3, CIFAR_CLASSES=10, args=None):
        super(LEAStereo2, self).__init__()
        p=0.0
        # network_path_fea = [1, 2, 3, 3, 3, 3, 3, 3]
        network_path_fea = [0, 0, 1, 1, 1, 2, 2, 2]
        network_path_fea = np.array(network_path_fea)
        network_arch_fea = network_layer_to_space(network_path_fea)

        # change your architecture here!
        cell_arch_fea = [[0, 1],
       [1, 0],
       [2, 0],
       [3, 1],
       [4, 0],
       [5, 1],
       [6, 1],
       [7, 0],
       [8, 0]]
        cell_spike_fea = [[0, 0],
       [1, 0],
       [2, 0],
       [3, 1],
       [4, 0],
       [5, 0],
       [6, 1],
       [7, 1],
       [8, 1]]
        cell_arch_fea = np.array(cell_arch_fea)
        cell_spike_fea = np.array(cell_spike_fea)

        self.feature = newFeature2(init_channels, CIFAR_CLASSES, network_arch_fea, cell_arch_fea, cell_spike_fea, args=args)
        # self.global_pooling = SNN_Adaptivepooling(1)
        # self.classifier = SNN_2d_fc(576, 10)

    def forward(self, input, timestamp = 6, param=None): 
        param['snn_output'] = 'mem'

        
        logits = None
        logits_aux_list = []
        # timestamp = 6
        A_s = 0
        T_s = 0
        input = input.transpose(0, 1)
        for i in range(timestamp):
            param['mixed_at_mem'] = False
            if i == 0:
                param['is_first'] = True
            else:
                param['is_first'] = False
            
            feature_out, logits_aux , A_spikes, T_spikes= self.feature(input[i], param) 
            logits_aux_list.append(logits_aux)
            A_s += A_spikes
            T_s += T_spikes
            if logits is None:
                logits = []
            logits.append(feature_out)

        test = torch.stack(logits)
        logits = torch.sum(test,dim=0) / timestamp

        if self.training:
            test2 = torch.stack(logits_aux_list)
            logits_aux_final = torch.mean(test2, dim=0)
            return logits, logits_aux_final, A_s, T_s
        else:
            return logits, None, A_s, T_s
        # return logits, None

class LEAStereo_ImageNet(nn.Module):
    def __init__(self, init_channels=3, CIFAR_CLASSES=10, args=None):
        super(LEAStereo_ImageNet, self).__init__()
        p=0.0
        network_path_fea = [0, 0, 1, 1, 2, 2, 2, 3, 3, 3]
        network_path_fea = np.array(network_path_fea)
        network_arch_fea = network_layer_to_space(network_path_fea)

        # change your architecture here!
        cell_arch_fea = [[0, 0],
       [1, 1],
       [2, 1],
       [3, 0],
       [4, 0],
       [5, 0],
       [6, 0],
       [7, 0],
       [8, 1]]
        cell_spike_fea = [[0, 0],
       [1, 0],
       [2, 0],
       [3, 1],
       [4, 0],
       [5, 1],
       [6, 0],
       [7, 1],
       [8, 1]]
        cell_arch_fea = np.array(cell_arch_fea)
        cell_spike_fea = np.array(cell_spike_fea)
        self.class_num = CIFAR_CLASSES
        self.feature = newFeature_ImageNet(init_channels, CIFAR_CLASSES,network_arch_fea, cell_arch_fea, cell_spike_fea, args=args)
        # self.global_pooling = SNN_Adaptivepooling(1)
        # self.classifier = SNN_2d_fc(1152, CIFAR_CLASSES)

    def forward(self, input, param=None): 
        param['snn_output'] = 'mem'

        
        logits = None
        logits_aux_list = []
        timestamp = 4
        logits = torch.zeros((timestamp,input.shape[0],self.class_num)).cuda()
        for i in range(timestamp):
            param['mixed_at_mem'] = False
            if i == 0:
                param['is_first'] = True
            else:
                param['is_first'] = False
            
            logits_buf, logits_aux = self.feature(input, param) 
            logits_aux_list.append(logits_aux)
 
            # pooling_out = self.global_pooling(feature_out) 
            # logits_buf = self.classifier(pooling_out.view(pooling_out.size(0),-1),param) 
            logits[i,:,:] = logits_buf
            
        logits = torch.sum(logits, dim=0) / timestamp
        

        if self.training:
            test2 = torch.stack(logits_aux_list)
            logits_aux_final = torch.mean(test2, dim=0)
            return logits, logits_aux_final
        else:
            return logits, None
        # return logits, None

def check_spike(input):
    input = input.cpu().detach().clone().reshape(-1)
    all_01 = torch.sum(input == 0) + torch.sum(input == 1)
    print(all_01 == input.shape[0])



class LEAStereo3(nn.Module):
    def __init__(self, init_channels=3, CIFAR_CLASSES=10, args=None):
        super(LEAStereo3, self).__init__()
        p=0.0
        # network_path_fea = [1, 2, 3, 3, 3, 3, 3, 3]
        network_path_fea = [0, 0, 1, 1, 1, 2, 2, 2]
        network_path_fea = np.array(network_path_fea)
        network_arch_fea = network_layer_to_space(network_path_fea)

        # change your architecture here!
        cell_arch_fea = [[0, 1],
       [1, 1],
       [2, 1],
       [3, 0],
       [4, 0],
       [5, 0],
       [6, 1],
       [7, 0],
       [8, 0]]
        cell_spike_fea = [[0, 0],
       [1, 1],
       [2, 1],
       [3, 1],
       [4, 1],
       [5, 1],
       [6, 1],
       [7, 1],
       [8, 1]]
        cell_arch_fea = np.array(cell_arch_fea)
        cell_spike_fea = np.array(cell_spike_fea)

        self.feature = newFeature3(init_channels, CIFAR_CLASSES, network_arch_fea, cell_arch_fea, cell_spike_fea, args=args)
        # self.global_pooling = SNN_Adaptivepooling(1)
        # self.classifier = SNN_2d_fc(576, 10)

    def forward(self, input, timestamp = 6, param=None): 
        param['snn_output'] = 'mem'

        
        logits = None
        logits_aux_list = []
        # timestamp = 6
        input = input.transpose(0, 1)
        for i in range(timestamp):
            param['mixed_at_mem'] = False
            if i == 0:
                param['is_first'] = True
            else:
                param['is_first'] = False
            
            feature_out, logits_aux = self.feature(input[i], param) 
            logits_aux_list.append(logits_aux)

            if logits is None:
                logits = []
            logits.append(feature_out)

        test = torch.stack(logits)
        logits = torch.sum(test,dim=0) / timestamp

        if self.training:
            test2 = torch.stack(logits_aux_list)
            logits_aux_final = torch.mean(test2, dim=0)
            return logits, logits_aux_final
        else:
            return logits, None
        # return logits, None

