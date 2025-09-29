import os
import sys
import numpy as np
import torch
import argparse
import pdb
import torch.nn.functional as F
from models.snndarts_search.build_model import AutoStereo
import utils
from models.snndarts_search.genotypes_2d import PRIMITIVES

def network_layer_to_space(net_arch):
    for i, layer in enumerate(net_arch):
        if i == 0:
            space = np.zeros((1, 4, 3))
            space[0][layer][0] = 1
            prev = layer
        else:
            if layer == prev + 1:
                sample = 0
            elif layer == prev:
                sample = 1
            elif layer == prev - 1:
                sample = 2
            space1 = np.zeros((1, 4, 3))
            space1[0][layer][sample] = 1
            space = np.concatenate([space, space1], axis=0)
            prev = layer
    """
        return:
        network_space[layer][level][sample]:
        layer: 0 - 12
        level: sample_level {0: 1, 1: 2, 2: 4, 3: 8}
        sample: 0: down 1: None 2: Up
    """
    return space


class Decoder2(object):
    def __init__(self,  betas):
        self._betas = betas
        self._num_layers = self._betas.shape[0]
        self.network_space = torch.zeros(self._num_layers, 4, 3)
        for layer in range(self._num_layers):
            if layer == 0:
                self.network_space[layer][0][1:] = betas[layer][0][:]
            elif layer == 1:
                self.network_space[layer][0][1:] = betas[layer][0][:]
                self.network_space[layer][1][1:] = betas[layer][1][:]
            elif layer == 2:
                self.network_space[layer][0][1:] = betas[layer][0][:]
                self.network_space[layer][1][1:] = betas[layer][1][:]
                self.network_space[layer][2][1:] = betas[layer][2][:]
            else:
                self.network_space[layer][0][1:] = betas[layer][0][:]
                self.network_space[layer][1][1:] = betas[layer][1][:]
                self.network_space[layer][2][1:] = betas[layer][2][:]
                self.network_space[layer][3][1:] = betas[layer][3][:]        
    def viterbi_decode(self):

        prob_space = np.zeros((self.network_space.shape[:2]))
        path_space = np.zeros((self.network_space.shape[:2])).astype('int8')

        for layer in range(self.network_space.shape[0]):
            if layer == 0:
                prob_space[layer][0] = self.network_space[layer][0][1]
                prob_space[layer][1] = self.network_space[layer][0][2]
                path_space[layer][0] = 0
                path_space[layer][1] = -1
            else:
                for sample in range(self.network_space.shape[1]):
                    if layer - sample < - 1:
                        continue
                    local_prob = []
                    for rate in range(self.network_space.shape[2]):  # k[0 : ➚, 1: ➙, 2 : ➘]
                        if (sample == 0 and rate == 2) or (sample == 3 and rate == 0):
                            continue
                        else:
                            local_prob.append(prob_space[layer - 1][sample + 1 - rate] *
                                              self.network_space[layer][sample + 1 - rate][rate])
                    prob_space[layer][sample] = np.max(local_prob, axis=0)
                    rate = np.argmax(local_prob, axis=0)
                    path = 1 - rate if sample != 3 else -rate
                    path_space[layer][sample] = path  # path[1 : ➚, 0: ➙, -1 : ➘]

        output_sample = prob_space[-1, :].argmax(axis=-1)
        actual_path = np.zeros(self._num_layers).astype('uint8')
        actual_path[-1] = output_sample
        for i in range(1, self._num_layers):
            actual_path[-i - 1] = actual_path[-i] + path_space[self._num_layers - i, actual_path[-i]]
        return actual_path, network_layer_to_space(actual_path)

    def genotype_decode(self):
        def _parse(alphas, steps):
            gene = []
            start = 0
            n = 2
            for i in range(steps):
                end = start + n
                edges = sorted(range(start, end), key=lambda x: -np.max(alphas[x, 1:]))  # ignore none value
                top2edges = edges[:2]
                for j in top2edges:
                    best_op_index = np.argmax(alphas[j])  # this can include none op
                    gene.append([j, best_op_index])
                start = end
                n += 1
            return np.array(gene)

        normalized_alphas = F.softmax(self._alphas, dim=-1).data.cpu().numpy()
        gene_cell = _parse(normalized_alphas, self._steps)
        return gene_cell


class Decoder(object):
    def __init__(self, alphas, steps):
        self._alphas = alphas
        self._steps = steps

    def genotype_decode(self):
        def _parse(alphas, steps):
            gene = []
            start = 0
            n = 2
            for i in range(steps):
                end = start + n
                edges = sorted(range(start, end), key=lambda x: -np.max(alphas[x, :]))  # ignore none value
                # top2edges = edges[:2]
                for j in edges:
                    best_op_index = np.argmax(alphas[j])  # this can include none op
                    gene.append([j, best_op_index])
                start = end
                n += 1
            return np.array(gene)

        # normalized_alphas = F.softmax(self._alphas, dim=-1).data.cpu().numpy()
        gene_cell = _parse(self._alphas.data.cpu().numpy(), self._steps)
        return gene_cell


def obtain_decode_args():
    parser = argparse.ArgumentParser(description="LEStereo Decoding..")
    parser.add_argument('--dataset', type=str, default='sceneflow',
                        choices=['sceneflow', 'kitti15', 'kitti12', 'middlebury'],
                        help='dataset name (default: sceneflow)') 
    parser.add_argument('--step', type=int, default=3)
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    ######### LEStereo params ##################
    parser.add_argument('--fea_num_layers', type=int, default=8)
    parser.add_argument('--fea_filter_multiplier', type=int, default=8)
    parser.add_argument('--fea_block_multiplier', type=int, default=3)
    parser.add_argument('--fea_step', type=int, default=3)
    parser.add_argument('--net_arch_fea', default=None, type=str)
    parser.add_argument('--cell_arch_fea', default=None, type=str)
    parser.add_argument('--drop_rate', default=0.5, type=float)
    parser.add_argument('--fitlog_path',type=str,default='debug')
    return parser.parse_args()

class Loader(object):
    def __init__(self, args):
        self.args = args
        # Resuming checkpoint
        assert args.resume is not None, RuntimeError("No model to decode in resume path: '{:}'".format(args.resume))
        assert os.path.isfile(args.resume), RuntimeError("=> no checkpoint found at '{}'".format(args.resume))

        # model = torch.load(args.resume,map_location='cuda:0')
        model = AutoStereo(2, 11, args = args)
        utils.load(model, args.resume)
        # self._alphas_fea = model.feature.normalized_alphas
        _alphas = model.feature.alphas
        self._alphas_fea = torch.nn.functional.one_hot(torch.argmax(_alphas, dim=1))
        self.decoder_fea = Decoder(alphas=self._alphas_fea, steps=self.args.step)
        print(_alphas)
        print(self._alphas_fea)
        _gammas = model.feature.gammas
        self._gammas_fea = torch.nn.functional.one_hot(torch.argmax(_gammas, dim=1))
        self.decoder_spike = Decoder(alphas=self._gammas_fea, steps=self.args.step)
        print(_gammas)
        print(self._gammas_fea)

    def decode_cell(self):
        fea_genotype = self.decoder_fea.genotype_decode()
        spike_genotype = self.decoder_spike.genotype_decode()
        return fea_genotype,spike_genotype

def get_new_network_cell():
    args = obtain_decode_args()
    load_model = Loader(args)
    fea_genotype = load_model.decode_cell()
    print('Feature Net cell structure:', fea_genotype)

    # dir_name = os.path.dirname(args.resume)
    # fea_genotype_filename = os.path.join(dir_name, 'feature_genotype')
    # np.save(fea_genotype_filename, fea_genotype)

    # fea_cell_name = os.path.join(dir_name, 'feature_cell_structure')  

if __name__ == '__main__':
    get_new_network_cell()