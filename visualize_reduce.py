import os
import sys
import numpy as np
import torch
import argparse
import pdb
import torch.nn.functional as F
from torch.distributions.one_hot_categorical import OneHotCategorical
from graphviz import Digraph
from models.snndarts_search.genotypes_2d import  *

def genotype(alphas_normal):

    def _parse(weights):
      gene = []
      n = 2
      start = 0
      steps = 3
      for i in range(steps):
        end = start + n
        W = weights[start:end].copy()
        edges = sorted(range(i + 2)) # we are going to consider all input nodes
        for j in edges:
          k_best = None
          for k in range(len(W[j])):
            #if k != PRIMITIVES.index('none'):
            if k_best is None or W[j][k] > W[j][k_best]:  ###   Choose best operation // We will see...
                k_best = k
          gene.append((PRIMITIVES[k_best], j))
        start = end
        n += 1
      return gene

    steps = 4; multiplier = 4
    concat = range(2 + steps - multiplier, steps+2) ## <==> range(2,6)
    genotype = Genotype(
      cell=_parse(alphas_normal), cell_concat=concat,
    )
    return genotype


def genotype2(alphas_normal):

    def _parse(weights):
      gene = []
      n = 2
      start = 0
      steps = 3
      for i in range(steps):
        end = start + n
        W = weights[start:end].copy()
        edges = sorted(range(i + 2)) # we are going to consider all input nodes
        for j in edges:
          k_best = None
          for k in range(len(W[j])):
            #if k != PRIMITIVES.index('none'):
            if k_best is None or W[j][k] > W[j][k_best]:  ###   Choose best operation // We will see...
                k_best = k
          gene.append((PRIMITIVES_SPIKE[k_best], j))
        start = end
        n += 1
      return gene

    steps = 4; multiplier = 4
    concat = range(2 + steps - multiplier, steps+2) ## <==> range(2,6)
    genotype = Genotype(
      cell=_parse(alphas_normal), cell_concat=concat,
    )
    return genotype

def plot(genotype, genotype2, filename):
  g = Digraph(
      format='pdf',
      edge_attr=dict(fontsize='20', fontname="times"),
      node_attr=dict(style='filled', shape='rect', align='center', fontsize='20', height='0.5', width='0.5', penwidth='2', fontname="times"),
      engine='dot')
  g.body.extend(['rankdir=LR'])

  g.node("c_{k-2}", fillcolor='darkseagreen2')
  g.node("c_{k-1}", fillcolor='darkseagreen2')
  # assert len(genotype) % 2 == 0
  #steps = len(genotype) // 2
  steps = 3
  for i in range(steps):
    g.node(str(i), fillcolor='lightblue')

  n = 2
  start = 0
  for i in range(steps):
    end = start + n
    #for k in [2*i, 2*i + 1]:
    for k in range(start , end):
      op, j = genotype[k]
      op2, j2 = genotype2[k]
      if op != 'none':
        if op == 'skip_connect':
            if j != 0:
                if j == 1:
                    u = "c_{k-1}"
                else:
                    u = str(j-2)
                v = str(i)
                g.edge(u, v, label=op, fillcolor="gray")
        else:
            if j != 0:
                if j == 1:
                    u = "c_{k-1}"
                else:
                    u = str(j-2)
                v = str(i)
                
                g.edge(u, v, label=op+' '+op2, fillcolor="gray")
        
        
    n +=1
    start = end
  g.node("c_{k}", fillcolor='palegoldenrod')
  for i in range(steps):
    g.edge(str(i), "c_{k}", fillcolor="gray")

  g.render(filename, view=True)


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


def obtain_decode_args():
    parser = argparse.ArgumentParser(description="LEStereo Decoding..")
    parser.add_argument('--dataset', type=str, default='sceneflow',
                        choices=['sceneflow', 'kitti15', 'kitti12', 'middlebury'],
                        help='dataset name (default: sceneflow)')
    parser.add_argument('--step', type=int, default=3)
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    return parser.parse_args()

class Loader(object):
    def __init__(self, args):
        self.args = args
        # Resuming checkpoint
        assert args.resume is not None, RuntimeError("No model to decode in resume path: '{:}'".format(args.resume))
        assert os.path.isfile(args.resume), RuntimeError("=> no checkpoint found at '{}'".format(args.resume))

        checkpoint = torch.load(args.resume)
        self._alphas_fea = checkpoint['module.feature.alphas']
        dis = OneHotCategorical(probs=self._alphas_fea)
        normal_alphas = dis.sample()
        self.decoder_fea = Decoder(alphas=normal_alphas, steps=self.args.step)

    def decode_cell(self):
        fea_genotype = self.decoder_fea.genotype_decode()
        return fea_genotype


if __name__ == '__main__':
    args = obtain_decode_args()

    assert args.resume is not None, RuntimeError("No model to decode in resume path: '{:}'".format(args.resume))
    assert os.path.isfile(args.resume), RuntimeError("=> no checkpoint found at '{}'".format(args.resume))

    model = torch.load(args.resume)
    # model = AutoStereo(3, 10, args = args)
    _alphas = model.feature.alphas
    _alphas_fea = torch.nn.functional.one_hot(torch.argmax(_alphas, dim=1))
    # _alphas_fea = model.module.feature.normalized_alphas
    _gammas = model.feature.gammas
    _gammas_fea = torch.nn.functional.one_hot(torch.argmax(_gammas, dim=1))
    # _gammas_fea = model.module.feature.normalized_gammas
    
    normal_alphas = _alphas_fea.cpu().numpy()
    normal_gammas = _gammas_fea.cpu().numpy()
    ex = genotype(normal_alphas)
    ex2 = genotype2(normal_gammas)
    plot(ex.cell, ex2.cell, 'cell_reduction.pdf')