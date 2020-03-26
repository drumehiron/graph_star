import sys

from torch_geometric.datasets import Planetoid
from utils.gsn_argparse import str2bool, str2actication
import os.path as osp
import torch_geometric.transforms as T
import torch_geometric.utils as gutils

import ssl
import torch
from torch_geometric.nn import GAE
import trainer
import utils.gsn_argparse as gap

from IPython import embed

import pickle

ssl._create_default_https_context = ssl._create_unverified_context


def main(_args):
    args = gap.parser.parse_args(_args)

    with open('data/big_company.pkl', 'rb') as f:
        data=pickle.load(f)    

    num_features = 500

    data.x = torch.ones(data.num_nodes,num_features)
    data.num_graphs = 1

    for i in range(5):
        print("===========================================")
        trainer.trainer(args, args.dataset, [data], [data], [data], transductive=True,
                        num_features=500, max_epoch=args.epochs,
                        num_node_class=0,
                        link_prediction=True,
                        modelname=args.modelname)

if __name__ == '__main__':
    main(sys.argv[1:])
