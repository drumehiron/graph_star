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

from run_lp import load_data
import pickle
import pandas as pd

import itertools

MOKUMOKU = False

if __name__ == "__main__":

    num_features = 500

    with open('data/big_company.pkl', 'rb') as f:
        data=pickle.load(f) 
    
    data.x = torch.ones(data.num_nodes,num_features)
    data.num_graphs = 1 
    data.to(0)
   
    
    if MOKUMOKU:
        with open('data/big_company_all_node.pkl', 'rb') as f:
            all_egde=pickle.load(f)
        all_egde.batch = torch.zeros((1, data.num_nodes), dtype=torch.int64).view(-1)
        all_egde.to(0)

    model = torch.load('model/sansan_big_company.pkl')

    result = model(edge_index=data.edge_index,batch=data.batch,x=data.x)

    pei, pet = trainer.get_edge_info(data, "test")

    nei, net = data.test_neg_edge_index, data.test_neg_edge_index.new_zeros(
                        (data.test_neg_edge_index.size(-1),))

    ei = torch.cat([pei, nei], dim=-1)
    et = torch.cat([pet, net], dim=-1)

    flag = [True] * len(pei[1]) + [False] * len(nei[1]) 

    pred = model.lp_score(result[2], ei, et)
    egde = ei.cpu().numpy()
    prob = pred.cpu().detach().numpy()
    df_f = pd.DataFrame(data=[egde[0],egde[1],list(itertools.chain.from_iterable(prob)),flag])


    if MOKUMOKU:
        sansan_i = all_egde.edge_index
        sansan_t = all_egde.edge_index.new_zeros(all_egde.edge_index.size(-1))

        pred = model.lp_score(result[2], sansan_i, sansan_t)
        egde = sansan_i.cpu().numpy()
        prob = pred.cpu().detach().numpy()
        
        df_f = pd.DataFrame(data=[egde[0],egde[1],list(itertools.chain.from_iterable(prob))])

    df_f.T.to_csv('output/all_big_company.csv')

    torch.cuda.empty_cache()
