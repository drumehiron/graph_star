import sys

import numpy as np

from torch_geometric.datasets import Planetoid
from utils.gsn_argparse import str2bool, str2actication
import os.path as osp
import torch_geometric.transforms as T
import torch_geometric.utils as gutils

import math
import random
import torch
from torch_geometric.utils import to_undirected

import ssl
import torch
from torch_geometric.nn import GAE
import trainer
import utils.gsn_argparse as gap

from torch_geometric.data import Data

from run_lp import load_data

import pandas as pd

import itertools

import pickle

def train_test_split_edges(data, val_ratio=0.05, test_ratio=0.1):
    r"""Splits the edges of a :obj:`torch_geometric.data.Data` object
    into positive and negative train/val/test edges, and adds attributes of
    `train_pos_edge_index`, `train_neg_adj_mask`, `val_pos_edge_index`,
    `val_neg_edge_index`, `test_pos_edge_index`, and `test_neg_edge_index`
    to :attr:`data`.

    Args:
        data (Data): The data object.
        val_ratio (float, optional): The ratio of positive validation
            edges. (default: :obj:`0.05`)
        test_ratio (float, optional): The ratio of positive test
            edges. (default: :obj:`0.1`)

    :rtype: :class:`torch_geometric.data.Data`
    """

    assert 'batch' not in data  # No batch-mode.

    row, col = data.edge_index
    data.edge_index = None

    # Return upper triangular portion.
    mask = row < col
    row, col = row[mask], col[mask]

    n_v = int(math.floor(val_ratio * row.size(0)))
    n_t = int(math.floor(test_ratio * row.size(0)))

    # Positive edges.
    perm = torch.randperm(row.size(0))
    row, col = row[perm], col[perm]

    r, c = row[:n_v], col[:n_v]
    data.val_pos_edge_index = torch.stack([r, c], dim=0)
    r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]
    data.test_pos_edge_index = torch.stack([r, c], dim=0)

    r, c = row[n_v + n_t:], col[n_v + n_t:]
    data.train_pos_edge_index = torch.stack([r, c], dim=0)
    data.train_pos_edge_index = to_undirected(data.train_pos_edge_index)

    # Negative edges.
    num_nodes = data.num_nodes
    neg_adj_mask = torch.ones(num_nodes, num_nodes, dtype=torch.uint8)
    neg_adj_mask = neg_adj_mask.triu(diagonal=1).to(torch.bool)
    neg_adj_mask[row, col] = 0

    neg_row, neg_col = neg_adj_mask.nonzero().t()
    perm = random.sample(range(neg_row.size(0)),
                         min(n_v + n_t, neg_row.size(0)))
    perm = torch.tensor(perm)
    perm = perm.to(torch.long)
    neg_row, neg_col = neg_row[perm], neg_col[perm]

    neg_adj_mask[neg_row, neg_col] = 0
    data.train_neg_adj_mask = neg_adj_mask

    row, col = neg_row[:n_v], neg_col[:n_v]
    data.val_neg_edge_index = torch.stack([row, col], dim=0)

    row, col = neg_row[n_v:n_v + n_t], neg_col[n_v:n_v + n_t]
    data.test_neg_edge_index = torch.stack([row, col], dim=0)

    return data

if __name__ == "__main__":

    # node_num
    num_nodes = 4317
    
    # raw_csv_path 
    edge_df = pd.read_csv('data/raw_data/filename.csv')
    
    # node column_name_1,column_name_2
    edge_company = torch.tensor(edge_df[["column_name_1","column_name_2"]].values, dtype=torch.long)

    sansan = Data(edge_index=edge_company.T)

    sansan.num_nodes = num_nodes

    train_test_split_edges(sansan)
    
    sansan.edge_index = torch.cat([sansan.train_pos_edge_index, sansan.val_pos_edge_index, sansan.test_pos_edge_index], dim=1)

    sansan.edge_train_mask = torch.cat([torch.ones((sansan.train_pos_edge_index.size(-1))),
                                      torch.zeros((sansan.val_pos_edge_index.size(-1))),
                                      torch.zeros((sansan.test_pos_edge_index.size(-1)))], dim=0).byte()
    sansan.edge_val_mask = torch.cat([torch.zeros((sansan.train_pos_edge_index.size(-1))),
                                    torch.ones((sansan.val_pos_edge_index.size(-1))),
                                    torch.zeros((sansan.test_pos_edge_index.size(-1)))], dim=0).byte()
    sansan.edge_test_mask = torch.cat([torch.zeros((sansan.train_pos_edge_index.size(-1))),
                                     torch.zeros((sansan.val_pos_edge_index.size(-1))),
                                     torch.ones((sansan.test_pos_edge_index.size(-1)))], dim=0).byte()

    sansan.edge_type = torch.zeros(((sansan.edge_index.size(-1)),)).long()
    sansan.batch = torch.zeros((1, sansan.num_nodes), dtype=torch.int64).view(-1)

    with open('data/big_company.pkl', 'wb') as f:
        pickle.dump(sansan,f)

    
