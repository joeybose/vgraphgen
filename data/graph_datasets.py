import os
import os.path as osp
import torch
import pickle
import glob
import time
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import shutil
import ipdb

import networkx as nx
import torch_geometric.transforms as T
from torch_geometric.data import Data,Dataset,DataLoader,DataListLoader
from torch_geometric.data import Data,Dataset,DataListLoader
from torch_geometric.utils import remove_self_loops
from typing import Any, List, Tuple
import urllib
from random import shuffle
import torch.nn.functional as F
from .data_helper import *
from .data_utils import *
from .gran_data import *


class NXDataset:
    def __init__(self, name, train_batch_size, num_fixed_features, node_order,
                 use_rand_feats, seed, train_ratio=0.8, val_ratio=0.1):
        path = osp.join(
            osp.dirname(osp.realpath(__file__)), '..', 'data', name)
        nx_dataset, self.max_nodes, self.max_edges, self.node_dist = create_nx_graphs(args,name,seed)
        self.train_batch_size = train_batch_size
        self.feats = 0.3*torch.randn(self.max_nodes,num_fixed_features,requires_grad=False)
        self.reconstruction_loss = None
        self.dataset = []
        for nx_graph in nx_dataset:
            num_nodes = len(nx_graph.nodes)
            nodes_to_pad = self.max_nodes - num_nodes
            perm = torch.randperm(self.feats.size(0))
            perm_idx = perm[:num_nodes]
            feats = self.feats[perm_idx]
            adj_mat = torch.Tensor(nx.to_numpy_matrix(nx_graph))
            col_zeros = torch.zeros(num_nodes, nodes_to_pad)
            adj_mat = torch.cat((adj_mat, col_zeros),dim=1)
            row_zeros = torch.zeros(nodes_to_pad, self.max_nodes)
            adj_mat = torch.cat((adj_mat, row_zeros),dim=0)
            edge_index = torch.tensor(list(nx_graph.edges)).t().contiguous()
            edge_index, _ = remove_self_loops(edge_index)
            if use_rand_feats:
                self.num_features = num_fixed_features
                self.dataset.append(Data(edge_index=edge_index, x=feats))
            else:
                self.dataset.append(Data(edge_index=edge_index, x=adj_mat))
                self.num_features = self.max_nodes

        train_cutoff = int(np.round(train_ratio*len(self.dataset)))
        val_cutoff = train_cutoff + int(np.round(val_ratio*len(self.dataset)))
        self.train_dataset = self.dataset[:train_cutoff]
        self.val_dataset = self.dataset[train_cutoff:val_cutoff]
        self.test_dataset = self.dataset[val_cutoff:]

    def create_loaders(self) -> Tuple[DataLoader, DataLoader]:
        train_loader = DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=False)
        val_loader = DataLoader(self.val_dataset, batch_size=self.train_batch_size, shuffle=False)
        test_loader = DataLoader(self.test_dataset, batch_size=2, shuffle=False)
        return train_loader, val_loader, test_loader


class GRANDataset(object):
    def __init__(self, args, config, name, seed, tag='train'):
        self.config = config
        self.graphs, self.max_nodes, self.max_edges, self.node_dist = create_nx_graphs(args, name,seed)
        self.train_ratio = config.dataset.train_ratio
        self.dev_ratio = config.dataset.dev_ratio
        self.block_size = config.model.block_size
        self.stride = config.model.sample_stride
        self.num_graphs = len(self.graphs)
        self.num_train = int(float(self.num_graphs) * self.train_ratio)
        self.num_dev = int(float(self.num_graphs) * self.dev_ratio)
        self.num_test_gt = self.num_graphs - self.num_train
        self.num_test_gen = config.test.num_test_gen
        self.args = args
        self.num_features = self.max_nodes
        self.train_batch_size = config.train.batch_size
        self.test_batch_size = config.test.batch_size
        self.num_workers = config.train.num_workers
        self.graphs_train = self.graphs[:self.num_train]
        self.graphs_dev = self.graphs[:self.num_dev]
        self.graphs_test = self.graphs[self.num_train:]
        self.num_nodes_pmf_train = np.bincount([len(gg.nodes) for gg in
                                                self.graphs_train])
        self.max_num_nodes = len(self.num_nodes_pmf_train)
        self.num_nodes_pmf_train = self.num_nodes_pmf_train / self.num_nodes_pmf_train.sum()

        self.train_dataset = GRANData(self.config, self.graphs_train, tag)
        self.val_dataset = GRANData(self.config, self.graphs_dev, tag='dev')
        self.test_dataset = GRANData(self.config, self.graphs_test, tag='test')

    def create_loaders(self):
        train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                   batch_size=self.train_batch_size,
                                                   shuffle=True,
                                                   num_workers=self.num_workers,
                                                   collate_fn=self.train_dataset.collate_fn,
                                                   drop_last=False)
        val_loader = torch.utils.data.DataLoader(self.val_dataset,
                                                   batch_size=self.train_batch_size,
                                                   shuffle=True,
                                                   num_workers=self.num_workers,
                                                   collate_fn=self.test_dataset.collate_fn,
                                                   drop_last=False)
        test_loader = torch.utils.data.DataLoader(self.test_dataset,
                                                   batch_size=self.train_batch_size,
                                                   shuffle=True,
                                                   num_workers=self.num_workers,
                                                   collate_fn=self.test_dataset.collate_fn,
                                                   drop_last=False)
        return train_loader, val_loader, test_loader
