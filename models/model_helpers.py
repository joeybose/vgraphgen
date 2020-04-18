import math
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as geom_nn
from torch_geometric.nn.conv import MessagePassing


EPS = np.finfo(np.float32).eps
import torch_geometric
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.utils import to_undirected
from torch_geometric.nn import GCNConv, GATConv, MessagePassing
from typing import Tuple
import sys
import numpy as np
import torch.nn.functional as F
from torch import nn, Tensor
from utils.utils import reset, log_mean_exp, extract_batch, MultiInputSequential

class MLP(nn.Module):
    # MLP with linear output
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            self.linear = nn.Identity()
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = torch.relu(self.batch_norms[layer](self.linears[layer](h)))
            return self.linears[self.num_layers - 1](h)

    def reset_parameters(self):
        if self.linear_or_not:
            reset(self.linear)
        else:
            reset(self.linears)
            reset(self.batch_norms)

class GRANat(MessagePassing):
    def __init__(self, msg_dim, node_state_dim, edge_feat_dim, max_nodes, num_prop=1,
                 has_attention=True, att_hidden_dim=128,
                 has_residual=False, has_graph_output=False,
                 output_hidden_dim=128, graph_output_dim=None, readout_type='add', combine_layers=1):
        super(GRANat, self).__init__(aggr='add', flow="target_to_source")  # "Add" aggregation.
        self.msg_dim = msg_dim
        self.node_state_dim = node_state_dim
        self.edge_feat_dim = edge_feat_dim
        self.num_prop = num_prop
        self.has_attention = has_attention
        self.has_residual = has_residual
        self.att_hidden_dim = att_hidden_dim
        self.has_graph_output = has_graph_output
        self.output_hidden_dim = output_hidden_dim
        self.graph_output_dim = graph_output_dim
        self.max_nodes = max_nodes

        self.msg_func = nn.Sequential(*[nn.Linear(self.node_state_dim +
                                                  self.edge_feat_dim,
                                                  self.msg_dim),nn.ReLU(),
                                        nn.Linear(self.msg_dim, self.msg_dim)])
        self.readout = self.__get_readout_fn(readout_type)

        if self.has_attention:
            self.att_head = nn.Sequential( *[
                nn.Linear(self.node_state_dim + self.edge_feat_dim,
                          self.att_hidden_dim), nn.ReLU(),
                nn.Linear(self.att_hidden_dim, self.msg_dim), nn.Sigmoid() ])

        if self.has_graph_output:
            self.graph_output_head_att = nn.Sequential(*[
                nn.Linear(self.node_state_dim, self.output_hidden_dim), nn.ReLU(),
                nn.Linear(self.output_hidden_dim, 1), nn.Sigmoid() ])

            self.graph_output_head = nn.Sequential(
                *[nn.Linear(self.node_state_dim, self.graph_output_dim)])

        self.update_func = nn.GRUCell(input_size=self.msg_dim,
                                      hidden_size=self.node_state_dim)
        self.update_func.reset_parameters()

        self.R = MLP(
            num_layers=combine_layers,
            input_dim=self.node_state_dim,
            hidden_dim=self.output_hidden_dim,
            output_dim=self.output_hidden_dim)

    def __get_readout_fn(self, readout_type):
        options = {
            "add": geom_nn.global_add_pool,
            "mean": geom_nn.global_mean_pool,
            "max": geom_nn.global_max_pool
        }
        if readout_type not in options:
            raise ValueError()
        return options[readout_type]

    def forward(self, node_feat, edge, edge_feat, graph_idx=None):
        """
        N.B.: merge a batch of graphs as a single graph
        node_feat: N X D, node feature
        edge: M X 2, edge indices
        edge_feat: M X D', edge feature
        graph_idx: N X 1, graph indices
        """

        state = node_feat
        prev_state = state
        max_node = self.max_nodes
        batch_size = node_feat.shape[0] // max_node
        batch = torch.zeros(node_feat.shape[0]).long().to(node_feat.device)
        for i in range(batch_size):
            batch[(i) * max_node:(i + 1) * max_node] = i
        # this give a (batch_size, features) tensor
        # this give a (nodes, features) tensor
        for jj in range(self.num_prop):
            readout = self.readout(x=state, batch=batch)
            readout = readout[batch]
            state = self.propagate(edge_index=edge.T, x=state, edge_feat=edge_feat, readout=readout)


        if self.has_residual:
            state = state + prev_state

        # if self.has_graph_output:
        #     num_graph = graph_idx.max() + 1
        #     node_att_weight = self.graph_output_head_att(state)
        #     node_output = self.graph_output_head(state)
        #
        #     # weighted average
        #     reduce_output = torch.zeros(num_graph,
        #                                 node_output.shape[1]).to(node_feat.device)
        #     reduce_output = reduce_output.scatter_add(0,
        #                                         graph_idx.unsqueeze(1).expand(
        #                                             -1, node_output.shape[1]),
        #                                         node_output * node_att_weight)
        #
        #     const = torch.zeros(num_graph).to(node_feat.device)
        #     const = const.scatter_add( 0, graph_idx,
        #                               torch.ones(node_output.shape[0]).to(node_feat.device))
        #
        #     reduce_output = reduce_output / const.view(-1, 1)
        #
        #     return reduce_output
        else:
            return state

    def message(self, x_i, x_j, edge_feat, readout):
        ### compute message
        state_diff = x_i - x_j
        if self.edge_feat_dim > 0:
            edge_input = torch.cat([state_diff, edge_feat], dim=1)
        else:
            edge_input = state_diff

        msg = self.msg_func(edge_input)

        ### attention on messages
        if self.has_attention:
            att_weight = self.att_head(edge_input)
            msg = msg * att_weight

        return msg

    def update(self, aggr_out, x, readout):
        aggr_out = aggr_out + self.R(readout)
        aggr_out = self.update_func(aggr_out, x)
        return aggr_out

def calc_mi(args, model, test_loader):
    mi = 0
    num_examples = 0
    for i, data_batch in enumerate(test_loader):
        batch = extract_batch(args, data_batch)
        # Correct shapes for VGAE processing
        if len(batch[0]['adj'].shape) > 2:
            adj = batch[0]['adj'] + batch[0]['adj'].transpose(2,3)
            node_feats = adj.view(-1, args.max_nodes)
            # node_feats = batch[0]['adj'].view(-1, args.max_nodes)
        else:
            node_feats = batch[0]['adj']

        if batch[0]['edges'].shape[0] != 2:
            edge_index = batch[0]['encoder_edges'].long()
        else:
            edge_index = batch[0]['edges']
        batch_size = batch[0]['adj'].size(0)
        num_examples += batch_size
        mutual_info = model.calc_mi_q(node_feats, edge_index)
        mi += mutual_info * batch_size

    return mi / num_examples

def calc_au(model, test_loader, delta=0.01):
    """compute the number of active units
    """
    means = []
    for datum in test_loader:
        batch_data, _ = datum
        mean, _ = model.encode_stats(batch_data)
        means.append(mean)

    means = torch.cat(means, dim=0)
    au_mean = means.mean(0, keepdim=True)

    # (batch_size, nz)
    au_var = means - au_mean
    ns = au_var.size(0)

    au_var = (au_var ** 2).sum(dim=0) / (ns - 1)

    return (au_var >= delta).sum().item(), au_var
