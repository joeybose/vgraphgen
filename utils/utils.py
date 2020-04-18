import torch
import torch.nn as nn
import os
import os.path as osp
import argparse
import ipdb
from collections import OrderedDict
import torch
import math
from statistics import median, mean
import random
import numpy as np
import copy
from torch._six import inf
import torch.nn as nn
import networkx as nx


def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)

class MultiInputSequential(nn.Sequential):
    def forward(self, *input):
        multi_inp = False
        if len(input) > 1:
            multi_inp = True
            _, edge_index = input[0], input[1]

        for module in self._modules.values():
            if multi_inp:
                if hasattr(module, 'weight'):
                    input = [module(*input)]
                # For ACR GNN's only
                elif hasattr(module.mlp.linear, 'weight'):
                    input = [module(*input)]
                else:
                    # Only pass in the features to the Non-linearity
                    input = [module(input[0]), edge_index]
            else:
                input = [module(*input)]
        return input[0]

class GRANMultiInputSequential(nn.Sequential):
    def forward(self, *input):
        multi_inp = False
        if len(input) > 1:
            multi_inp = True
            node_feat, edges, edge_feat = input[0], input[1], input[2]

        for module in self._modules.values():
            if multi_inp:
                if len(module.__dict__['_modules']) > 1:
                    input = [module(*input)]
                else:
                    # Only pass in the features to the Non-linearity
                    input = [module(input[0]), edges,
                             edge_feat]
            else:
                input = [module(*input)]
        return input[0]

def extract_batch(args, data_batch):
    batch_fwd = []
    data = {}
    if args.use_gran_data and not args.is_ar:
        data_batch = data_batch[0]
        data['adj'] = data_batch['adj'].pin_memory().to(args.dev, non_blocking=True)
        data['edges'] = data_batch['edges'].pin_memory().to(args.dev, non_blocking=True)
        data['node_idx_gnn'] = data_batch['node_idx_gnn'].pin_memory().to(args.dev, non_blocking=True)
        data['node_idx_feat'] = data_batch['node_idx_feat'].pin_memory().to(args.dev, non_blocking=True)
        data['label'] = data_batch['label'].pin_memory().to(args.dev, non_blocking=True)
        data['att_idx'] = data_batch['att_idx'].pin_memory().to(args.dev, non_blocking=True)
        data['subgraph_idx'] = data_batch['subgraph_idx'].pin_memory().to(args.dev, non_blocking=True)
        data['encoder_edges'] = data_batch['encoder_edges'].pin_memory().to(args.dev, non_blocking=True)
        data['adj_pad_idx'] = data_batch['adj_pad_idx'].pin_memory().to(args.dev, non_blocking=True)
        data['num_nodes_gt'] = data_batch['num_nodes_gt']
        batch_fwd.append(data)
        return batch_fwd
    elif args.use_gran_data and args.is_ar:
        data_batch = data_batch[0]
        data['adj'] = data_batch['adj'].pin_memory().to(args.dev, non_blocking=True)
        data['node_idx_feat'] = data_batch['node_idx_feat'].pin_memory().to(args.dev, non_blocking=True)
        data['subgraph_idx'] = data_batch['subgraph_idx'].pin_memory().to(args.dev, non_blocking=True)
        data['encoder_edges'] = data_batch['encoder_edges'].pin_memory().to(args.dev, non_blocking=True)
        data['num_nodes_gt'] = data_batch['num_nodes_gt']
        data['idx_base'] = data_batch['idx_base']
        # AR
        data['edges_ar'] = data_batch['edges_ar']
        data['node_idx_gnn_ar'] = data_batch['node_idx_gnn_ar']
        data['att_idx_ar'] = data_batch['att_idx_ar']
        data['label_ar'] = data_batch['label_ar']
        batch_fwd.append(data)
        return batch_fwd
    else:
        data['adj'] = data_batch.x.pin_memory().to(args.dev, non_blocking=True)
        data['edges'] = data_batch.edge_index.pin_memory().to(args.dev, non_blocking=True)
        batch_fwd.append(data)
        return batch_fwd

def get_graph(adj):
    """ get a graph from zero-padded adj """
    # remove all zeros rows and columns
    adj = adj[~np.all(adj == 0, axis=1)]
    adj = adj[:, ~np.all(adj == 0, axis=0)]
    adj = np.asmatrix(adj)
    G = nx.from_numpy_matrix(adj)
    return G

def create_selfloop_edges(num_nodes):
    edges = []
    for i in range(0, num_nodes):
        edges.append((int(i),int(i)))

    return edges

def perm_node_feats(feats):
    num_nodes = feats.size(0)
    perm = torch.randperm(feats.size(0))
    perm_idx = perm[:num_nodes]
    feats = feats[perm_idx]
    return feats

def log_mean_exp(value, dim=0, keepdim=False):
    return log_sum_exp(value, dim, keepdim) - math.log(value.size(dim))

def log_sum_exp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation
    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0), dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        return m + torch.log(sum_exp)

def log_sum_exp_signs(value, signs, dim=0, keepdim=False):
    m, _ = torch.max(value, dim=dim, keepdim=True)
    value0 = value - m
    if keepdim is False:
        m = m.squeeze(dim)
    return m + torch.log(torch.sum(signs * torch.exp(value0), dim=dim, keepdim=keepdim))

def str2bool(v: str) -> bool:
    v = v.lower()
    if v == "true":
        return True
    elif v == "false":
        return False
    else:
        raise argparse.ArgumentTypeError(f"Boolean value expected, got '{v}'.")

def filter_state_dict(state_dict,name):
    keys_to_del = []
    for key in state_dict.keys():
        if name in key:
            keys_to_del.append(key)
    for key in sorted(keys_to_del, reverse=True):
        del state_dict[key]
    return state_dict

''' Set Random Seed '''
def seed_everything(seed):
    if seed:
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def custom_clip_grad_norm_(parameters, max_norm, norm_type=2):
    r"""Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p is not int(0), parameters))
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if norm_type == inf:
        total_norm = max(p.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            param_norm = p.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.data.mul_(clip_coef)
    return total_norm

'''Monitor Norm of gradients'''
def monitor_grad_norm(model):
    parameters = list(filter(lambda p: p.grad is not None, model.parameters()))
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm

'''Monitor Norm of gradients'''
def monitor_grad_norm_2(gradients):
    total_norm = 0
    for p in gradients:
        if p is not int(0):
            param_norm = p.data.norm(2)
            total_norm += param_norm ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm

'''Monitor Norm of weights'''
def monitor_weight_norm(model):
    parameters = list(filter(lambda p: p is not None, model.parameters()))
    total_norm = 0
    for p in parameters:
        param_norm = p.data.norm(2)
        total_norm += param_norm ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm

def uniform(size, tensor):
    bound = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=20, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
        elif score < self.best_score - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss

def project_name(dataset_name):
    if dataset_name:
        return "dl4gg-{}".format(dataset_name)
    else:
        return "dl4gg"

def constraint_mask(args, model):
    model_mask = model.mask.view(-1, args.max_nodes)
    num_nodes = [graph_mask.sum() for graph_mask in model_mask]
    fully_connected_list = [nx.complete_graph(nodes2gen) for nodes2gen in num_nodes]
    edges_list = [torch.tensor(list(G.edges)).to(args.dev).t().contiguous() for G in fully_connected_list]
    constraint_edge_index = torch.cat([edges + ii * args.max_nodes for ii,
                                       edges in enumerate(edges_list)], dim=1)

    return constraint_edge_index

def pad_adj_mat(args, A_list):
    padded_adj_list = []
    for adj_mat in A_list:
        num_nodes = adj_mat.shape[-1]
        nodes_to_pad = args.max_nodes - num_nodes
        col_zeros = torch.zeros(num_nodes, nodes_to_pad).to(adj_mat.device)
        adj_mat = torch.cat((adj_mat, col_zeros),dim=1)
        row_zeros = torch.zeros(nodes_to_pad,
                args.max_nodes).to(adj_mat.device)
        adj_mat = torch.cat((adj_mat, row_zeros),dim=0)
        padded_adj_list.append(adj_mat)
    return torch.stack(padded_adj_list)


class Constants(object):
    eta = 1e-5
    log2 = math.log(2)
    logpi = math.log(math.pi)
    log2pi = math.log(2 * math.pi)
    logceilc = 88                # largest cuda v s.t. exp(v) < inf
    logfloorc = -104             # smallest cuda v s.t. exp(v) > 0
    invsqrtsubsub2pi = 1. / math.sqrt(2 * math.pi)
    sqrthalfpi = math.sqrt(math.pi/2)


def get_subgraph(args, b, t):
    sub_batch = []
    # t = 10
    valid_graphs = (t < b['num_nodes_gt']).nonzero()
    num_batch = len(valid_graphs)
    assert len(valid_graphs.size()) > 0
    subgraph_size = [t for i in range(len(valid_graphs))]
    cum_size = torch.tensor(np.cumsum([0] + subgraph_size))

    # There are (t+1) nodes in the graph at timestep t
    total_idx = [torch.arange((t*(t+1)),(t+1)*(t+1)) + jj * (t+1)*(t+1) for jj in
                 range(num_batch)]
    adj_pad_idx = torch.cat(total_idx).long().pin_memory().to(args.dev,
                                                              non_blocking=True)

    edges_ar = torch.cat([b['edges_ar'][i][t] + cum_size[cum_idx] for cum_idx, i in
                          enumerate(valid_graphs)],
                         dim=1).pin_memory().to(args.dev, non_blocking=True)

    node_idx_gnn_ar = torch.cat([torch.tensor(b['node_idx_gnn_ar'][i][t]) +
                                 cum_size[cum_idx] for cum_idx, i in
                                 enumerate(valid_graphs)],
                                dim=0).long().pin_memory().to(args.dev,
                                                              non_blocking=True)

    subgraph_idx_base = np.array([0] + [1 for i in range(num_batch)])
    subgraph_idx =[np.zeros(t+1).astype(np.int64) for i in range(num_batch)]
    subgraph_idx = np.concatenate([subgraph_idx[ii] + subgraph_idx_base[ii] for
                                   ii in range(num_batch) ])
    subgraph_idx = torch.from_numpy(subgraph_idx).long().pin_memory().to(args.dev,
                                                          non_blocking=True)

    node_idx_gnn_ar = torch.cat([torch.tensor(b['node_idx_gnn_ar'][i][t]) +
                                 cum_size[cum_idx] for cum_idx, i in
                                 enumerate(valid_graphs)],
                                dim=0).long().pin_memory().to(args.dev,
                                                              non_blocking=True)

    label_ar = torch.cat([torch.tensor(b['label_ar'][i][t]) for i in
                          valid_graphs],
                         dim=0).float().pin_memory().to(args.dev,
                                                       non_blocking=True)
    att_idx_ar = torch.cat([torch.tensor(b['att_idx_ar'][i][t]) for i in
                          valid_graphs],
                         dim=0).long().pin_memory().to(args.dev,
                                                       non_blocking=True)
    node_idx_feat = b['node_idx_feat']

    sub_graph = dict(adj=b['adj'][:, :, :(t+1), :(t+1)],
                     adj_pad_idx=adj_pad_idx, edges=edges_ar,
                     node_idx_gnn=node_idx_gnn_ar, att_idx=att_idx_ar,
                     label=label_ar, node_idx_feat=node_idx_feat,
                     subgraph_idx=subgraph_idx, timestep=t)

    sub_batch.append(sub_graph)

    return sub_batch


def aggressive_burnin(args, epoch, kl_weight, anneal_rate, aggressive_flag,
                      data_batch, model, enc_opt, dec_opt, train_loader,
                      constraints, oracle):
    burn_num_examples = 0
    burn_pre_loss = 1e4
    burn_cur_loss = 0
    sub_iter = 1
    collate_fn = train_loader.collate_fn
    dataset_iter = iter(train_loader.dataset)

    while aggressive_flag and sub_iter < args.sub_iter_limit:
        batch = extract_batch(args, data_batch)
        # Correct shapes for VGAE processing
        if len(batch[0]['adj'].shape) > 2:
            # We give the full Adj to the encoder
            adj = batch[0]['adj'] + batch[0]['adj'].transpose(2,3)
            node_feats = adj.view(-1, args.max_nodes)
            # node_feats = batch[0]['adj'].view(-1, args.max_nodes)
        else:
            node_feats = batch[0]['adj']

        if batch[0]['edges'].shape[0] != 2:
            edge_index = batch[0]['encoder_edges'].long()
        else:
            edge_index = batch[0]['edges']

        enc_opt.zero_grad()
        dec_opt.zero_grad()

        burn_num_examples += batch[0]['adj'].size(0)
        z, z_k = model.encode(node_feats, edge_index)
        batch[0]['node_latents'] = z_k
        batch[0]['return_adj'] = True
        recon_loss, soft_adj_mat = model.decode(batch)

        # compute KL for Real nodes
        kl_weight = min(1.0, kl_weight + anneal_rate)
        kl_loss = model.kl_loss(z, z_k, free_bits=args.free_bits)
        # Check whether we want to enforce constraints at Train time
        eval_const_cond = constraints is not None and not args.eval_const_test
        if eval_const_cond:
            input_batches = z_k.view(-1, args.max_nodes, args.z_dim)
            domains, max_node_batches = oracle.constraint.get_domains(input_batches)
            if args.decoder == 'gran':
                avg_neg_loss, z_star_batches = oracle.general_pgd(input_batches, domains,
                        max_node_batches, batch, num_restarts=1,
                        num_iters=args.num_iters, args=args)
                batch[0]['node_latents'] = z_star_batches
                _, adj_star = model.decode(batch)
            else:
                constraint_edge_index = constraint_mask(args, model)
                avg_neg_loss, z_star_batches = oracle.general_pgd(input_batches, domains,
                        max_node_batches, constraint_edge_index,
                        num_restarts=1, num_iters=args.num_iters,
                        args=args)
                _, adj_star = model.decoder(z_star_batches, constraint_edge_index, return_adj=True)
            _, constraint_loss, constr_sat = oracle.evaluate(args, max_node_batches,  adj_star)
        else:
            constraint_loss = recon_loss * 0
            constr_sat = recon_loss * 0
            avg_neg_loss = recon_loss * 0

        train_loss = recon_loss + kl_weight*kl_loss + args.constraint_weight*constraint_loss
        burn_cur_loss += train_loss.item()
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

        enc_opt.step()
        ids = np.random.choice(len(train_loader.dataset), args.batch_size, replace=False)
        data_batch = collate_fn([train_loader.dataset[idx] for idx in ids])

        if sub_iter % 10 == 0:
            burn_cur_loss = burn_cur_loss / burn_num_examples
            if burn_pre_loss - burn_cur_loss < 0:
                break
            burn_pre_loss = burn_cur_loss
            burn_cur_loss = burn_num_examples = 0

        sub_iter += 1

    return sub_iter
