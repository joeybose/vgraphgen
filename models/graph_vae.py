import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from model.gcn import GCN
from utils.model_helper import *


class GraphVAE(nn.Module):
  """
    Variational Graph Generative Model
  """

  def __init__(self, config):
    super(GraphVAE, self).__init__()
    self.config = config
    self.device = config.device
    self.latent_dim = config.model.latent_dim
    self.max_num_nodes = config.model.max_num_nodes
    self.is_sym = config.model.is_sym
    self.has_node_feat = config.dataset.has_node_feat
    self.encoder_dim = config.model.encoder_dim
    self.decoder_dim = config.model.decoder_dim
    self.output_dim = self.max_num_nodes**2
    self.edge_weight = 1.0

    # Distribution over number of nodes
    self.num_nodes_prob = nn.Parameter(
        torch.zeros(self.max_num_nodes).float().to(
            torch.device(self.device)))  # shape N_max

    ### Encoder
    self.encoder = nn.ModuleList([
        nn.Linear(self.max_num_nodes, self.encoder_dim),
        nn.Linear(self.encoder_dim, self.encoder_dim),
        nn.Linear(self.encoder_dim, self.latent_dim * 2)
    ])

    ### Decoder
    self.decoder = nn.Sequential(
        nn.Linear(self.latent_dim, self.decoder_dim),
        nn.ReLU(inplace=True),
        nn.Linear(self.decoder_dim, self.decoder_dim),
        nn.ReLU(inplace=True),
        nn.Linear(self.decoder_dim, self.output_dim))

    ### Loss functions
    pos_weight = torch.ones([1]) * self.edge_weight
    self.adj_loss_func = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight, reduction='none')

  def _graph_conv_encoder(self, adj, state_in):
    sB, sN, _ = state_in.shape
    L = graph_laplacian(adj)
    state_out = state_in
    for tt in range(len(self.encoder) - 1):
      state_out = L.bmm(state_out)  # shape B X N X D
      state_out = torch.tanh(self.encoder[tt](state_out.view(sB * sN,
                                                             -1))).view(
                                                                 sB, sN, -1)

    num_nodes = (torch.sum(adj, dim=2) > 0).float().sum(
        dim=1, keepdim=True)  # B X 1
    state_out = torch.sum(state_out, dim=1) / num_nodes
    latent = self.encoder[-1](state_out)

    mu, logvar = latent.split(self.latent_dim, dim=1)

    return mu, logvar

  def reparameterize(self, mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

  def decode(self, z):
    ### output log theta
    return self.decoder(z)

  def forward(self,
              is_sampling=False,
              batch_size=None,
              A_gt=None,
              num_nodes_gt=None,
              node_mask=None):
    """

      M : initiator graph size
      N: maximum number of nodes per batch
      N_max: maximum number of nodes

      adj shape = N X N training
      adj shape = N_max X N_max inference

    Args:
      is_sampling: bool, whether do sampling
      batch_size: int, only useful during inference
      A_gt: B X N X N, ground-truth adjacency matrices
      num_nodes_gt: B X 1, gt number of nodes
    """
    if not is_sampling:
      assert num_nodes_gt is not None
      batch_size = A_gt.shape[0]
      batch_max_num_nodes = A_gt.shape[1]  # N
    else:
      assert self.training == False
      assert A_gt is None
      assert num_nodes_gt is None
      assert batch_size is not None

    ###########################################################################
    # Encoder
    ###########################################################################
    if not is_sampling:
      mu, logvar = self._graph_conv_encoder(A_gt, A_gt)
      z = self.reparameterize(mu, logvar)
    else:
      z = torch.randn(batch_size, self.latent_dim).to(self.device)

    log_theta = self.decode(z)
    log_theta = log_theta.view(batch_size, self.max_num_nodes,
                               self.max_num_nodes)

    if not is_sampling:
      KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) -
                             logvar.exp()) / float(batch_size)

      log_theta = log_theta[node_mask == 1].view(-1)
      A_gt = A_gt[node_mask == 1].view(-1)
      adj_loss = self.adj_loss_func(log_theta, A_gt).sum() / float(batch_size)

      self.num_nodes_prob.data[num_nodes_gt-1] += 1

      return adj_loss, KLD
    else:
      A = torch.bernoulli(torch.sigmoid(log_theta))

      # make it symmetric
      if self.is_sym:
        A = torch.tril(A, diagonal=-1)
        A = A + A.transpose(1, 2)

      # sampling during inference
      num_nodes_prob = self.num_nodes_prob / self.num_nodes_prob.sum()
      num_nodes = torch.multinomial(
          num_nodes_prob, batch_size, replacement=True) + 1  # shape B X 1

      A_list = [
          A[ii, :num_nodes[ii], :num_nodes[ii]] for ii in range(batch_size)
      ]
      return A_list


def graph_laplacian(A):
  ### remove diagonal & make it symmetric
  A = torch.tril(A, diagonal=-1)
  A = A + A.transpose(1, 2)

  ### add self loops
  A = A + torch.eye(A.shape[1]).expand(A.shape[0], -1, -1).to(A.device)

  ### normalize
  row_sum = torch.sum(A, dim=2, keepdim=True)  # shape B X N X 1
  D = 1.0 / row_sum.pow(0.5)  # shape B X N X 1
  L = D * A * D.transpose(1, 2)  # shape B X N X N

  return L


def draw_rand_permutation_mat(n):
  idx = torch.arange(n).long()
  idx_perm = torch.randperm(n).long()
  P = torch.zeros(n, n).float()  # shape N X N
  P[idx, idx_perm] = 1
  return P


