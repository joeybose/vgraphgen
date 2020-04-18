import math
import random

import torch
import numpy as np
import torch_geometric
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.utils import to_undirected

from typing import Tuple
from .gnn_layers import *
import sys
import numpy as np
import torch.nn.functional as F
from torch import nn, Tensor
from distributions.normal import EuclideanNormal, MultivariateEuclideanNormal
from utils.utils import log_sum_exp, filter_state_dict, MultiInputSequential
from .gran_mixture_bernoulli import GRANMixtureBernoulli

sys.path.append("..")  # Adds higher directory to python modules path.
from flows.flows import MAFRealNVP, RealNVP

kwargs_flows = {'MAFRealNVP': MAFRealNVP, 'RealNVP': RealNVP}
kwargs_enc_conv = {'GCN': GCNLayer, 'GAT': GATLayer,'ACR':ACRConv}

EPS = 1e-15
LOG_VAR_MAX = 10
LOG_VAR_MIN = EPS

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

def negative_sampling(pos_edge_index, num_nodes):
    idx = (pos_edge_index[0] * num_nodes + pos_edge_index[1])
    idx = idx.to(torch.device('cpu'))

    rng = range(num_nodes**2)
    perm = torch.tensor(random.sample(rng, idx.size(0)))
    mask = torch.from_numpy(np.isin(perm, idx).astype(np.uint8))
    rest = mask.nonzero().view(-1)
    while rest.numel() > 0:  # pragma: no cover
        tmp = torch.tensor(random.sample(rng, rest.size(0)))
        mask = torch.from_numpy(np.isin(tmp, idx).astype(np.uint8))
        perm[rest] = tmp
        rest = mask.nonzero().view(-1)

    row, col = perm / num_nodes, perm % num_nodes
    return torch.stack([row, col], dim=0).to(pos_edge_index.device)


def negative_sampling_mask(pos_edge_index, num_nodes, batch_size, mask):

    N = int(num_nodes // batch_size)
    assert N * batch_size == num_nodes

    ### create batched adj. matrix
    bs_idx   = pos_edge_index[0] / N
    edge_idx = pos_edge_index % N

    # bs_idx[i] * (N * N) + edge_idx[0, i] * N + edge_idx[1, i]
    flat_adj      = torch.ByteTensor(batch_size * N * N).to(mask.device).fill_(0)
    flattened_idx = bs_idx * (N * N) + edge_idx[0] * N + edge_idx[1]

    flat_adj[flattened_idx] = 1
    adj = flat_adj.view(batch_size, N, N)

    # make symmetric
    adj = (adj + adj.transpose(-2, -1)).clamp_(max=1)

    # first we need to find these padded nodes. To do so, find the lenght of each ex.
    lens = (adj.sum(-1) > 0).sum(-1)

    # note that adj[i, lens[i]].sum() == 0, but adj[i, lens[i] - 1].sum() > 0

    # in the case that lens.max() == N
    # (this is incorrect, as edges not existing for the last row won't be chosen for
    # negative sampling)
    lens.clamp_(max=N - 1)

    mask = torch.zeros_like(adj)
    batch_arange = torch.arange(batch_size).to(lens.device)
    mask[batch_arange, lens, :] = 1
    mask[batch_arange, :, lens] = 1
    mask = 1 - (mask.cumsum(-1).cumsum(-2) > 0).byte()

    # at this point, mask[b, i, j] = 1 i.f.f node_i and node_j exist for the b'th graph
    # now, mask out already existing edges

    valid_neg_edge = mask.byte() * (1 - adj).byte()

    # for simplicity, let's count every edge only once, and consider the lower triang. part
    valid_neg_edge = torch.tril(valid_neg_edge)

    # at this point, valid_neg_edge[b, i, j] = 0 i.f.f b'th graph has an edge from (i,j)
    # finally, all that remains is to sample from this collection of negative edges

    # TODO: how many samples to draw ?
    avg_pos_edge = int(pos_edge_index.size(1) // batch_size + 1)

    # flatten the 2D adj so that we sample the same amount per elem in batch size
    valid_neg_edge_flat = valid_neg_edge.view(batch_size, N * N)
    valid_neg_edge_flat_p = valid_neg_edge_flat.float() / valid_neg_edge_flat.sum(-1).unsqueeze(-1)
    neg_samples = torch.multinomial(valid_neg_edge_flat_p, num_samples=avg_pos_edge, replacement=True)

    # 'neg_samples` is a (bs, K) matrix, where K ranges in 0 to N^2.
    # neg_samples[0, 2] = 12 and N == 10, it means you have an edge from node 1 --> 2

    neg_edge_i = batch_arange.unsqueeze(1) * N + neg_samples / N
    neg_edge_j = batch_arange.unsqueeze(1) * N + neg_samples % N

    neg_edge_i = neg_edge_i.flatten()
    neg_edge_j = neg_edge_j.flatten()

    neg_edge_index = torch.stack((neg_edge_i, neg_edge_j))
    return neg_edge_index


class Encoder(torch.nn.Module):
    def __init__(self, args, n_blocks, conv_type, input_dim, hidden_dim, z_dim,
                 ll_estimate, K, flow_args, dev):
        super(Encoder, self).__init__()
        self.args = args
        self.n_blocks = n_blocks
        self.conv1, self.conv_mu, self.conv_logvar = self.create_enc_blocks(input_dim, hidden_dim, z_dim, n_blocks,
                                                                            conv_type)
        self.type = 'euclidean'
        self.dev = dev
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.ll_estimate = ll_estimate
        self.K = K
        n_blocks, flow_hidden_size, n_hidden = flow_args[0], flow_args[1], flow_args[2]
        flow_model, flow_layer_type = flow_args[3], flow_args[4]
        if flow_model is not None and not args.fit_prior:
            self.flow_model = kwargs_flows[flow_model](n_blocks, self.z_dim,
                                                       flow_hidden_size,
                                                       n_hidden,
                                                       flow_layer_type).to(self.dev)
            self.analytic_kl = False
        else:
            self.flow_model = None
            self.analytic_kl = True

    def create_enc_blocks(self, input_dim, hidden_dim, z_dim, n_blocks,
                          conv_type='GCN'):
        # Build the Flow Block by Block
        if conv_type == 'GCN':
            GCNConv.cached = False
        elif conv_type == 'GAT':
            GATConv.heads = 8

        block_net_enc = [kwargs_enc_conv[conv_type](input_dim, hidden_dim, self.args.max_nodes)]
        for i in range(n_blocks):
            block_net_enc += [nn.ReLU(), kwargs_enc_conv[conv_type](hidden_dim, hidden_dim, self.args.max_nodes)]

        enc = MultiInputSequential(*block_net_enc)
        conv_mu = kwargs_enc_conv[conv_type](hidden_dim, z_dim, self.args.max_nodes)
        conv_logvar = kwargs_enc_conv[conv_type](hidden_dim, z_dim, self.args.max_nodes)
        return enc, conv_mu, conv_logvar

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        logvar = self.conv_logvar(x, edge_index)
        return self.conv_mu(x, edge_index), self.conv_logvar(x, edge_index)


class InnerProductDecoder(torch.nn.Module):
    # TODO: InnerProduct currently broken.
    r"""The inner product decoder from the `"Variational Graph Auto-Encoders"
    <https://arxiv.org/abs/1611.07308>`_ paper
    .. math::
        \sigma(\mathbf{Z}\mathbf{Z}^{\top})
    where :math:`\mathbf{Z} \in \mathbb{R}^{N \times d}` denotes the latent
    space produced by the encoder."""

    def forward(self, z, edge_index, sigmoid=True, return_adj=False):
        r"""Decodes the latent variables :obj:`z` into edge probabilties for
        the given node-pairs :obj:`edge_index`.
        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        if sigmoid: value = torch.sigmoid(value)

        if return_adj:
            adj = torch.matmul(z, z.t())
            if sigmoid: adj = torch.sigmoid(adj)

            return value, adj

        return value


class TanhDecoder(torch.nn.Module):
    """TanH to compute edge probabilities based on distances."""

    def __init__(self):
        super(TanhDecoder, self).__init__()

    def forward(self, z, edge_index, return_adj=False):
        dist = -1*F.pairwise_distance(z[edge_index[0]], z[edge_index[1]])
        scores = torch.tanh(dist).squeeze()

        if return_adj:
            raise NotImplementedError
            adj = torch.tanh(-1 * F.pairwise_distance(z, z))
            return scores, adj

        return scores


class SoftmaxDecoder(torch.nn.Module):
    """Distance to compute edge probabilities based on distances."""

    def __init__(self, p):
        super(SoftmaxDecoder, self).__init__()
        self.p = torch.nn.Parameter(torch.tensor(p))

    def forward(self, z, edge_index, return_adj=False):
        dist = 1./ F.pairwise_distance(z[edge_index[0]], z[edge_index[1]])
        scores = torch.sigmoid(self.p) * F.softmax(dist)
        probs = scores * 1. / max(scores)

        if return_adj:
            raise NotImplementedError
            adj = torch.eye(z.shape[0])
            for i in range(0, len(edge_index[0])):
                adj[edge_index[0][i]][edge_index[1][i]] = probs[i]

            return probs, adj

        return probs


class DistanceDecoder(torch.nn.Module):
    """Distance to compute edge probabilities based on distances."""

    def __init__(self, conv_type, r, t, hidden_dim, z_dim, n_blocks, max_nodes):
        super(DistanceDecoder, self).__init__()

        self.max_nodes = max_nodes

        block_net_r = [kwargs_enc_conv[conv_type](z_dim, hidden_dim)]
        block_net_t = [kwargs_enc_conv[conv_type](z_dim, hidden_dim)]
        for i in range(n_blocks):
            block_net_r += [nn.ReLU(), kwargs_enc_conv[conv_type](hidden_dim, hidden_dim)]
            block_net_t += [nn.ReLU(), kwargs_enc_conv[conv_type](hidden_dim, hidden_dim)]

        block_net_r += [nn.ReLU(), kwargs_enc_conv[conv_type](hidden_dim,
                                                              int(hidden_dim/2))]
        block_net_t += [nn.ReLU(), kwargs_enc_conv[conv_type](hidden_dim,
                                                              int(hidden_dim/2))]
        self.r = MultiInputSequential(*block_net_r)
        self.r_mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.2),
                                     nn.Linear(hidden_dim, 1))
        self.t_mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.2),
                                     nn.Linear(hidden_dim, 1))
        self.t = MultiInputSequential(*block_net_r)


    def forward(self, z, edge_index, return_adj=False):
        dist = -1*F.pairwise_distance(z[edge_index[0]], z[edge_index[1]])
        r_gnn = self.r(z, edge_index)
        t_gnn = self.t(z, edge_index)
        inp_r = torch.cat((r_gnn[edge_index[0]],r_gnn[edge_index[1]]), dim=1)
        inp_t = torch.cat((t_gnn[edge_index[0]],t_gnn[edge_index[1]]), dim=1)
        r = self.r_mlp(inp_r).squeeze()
        t = self.t_mlp(inp_t).squeeze()
        probs = torch.sigmoid((dist - r) / t).squeeze()

        if return_adj:
            N = self.max_nodes
            B = max(1, z.size(0) // N)
            adj = torch.zeros(B, N, N).to(z.device)

            bs_idx   = edge_index[0] / N
            edge_idx = edge_index % N

            # when adj is flattened, index of every slot will be
            # bs_idx[i] * (N * N) + edge_idx[0, i] * N + edge_idx[1, i]
            flattened_idx = bs_idx * (N * N) + edge_idx[0] * N + edge_idx[1]
            flat_adj = adj.flatten()

            flat_adj[flattened_idx] = probs

            return probs, flat_adj.view_as(adj)

        return probs


class GAE(torch.nn.Module):
    r"""The Graph Auto-Encoder model from the
    `"Variational Graph Auto-Encoders" <https://arxiv.org/abs/1611.07308>`_
    paper based on user-defined encoder and decoder models.
    Args:
        encoder (Module): The encoder module.
        decoder (Module, optional): The decoder module. If set to :obj:`None`,
            will default to the
            :class:`torch_geometric.nn.models.InnerProductDecoder`.
            (default: :obj:`None`)
    """

    def __init__(self, encoder, decoder=None):
        super(GAE, self).__init__()
        self.encoder = encoder
        self.decoder = InnerProductDecoder() if decoder is None else decoder
        self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.encoder)
        reset(self.decoder)

    def encode(self, *args, **kwargs):
        r"""Runs the encoder and computes node-wise latent variables."""
        return self.encoder(*args, **kwargs)

    def decode(self, *args, **kwargs):
        r"""Runs the decoder and computes edge probabilties."""
        return self.decoder(*args, **kwargs)

    def kl_loss(self, z_0, z_k, mu=None, logvar=None):
        r"""There is no KL here as everything is deterministic"""
        return torch.Tensor([0]).to(self.dev)

    def split_edges(self, data, val_ratio=0.05, test_ratio=0.1):
        r"""Splits the edges of a :obj:`torch_geometric.data.Data` object
        into positve and negative train/val/test edges.
        Args:
            data (Data): The data object.
            val_ratio (float, optional): The ratio of positive validation
                edges. (default: :obj:`0.05`)
            test_ratio (float, optional): The ratio of positive test
                edges. (default: :obj:`0.1`)
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
        neg_adj_mask = neg_adj_mask.triu(diagonal=1)
        neg_adj_mask[row, col] = 0

        neg_row, neg_col = neg_adj_mask.nonzero().t()
        perm = torch.tensor(random.sample(range(neg_row.size(0)), n_v + n_t))
        perm = perm.to(torch.long)
        neg_row, neg_col = neg_row[perm], neg_col[perm]

        neg_adj_mask[neg_row, neg_col] = 0
        data.train_neg_adj_mask = neg_adj_mask

        row, col = neg_row[:n_v], neg_col[:n_v]
        data.val_neg_edge_index = torch.stack([row, col], dim=0)

        row, col = neg_row[n_v:n_v + n_t], neg_col[n_v:n_v + n_t]
        data.test_neg_edge_index = torch.stack([row, col], dim=0)

        return data

    def decode(self, decoder_args):
        r""" Rewrap decoders """
        if self.decoder_name != 'gran':
            loss = self.recon_loss(decoder_args[0]['node_latents'],
                                   decoder_args[0]['edges'])
            if decoder_args[0]['return_adj']:
                adj_mat = self.decoder(decoder_args[0]['node_latents'],
                                       decoder_args[0]['edges'],
                                       return_adj=True)[-1]
                return loss, adj_mat

            return loss
        else:
            return self.decoder(*decoder_args)

    def recon_loss(self, z, pos_edge_index):
        r"""Given latent variables :obj:`z`, computes the binary cross
        entropy loss for positive edges :obj:`pos_edge_index` and negative
        sampled edges.
        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (LongTensor): The positive edges to train against.
        """

        pos_loss = -torch.log(
            self.decoder(z, pos_edge_index) + EPS).mean()

        # make sure the negative sampling step only targets valid non-edges

        if hasattr(self, 'mask'):
            neg_edge_index = negative_sampling_mask(pos_edge_index, z.size(0),
                    self.args.batch_size, self.mask.squeeze(1))
        else:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))

        neg_loss = -torch.log(
            1 - self.decoder(z, neg_edge_index) + EPS).mean()

        return pos_loss + neg_loss

    def ranking_metrics(self, logits, y):
        """Given ground-truth :obj:`y`, computes Mean Reciprocal Rank (MRR)
        and Hits at 1/3/10."""

        y = y.to(logits.device)
        adj = torch.mm(logits, logits.t())
        adj = adj[y[0]]
        _, perm = adj.sort(dim=1, descending=True)

        mask = (y[1].view(-1, 1) == perm)

        mrr = (1 / (mask.nonzero()[:, -1] + 1).to(torch.float)).mean().item()
        hits1 = mask[:, :1].sum().item() / y.size(1)
        hits3 = mask[:, :3].sum().item() / y.size(1)
        hits10 = mask[:, :10].sum().item() / y.size(1)

        return mrr, hits1, hits3, hits10

    def test(self, z, pos_edge_index, neg_edge_index):
        r"""Given latent variables :obj:`z`, positive edges
        :obj:`pos_edge_index` and negative edges :obj:`neg_edge_index`,
        computes area under the ROC curve (AUC) and average precision (AP)
        scores.
        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (LongTensor): The positive edges to evaluate
                against.
            neg_edge_index (LongTensor): The negative edges to evaluate
                against.
        """
        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = self.decoder(z, pos_edge_index)
        neg_pred = self.decoder(z, neg_edge_index)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()
        return roc_auc_score(y, pred), average_precision_score(y, pred)

    def save(self, fn_enc):
        torch.save(self.encoder.state_dict(), fn_enc)

    def load(self, fn_enc):
        self.encoder.load_state_dict(torch.load(fn_enc))

    def load_no_flow(self, fn_enc):
        self.encoder.load_state_dict(torch.load(fn_enc),strict=False)
        pretrained_dict = self.encoder.state_dict()
        filtered_dict = filter_state_dict(pretrained_dict, "flow")
        pretrained_dict.update(filtered_dict)

class VGAE(GAE):
    r"""The Variational Graph Auto-Encoder model from the
    `"Variational Graph Auto-Encoders" <https://arxiv.org/abs/1611.07308>`_
    paper.
    Args:
        encoder (Module): The encoder module to compute :math:`\mu` and
            :math:`\log\sigma^2`.
        decoder (Module, optional): The decoder module. If set to :obj:`None`,
            will default to the
            :class:`torch_geometric.nn.models.InnerProductDecoder`.
            (default: :obj:`None`)
    """

    def __init__(self, args, encoder, decoder, r, temperature):
        self.sum_log_det_jac = 0
        self. args = args
        # prior distribution
        self.p_z = EuclideanNormal
        # posterior distribution
        self.qz_x = EuclideanNormal
        # likelihood distribution
        self.px_z = EuclideanNormal
        self.r = r
        self.temperature = temperature
        self.z_dim = encoder.z_dim
        self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.decoder_name = decoder
        if self.decoder_name == 'tanh':
            decoder = TanhDecoder()
        elif self.decoder_name == 'distance':
            decoder = DistanceDecoder('GAT', self.r, self.temperature,
                                      encoder.hidden_dim, self.z_dim, 1, args.max_nodes)
        elif self.decoder_name == 'softmax':
            decoder = SoftmaxDecoder(self.r)
        elif self.decoder_name == 'gran':
            if args.is_ar:
                decoder = GRAN_AR_MixtureBernoulli(args)
            else:
                decoder = GRANMixtureBernoulli(args)
        else:
            decoder = InnerProductDecoder()
        super(VGAE, self).__init__(encoder, decoder=decoder)

    def reparametrize(self, mu, std):
        if self.training:
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def encode(self, *args, **kwargs):
        """"""
        self.sum_log_det_jac = 0
        node_feats, edge_index = args[0], args[1]
        self.__mu__, self.__logvar__ = self.encoder(*args, **kwargs)
        # Create Masking if necessary (Only for padded rows)
        self.mask = (node_feats.sum(dim=-1) != 0).int().unsqueeze(1)
        self.__std__ = self.__logvar__.mul(0.5).exp_()
        z = self.mask * self.reparametrize(self.__mu__, self.__std__)
        if self.encoder.flow_model:
            self.encoder.flow_model.base_dist_mean = self.__mu__
            self.encoder.flow_model.base_dist_var = self.__logvar__
            z_k, sum_log_det_jac = self.encoder.flow_model.inverse(z, edge_index)
            self.sum_log_det_jac = sum_log_det_jac
            self.z_k = z_k
        else:
            self.z_k = z
            z_k = z

        return z, z_k

    def calc_mi_q(self, x, edge_index):
        """Approximate the mutual information between x and z
        I(x, z) = E_xE_{q(z|x)}log(q(z|x)) - E_xE_{q(z|x)}log(q(z))
        Returns: Float
        Code taken from:
        https://github.com/jxhe/vae-lagging-encoder/blob/master/modules/encoders/encoder.py#L111
        """

        # [x_batch, nz]
        mask = (x.sum(dim=-1) != 0).nonzero().squeeze().long()
        mu, logvar = self.encode(x, edge_index)
        std = logvar.mul(0.5).exp_()
        nz = mu.size(1)
        x_batch = len(mask)

        # E_{q(z|x)}log(q(z|x)) = -0.5*nz*log(2*\pi) - 0.5*(1+logvar).sum(-1)
        masked_mu = mu[mask]
        masked_logvar = logvar[mask]
        masked_std = std[mask]
        masked_var = masked_std ** 2
        neg_entropy = (-0.5 * nz * math.log(2 * math.pi)- 0.5 * (1 + masked_logvar).sum(-1)).mean()

        # [z_batch, 1, nz]
        z = self.reparametrize(mu, std)
        z = z[mask].unsqueeze(1)

        # [1, x_batch, nz]
        masked_mu, masked_logvar = masked_mu.unsqueeze(0), masked_logvar.unsqueeze(0)
        # var = logvar.exp()

        # (z_batch, x_batch, nz)
        dev = z - masked_mu

        # (z_batch, x_batch)
        log_density = -0.5 * ((dev ** 2) / masked_var).sum(dim=-1) - \
            0.5 * (nz * math.log(2 * math.pi) + masked_logvar.sum(-1))

        # log q(z): aggregate posterior
        # [z_batch]
        log_qz = log_sum_exp(log_density, dim=1) - math.log(x_batch)

        return (neg_entropy - log_qz.mean(-1)).item()

    def kl_loss(self, z_0, z_k, mu=None, std=None, free_bits=0.):
        r"""Computes the KL loss, either for the passed arguments :obj:`mu`
        and :obj:`std`, or based on latent variables from last encoding.
        Args:
            z_0 (Tensor): The latent sample prior to going through the flow
            z_k (Tensor): The latent sample after the flow. If there is no flow
            then z_0 == z_k
            mu (Tensor, optional): The latent space for :math:`\mu`. If set to
                :obj:`None`, uses the last computation of :math:`mu`.
                (default: :obj:`None`)
            std (Tensor, optional): The latent space for
                :math:`\log\sigma^2`.  If set to :obj:`None`, uses the last
                computation of :math:`\log\sigma^2`.(default: :obj:`None`)
        """
        base_mu = self.__mu__ if mu is None else mu
        base_std = self.__std__ if std is None else std
        base_logvar = self.__logvar__ if std is None else torch.log(std)
        mask = self.mask.squeeze().nonzero().squeeze()
        base_mu = base_mu[mask]
        base_std = base_std[mask]
        base_logvar = base_logvar[mask]
        if self.encoder.analytic_kl:
            kld = -0.5 * torch.mean( torch.sum(1 + base_logvar - base_mu**2 -
                                               base_logvar.exp(), dim=1))

            # clamp to avoid kl collapse
            kld.clamp_(min=free_bits)
            kld = 1. / len(mask) * kld
            return kld

        q_z_0_x = self.qz_x(base_mu, base_std)
        p_z_k = self.p_z(torch.zeros_like(base_mu, device=self.dev),
                         torch.ones_like(base_std))
        log_q_z_0_x = q_z_0_x.log_prob(z_0)
        log_p_z_k = p_z_k.log_prob(z_k)
        kld = log_q_z_0_x - log_p_z_k
        kld = kld - self.sum_log_det_jac
        mean_kld = torch.mean(kld.clamp_(min=free_bits))
        return mean_kld

    def save(self, fn_enc):
        torch.save(self.encoder.state_dict(), fn_enc)

    def load(self, fn_enc):
        self.encoder.load_state_dict(torch.load(fn_enc))

    def load_no_flow(self, fn_enc):
        self.encoder.load_state_dict(torch.load(fn_enc),strict=False)
        pretrained_dict = self.encoder.state_dict()
        filtered_dict = filter_state_dict(pretrained_dict, "flow")
        pretrained_dict.update(filtered_dict)




