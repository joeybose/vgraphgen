import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.distributions import Normal
from torch import distributions
from torch.nn.parameter import Parameter
from torch_geometric.nn import GCNConv, GATConv
import ipdb
from sklearn import cluster, datasets, mixture
from sklearn.preprocessing import StandardScaler
from flows.flow_helpers import *
from utils.math_ops import clamp
from distributions.normal import EuclideanNormal


#Reference: https://github.com/ritheshkumar95/pytorch-normalizing-flows/blob/master/modules.py
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6
max_clamp_norm = 40
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def weights_init_(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

# All code below this line is taken from
# https://github.com/kamenbliznashki/normalizing_flows/blob/master/maf.py

class FlowSequential(nn.Sequential):
    """ Container for layers of a normalizing flow """
    def forward(self, x, y):
        sum_log_abs_det_jacobians = 0
        i = len(self)
        for module in self:
            x, log_abs_det_jacobian = module(x, y)
            sum_log_abs_det_jacobians = sum_log_abs_det_jacobians + log_abs_det_jacobian
            i -= 1
        return x, sum_log_abs_det_jacobians

    def inverse(self, u, y):
        i = 0
        sum_log_abs_det_jacobians = 0
        for module in reversed(self):
            u, log_abs_det_jacobian = module.inverse(u, y)
            sum_log_abs_det_jacobians = sum_log_abs_det_jacobians + log_abs_det_jacobian
            i += 1
        return u, sum_log_abs_det_jacobians

class FCNN(nn.Module):
    """
    Simple fully connected neural network.
    """
    def __init__(self, in_dim, out_dim, hidden_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.network(x)

# --------------------
# Models
# --------------------

class MAFRealNVP(nn.Module):
    def __init__(self, n_blocks, input_size, hidden_size, n_hidden,
                 cond_label_size=None, batch_norm=False):
        super().__init__()

        # base distribution for calculation of log prob under the model
        self.register_buffer('base_dist_mean', torch.zeros(input_size))
        self.register_buffer('base_dist_var', torch.ones(input_size))
        self.p_z = EuclideanNormal

        # construct model
        modules = []
        mask = torch.arange(input_size).float() % 2
        for i in range(n_blocks):
            modules += [LinearMaskedCoupling(input_size, hidden_size, n_hidden, mask, cond_label_size)]
            mask = 1 - mask
            # modules += batch_norm * [BatchNorm(input_size)]

        self.net = FlowSequential(*modules)

    @property
    def base_dist(self):
        return D.Normal(self.base_dist_mean, self.base_dist_var)

    def forward(self, x, y=None):
        return self.net(x, y)

    def inverse(self, u, y=None):
        return self.net.inverse(u, y)

    def log_prob(self, x, y=None):
        u, sum_log_abs_det_jacobians = self.forward(x, y)
        return torch.sum(self.base_dist.log_prob(u) + sum_log_abs_det_jacobians, dim=1)

## Taken from: https://github.com/senya-ashukha/real-nvp-pytorch/blob/master/real-nvp-pytorch.ipynb
class RealNVP(nn.Module):
    def __init__(self, n_blocks, input_size, hidden_size, n_hidden,
                 layer_type='Linear'):
        super(RealNVP, self).__init__()
        mask = torch.arange(input_size).float() % 2
        self.n_blocks = n_blocks
        self.n_hidden = n_hidden
        self.n_components = 1
        self.layer_type = layer_type
        self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        i_mask = 1 - mask
        mask = torch.stack([mask,i_mask]).repeat(int(n_blocks/2),1)
        self.p_z = EuclideanNormal
        self.s, self.t = create_real_nvp_blocks(input_size, hidden_size,
                                                n_blocks, n_hidden, layer_type)
        # base distribution for calculation of log prob under the model
        self.register_buffer('base_dist_mean', torch.zeros(input_size))
        self.register_buffer('base_dist_var', torch.ones(input_size))
        self.mask = nn.Parameter(mask, requires_grad=False)

    @property
    def base_dist(self):
        return self.p_z(self.base_dist_mean, self.base_dist_var)

    def inverse(self, z, edge_index=None):
        log_det_J, x = z.new_zeros(z.shape[0]), z
        for i in range(0,self.n_blocks):
            x_ = x*self.mask[i]
            if self.layer_type != 'Linear':
                s = self.s[i](x_, edge_index)
                t = self.t[i](x_, edge_index)
            else:
                s = self.s[i](x_)
                t = self.t[i](x_)
            x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)
            log_det_J += ((1-self.mask[i])*s).sum(dim=1)  # log det dx/du
        return x, log_det_J

    def forward(self, x, edge_index=None):
        log_det_J, z = x.new_zeros(x.shape[0]), x
        for i in reversed(range(0,self.n_blocks)):
            z_ = self.mask[i] * z
            if self.layer_type != 'Linear':
                s = self.s[i](z_, edge_index)
                t = self.t[i](z_, edge_index)
            else:
                s = self.s[i](z_)
                t = self.t[i](z_)
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_J -= ((1-self.mask[i])*s).sum(dim=1)
        return z, log_det_J

    def log_prob(self, x, edge_index=None):
        z, logp = self.forward(x, edge_index)
        p_z = self.p_z(torch.zeros_like(x, device=self.dev),
                         torch.ones_like(x))
        return p_z.log_prob(z) + logp

    def sample(self, batchSize):
        # TODO: Update this method for edge_index
        z = self.prior.sample((batchSize, 1))
        logp = self.prior.log_prob(z)
        x = self.inverse(z)
        return x

class MADE(nn.Module):
    def __init__(self, input_size, hidden_size, n_hidden, cond_label_size=None,
                 activation='relu', input_order='sequential',
                 input_degrees=None):
        """
        Args:
            input_size -- scalar; dim of inputs
            hidden_size -- scalar; dim of hidden layers
            n_hidden -- scalar; number of hidden layers
            activation -- str; activation function to use
            input_order -- str or tensor; variable order for creating the autoregressive masks (sequential|random)
                            or the order flipped from the previous layer in a stack of mades
            conditional -- bool; whether model is conditional
        """
        super().__init__()
        # base distribution for calculation of log prob under the model
        self.register_buffer('base_dist_mean', torch.zeros(input_size))
        self.register_buffer('base_dist_var', torch.ones(input_size))

        # create masks
        masks, self.input_degrees = create_masks(input_size, hidden_size,
                                                 n_hidden, input_order,
                                                 input_degrees)

        # setup activation
        if activation == 'relu':
            activation_fn = nn.ReLU()
        elif activation == 'tanh':
            activation_fn = nn.Tanh()
        else:
            raise ValueError('Check activation function.')

        # construct model
        self.net_input = MaskedLinear(input_size, hidden_size, masks[0], cond_label_size)
        self.net = []
        for m in masks[1:-1]:
            self.net += [activation_fn, MaskedLinear(hidden_size, hidden_size, m)]
        self.net += [activation_fn, MaskedLinear(hidden_size, 2 * input_size,
                                                 masks[-1].repeat(2,1))]
        self.net = nn.Sequential(*self.net)

    @property
    def base_dist(self):
        return D.Normal(self.base_dist_mean, self.base_dist_var)

    def forward(self, x, y=None, edge_index=None):
        # MAF eq 4 -- return mean and log std
        m, loga = self.net(self.net_input(x, y)).chunk(chunks=2, dim=1)
        u = (x - m) * torch.exp(-loga)
        # MAF eq 5
        log_abs_det_jacobian = - loga
        return u, log_abs_det_jacobian

    def inverse(self, u, y=None, sum_log_abs_det_jacobians=None):
        # MAF eq 3
        D = u.shape[1]
        x = torch.zeros_like(u)
        # run through reverse model
        for i in self.input_degrees:
            m, loga = self.net(self.net_input(x, y)).chunk(chunks=2, dim=1)
            x[:,i] = u[:,i] * torch.exp(loga[:,i]) + m[:,i]
        log_abs_det_jacobian = loga
        return x, log_abs_det_jacobian

    def log_prob(self, x, y=None):
        u, log_abs_det_jacobian = self.forward(x, y)
        return torch.sum(self.base_dist.log_prob(u) + log_abs_det_jacobian,
                         dim=1)

class MADEMOG(nn.Module):
    """ Mixture of Gaussians MADE """
    def __init__(self, n_components, input_size, hidden_size, n_hidden,
                 cond_label_size=None, activation='relu',
                 input_order='sequential', input_degrees=None):
        """
        Args:
            n_components -- scalar; number of gauassian components in the mixture
            input_size -- scalar; dim of inputs
            hidden_size -- scalar; dim of hidden layers
            n_hidden -- scalar; number of hidden layers
            activation -- str; activation function to use
            input_order -- str or tensor; variable order for creating the autoregressive masks (sequential|random)
                            or the order flipped from the previous layer in a stack of mades
            conditional -- bool; whether model is conditional
        """
        super().__init__()
        self.n_components = n_components

        # base distribution for calculation of log prob under the model
        self.register_buffer('base_dist_mean', torch.zeros(input_size))
        self.register_buffer('base_dist_var', torch.ones(input_size))

        # create masks
        masks, self.input_degrees = create_masks(input_size, hidden_size,
                                                 n_hidden, input_order,
                                                 input_degrees)

        # setup activation
        if activation == 'relu':
            activation_fn = nn.ReLU()
        elif activation == 'tanh':
            activation_fn = nn.Tanh()
        else:
            raise ValueError('Check activation function.')

        # construct model
        self.net_input = MaskedLinear(input_size, hidden_size, masks[0], cond_label_size)
        self.net = []
        for m in masks[1:-1]:
            self.net += [activation_fn, MaskedLinear(hidden_size, hidden_size, m)]
        self.net += [activation_fn, MaskedLinear(hidden_size, n_components * 3
                                                 * input_size,
                                                 masks[-1].repeat(n_components
                                                                  * 3,1))]
        self.net = nn.Sequential(*self.net)

    @property
    def base_dist(self):
        return D.Normal(self.base_dist_mean, self.base_dist_var)

    def forward(self, x, y=None, edge_index=None):
        # shapes
        N, L = x.shape
        C = self.n_components
        # MAF eq 2 -- parameters of Gaussians - mean, logsigma, log unnormalized cluster probabilities
        m, loga, logr = self.net(self.net_input(x, y)).view(N, C, 3 * L).chunk(chunks=3, dim=-1)  # out 3 x (N, C, L)
        # MAF eq 4
        x = x.repeat(1, C).view(N, C, L)  # out (N, C, L)
        u = (x - m) * torch.exp(-loga)  # out (N, C, L)
        # MAF eq 5
        log_abs_det_jacobian = - loga  # out (N, C, L)
        # normalize cluster responsibilities
        self.logr = logr - logr.logsumexp(1, keepdim=True)  # out (N, C, L)
        return u, log_abs_det_jacobian

    def inverse(self, u, y=None, sum_log_abs_det_jacobians=None):
        # shapes
        N, C, L = u.shape
        # init output
        x = torch.zeros(N, L).to(u.device)
        # MAF eq 3
        # run through reverse model along each L
        for i in self.input_degrees:
            m, loga, logr = self.net(self.net_input(x, y)).view(N, C, 3 * L).chunk(chunks=3, dim=-1)  # out 3 x (N, C, L)
            # normalize cluster responsibilities and sample cluster assignments from a categorical dist
            logr = logr - logr.logsumexp(1, keepdim=True)  # out (N, C, L)
            z = D.Categorical(logits=logr[:,:,i]).sample().unsqueeze(-1)  # out (N, 1)
            u_z = torch.gather(u[:,:,i], 1, z).squeeze()  # out (N, 1)
            m_z = torch.gather(m[:,:,i], 1, z).squeeze()  # out (N, 1)
            loga_z = torch.gather(loga[:,:,i], 1, z).squeeze()
            x[:,i] = u_z * torch.exp(loga_z) + m_z
        log_abs_det_jacobian = loga
        return x, log_abs_det_jacobian

    def log_prob(self, x, y=None):
        u, log_abs_det_jacobian = self.forward(x, y)  # u = (N,C,L); log_abs_det_jacobian = (N,C,L)
        # marginalize cluster probs
        # sum over C; out (N, L)
        log_probs = torch.logsumexp(self.logr + self.base_dist.log_prob(u) +
                                    log_abs_det_jacobian, dim=1)
        return log_probs.sum(1)  # sum over L; out (N,)

class MAF(nn.Module):
    def __init__(self, n_blocks, input_size, hidden_size, n_hidden,
                 cond_label_size=None, activation='relu',
                 input_order='sequential', batch_norm=True, layer_type=None):
        super().__init__()
        # base distribution for calculation of log prob under the model
        self.register_buffer('base_dist_mean', torch.zeros(input_size))
        self.register_buffer('base_dist_var', torch.ones(input_size))

        # construct model
        modules = []
        self.input_degrees = None
        for i in range(n_blocks):
            modules += [MADE(input_size, hidden_size, n_hidden,
                             cond_label_size, activation, input_order,
                             self.input_degrees)]
            self.input_degrees = modules[-1].input_degrees.flip(0)
            modules += batch_norm * [BatchNorm(input_size)]

        self.net = FlowSequential(*modules)

    @property
    def base_dist(self):
        return D.Normal(self.base_dist_mean, self.base_dist_var)

    def forward(self, x, y=None):
        return self.net(x, y)

    def inverse(self, u, y=None):
        return self.net.inverse(u, y)

    def log_prob(self, x, y=None, edge_index=None):
        u, sum_log_abs_det_jacobians = self.forward(x, y)
        return torch.sum(self.base_dist.log_prob(u) +
                         sum_log_abs_det_jacobians, dim=1)

class MAFMOG(nn.Module):
    """ MAF on mixture of gaussian MADE """
    def __init__(self, n_blocks, input_size, hidden_size, n_hidden,
                 n_components=2, cond_label_size=None, activation='relu',
                 input_order='sequential', batch_norm=True, layer_type=None):
        super().__init__()
        self.n_components = n_components
        # base distribution for calculation of log prob under the model
        self.register_buffer('base_dist_mean', torch.zeros(input_size))
        self.register_buffer('base_dist_var', torch.ones(input_size))

        self.maf = MAF(n_blocks, input_size, hidden_size, n_hidden,
                       cond_label_size, activation, input_order, batch_norm)
        # get reversed input order from the last layer (note in maf model,
        # input_degrees are already flipped in for-loop model constructor
        input_degrees = self.maf.input_degrees#.flip(0)
        self.mademog = MADEMOG(n_components, input_size, hidden_size, n_hidden,
                               cond_label_size, activation, input_order,
                               input_degrees)

    @property
    def base_dist(self):
        return D.Normal(self.base_dist_mean, self.base_dist_var)

    def forward(self, x, y=None, edge_index=None):
        u, maf_log_abs_dets = self.maf(x, y)
        u, made_log_abs_dets = self.mademog(u, y)
        sum_log_abs_det_jacobians = maf_log_abs_dets.unsqueeze(1) + made_log_abs_dets
        return u, sum_log_abs_det_jacobians

    def inverse(self, u, y=None, edge_index=None):
        x, made_log_abs_dets = self.mademog.inverse(u, y)
        x, maf_log_abs_dets = self.maf.inverse(x, y)
        sum_log_abs_det_jacobians = maf_log_abs_dets.unsqueeze(1) + made_log_abs_dets
        return x, sum_log_abs_det_jacobians

    def log_prob(self, x, y=None, edge_index=None):
        u, log_abs_det_jacobian = self.forward(x, y)  # u = (N,C,L); log_abs_det_jacobian = (N,C,L)
        # marginalize cluster probs
        log_probs = torch.logsumexp(self.mademog.logr +
                                    self.base_dist.log_prob(u) +
                                    log_abs_det_jacobian, dim=1)  # out (N, L)
        return log_probs.sum(1)  # out (N,)

class NSF_AR(nn.Module):
    """
    Neural spline flow, auto-regressive.
    [Durkan et al. 2019]
    """
    def __init__(self, dim, K = 5, B = 3, hidden_dim = 8, base_network = FCNN):
        super().__init__()
        self.dim = dim
        self.K = K
        self.B = B
        self.layers = nn.ModuleList()
        self.init_param = nn.Parameter(torch.Tensor(3 * K - 1))
        for i in range(1, dim):
            self.layers += [base_network(i, 3 * K - 1, hidden_dim)]
        self.reset_parameters()

    def reset_parameters(self):
        init.uniform_(self.init_param, - 1 / 2, 1 / 2)

    def forward(self, x):
        z = torch.zeros_like(x)
        log_det = torch.zeros(z.shape[0])
        for i in range(self.dim):
            if i == 0:
                init_param = self.init_param.expand(x.shape[0], 3 * self.K - 1)
                W, H, D = torch.split(init_param, self.K, dim = 1)
            else:
                out = self.layers[i - 1](x[:, :i])
                W, H, D = torch.split(out, self.K, dim = 1)
            W, H = torch.softmax(W, dim = 1), torch.softmax(H, dim = 1)
            W, H = 2 * self.B * W, 2 * self.B * H
            D = F.softplus(D)
            z[:, i], ld = unconstrained_RQS(
                x[:, i], W, H, D, inverse=False, tail_bound=self.B)
            log_det += ld
        return z, log_det

    def inverse(self, z):
        x = torch.zeros_like(z)
        log_det = torch.zeros(x.shape[0])
        for i in range(self.dim):
            if i == 0:
                init_param = self.init_param.expand(x.shape[0], 3 * self.K - 1)
                W, H, D = torch.split(init_param, self.K, dim = 1)
            else:
                out = self.layers[i - 1](x[:, :i])
                W, H, D = torch.split(out, self.K, dim = 1)
            W, H = torch.softmax(W, dim = 1), torch.softmax(H, dim = 1)
            W, H = 2 * self.B * W, 2 * self.B * H
            D = F.softplus(D)
            x[:, i], ld = unconstrained_RQS(
                z[:, i], W, H, D, inverse = True, tail_bound = self.B)
            log_det += ld
        return x, log_det
