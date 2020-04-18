from torch_geometric.nn import GCNConv, GATConv, MessagePassing
from .model_helpers import MLP
from torch_geometric.utils import remove_self_loops, softmax
from torch_scatter import scatter_add
import torch_geometric.nn as geom_nn
import torch
import torch.nn.functional as F
import ipdb

def my_add_self_loops(edge_index, edge_weight=None, fill_value=1, num_nodes=None):
    r"""Adds a self-loop :math:`(i,i) \in \mathcal{E}` to every node
    :math:`i \in \mathcal{V}` in the graph given by :attr:`edge_index`.
    In case the graph is weighted, self-loops will be added with edge weights
    denoted by :obj:`fill_value`.
    Args:
        edge_index (LongTensor): The edge indices.
        edge_weight (Tensor, optional): One-dimensional edge weights.
            (default: :obj:`None`)
        fill_value (int, optional): If :obj:`edge_weight` is not :obj:`None`,
            will add self-loops with edge weights of :obj:`fill_value` to the
            graph. (default: :obj:`1`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """
    num_nodes = len(edge_index.unique())
    loop_index = edge_index.unique().repeat(2, 1)

    if edge_weight is not None:
        assert edge_weight.numel() == edge_index.size(1)
        loop_weight = edge_weight.new_full((num_nodes, ), fill_value)
        edge_weight = torch.cat([edge_weight, loop_weight], dim=0)

    edge_index = torch.cat([edge_index, loop_index], dim=1)

    return edge_index, edge_weight

class GCNLayer(GCNConv):
    def __init__(self, input_dim, hidden_dim, max_node):
        super(GCNLayer, self).__init__(input_dim, hidden_dim)
        self.max_node = max_node

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, improved=False,
             dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        fill_value = 1 if not improved else 2
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, edge_weight = my_add_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        x = torch.matmul(x, self.weight)
        num_nodes = x.size(self.node_dim)
        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}. Please '
                    'disable the caching behavior of this layer by removing '
                    'the `cached=True` argument in its constructor.'.format(
                        self.cached_num_edges, edge_index.size(1)))

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            if self.normalize:
                edge_index, norm = self.norm(edge_index, num_nodes,
                                             edge_weight, self.improved,
                                             x.dtype)
            else:
                norm = edge_weight

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j if norm is not None else x_j

    def update(self, aggr_out):
        mask = (aggr_out.sum(dim=-1) != 0).int().unsqueeze(1)
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
            aggr_out = mask*aggr_out
        return aggr_out

class GATLayer(GATConv):
    def __init__(self, input_dim, hidden_dim, max_node):
        super(GATLayer, self).__init__(input_dim, hidden_dim)
        self.max_node = max_node

    def forward(self, x, edge_index, size=None):
        """"""
        num_nodes = x.size(self.node_dim)
        if size is None and torch.is_tensor(x):
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = my_add_self_loops(edge_index, num_nodes=num_nodes)

        if torch.is_tensor(x):
            x = torch.matmul(x, self.weight)
        else:
            x = (None if x[0] is None else torch.matmul(x[0], self.weight),
                 None if x[1] is None else torch.matmul(x[1], self.weight))
        return self.propagate(edge_index, size=size, x=x)

    def message(self, edge_index_i, x_i, x_j, size_i):
        # Compute attention coefficients.
        x_j = x_j.view(-1, self.heads, self.out_channels)
        if x_i is None:
            alpha = (x_j * self.att[:, :, self.out_channels:]).sum(dim=-1)
        else:
            x_i = x_i.view(-1, self.heads, self.out_channels)
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, size_i)

        # Sample attention coefficients stochastically.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return (x_j * alpha.view(-1, self.heads, 1))

    def update(self, aggr_out):
        mask = (aggr_out.sum(dim=-1) != 0).int()
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
            aggr_out = mask*aggr_out
        return aggr_out


class ACRConv(MessagePassing):
    def __init__(self, input_dim, output_dim, max_nodes, aggregate_type="add",
                 readout_type='add', combine_type='mlp', combine_layers=1,
                 num_mlp_layers=1):

        assert aggregate_type in ["add", "mean", "max"]
        assert combine_type in ["simple", "mlp"]
        assert readout_type in ["add", "mean", "max"]

        super(ACRConv, self).__init__(aggr=aggregate_type)

        self.mlp_combine = False
        self.max_nodes = max_nodes
        if combine_type == "mlp":
            self.mlp = MLP(
                num_layers=num_mlp_layers,
                input_dim=output_dim,
                hidden_dim=output_dim,
                output_dim=output_dim)

            self.mlp_combine = True

        self.V = MLP(
            num_layers=combine_layers,
            input_dim=input_dim,
            hidden_dim=output_dim,
            output_dim=output_dim)
        self.A = MLP(
            num_layers=combine_layers,
            input_dim=input_dim,
            hidden_dim=output_dim,
            output_dim=output_dim)
        self.R = MLP(
            num_layers=combine_layers,
            input_dim=input_dim,
            hidden_dim=output_dim,
            output_dim=output_dim)

        self.readout = self.__get_readout_fn(readout_type)

    def __get_readout_fn(self, readout_type):
        options = {
            "add": geom_nn.global_add_pool,
            "mean": geom_nn.global_mean_pool,
            "max": geom_nn.global_max_pool
        }
        if readout_type not in options:
            raise ValueError()
        return options[readout_type]

    def forward(self, h, edge_index):
        max_node = self.max_nodes
        batch_size = h.shape[0] // max_node
        batch = torch.zeros(h.shape[0]).long().to(h.device)
        for i in range(batch_size):
            batch[(i)*max_node:(i+1)*max_node] = i
        # this give a (batch_size, features) tensor
        readout = self.readout(x=h, batch=batch)
        # this give a (nodes, features) tensor
        readout = readout[batch]

        return self.propagate(
            edge_index=edge_index,
            h=h,
            readout=readout)

    def message(self, h_j):
        return h_j

    def update(self, aggr, h, readout):
        mask = (h.sum(dim=-1) != 0).int().unsqueeze(1)
        updated = mask*self.V(h) + mask*self.A(aggr) + mask*self.R(readout)

        if self.mlp_combine:
            updated = mask*self.mlp(updated)

        return updated

    def reset_parameters(self):
        self.V.reset_parameters()
        self.A.reset_parameters()
        self.R.reset_parameters()
        if hasattr(self, "mlp"):
            self.mlp.reset_parameters()
