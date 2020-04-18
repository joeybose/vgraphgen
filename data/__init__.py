from typing import Any
import argparse

from .graph_datasets import *

__all__ = [
    "create_dataset",
    "NXDataset",
    "GRANDataset",
]


def create_dataset(arg_parse, config):
    dataset_type = config.dataset.name.lower()
    if dataset_type in ["lobster", "grid", "prufer", "fc"]:
        if config.dataset.use_gran_data:
            dataset = GRANDataset(arg_parse, config, dataset_type, arg_parse.seed, tag='train')
            arg_parse.input_dim = dataset.num_features
            arg_parse.num_features = dataset.num_features
            arg_parse.node_dist = dataset.node_dist
            arg_parse.max_nodes = dataset.max_nodes
            return dataset
        else:
            dataset = NXDataset(dataset_type, arg_parse.batch_size,
                                arg_parse.num_fixed_features, arg_parse.node_order,
                                arg_parse.use_rand_feats, arg_parse.seed)
            arg_parse.input_dim = dataset.num_features
            arg_parse.num_features = dataset.num_features
            arg_parse.node_dist = dataset.node_dist
            arg_parse.max_nodes = dataset.max_nodes
            return dataset
    else:
        raise ValueError(f"Unknown dataset type: '{dataset_type}'.")
