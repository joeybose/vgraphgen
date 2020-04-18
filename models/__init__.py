from typing import Any
import argparse

from .euclidean_vgae import Encoder, GAE, VGAE
from .gran_mixture_bernoulli import GRANMixtureBernoulli

__all__ = [
    "VGAE",
    "GRANMixtureBernoulli",
    "Encoder"
]


def create_model(arg_parse, config):
    if config.dataset.name in ["lobster", "grid", "prufer", "fc"]:
        if arg_parse.model == 'vgae':
            model = VGAE(arg_parse, Encoder(arg_parse, arg_parse.enc_blocks,
                                            arg_parse.conv_type,
                                            arg_parse.input_dim,
                                            arg_parse.hidden_dim,
                                            arg_parse.z_dim,
                                            arg_parse.ll_estimate, arg_parse.K,
                                            arg_parse.flow_args,
                                            arg_parse.dev), arg_parse.decoder,
                         arg_parse.r, arg_parse.temperature)
        else:
            model = GRANMixtureBernoulli(config)
        return model
    else:
        raise ValueError(f"Unknown dataset type: '{arg_parse.dataset}'.")
