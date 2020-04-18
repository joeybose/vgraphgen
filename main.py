import numpy as np
import argparse
import json
import os
import wandb
import torch
import ipdb
from data import create_dataset
from models import create_model
from train_helper import train_graph_generation
from utils import utils
from utils.utils import seed_everything, str2bool
from utils.arg_helper import parse_arguments, get_config

def main(args, config):
    ipdb.set_trace()
    dataset = create_dataset(args, config)
    train_loader, val_loader, test_loader = dataset.create_loaders()
    args.flow_args = [args.n_blocks, args.flow_hidden_size, args.n_hidden,
                 args.flow_model, args.flow_layer_type]
    model = create_model(args, config).to(args.dev)
    print(vars(args))
    print(model)
    print('number of parameters : {}'.format(sum([np.prod(x.shape) for x in model.parameters()])))
    trained_model = train_graph_generation(args, train_loader, val_loader,
                                           test_loader, model)


if __name__ == '__main__':
    """
    Process command-line arguments, then call main()
    """

    args = parse_arguments()
    config = get_config(args.config_file, is_test=args.test)
    p_name = utils.project_name(config.dataset.name)

    if args.wandb:
        os.environ['WANDB_API_KEY'] = args.wandb_apikey
        wandb.init(project='{}'.format(p_name),
                   name='{}-{}'.format(args.namestr, args.model_name))

    ''' Fix Random Seed '''
    seed_everything(args.seed)
    # Check if settings file
    if os.path.isfile("settings.json"):
        with open('settings.json') as f:
            data = json.load(f)
        args.wandb_apikey = data.get("wandbapikey")


    main(args, config)
