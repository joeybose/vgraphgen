import os
import yaml
import time
import argparse
from easydict import EasyDict as edict
import torch

def parse_arguments():
  parser = argparse.ArgumentParser(
      description="Running Experiments of Deep Prediction")
  parser.add_argument(
      '-c',
      '--config_file',
      type=str,
      default="config/resnet101_cifar.json",
      required=True,
      help="Path of config file")
  parser.add_argument(
      '-l',
      '--log_level',
      type=str,
      default='INFO',
      help="Logging Level, \
        DEBUG, \
        INFO, \
        WARNING, \
        ERROR, \
        CRITICAL")
  parser.add_argument('-model', type=str, default='gran')
  parser.add_argument('-m', '--comment', help="Experiment comment")
  parser.add_argument('-t', '--test', help="Test model", action='store_true')
  """ new args """
  parser.add_argument('--z_dim', type=int, default='128')
  parser.add_argument('--flow_lr', type=float, default='1e-3')
  parser.add_argument("--data", type=str, default="data", help="Data directory.")
  parser.add_argument('--flow_epochs', default=50, type=int)
  parser.add_argument('--log_freq', default=100, type=int)
  parser.add_argument('--test_batch_size', default=20, type=int)
  parser.add_argument('--clip_grad', type=float, default='5.0')
  parser.add_argument('--decoder', type=str, default=None)
  parser.add_argument('--num_gen_samples', type=int, default=15)
  parser.add_argument('--flow_hidden_size', type=int, default=128, help='Hidden layer size for Flows.')
  parser.add_argument('--n_blocks', type=int, default=2,
                      help='Number of blocks to stack in a model (MADE in MAF; Coupling+BN in RealNVP).')
  parser.add_argument('--enc_blocks', type=int, default=0, help='Number of Additional blocks in VGAE Encoder.')
  parser.add_argument('--n_hidden', type=int, default=1, help='Number of hidden layers in each Flow.')
  parser.add_argument('--num_channels', type=int, default=16, help='Number of channels in VGAE.')
  parser.add_argument('--flow_model', default=None, help='Which model to use')
  parser.add_argument('--flow_layer_type', type=str, default='Linear',
                help='Which type of layer to use ---i.e. GRevNet or Linear')
  parser.add_argument('--do_kl_anneal', action="store_true", default=False, help='Do KL Annealing')
  parser.add_argument("--wandb", action="store_true", default=False, help='Use wandb for logging')
  parser.add_argument('--seed', type=int, metavar='S', help='random seed (default: None)')
  parser.add_argument('--namestr', type=str, default='dl4gg',
                help='additional info in output filename to describe experiments')

  args = parser.parse_args()
  args.model_name = '{}_{}_DD'.format(args.model,
                                   args.decoder)
  args.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  return args


def get_config(config_file, exp_dir=None, is_test=False):
  """ Construct and snapshot hyper parameters """
  # config = edict(yaml.load(open(config_file, 'r'), Loader=yaml.FullLoader))
  config = edict(yaml.load(open(config_file, 'r')))

  # create hyper parameters
  config.run_id = str(os.getpid())
  config.exp_name = '_'.join([
      config.model.name, config.dataset.name,
      time.strftime('%Y-%b-%d-%H-%M-%S'), config.run_id
  ])

  if exp_dir is not None:
    config.exp_dir = exp_dir

  if config.train.is_resume and not is_test:
    config.save_dir = config.train.resume_dir
    save_name = os.path.join(config.save_dir, 'config_resume_{}.yaml'.format(config.run_id))
  else:
    config.save_dir = os.path.join(config.exp_dir, config.exp_name)
    save_name = os.path.join(config.save_dir, 'config.yaml')

  # snapshot hyperparameters
  mkdir(config.exp_dir)
  mkdir(config.save_dir)

  yaml.dump(edict2dict(config), open(save_name, 'w'), default_flow_style=False)

  return config


def edict2dict(edict_obj):
  dict_obj = {}

  for key, vals in edict_obj.items():
    if isinstance(vals, edict):
      dict_obj[key] = edict2dict(vals)
    else:
      dict_obj[key] = vals

  return dict_obj


def mkdir(folder):
  if not os.path.isdir(folder):
    os.makedirs(folder)
