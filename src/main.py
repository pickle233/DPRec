import argparse
import torch
import numpy as np
from data_loader import load_data
from DPRec import train
import random

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=str, default='music')
parser.add_argument('--n_epoch', type=int, default=60)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--n_layer', type=int, default=2)
parser.add_argument('--lr', type=float, default=0.004)
parser.add_argument('--l2_weight', type=float, default=1e-4)

parser.add_argument('--dim', type=int, default=64)
parser.add_argument('--user_triple_set_size', type=int, default=128)
parser.add_argument('--user_potential_triple_set_size', type=int, default=128)
parser.add_argument('--item_origin_triple_set_size', type=int, default=128)
parser.add_argument('--item_triple_set_size', type=int, default=128)

parser.add_argument('--agg', type=str, default='concat')

parser.add_argument('--use_cuda', type=bool, default=True)
parser.add_argument('--seed', type=int, default=2025)

args = parser.parse_args()


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_random_seed(args.seed)

data_info = load_data(args)
result = train(args, data_info)

print("Final result:", result)