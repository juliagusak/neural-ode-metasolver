import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets


import numpy as np
import glob
import pandas as pd
from collections import defaultdict
from argparse import Namespace

import os
import sys
sys.path.append('/workspace/home/jgusak/neural-ode-sopa')

import dataloaders

from sopa.src.models.odenet_mnist.layers import  MetaNODE
from sopa.src.solvers.utils import create_solver

from sopa.src.models.utils import load_model
from sopa.src.models.odenet_mnist.attacks_utils import run_attack

from MegaAdversarial.src.utils.runner import   fix_seeds

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default='')
parser.add_argument('--models_root', type=str, default='')
parser.add_argument('--save_path', type=str, default='')
parser.add_argument('--key_path_word', type=str, default='')

parser.add_argument('--min_eps', type=float, default=0.)
parser.add_argument('--max_eps', type=float, default=0.3)
parser.add_argument('--num_eps', type=int, default=20)
parser.add_argument('--epsilons', type=lambda s: [float(eps) for eps in s.split(',')], default=None)

args = parser.parse_args()


def run_set_of_attacks(epsilons, attack_modes, loaders, models_root, save_path = None, device = 'cuda', key_path_word = ''):
    fix_seeds()
   
    df = pd.DataFrame()
    
    if key_path_word == 'u2':
        meta_model_path = "{}/*/*/*.pth".format(models_root)
    elif key_path_word == 'euler':
        meta_model_path = "{}/*/*.pth".format(models_root)
    else:
        meta_model_path = "{}/*/*.pth".format(models_root)
        
    
    i = 0
    for state_dict_path in glob.glob(meta_model_path, recursive = True):
        if not  (key_path_word in state_dict_path):
            continue
            
        model, model_args = load_model(state_dict_path)
        model.eval()
        model.cuda()
        
        val_solver = create_solver(*model_args.solvers[0], dtype = torch.float32, device = device)
        val_solver_options = Namespace(solver_mode = 'standalone')
        solvers_kwargs = {'solvers':[val_solver], 'solver_options':val_solver_options}

        robust_accuracies = run_attack(model, epsilons, attack_modes, loaders, device, solvers_kwargs=solvers_kwargs)   
        robust_accuracies = {k : np.array(v) for k,v in robust_accuracies.items()}

        data = [list(dict(model_args._get_kwargs()).values()) +\
                list(robust_accuracies.values()) +\
                [epsilons]
               ]
        columns = list(dict(model_args._get_kwargs()).keys()) +\
                  list(robust_accuracies.keys()) +\
                  ['epsilons']
        
        df_tmp = pd.DataFrame(data = data, columns = columns) 
        df = df.append(df_tmp)
    
        if save_path is not None:
            df.to_csv(save_path, index = False)
            
        i += 1
        print('{} models have been processed'.format(i))
        


if __name__=="__main__":
    loaders = dataloaders.get_loader(batch_size=256,
                                     data_name='mnist',
                                     data_root=args.data_root,
                                     num_workers = 4,
                                     train=False, val=True)
    device = 'cuda'
    
    if args.epsilons is not None:
        epsilons = args.epsilons
    else:
        epsilons = np.linspace(args.min_eps, args.max_eps, num=args.num_eps) 
        
    run_set_of_attacks(epsilons=epsilons,
                   attack_modes = ["fgsm", "at", "at_ls", "av", "fs"][:1],
                   loaders = loaders,
                   models_root = args.models_root,
                   save_path = args.save_path,
                   key_path_word = args.key_path_word,
                   device = device)


# CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python3 attacks_runner.py --data_root "/workspace/raid/data/datasets" --models_root "/workspace/raid/data/jgusak/neural-ode-sopa/odenet_mnist_noise/" --save_path '/workspace/home/jgusak/neural-ode-sopa/experiments/odenet_mnist/results/robust_accuracies_fgsm/test_part1.csv' --epsilons 0.15,0.3,0.5 
