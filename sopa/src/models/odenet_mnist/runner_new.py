import os
import argparse
from argparse import Namespace
import logging
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy

from decimal import Decimal
import wandb
import sys

sys.path.append('../../../../')

from sopa.src.solvers.utils import create_solver
from sopa.src.models.utils import fix_seeds, RunningAverageMeter
from sopa.src.models.odenet_mnist.layers import MetaNODE
from sopa.src.models.odenet_mnist.utils import makedirs, get_logger, count_parameters, learning_rate_with_decay
from sopa.src.models.odenet_mnist.data import get_mnist_loaders, inf_generator
# from sopa.src.models.odenet_mnist.metrics import accuracy
from sopa.src.models.odenet_mnist.train_validate import train, validate

parser = argparse.ArgumentParser()
parser.add_argument('--network', type=str, choices=['resnet', 'odenet'], default='odenet')
parser.add_argument('--downsampling-method', type=str, default='conv', choices=['conv', 'res'])
parser.add_argument('--activation', type=str, choices=['tanh', 'softplus', 'softsign', 'relu'], default='relu')
parser.add_argument('--in_channels', type=int, default=1)

parser.add_argument('--solvers',
                    type=lambda s: [tuple(map(lambda iparam: str(iparam[1]) if iparam[0] <= 1 else (
                        int(iparam[1]) if iparam[0] == 2 else (
                            float(iparam[1]) if iparam[0] == 3 else Decimal(iparam[1]))),
                                              enumerate(item.split(',')))) for item in s.strip().split(';')],
                    default=None,
                    help='Each solver is represented with (method,parameterization,n_steps,step_size,u0,v0) \n' +
                         'If the solver has only one parameter u0, set v0 to -1; \n' +
                         'n_steps and step_size are exclusive parameters, only one of them can be != -1, \n'
                         'If n_steps = step_size = -1, automatic time grid_constructor is used \n;'
                         'For example, --solvers rk4,uv,2,-1,0.3,0.6;rk3,uv,-1,0.1,0.4,0.6;rk2,u,4,-1,0.3,-1')

parser.add_argument('--solver_mode', type=str, choices=['switch', 'ensemble', 'standalone'], default='standalone')
parser.add_argument('--val_solver_modes',
                    type=lambda s: s.strip().split(','),
                    default=['standalone', 'ensemble', 'switch'],
                    help='Solver modes to use for validation step')

parser.add_argument('--switch_probs', type=lambda s: [float(item) for item in s.split(',')], default=None,
                    help="--switch_probs 0.8,0.1,0.1")
parser.add_argument('--ensemble_weights', type=lambda s: [float(item) for item in s.split(',')], default=None,
                    help="ensemble_weights 0.6,0.2,0.2")
parser.add_argument('--ensemble_prob', type=float, default=1.)

parser.add_argument('--noise_type', type=str, choices=['cauchy', 'normal'], default=None)
parser.add_argument('--noise_sigma', type=float, default=0.001)
parser.add_argument('--noise_prob', type=float, default=0.)
parser.add_argument('--minimize_rk2_error', type=eval, default=False, choices=[True, False])

parser.add_argument('--nepochs_nn', type=int, default=160)
parser.add_argument('--nepochs_solver', type=int, default=0)
parser.add_argument('--nstages', type=int, default=1)

parser.add_argument('--data_aug', type=eval, default=True, choices=[True, False])
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--test_batch_size', type=int, default=1000)

parser.add_argument('--lr_uv', type=float, default=1e-3)
parser.add_argument('--torch_dtype', type=str, default='float32')
parser.add_argument('--wandb_name', type=str, default='mnist_tmp')

parser.add_argument('--data_root', type=str, default='/gpfs/gpfs0/t.daulbaev/data/MNIST')
parser.add_argument('--save', type=str, default='../../../rk2_tmp')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--gpu', type=int, default=0)

parser.add_argument('--seed', type=int, default=502)
# Noise and adversarial attacks parameters:
parser.add_argument('--data_noise_std', type=float, default=0.,
                    help='Applies Norm(0, std) gaussian noise to each batch entry')
parser.add_argument('--eps_adv_training', type=float, default=0.3,
                    help='Epsilon for adversarial training')
parser.add_argument(
        "--adv_training_mode",
        default="clean",
        choices=["clean", "fgsm", "at"], #, "at_ls", "av", "fs", "nce", "nce_moco", "moco", "av_extra", "meta"],
        help='''Adverarial training method/mode, by default there is no adversarial training (clean).
        For further details see MegaAdversarial/train in this repository.
        '''
    )
parser.add_argument('--use_wandb', type=eval, default=True, choices=[True, False])
parser.add_argument('--use_logger', type=eval, default=False, choices=[True, False])
parser.add_argument('--ss_loss', type=eval, default=False, choices=[True, False])
parser.add_argument('--ss_loss_reg', type=float, default=0.1)
parser.add_argument('--timestamp', type=int, default=int(1e6 * time.time()))

args = parser.parse_args()

sys.path.append('../../')

if args.use_logger:
    makedirs(args.save)
    logger = get_logger(logpath=os.path.join(args.save, 'logs'), filepath=None)
    logger.info(args)
if args.use_wandb:
    wandb.init(project=args.wandb_name, entity="sopa_node")
    makedirs(args.save)
    wandb.config.update(args)
    os.makedirs(os.path.join(args.save, str(args.timestamp)))

if args.torch_dtype == 'float64':
    dtype = torch.float64
elif args.torch_dtype == 'float32':
    dtype = torch.float32
else:
    raise ValueError('torch_type should be either float64 or float32')

if __name__ == "__main__":
    print(args.solvers)
    fix_seeds(args.seed)

    if args.torch_dtype == 'float64':
        dtype = torch.float64
    elif args.torch_dtype == 'float32':
        dtype = torch.float32
    else:
        raise ValueError('torch_type should be either float64 or float32')

    makedirs(args.save)
    logger = get_logger(logpath=os.path.join(args.save, 'logs'), filepath=None)
    logger.info(args)

    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

    ########### Create train / val solvers
    train_solvers = [create_solver(*solver_params, dtype=dtype, device=device) for solver_params in args.solvers]
    for solver in train_solvers:
        solver.freeze_params()

    train_solver_options = Namespace(**{key: vars(args)[key] for key in ['solver_mode', 'switch_probs',
                                                                         'ensemble_prob', 'ensemble_weights']})

    val_solver_modes = args.val_solver_modes

    ########## Build the model
    is_odenet = args.network == 'odenet'

    model = MetaNODE(downsampling_method=args.downsampling_method,
                     is_odenet=is_odenet,
                     activation_type=args.activation,
                     in_channels=args.in_channels)
    model.to(device)
    if args.use_wandb:
        wandb.watch(model)
    if args.use_logger:
        logger.info(model)

    ########### Create data loaders
    train_loader, test_loader, train_eval_loader = get_mnist_loaders(args.data_aug,
                                                                     args.batch_size,
                                                                     args.test_batch_size,
                                                                     data_root=args.data_root)

    data_gen = inf_generator(train_loader)
    batches_per_epoch = len(train_loader)

    ########### Creare criterion and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    loss_options = Namespace(ss_loss=args.ss_loss)

    lr_fn = learning_rate_with_decay(
        args.batch_size, batch_denom=128, batches_per_epoch=batches_per_epoch, boundary_epochs=[60, 100, 140],
        decay_rates=[1, 0.1, 0.01, 0.001], lr0=args.lr)

    optimizer = optim.RMSprop([{"params": model.parameters(), 'lr': args.lr}, ], lr=args.lr,
                              weight_decay=args.weight_decay)

    ########### Train the model
    best_acc = {'standalone': [0] * len(train_solvers),
                'ensemble': 0,
                'switch': 0}

    batch_time_meter = RunningAverageMeter()
    f_nfe_meter = RunningAverageMeter()
    b_nfe_meter = RunningAverageMeter()

    for itr in range(args.nepochs_nn * batches_per_epoch):

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_fn(itr)

        if itr % batches_per_epoch != 0:
            train(itr,
                  model,
                  data_gen,
                  solvers=train_solvers,
                  solver_options=train_solver_options,
                  criterion=criterion,
                  optimizer=optimizer,
                  batch_time_meter=batch_time_meter,
                  f_nfe_meter=f_nfe_meter,
                  b_nfe_meter=b_nfe_meter,
                  device=device,
                  dtype=dtype,
                  is_odenet=is_odenet,
                  args=args,
                  logger=None,
                  wandb_logger=None)
        else:
            train(itr,
                  model,
                  data_gen,
                  solvers=train_solvers,
                  solver_options=train_solver_options,
                  criterion=criterion,
                  optimizer=optimizer,
                  batch_time_meter=batch_time_meter,
                  f_nfe_meter=f_nfe_meter,
                  b_nfe_meter=b_nfe_meter,
                  device=device,
                  dtype=dtype,
                  is_odenet=is_odenet,
                  args=args,
                  logger=logger,
                  wandb_logger=wandb)

            best_acc = validate(best_acc,
                                itr,
                                model,
                                train_eval_loader,
                                test_loader,
                                batches_per_epoch,
                                solvers=train_solvers,
                                val_solver_modes=val_solver_modes,
                                batch_time_meter=batch_time_meter,
                                f_nfe_meter=f_nfe_meter,
                                b_nfe_meter=b_nfe_meter,
                                device=device,
                                dtype=dtype,
                                args=args,
                                logger=logger,
                                wandb_logger=wandb)

        # # How to run
# CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 python3 runner.py
# --data_root "/workspace/raid/data/datasets"
# --save "./experiment1"
# --network odenet
# --downsampling_method conv
# --solvers rk2,u,-1,-1,0.6,-1;rk2,u,-1,-1,0.5,-1
# --solver_mode switch
# --switch_probs 0.8,0.2
# --nepochs_nn 160
# --nepochs_solver 0
# --nstages 1
# --lr 0.1
# --seed 502

# CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python3 runner_new.py --data_root /workspace/raid/data/datasets/mnist --save ./experiment2_new --network 'odenet' --downsampling-method 'conv' --solvers "rk2,u,1,-1,0.5,-1;rk2,u,1,-1,1.0,-1" --solver_mode "switch" --activation "relu" --seed 702  --nepochs_nn 160 --nepochs_solver 0 --nstages 1 --lr 0.1 

# CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4  python3 runner_new.py --data_root /workspace/raid/data/datasets/mnist --save ./experiment2_new --network 'odenet' --downsampling-method 'conv' --solvers "rk2,u,1,-1,0.66666666,-1" --solver_mode standalone --activation relu --seed 702  --nepochs_nn 160 --nepochs_solver 0 --nstages 1 --lr 0.1 --noise_type 'cauchy' --noise_sigma 0.001 --noise_prob 1.

# Пересчитать MNIST