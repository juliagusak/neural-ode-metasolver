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

import sys
sys.path.append('/workspace/home/jgusak/neural-ode-sopa/')

from sopa.src.solvers.utils import create_solver, noise_params
from sopa.src.models.utils import fix_seeds, RunningAverageMeter
from sopa.src.models.odenet_mnist.layers import MetaNODE
from sopa.src.models.odenet_mnist.utils import makedirs, get_logger, count_parameters, learning_rate_with_decay
from sopa.src.models.odenet_mnist.data import get_mnist_loaders, inf_generator
from sopa.src.models.odenet_mnist.metrics import accuracy


parser = argparse.ArgumentParser()
parser.add_argument('--network', type=str, choices=['resnet', 'odenet'], default='odenet')
parser.add_argument('--downsampling-method', type=str, default='conv', choices=['conv', 'res'])
parser.add_argument('--activation', type=str, choices=['tanh', 'softplus', 'softsign', 'relu'], default='relu')
parser.add_argument('--in_channels', type=int, default=1)

parser.add_argument('--solvers',
                    type=lambda s:[tuple(map(lambda iparam: str(iparam[1]) if iparam[0] <= 1 else (
                                    int(iparam[1]) if iparam[0]==2 else (
                                        float(iparam[1]) if iparam[0] == 3 else Decimal(iparam[1]))),
                                           enumerate(item.split(',')))) for item in s.strip().split(';')],
                    default = None,
                    help='Each solver is represented with (method,parameterization,n_steps,step_size,u0,v0) \n' +
                         'If the solver has only one parameter u0, set v0 to -1; \n' +
                         'n_steps and step_size are exclusive parameters, only one of them can be != -1, \n'
                         'If n_steps = step_size = -1, automatic time grid_constructor is used \n;'  
                         'For example, --solvers rk4,uv,2,-1,0.3,0.6;rk3,uv,-1,0.1,0.4,0.6;rk2,u,4,-1,0.3,-1')

parser.add_argument('--solver_mode', type=str,  choices=['switch', 'ensemble', 'standalone'], default='standalone')
parser.add_argument('--switch_probs',type=lambda s: [float(item) for item in s.split(',')], default=None,
                    help="--switch_probs 0.8,0.1,0.1")
parser.add_argument('--ensemble_weights', type=lambda s: [float(item) for item in s.split(',')], default=None,
                    help="ensemble_weights 0.6,0.2,0.2")
parser.add_argument('--ensemble_prob', type=float, default=1.)

parser.add_argument('--noise_type', type=str,  choices=['cauchy', 'normal'], default=None)
parser.add_argument('--noise_sigma', type=float, default = 0.001)
parser.add_argument('--noise_prob', type=float, default = 0.)
parser.add_argument('--minimize_rk2_error', type=eval, default=False, choices=[True, False])

parser.add_argument('--nepochs_nn', type=int, default=160)
parser.add_argument('--nepochs_solver', type=int, default=0)
parser.add_argument('--nstages', type=int, default=1)

parser.add_argument('--ss_loss', type=eval, default=False, choices=[True, False])
parser.add_argument('--ss_loss_reg', type=float, default=0.1)

parser.add_argument('--data_aug', type=eval, default=True, choices=[True, False])
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--test_batch_size', type=int, default=1000)

parser.add_argument('--lr_uv', type=float, default=1e-3)
parser.add_argument('--torch_dtype', type=str,  default='float32')

parser.add_argument('--data_root', type=str, default='/workspace/home/jgusak/neural-ode-sopa/.data/mnist')
parser.add_argument('--save', type=str, default='./experiment2')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--gpu', type=int, default=0)

parser.add_argument('--seed', type=int, default=502)


args = parser.parse_args()


if args.torch_dtype == 'float64':
    dtype = torch.float64
elif args.torch_dtype == 'float32':
    dtype = torch.float32
else:
    raise ValueError('torch_type should be either float64 or float32')
    
    
if __name__=="__main__":
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
    train_solvers = [create_solver(*solver_params, dtype =dtype, device = device) for solver_params in args.solvers]
    for solver in train_solvers:
        print(solver.u)
        solver.freeze_params()

    train_solver_options = Namespace(**{key:vars(args)[key] for key in ['solver_mode','switch_probs',
                                                                        'ensemble_prob','ensemble_weights']})
    val_solver_options = Namespace(solver_mode = 'standalone')

    ########## Build the model
    is_odenet = args.network == 'odenet'

    model = MetaNODE(downsampling_method = args.downsampling_method, is_odenet=is_odenet,
                     activation_type=args.activation, in_channels = args.in_channels)
    model.to(device)

    logger.info(model)

    ########### Create data loaders
    train_loader, test_loader, train_eval_loader = get_mnist_loaders(
        args.data_aug, args.batch_size, args.test_batch_size, data_root = args.data_root)

    data_gen = inf_generator(train_loader)
    batches_per_epoch = len(train_loader)

    ########### Creare criterion and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    loss_options = Namespace(ss_loss=args.ss_loss)

    lr_fn = learning_rate_with_decay(
        args.batch_size, batch_denom=128, batches_per_epoch=batches_per_epoch, boundary_epochs=[60, 100, 140],
        decay_rates=[1, 0.1, 0.01, 0.001], lr0 = args.lr)

    optimizer = optim.RMSprop([{"params" : model.parameters(), 'lr' : args.lr},], lr=args.lr, weight_decay=args.weight_decay)

    ########### Train the model
    nsolvers = len(train_solvers)
    best_acc = [0]*nsolvers

    batch_time_meter = RunningAverageMeter()
    f_nfe_meter = RunningAverageMeter()
    b_nfe_meter = RunningAverageMeter()
    end = time.time()


    for itr in range(args.nepochs_nn * batches_per_epoch):

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_fn(itr)

        optimizer.zero_grad()
        x, y = data_gen.__next__()
        x = x.to(device)
        y = y.to(device)

        ##### Noise params
        if args.noise_type is not None:
            for i in range(len(train_solvers)):
                train_solvers[i].u, train_solvers[i].v = noise_params(train_solvers[i].u0,
                                                                      train_solvers[i].v0,
                                                                      std = args.noise_sigma,
                                                                      bernoulli_p = args.noise_prob,
                                                                      noise_type = args.noise_type)
                train_solvers[i].build_ButcherTableau()
        
        ##### Forward pass
        if is_odenet:
            logits = model(x, train_solvers, train_solver_options, loss_options)
        else:
            logits = model(x)

        loss = criterion(logits, y)
        if (loss_options is not None) and loss_options.ss_loss:
            loss += args.ss_loss_reg * model.get_ss_loss()
        
        ##### Compute NFE-forward
        if is_odenet:
            nfe_forward = 0
            for i in range(len(model.blocks)):
                nfe_forward += model.blocks[i].rhs_func.nfe
                model.blocks[i].rhs_func.nfe = 0

        loss.backward()
        optimizer.step()

        ##### Compute NFE-backward
        if is_odenet:
            nfe_backward = 0
            for i in range(len(model.blocks)):
                nfe_backward += model.blocks[i].rhs_func.nfe
                model.blocks[i].rhs_func.nfe = 0
                
        ##### Denoise params
        if args.noise_type is not None:
            for i in range(len(train_solvers)):
                train_solvers[i].u, train_solvers[i].v = train_solvers[i].u0, train_solvers[i].v0
                train_solvers[i].build_ButcherTableau()

        batch_time_meter.update(time.time() - end)
        if is_odenet:
            f_nfe_meter.update(nfe_forward)
            b_nfe_meter.update(nfe_backward)
        end = time.time()

        if itr % batches_per_epoch == 0:
            with torch.no_grad():
                train_acc = [0]*nsolvers
                val_acc = [0]*nsolvers
                
                for solver_id, val_solver in enumerate(train_solvers):
                    train_acc_id = accuracy(model, train_eval_loader, device, [val_solver], val_solver_options)
                    val_acc_id = accuracy(model, test_loader, device, [val_solver], val_solver_options)

                    train_acc[solver_id] = train_acc_id
                    val_acc[solver_id] = val_acc_id
                    
                    if val_acc_id > best_acc[solver_id]:
                        torch.save({'state_dict': model.state_dict(), 'args': args, 'solver_id':solver_id},
                                   os.path.join(args.save,'model_best_{}.pth'.format(solver_id)))
                        best_acc[solver_id] = val_acc_id
                    del train_acc_id, val_acc_id

                if is_odenet:
                    for i in range(len(model.blocks)):
                        model.blocks[i].rhs_func.nfe = 0

                logger.info(
                    "Epoch {:04d} | Time {:.3f} ({:.3f}) | NFE-F {:.1f} | NFE-B {:.1f} | "
                    "TrainAcc {} | TestAcc {} ".format(
                        itr // batches_per_epoch, batch_time_meter.val, batch_time_meter.avg, f_nfe_meter.avg,
                        b_nfe_meter.avg, train_acc, val_acc)) 
                    
                
## How to run 
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

# CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python3 runner.py --data_root /workspace/raid/data/datasets/mnist --save ./experiment2_new --network 'odenet' --downsampling-method 'conv' --solvers "rk2,u,4,-1,0.5,-1;rk2,u,4,-1,1.0,-1" --solver_mode "switch" --activation "relu" --seed 702  --nepochs_nn 160 --nepochs_solver 0 --nstages 1 --lr 0.01 --ss_loss True

# CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4  python3 runner.py --data_root /workspace/raid/data/datasets/mnist --save ./experiment2_new --network 'odenet' --downsampling-method 'conv' --solvers "rk2,u,1,-1,0.66666666,-1" --solver_mode standalone --activation relu --seed 702  --nepochs_nn 160 --nepochs_solver 0 --nstages 1 --lr 0.1 --noise_type 'cauchy' --noise_sigma 0.001 --noise_prob 1.
