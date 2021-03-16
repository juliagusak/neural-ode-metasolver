import os
import argparse
from argparse import Namespace
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy

from decimal import Decimal
import wandb
import sys

sys.path.append('../../')

from sopa.src.solvers.utils import create_solver
from sopa.src.models.utils import fix_seeds, RunningAverageMeter
from sopa.src.models.odenet_mnist.layers import MetaNODE
from sopa.src.models.odenet_mnist.utils import makedirs, learning_rate_with_decay
from sopa.src.models.odenet_mnist.data import get_mnist_loaders, inf_generator
from MegaAdversarial.src.attacks import (
    Clean,
    PGD,
    FGSM
)

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
                    default=['standalone'],
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

parser.add_argument('--nepochs_nn', type=int, default=50)
parser.add_argument('--nepochs_solver', type=int, default=0)
parser.add_argument('--nstages', type=int, default=1)

parser.add_argument('--data_aug', type=eval, default=True, choices=[True, False])
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--test_batch_size', type=int, default=1000)
parser.add_argument('--base_lr', type=float, default=1e-5, help='base_lr for CyclicLR scheduler')
parser.add_argument('--max_lr', type=float, default=1e-3, help='max_lr for CyclicLR scheduler')
parser.add_argument('--step_size_up', type=int, default=2000, help='step_size_up for CyclicLR scheduler')
parser.add_argument('--cyclic_lr_mode', type=str, default='triangular2', help='mode for CyclicLR scheduler')
parser.add_argument('--lr_uv', type=float, default=1e-3)
parser.add_argument('--torch_dtype', type=str, default='float32')
parser.add_argument('--wandb_name', type=str, default='find_best_solver')

parser.add_argument('--data_root', type=str, default='./')
parser.add_argument('--save_dir', type=str, default='./')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--gpu', type=int, default=0)

parser.add_argument('--seed', type=int, default=502)
# Noise and adversarial attacks parameters:
parser.add_argument('--data_noise_std', type=float, default=0.,
                    help='Applies Norm(0, std) gaussian noise to each training batch')
parser.add_argument('--eps_adv_training', type=float, default=0.3,
                    help='Epsilon for adversarial training')
parser.add_argument(
    "--adv_training_mode",
    default="clean",
    choices=["clean", "fgsm", "at"],  # , "at_ls", "av", "fs", "nce", "nce_moco", "moco", "av_extra", "meta"],
    help='''Adverarial training method/mode, by default there is no adversarial training (clean).
        For further details see MegaAdversarial/train in this repository.
        '''
)
parser.add_argument('--use_wandb', type=eval, default=True, choices=[True, False])
parser.add_argument('--ss_loss', type=eval, default=False, choices=[True, False])
parser.add_argument('--ss_loss_reg', type=float, default=0.1)
parser.add_argument('--timestamp', type=int, default=int(1e6 * time.time()))

parser.add_argument('--eps_adv_testing', type=float, default=0.3,
                    help='Epsilon for adversarial testing')
parser.add_argument('--adv_testing_mode',
                    default="clean",
                    choices=["clean", "fgsm", "at"],
                    help='''Adversarsarial testing mode''')

args = parser.parse_args()

sys.path.append('../../')

makedirs(args.save_dir)
if args.use_wandb:
    wandb.init(project=args.wandb_name, entity="sopa_node", anonymous="allow")
    wandb.config.update(args)
    wandb.config.update({'u': float(args.solvers[0][-2])})  # a dirty way to extract u from the rk2 solver
    # Path to save checkpoints locally in <args.save_dir>/<entity>/<project>/<run_id> [Julia style]
    makedirs(wandb.config.save_dir)
    makedirs(os.path.join(wandb.config.save_dir,  wandb.run.path))

if args.torch_dtype == 'float64':
    dtype = torch.float64
elif args.torch_dtype == 'float32':
    dtype = torch.float32
else:
    raise ValueError('torch_type should be either float64 or float32')

# I've decided to copy and modify functions from src/models/odenet_mnist/train_validate.py

CONFIG_PGD_TRAIN = {"eps": 0.3, "lr": 2.0 / 255, "n_iter": 7}
CONFIG_FGSM_TRAIN = {"alpha": 0.3, "epsilon": 0.05}


def one_hot(x, K):
    return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)


def accuracy(model, dataset_loader, device, solvers=None, solver_options=None):
    model.eval()
    total_correct = 0
    for x, y in dataset_loader:
        x = x.to(device)
        y = one_hot(np.array(y.numpy()), 10)
        target_class = np.argmax(y, axis=1)
        with torch.no_grad():
            if solver is not None:
                out = model(x, solvers, solver_options).cpu().detach().numpy()
            else:
                out = model(x).cpu().detach().numpy()
        predicted_class = np.argmax(out, axis=1)
        total_correct += np.sum(predicted_class == target_class)
    return total_correct / len(dataset_loader.dataset)


def adversarial_accuracy(model, dataset_loader, device, solvers=None, solver_options=None, args=None):
    model.eval()
    total_correct = 0
    if args.adv_testing_mode == "clean":
        test_attack = Clean(model)
    elif args.adv_testing_mode == "fgsm":
        test_attack = FGSM(model, mean=[0.], std=[1.], **CONFIG_FGSM_TRAIN)
    elif args.adv_testing_mode == "at":
        test_attack = PGD(model, mean=[0.], std=[1.], **CONFIG_PGD_TRAIN)
    else:
        raise ValueError("Attack type not understood.")
    for x, y in dataset_loader:
        x, y = x.to(device), y.to(device)
        x, y = test_attack(x, y, {"solvers": solvers, "solver_options": solver_options})
        y = one_hot(np.array(y.cpu().numpy()), 10)
        target_class = np.argmax(y, axis=1)
        with torch.no_grad():
            if solver is not None:
                out = model(x, solvers, solver_options).cpu().detach().numpy()
            else:
                out = model(x).cpu().detach().numpy()
        predicted_class = np.argmax(out, axis=1)
        total_correct += np.sum(predicted_class == target_class)
    return total_correct / len(dataset_loader.dataset)


def train(model,
          data_gen,
          solvers,
          solver_options,
          criterion,
          optimizer,
          device,
          is_odenet=True,
          args=None):
    model.train()
    optimizer.zero_grad()
    x, y = data_gen.__next__()
    x = x.to(device)
    y = y.to(device)

    if args.adv_training_mode == "clean":
        train_attack = Clean(model)
    elif args.adv_training_mode == "fgsm":
        train_attack = FGSM(model, **CONFIG_FGSM_TRAIN)
    elif args.adv_training_mode == "at":
        train_attack = PGD(model, **CONFIG_PGD_TRAIN)
    else:
        raise ValueError("Attack type not understood.")
    x, y = train_attack(x, y, {"solvers": solvers, "solver_options": solver_options})

    # Add noise:
    if args.data_noise_std > 1e-12:
        with torch.no_grad():
            x = x + args.data_noise_std * torch.randn_like(x)
    ##### Forward pass
    if is_odenet:
        logits = model(x, solvers, solver_options, Namespace(ss_loss=args.ss_loss))
    else:
        logits = model(x)

    xentropy = criterion(logits, y)
    if args.ss_loss:
        ss_loss = model.get_ss_loss()
        loss = xentropy + args.ss_loss_reg * ss_loss
    else:
        ss_loss = 0.
        loss = xentropy

    loss.backward()
    optimizer.step()
    if args.ss_loss:
        return {'xentropy': xentropy.item(), 'ss_loss': ss_loss.item()}
    return {'xentropy': xentropy.item()}


if __name__ == "__main__":
    print(f'CUDA is available: {torch.cuda.is_available()}')
    print(args.solvers)
    fix_seeds(args.seed)

    if args.torch_dtype == 'float64':
        dtype = torch.float64
    elif args.torch_dtype == 'float32':
        dtype = torch.float32
    else:
        raise ValueError('torch_type should be either float64 or float32')

    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

    ########### Create train / val solvers
    print(args.solvers)
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

    ########### Create data loaders
    train_loader, test_loader, train_eval_loader = get_mnist_loaders(args.data_aug,
                                                                     args.batch_size,
                                                                     args.test_batch_size,
                                                                     data_root=args.data_root)
    data_gen = inf_generator(train_loader)
    batches_per_epoch = len(train_loader)

    ########### Create criterion and optimizer

    criterion = nn.CrossEntropyLoss().to(device)
    loss_options = Namespace(ss_loss=args.ss_loss)

    ##### We exchange the learning rate with a circular learning rate

    optimizer = optim.RMSprop([{"params": model.parameters(), 'lr': args.lr}, ], lr=args.lr,
                              weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=args.base_lr,
                                            max_lr=args.max_lr, step_size_up=args.step_size_up, mode=args.cyclic_lr_mode)

    ########### Train the model

    for itr in range(args.nepochs_nn * batches_per_epoch):

        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = lr_fn(itr)

        train_loss = train(model,
                           data_gen,
                           solvers=train_solvers,
                           solver_options=train_solver_options,
                           criterion=criterion,
                           optimizer=optimizer,
                           device=device,
                           is_odenet=is_odenet,
                           args=args)
        scheduler.step()

        if itr % batches_per_epoch == 0:
            train_acc = accuracy(model, train_loader, device, solvers=train_solvers, solver_options=train_solver_options)
            test_acc = accuracy(model, test_loader, device, solvers=train_solvers, solver_options=train_solver_options)
            adv_test_acc = adversarial_accuracy(model, test_loader, device, solvers=train_solvers,
                                                    solver_options=train_solver_options, args=args)
            adv_train_acc = adversarial_accuracy(model, train_loader, device, solvers=train_solvers,
                                                     solver_options=train_solver_options, args=args)

            makedirs(os.path.join(wandb.config.save_dir, wandb.run.path))
            save_path = os.path.join(wandb.config.save_dir, wandb.run.path, "checkpoint_{}.pth".format(itr))
            print(save_path)
            torch.save(model, save_path)
            wandb.save(save_path)

            wandb.log({'train_acc': train_acc,
                       'test_acc': test_acc,
                       'adv_test_acc': adv_test_acc,
                       'adv_train_acc': adv_train_acc,
                       'train_loss': train_loss['xentropy']})