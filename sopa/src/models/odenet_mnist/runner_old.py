import os
import argparse
import logging
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torchdiffeq._impl.rk_common import _ButcherTableau

import sys
sys.path.append('/workspace/home/jgusak/neural-ode-sopa/')
from sopa.src.solvers.rk_parametric_old import rk_param_tableau, odeint_plus

from sopa.src.models.odenet_mnist.layers import ResBlock, ODEfunc, build_downsampling_layers, build_fc_layers
from sopa.src.models.odenet_mnist.utils import makedirs, get_logger, count_parameters, learning_rate_with_decay, RunningAverageMeter
from sopa.src.models.odenet_mnist.data import get_mnist_loaders, inf_generator
from sopa.src.models.odenet_mnist.metrics import accuracy

parser = argparse.ArgumentParser()
parser.add_argument('--network', type=str, choices=['resnet', 'odenet'], default='odenet')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams', 'rk4', 'rk4_param', 'rk3_param','euler'], default='rk4')
parser.add_argument('--step_size', type=float, default=None)
parser.add_argument('--tol', type=float, default=1e-3)
parser.add_argument('--adjoint', type=eval, default=False, choices=[True, False])
parser.add_argument('--downsampling-method', type=str, default='conv', choices=['conv', 'res'])
parser.add_argument('--nepochs', type=int, default=160)
parser.add_argument('--data_aug', type=eval, default=True, choices=[True, False])
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--test_batch_size', type=int, default=1000)

parser.add_argument('--lr_uv', type=float, default=1e-3)
parser.add_argument('--parameterization', type=str, choices=['uv', 'u1', 'u2', 'u3'],  default='uv')
parser.add_argument('--u0', type=float, default=1/3.)
parser.add_argument('--v0', type=float, default=2/3.)
parser.add_argument('--fix_param', action='store_true')

parser.add_argument('--data_root', type=str, default='/workspace/home/jgusak/neural-ode-sopa/.data/mnist')
parser.add_argument('--save', type=str, default='./experiment1')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--gpu', type=int, default=0)

parser.add_argument('--torch_seed', type=int, default=502)
parser.add_argument('--numpy_seed', type=int, default=502)

args = parser.parse_args([])

# Set random seed for reproducibility
np.random.seed(args.numpy_seed)
torch.manual_seed(args.torch_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint
    
class ODEBlock(nn.Module):

    def __init__(self, odefunc, args, device = 'cpu'):
        super(ODEBlock, self).__init__()
            
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 1]).float()
        
        # make trainable parameters as attributes of ODE block,
        # recompute tableau at each forward step
        self.step_size = args.step_size

        self.method = args.method
        self.fix_param = None
        self.parameterization = None
        self.u0, self.v0 = None, None
        self.u_, self.v_ = None, None
        self.u, self.v = None, None
        
        
        self.eps = torch.finfo(torch.float32).eps
        self.device = device

        
        if self.method in ['rk4_param', 'rk3_param']:
            self.fix_param = args.fix_param
            self.parameterization = args.parameterization
            self.u0 = args.u0

            if self.fix_param:
                self.u = torch.tensor(self.u0)

                if self.parameterization == 'uv':
                    self.v0 = args.v0
                    self.v = torch.tensor(self.v0)

            else:
                # an important issue about putting leaf variables to the device https://discuss.pytorch.org/t/tensor-to-device-changes-is-leaf-causing-cant-optimize-a-non-leaf-tensor/37659
                self.u_ = nn.Parameter(torch.tensor(self.u0)).to(self.device)
                self.u = torch.clamp(self.u_, self.eps, 1. - self.eps).detach().requires_grad_(True)
                
                if self.parameterization == 'uv':
                    self.v0 = args.v0
                    self.v_ = nn.Parameter(torch.tensor(self.v0)).to(self.device)
                    self.v = torch.clamp(self.v_, self.eps, 1. - self.eps).detach().requires_grad_(True)

            logger.info('Init | u {} | v {}'.format(self.u.data, (self.v if self.v is None else self.v.data)))

            self.alpha, self.beta, self.c_sol = rk_param_tableau(self.u, self.v, device = self.device,
                                                                 parameterization=self.parameterization,
                                                                 method = self.method)
            self.tableau = _ButcherTableau(alpha = self.alpha,
                                           beta = self.beta,
                                           c_sol = self.c_sol,
                                           c_error = torch.zeros((len(self.c_sol),), device = self.device))
    
    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        
        if self.method in ['rk4_param', 'rk3_param']:
            out = odeint_plus(self.odefunc, x, self.integration_time,
                              method=self.method, options = {'tableau':self.tableau, 'step_size':self.step_size})
        else:
            out = odeint(self.odefunc, x, self.integration_time, rtol=args.tol, atol=args.tol,
                         method=self.method, options = {'step_size':self.step_size})
                
        return out[1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value
        
        
if __name__=="__main__":
    makedirs(args.save)
    logger = get_logger(logpath=os.path.join(args.save, 'logs'), filepath=None)
    logger.info(args)

    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

    ########### Build model
    is_odenet = args.network == 'odenet'

    downsampling_layers = build_downsampling_layers(args.downsampling_method)
    fc_layers = build_fc_layers()
    
    feature_layers = [ODEBlock(ODEfunc(64), args, device)] if is_odenet else [ResBlock(64, 64) for _ in range(6)]
    model = nn.Sequential(*downsampling_layers, *feature_layers, *fc_layers).to(device)

    logger.info(model)
    logger.info('Number of parameters: {}'.format(count_parameters(model)))

    ########### Create data loaders
    train_loader, test_loader, train_eval_loader = get_mnist_loaders(
        args.data_aug, args.batch_size, args.test_batch_size, data_root = args.data_root)

    data_gen = inf_generator(train_loader)
    batches_per_epoch = len(train_loader)
    
    ########### Creare criterion and optimizer
    criterion = nn.CrossEntropyLoss().to(device)

    lr_fn = learning_rate_with_decay(
        args.batch_size, batch_denom=128, batches_per_epoch=batches_per_epoch, boundary_epochs=[60, 100, 140],
        decay_rates=[1, 0.1, 0.01, 0.001], lr0 = args.lr)

    if is_odenet and (args.method in ['rk4_param', 'rk3_param']) and not args.fix_param:
        params_uv = []

        for mname, m in model.named_modules():
            if isinstance(m, ODEBlock):
                if m.u_ is not None:
                    params_uv.append(m.u_)
                    if m.v_ is not None:
                        params_uv.append(m.v_)

        optimizer = optim.SGD([{"params" : model.parameters()},
                               {"params" : params_uv, 'lr' : args.lr_uv}], lr=args.lr, momentum = 0.9)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum = 0.9)

    
    ########### Train the model
    best_acc = 0
    batch_time_meter = RunningAverageMeter()
    f_nfe_meter = RunningAverageMeter()
    b_nfe_meter = RunningAverageMeter()
    end = time.time()

    for itr in range(args.nepochs * batches_per_epoch):

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_fn(itr)

        if is_odenet and (args.method in ['rk4_param', 'rk3_param']) and not args.fix_param:

            ### Iterate along the model, find ODEBlock, recalculate the tableau
            for mname, m in model.named_modules():
                if isinstance(m, ODEBlock):
                    m.alpha, m.beta, m.c_sol = rk_param_tableau(m.u, m.v, device = device,
                                                                parameterization=args.parameterization,
                                                                method = args.method)
                    m.tableau = _ButcherTableau(alpha = m.alpha, beta = m.beta, c_sol = m.c_sol,
                                                c_error = torch.zeros((len(m.c_sol),), device = device))

        optimizer.zero_grad()
        x, y = data_gen.__next__()
        x = x.to(device)
        y = y.to(device)

        logits = model(x)

        loss = criterion(logits, y)

        if is_odenet:
            nfe_forward = feature_layers[0].nfe
            feature_layers[0].nfe = 0

        loss.backward()
        optimizer.step()

        if is_odenet:
            nfe_backward = feature_layers[0].nfe
            feature_layers[0].nfe = 0

        batch_time_meter.update(time.time() - end)
        if is_odenet:
            f_nfe_meter.update(nfe_forward)
            b_nfe_meter.update(nfe_backward)
        end = time.time()

        if itr % batches_per_epoch == 0:
            with torch.no_grad():
                train_acc = accuracy(model, train_eval_loader, device)
                val_acc = accuracy(model, test_loader, device)
                if val_acc > best_acc:
                    torch.save({'state_dict': model.state_dict(), 'args': args}, os.path.join(args.save, 'model.pth'))
                    best_acc = val_acc

                logger.info(
                    "Epoch {:04d} | Time {:.3f} ({:.3f}) | NFE-F {:.1f} | NFE-B {:.1f} | "
                    "TrainAcc {:.6f} | TestAcc {:.6f}".format(
                        itr // batches_per_epoch, batch_time_meter.val, batch_time_meter.avg, f_nfe_meter.avg,
                        b_nfe_meter.avg, train_acc, val_acc))    

#CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 python3 runner_old.py --network 'odenet' --data_root "/workspace/raid/data/datasets" --batch_size 128 --test_batch_size 1000 --nepochs 160 --lr 0.1   --save "./experiment1"  --method 'euler' --step_size 1.0 --gpu 0 --torch_seed 502 --numpy_seed 502 --fix_param