import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial


class MetaODEBlock(nn.Module):
    def __init__(self, activation_type = 'relu'):
        super(MetaODEBlock, self).__init__()
        
        self.rhs_func = ODEfunc(64, activation_type)
        self.integration_time = torch.tensor([0, 1]).float()
        
    
    def forward(self, x, solvers, solver_options):
        nsolvers = len(solvers)
        
        if solver_options.solver_mode == 'standalone':
            y = solvers[0].integrate(self.rhs_func, x = x, t = self.integration_time)

        elif solver_options.solver_mode == 'switch':
            if solver_options.switch_probs is not None:
                switch_probs = solver_options.switch_probs
            else:
                switch_probs = [1./nsolvers for _ in range(nsolvers)]
            solver_id = np.random.choice(range(nsolvers), p = switch_probs)
            solver_options.switch_solver_id = solver_id

            y = solvers[solver_id].integrate(self.rhs_func, x = x, t = self.integration_time)

        elif solver_options.solver_mode == 'ensemble':
            coin_flip = torch.bernoulli(torch.tensor((1,)), solver_options.ensemble_prob)
            solver_options.ensemble_coin_flip = coin_flip

            if coin_flip :
                if solver_options.ensemble_weights is not None:
                    ensemble_weights = solver_options.ensemble_weights
                else:    
                    ensemble_weights = [1./nsolvers for _ in range(nsolvers)]

                for i, (wi, solver) in enumerate(zip(ensemble_weights, solvers)):
                    if i == 0:
                        y = wi * solver.integrate(self.rhs_func, x = x, t = self.integration_time)
                    else:
                        y += wi * solver.integrate(self.rhs_func, x = x, t = self.integration_time)
            else:
                y = solvers[0].integrate(self.rhs_func, x = x, t = self.integration_time)
        
        return y[-1,:,:,:,:]
        
        
    def ss_loss(self, y, solvers, solver_options):
        z0 = y
        rhs_func_ss = partial(self.rhs_func, ss_loss = True)
        integration_time_ss = self.integration_time + 1
        
        nsolvers = len(solvers)
        
        if solver_options.solver_mode == 'standalone':
            z = solvers[0].integrate(rhs_func_ss.func, x = y, t = integration_time_ss)

        elif solver_options.solver_mode == 'switch':
            if solver_options.switch_probs is not None:
                switch_probs = solver_options.switch_probs
            else:
                switch_probs = [1./nsolvers for _ in range(nsolvers)]
                solver_id = solver_options.switch_solver_id

            z = solvers[solver_id].integrate(rhs_func_ss.func, x = y, t = integration_time_ss)

        elif solver_options.solver_mode == 'ensemble':
            coin_flip = solver_options.ensemble_coin_flip

            if coin_flip :
                if solver_options.ensemble_weights is not None:
                    ensemble_weights = solver_options.ensemble_weights
                else:    
                    ensemble_weights = [1./nsolvers for _ in range(nsolvers)]

                for i, (wi, solver) in enumerate(zip(ensemble_weights, solvers)):
                    if i == 0:
                        z = wi * solver.integrate(rhs_func_ss.func, x = y, t = integration_time_ss)
                    else:
                        z += wi * solver.integrate(rhs_func_ss.func, x = y, t = integration_time_ss)
            else:
                z = solvers[0].integrate(rhs_func_ss.func, x = y, t = integration_time_ss)
        
        z = z[-1,:,:,:,:] - z0
        z = torch.norm(z.reshape((z.shape[0], -1)), dim = 1)
        z = torch.mean(z)
        
        return z


class MetaNODE(nn.Module):
    
    def __init__(self, downsampling_method = 'conv', is_odenet = True, activation_type = 'relu', in_channels = 1):
        super(MetaNODE, self).__init__()
        
        self.is_odenet = is_odenet
        
        self.downsampling_layers = nn.Sequential(*build_downsampling_layers(downsampling_method, in_channels))
        self.fc_layers = nn.Sequential(*build_fc_layers())
        
        if is_odenet:
            self.blocks = nn.ModuleList([MetaODEBlock(activation_type)])
        else:
            self.blocks = nn.ModuleList([ResBlock(64, 64) for _ in range(6)])
    
    
    def forward(self, x, solvers=None, solver_options=None, loss_options = None):
        self.ss_loss = 0
        
        x = self.downsampling_layers(x)
        
        for block in self.blocks:
            if self.is_odenet:
                x = block(x, solvers, solver_options)
                
                if (loss_options is not None) and loss_options.ss_loss:
                    z = block.ss_loss(x, solvers, solver_options)
                    self.ss_loss += z
            else:
                x = block(x)

        x = self.fc_layers(x)
        return x
    
    def get_ss_loss(self):
        return self.ss_loss 
    

class ODEfunc(nn.Module):

    def __init__(self, dim, activation_type = 'relu'):
        super(ODEfunc, self).__init__()
        
        if activation_type == 'tanh':
            activation = nn.Tanh()
        elif activation_type == 'softplus':
            activation = nn.Softplus()
        elif activation_type == 'softsign':
            activation = nn.Softsign()
        elif activation_type == 'relu':
            activation = nn.ReLU()
        else:
            raise NotImplementedError('{} activation is not implemented'.format(activation_type))

        self.norm1 = norm(dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm2 = norm(dim)
        self.conv2 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm3 = norm(dim)
        self.nfe = 0

    def forward(self, t, x, ss_loss = False):
        self.nfe += 1
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(t, out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(t, out)
        out = self.norm3(out)
        
        if ss_loss:
            out = torch.abs(out)

        return out
    
def build_downsampling_layers(downsampling_method = 'conv', in_channels = 1):
    if downsampling_method == 'conv':
        downsampling_layers = [
            nn.Conv2d(in_channels, 64, 3, 1),
            norm(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 4, 2, 1),
            norm(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 4, 2, 1),
        ]
    elif downsampling_method == 'res':
        downsampling_layers = [
            nn.Conv2d(in_channels, 64, 3, 1),
            ResBlock(64, 64, stride=2, downsample=conv1x1(64, 64, 2)),
            ResBlock(64, 64, stride=2, downsample=conv1x1(64, 64, 2)),
        ]
    return downsampling_layers


def build_fc_layers():
    fc_layers = [norm(64), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)), Flatten(), nn.Linear(64, 10)]
    return fc_layers


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def norm(dim):
    return nn.GroupNorm(min(32, dim), dim)


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.norm1 = norm(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.norm2 = norm(planes)
        self.conv2 = conv3x3(planes, planes)

    def forward(self, x):
        shortcut = x

        out = self.relu(self.norm1(x))

        if self.downsample is not None:
            shortcut = self.downsample(out)

        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + shortcut


class ConcatConv2d(nn.Module):

    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(ConcatConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in + 1, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1, :, :]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)
    
    
