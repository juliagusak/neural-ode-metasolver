import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

import numpy as np
from copy import deepcopy

__all__ = ['MetaNODE', 'metanode4', 'metanode6', 'metanode10', 'metanode18', 'metanode34',
           'premetanode4', 'premetanode6', 'premetanode10', 'premetanode18', 'premetanode34']

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)


class BasicBlock(nn.Module):
    '''Standard ResNet block
    '''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1,\
                 norm_layer=None, act_layer=None, param_norm=lambda x: x
                 ):
        super(BasicBlock, self).__init__()
        self.conv1 = param_norm(nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False))
        self.bn1 = norm_layer(planes)
    
        self.conv2 = param_norm(nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False))
        self.bn2 = norm_layer(planes)
    
        self.act = act_layer

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                param_norm(nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)),
                norm_layer(self.expansion * planes)
            )

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.act(out)
        return out


class PreBasicBlock(nn.Module):
    '''Standard PreResNet block
    '''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, \
                 norm_layer=None, act_layer=None, param_norm=lambda x: x
                 ):
        super(PreBasicBlock, self).__init__()
        self.bn1 = norm_layer(in_planes)
        self.conv1 = param_norm(nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False))

        self.bn2 = norm_layer(planes)
        self.conv2 = param_norm(nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False))

        self.act = act_layer

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                param_norm(nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)),
            )

    def forward(self, x):
        out = self.conv1(self.act(self.bn1(x)))
        out = self.conv2(self.act(self.bn2(out)))
        out += self.shortcut(x)
        return out
    
    
class BasicBlock2(nn.Module):
    '''Odefunc to use inside MetaODEBlock
    '''
    expansion = 1

    def __init__(self, dim,
                 norm_layer=None, act_layer=None, param_norm=lambda x: x):
        super(BasicBlock2, self).__init__()
        in_planes = dim
        planes = dim
        stride = 1
        self.nfe = 0

        self.conv1 = param_norm(nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False))
        # Replace BN to GN because BN doesn't work with our method normaly
        self.bn1 = norm_layer(planes)

        self.conv2 = param_norm(nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False))
        self.bn2 = norm_layer(planes)

        self.act = act_layer

        self.shortcut = nn.Sequential()

    def forward(self, t, x, ss_loss = False):
        self.nfe += 1
        if isinstance(x, tuple):
            x = x[0]
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act(out)
        
        if ss_loss:
            out = torch.abs(out)
        return out


class PreBasicBlock2(nn.Module):
    '''Odefunc to use inside MetaODEBlock
    '''
    expansion = 1

    def __init__(self, dim,
                 norm_layer=None, act_layer=None, param_norm=lambda x: x):
        super(PreBasicBlock2, self).__init__()
        in_planes = dim
        planes = dim
        stride = 1
        self.nfe = 0

        # Replace BN to GN because BN doesn't work with our method normaly
        self.bn1 = norm_layer(in_planes)
        self.conv1 = param_norm(nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False))

        self.bn2 = norm_layer(planes)
        self.conv2 = param_norm(nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False))

        self.act = act_layer

        self.shortcut = nn.Sequential()

    def forward(self, t, x, ss_loss=False):
        self.nfe += 1
        if isinstance(x, tuple):
            x = x[0]
        out = self.bn1(x)
        out = self.act(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.act(out)
        out = self.conv2(out)

        if ss_loss:
            out = torch.abs(out)
        return out


class MetaODEBlock(nn.Module):
    '''The same as MetaODEBlock for MNIST. Only difference is that odefunc is passed as an keyword argument'''
    def __init__(self, odefunc=None):
        super(MetaODEBlock, self).__init__()
        
        self.rhs_func = odefunc
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
    
    
class MetaLayer(nn.Module):
    '''
    norm_layers_: tuple of normalization layers for (BasicBlock, BasicBlock2, bn1)
    param_norm_layers_: tuple of normalizations for weights in (BasicBlock, BasicBlock2, conv1)
    act_layers_: tuple of activation layers for (BasicBlock, BasicBlock2, activation after bn1)
    resblock: BasicBlock or PreBasicBlock
    odefunc: BasicBlock2 or PreBasicBlock2

    '''
    def __init__(self, planes, num_blocks, stride, norm_layers_, param_norm_layers_, act_layers_,
                in_planes, resblock=None, odefunc=None):
        super(MetaLayer, self).__init__()

        num_resblocks, num_odeblocks = num_blocks
        
        strides = [stride] + [1] * (num_resblocks + num_odeblocks - 1)
        layers_res = []
        layers_ode = []
        
        self.in_planes = in_planes
        for stride in strides[:num_resblocks]:
            layers_res.append(resblock(self.in_planes, planes, stride,
                                         norm_layer = norm_layers_[0],
                                         param_norm=param_norm_layers_[0],
                                         act_layer = act_layers_[0]))
            self.in_planes = planes * resblock.expansion
            
        for stride in strides[num_resblocks:]:
            layers_ode.append(MetaODEBlock(odefunc(self.in_planes,
                                                    norm_layer=norm_layers_[1],
                                                    param_norm=param_norm_layers_[1],
                                                    act_layer=act_layers_[1])))
            
        self.blocks_res = nn.Sequential(*layers_res)
        self.blocks_ode = nn.ModuleList(layers_ode)
        

    def forward(self, x, solvers=None, solver_options=None, loss_options = None):
        x = self.blocks_res(x)
        
        self.ss_loss = 0
        
        for block in self.blocks_ode:
            x = block(x, solvers, solver_options)

            if (loss_options is not None) and loss_options.ss_loss:
                z = block.ss_loss(x, solvers, solver_options)
                self.ss_loss += z
                
        return x
    
    def get_ss_loss(self):
        return self.ss_loss
    
    @property
    def nfe(self):
        per_block_nfe = {idx: block.nfe for idx, block in enumerate(self.blocks_ode)}
        return sum(per_block_nfe)

    @nfe.setter
    def nfe(self, value):
        for block in self.blocks_ode:
            block.nfe = value
            
            
class MetaNODE(nn.Module):
    def __init__(self, num_blocks, num_classes=10,
                 norm_layers_=(None, None, None),
                 param_norm_layers_=(lambda x: x, lambda x: x, lambda x: x),
                 act_layers_=(None, None, None),
                 in_planes_=64,
                 resblock=None,
                 odefunc=None):
        '''
        norm_layers_: tuple of normalization layers for (BasicBlock, BasicBlock2, bn1)
        param_norm_layers_: tuple of normalizations for weights in (BasicBlock, BasicBlock2, conv1)
        act_layers_: tuple of activation layers for (BasicBlock, BasicBlock2, activation after bn1)
        resblock: BasicBlock or PreBasicBlock
        odefunc: BasicBlock2 or PreBasicBlock2
        '''
        
        super(MetaNODE, self).__init__()
        self.in_planes = in_planes_
        
        self.n_layers = len(num_blocks)
        self.n_features_linear = in_planes_
        
        self.is_preactivation = False
        if (((resblock is not None) and isinstance(resblock, PreBasicBlock)) or
                ((odefunc is not None) and isinstance(odefunc, PreBasicBlock2))):
            self.is_preactivation = True

        self.conv1 = param_norm_layers_[2](nn.Conv2d(3, in_planes_, kernel_size=3, stride=1, padding=1, bias=False))
        self.bn1 = norm_layers_[2](in_planes_)
        self.act = act_layers_[2]
        
        self.layer1 = MetaLayer(in_planes_, num_blocks[0], stride=1,
                                norm_layers_=norm_layers_[:2],
                                param_norm_layers_=param_norm_layers_[:2],
                                act_layers_=act_layers_[:2],
                                in_planes=self.in_planes,
                                resblock=resblock,
                                odefunc=odefunc
                                )
        
        if self.n_layers >= 2:
            self.n_features_linear *= 2
            self.layer2 = MetaLayer(in_planes_*2, num_blocks[1], stride=2,
                                    norm_layers_=norm_layers_[:2],
                                    param_norm_layers_=param_norm_layers_[:2],
                                    act_layers_=act_layers_[:2],
                                    in_planes=self.layer1.in_planes,
                                    resblock=resblock,
                                    odefunc=odefunc
                                    )
   
        if self.n_layers >= 3:
            self.n_features_linear *= 2
            self.layer3 = MetaLayer(in_planes_*4, num_blocks[2], stride=2,
                                    norm_layers_=norm_layers_[:2],
                                    param_norm_layers_=param_norm_layers_[:2],
                                    act_layers_=act_layers_[:2],
                                    in_planes=self.layer2.in_planes,
                                    resblock=resblock,
                                    odefunc=odefunc
                                    )

        if self.n_layers >= 4:
            self.n_features_linear *= 2
            self.layer4 = MetaLayer(in_planes_*8, num_blocks[3], stride=2,
                                    norm_layers_=norm_layers_[:2],
                                    param_norm_layers_=param_norm_layers_[:2],
                                    act_layers_=act_layers_[:2],
                                    in_planes=self.layer3.in_planes,
                                    resblock=resblock,
                                    odefunc=odefunc
                                    )

        self.fc_layers = nn.Sequential(*[nn.AdaptiveAvgPool2d((1, 1)),
                                         Flatten(),
                                         nn.Linear(self.n_features_linear * resblock.expansion, num_classes)])
        self.nfe = 0

    @property
    def nfe(self):
        nfe = 0
        for idx in range(1, self.n_layers+1):
            nfe += self.__getattr__('layer{}'.format(idx)).nfe
        return nfe

    @nfe.setter
    def nfe(self, value):
        for idx in range(1, self.n_layers+1):
            self.__getattr__('layer{}'.format(idx)).nfe = value

    
    def forward(self, x, solvers=None, solver_options=None, loss_options = None):
        self.ss_loss = 0
        
        out = self.conv1(x)
        if not self.is_preactivation:
            out = self.act(self.bn1(out))

        for idx in range(1, self.n_layers + 1):
            out = self.__getattr__('layer{}'.format(idx))(out,
                  solvers=solvers, solver_options=solver_options,
                  loss_options = loss_options)
            
            self.ss_loss += self.__getattr__('layer{}'.format(idx)).ss_loss
            
        if self.is_preactivation:
            out = self.act(self.bn1(out))

        out = self.fc_layers(out)
        return out
    

def metanode4(norm_layers, param_norm_layers, act_layers, in_planes, is_odenet = True):
    if is_odenet:
        num_blocks = [(0, 1)]
    else:
        num_blocks = [(1, 0)]
    return MetaNODE(num_blocks,
                    norm_layers_= norm_layers, param_norm_layers_=param_norm_layers, act_layers_=act_layers, 
                    in_planes_ = in_planes,
                    resblock=BasicBlock,
                    odefunc=BasicBlock2
                    )

    
def metanode6(norm_layers, param_norm_layers, act_layers, in_planes, is_odenet = True):
    if is_odenet:
        num_blocks = [(1, 1)]
    else:
        num_blocks = [(2, 0)]
    return MetaNODE(num_blocks,
                    norm_layers_= norm_layers, param_norm_layers_=param_norm_layers, act_layers_=act_layers, 
                    in_planes_ = in_planes,
                    resblock=BasicBlock,
                    odefunc=BasicBlock2
                    )

    
def metanode10(norm_layers, param_norm_layers, act_layers, in_planes, is_odenet = True):
    if is_odenet:
        num_blocks = [(1, 1), (1, 1)]
    else:
        num_blocks = [(2, 0), (2, 0)]
    return MetaNODE(num_blocks,
                    norm_layers_= norm_layers, param_norm_layers_=param_norm_layers, act_layers_=act_layers, 
                    in_planes_ = in_planes,
                    resblock=BasicBlock,
                    odefunc=BasicBlock2
                    )

    
def metanode18(norm_layers, param_norm_layers, act_layers, in_planes, is_odenet = True):
    if is_odenet:
        num_blocks =  [(1, 1), (1, 1), (1, 1), (1, 1)]
    else:
        num_blocks = [(2, 0), (2, 0), (2, 0), (2, 0)]
    return MetaNODE(num_blocks,
                    norm_layers_= norm_layers, param_norm_layers_=param_norm_layers, act_layers_=act_layers, 
                    in_planes_ = in_planes,
                    resblock=BasicBlock,
                    odefunc=BasicBlock2
                    )

    
def metanode34(norm_layers, param_norm_layers, act_layers, in_planes, is_odenet = True):
    if is_odenet:
        num_blocks = [(1, 2), (1, 3), (1, 5), (1, 2)]
    else:
        num_blocks = [(3, 0), (4, 0), (6, 0), (3, 0)]
    return MetaNODE(num_blocks,
                    norm_layers_= norm_layers, param_norm_layers_=param_norm_layers, act_layers_=act_layers, 
                    in_planes_ = in_planes,
                    resblock=BasicBlock,
                    odefunc=BasicBlock2
                    )


def premetanode4(norm_layers, param_norm_layers, act_layers, in_planes, is_odenet=True):
    if is_odenet:
        num_blocks = [(0, 1)]
    else:
        num_blocks = [(1, 0)]
    return MetaNODE(num_blocks,
                    norm_layers_=norm_layers, param_norm_layers_=param_norm_layers, act_layers_=act_layers,
                    in_planes_=in_planes,
                    resblock=PreBasicBlock,
                    odefunc=PreBasicBlock2
                    )


def premetanode6(norm_layers, param_norm_layers, act_layers, in_planes, is_odenet=True):
    if is_odenet:
        num_blocks = [(1, 1)]
    else:
        num_blocks = [(2, 0)]
    return MetaNODE(num_blocks,
                    norm_layers_=norm_layers, param_norm_layers_=param_norm_layers, act_layers_=act_layers,
                    in_planes_=in_planes,
                    resblock=PreBasicBlock,
                    odefunc=PreBasicBlock2
                    )


def premetanode10(norm_layers, param_norm_layers, act_layers, in_planes, is_odenet=True):
    if is_odenet:
        num_blocks = [(1, 1), (1, 1)]
    else:
        num_blocks = [(2, 0), (2, 0)]
    return MetaNODE(num_blocks,
                    norm_layers_=norm_layers, param_norm_layers_=param_norm_layers, act_layers_=act_layers,
                    in_planes_=in_planes,
                    resblock=PreBasicBlock,
                    odefunc=PreBasicBlock2
                    )


def premetanode18(norm_layers, param_norm_layers, act_layers, in_planes, is_odenet=True):
    if is_odenet:
        num_blocks = [(1, 1), (1, 1), (1, 1), (1, 1)]
    else:
        num_blocks = [(2, 0), (2, 0), (2, 0), (2, 0)]
    return MetaNODE(num_blocks,
                    norm_layers_=norm_layers, param_norm_layers_=param_norm_layers, act_layers_=act_layers,
                    in_planes_=in_planes,
                    resblock=PreBasicBlock,
                    odefunc=PreBasicBlock2
                    )


def premetanode34(norm_layers, param_norm_layers, act_layers, in_planes, is_odenet=True):
    if is_odenet:
        num_blocks = [(1, 2), (1, 3), (1, 5), (1, 2)]
    else:
        num_blocks = [(3, 0), (4, 0), (6, 0), (3, 0)]
    return MetaNODE(num_blocks,
                    norm_layers_=norm_layers, param_norm_layers_=param_norm_layers, act_layers_=act_layers,
                    in_planes_=in_planes,
                    resblock=PreBasicBlock,
                    odefunc=PreBasicBlock2
                    )
