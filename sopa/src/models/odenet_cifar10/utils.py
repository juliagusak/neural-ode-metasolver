import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.utils import spectral_norm, weight_norm
from functools import partial


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x

def get_normalization(key, num_groups=32):
    '''Create a normalization layer given name of layer type.

    :param key: str
        Type of normalization layer to use after convolutional layer (e.g. type of layer output normalization).
        Can be one of BN (batch normalization), GN (group normalization), LN (layer normalization),
        IN (instance  normalization), NF(no normalization)
    :param num_groups: int
        Number of groups for GN normalization
    :return: nn.Module
        Normalization layer
    '''
    if key == 'BN':
        return nn.BatchNorm2d
    elif key == 'LN':
        return partial(nn.GroupNorm, 1)
    elif key == 'GN':
        return partial(nn.GroupNorm, num_groups)
    elif key == 'IN':
        return nn.InstanceNorm2d
    elif key == 'NF':
        return Identity
    else:
        raise NameError('Unknown layer normalization type')

def get_param_normalization(key):
    '''Create a function to normalize layer weights given name of normalization.

    :param key: str
        Type of normalization applied  to layer's weights.
        Can be one of SN (spectral normalization), WN (weight normalization), PNF (no weight normalization).
    :return: function
    '''
    if key == 'SN':
        return spectral_norm
    elif key == 'WN':
        return weight_norm
    elif key == 'PNF':
        return lambda x: x 
    else:
        raise NameError('Unknown param normalization type')

def get_activation(key):
    '''Create an activation function given name of function type.
    
    :param key: str
        Type of activation layer.
        Can be one of: ReLU, AF (no activation/linear activation)
    :return: function
    '''
    if key == 'ReLU':
        return F.relu
    elif key == 'GeLU':
        return F.gelu
    elif key == 'Softsign':
        return F.softsign
    elif key == 'Tanh':
        return F.tanh
    elif key == 'AF':
        return partial(F.leaky_relu, negative_slope=1)
    else:
        raise NameError('Unknown activation type')
        
def conv_init(m):
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1 and m.bias is not None:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif class_name.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)

def conv_init_orthogonal(m):
    if isinstance(m, nn.Conv2d):
        init.orthogonal_(m.weight)

def fc_init_orthogonal(m):
    if isinstance(m, nn.Linear):
        init.orthogonal_(m.weight)
        init.constant_(m.bias, 1e-3)

