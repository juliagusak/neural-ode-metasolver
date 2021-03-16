import numpy as np
import torch
import random

from .odenet_mnist.layers import MetaNODE
    
def fix_seeds(seed=502):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_printoptions(precision=10)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val
        
        
def load_model(path):
    (_, state_dict), (_, model_args), (_, slover_id) = torch.load(path, map_location='cpu').items()

    is_odenet = model_args.network == 'odenet'
    
    if not hasattr(model_args, 'in_channels'):
        model_args.in_channels = 1
    
    model = MetaNODE(downsampling_method=model_args.downsampling_method,
                     is_odenet=is_odenet,
                     in_channels=model_args.in_channels)
    model.load_state_dict(state_dict) 
    
    return model, model_args