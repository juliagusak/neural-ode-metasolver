import torch
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Attack(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.device = device

    def _project(self, x):
        return torch.clamp(x, 0, 1)

    def _clamp(self, x, min, max):
        return torch.max(torch.min(x, max), min)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

        
class Attack2Ensemble(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = models
        self.device = device

    def _project(self, x):
        return torch.clamp(x, 0, 1)

    def _clamp(self, x, min, max):
        return torch.max(torch.min(x, max), min)

    def forward(self, *args, **kwargs):
        raise NotImplementedError