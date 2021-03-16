from .attack import Attack, Attack2Ensemble
import torch
import torch.nn as nn
import torchvision.transforms as transforms



class FGSM(Attack):
    """
    The standard FGSM attack. Assumes (0, 1) normalization.
    """

    def __init__(self, model, eps=None, mean=None, std=None):
        super(FGSM, self).__init__(model)
        self.eps = eps
        self.loss_fn = nn.CrossEntropyLoss().to(self.device)
        self.mean = mean if mean is not None else (0., 0., 0.)
        self.std = std if std is not None else (1., 1., 1.)


    def forward(self, x, y, kwargs):

        training = self.model.training
        if training:
            self.model.eval()

        inv_normalize = transforms.Normalize(mean=[-m/s for m, s in zip(self.mean, self.std)], std=[1/s for s in self.std])
        normalize = transforms.Normalize(mean=self.mean, std=self.std)
        x = inv_normalize(x) # x in [0, 1]

        x_attacked = x.clone().detach()
        x_attacked.requires_grad_(True)
        loss = self.loss_fn(self.model(normalize(x_attacked), **kwargs), y)
        grad = torch.autograd.grad(
            [loss], [x_attacked], create_graph=False, retain_graph=False
        )[0]
        x_attacked = x_attacked + self.eps * grad.sign()
        x_attacked = self._project(x_attacked)
        x_attacked = normalize(x_attacked)
        x_attacked = x_attacked.detach()
        if training:
            self.model.train()
        return x_attacked, y


def clamp(X, lower_limit, upper_limit):
    if not isinstance(upper_limit, torch.Tensor):
        upper_limit = torch.tensor(upper_limit, device=X.device, dtype=X.dtype)
    if not isinstance(lower_limit, torch.Tensor):
        lower_limit = torch.tensor(lower_limit, device=X.device, dtype=X.dtype)
    return torch.max(torch.min(X, upper_limit), lower_limit)


class FGSMRandom(Attack):
    """
    The standard FGSM attack. Assumes (0, 1) normalization.
    This implementation is inspired by the implementation from here:
    https://github.com/locuslab/fast_adversarial/blob/master/CIFAR10/train_fgsm.py
    """

    def __init__(self, model, alpha, epsilon=None, mu=None, std=None):
        '''
        Args:
            model: the neural network model
            alpha: the step size
            epsilon: the radius of the random noise
            mu: the mean value of all dataset samples
            std: the std value of all dataset samples
        '''
        super(FGSMRandom, self).__init__(model)
        self.epsilon = epsilon
        self.alpha = alpha
        if (mu is not None) and (std is not None):
            mu = torch.tensor(mu, device=self.device).view(1, 3, 1, 1)
            std = torch.tensor(std, device=self.device).view(1, 3, 1, 1)

            self.lower_limit = (0. - mu) / std
            self.upper_limit = (1. - mu) / std # lower = -mu/std, upper=(1-mu)/std

            self.epsilon = self.epsilon / std
            self.alpha = self.alpha / std
        else:
            self.lower_limit = 0.
            self.upper_limit = 1.

        self.loss_fn = nn.CrossEntropyLoss().to(self.device)

    def forward(self, x, y, kwargs):
        training = self.model.training
        if training:
            self.model.eval()

        delta = self.epsilon - (2 * self.epsilon) * torch.rand_like(x)  # Uniform[-eps, eps]
        delta.data = clamp(delta, self.lower_limit - x, self.upper_limit - x)
        delta.requires_grad = True
        output = self.model(x + delta, **kwargs)
        loss = self.loss_fn(output, y)
        loss.backward()
        grad = delta.grad.detach()
        delta.data = clamp(delta + self.alpha * torch.sign(grad), -self.epsilon, self.epsilon)
        delta.data = clamp(delta, self.lower_limit - x, self.upper_limit - x)
        delta = delta.detach()

        if training:
            self.model.train()
        return x + delta, y


class FGSM2Ensemble(Attack2Ensemble):
    """
    The standard FGSM attack. Assumes (0, 1) normalization.
    """

    def __init__(self, models, eps=None, mean=None, std=None):
        super(FGSM2Ensemble, self).__init__(models)
        self.eps = eps
        self.loss_fn = nn.NLLLoss().to(self.device)
        self.mean = mean if mean is not None else (0., 0., 0.)
        self.std = std if std is not None else (1., 1., 1.)

    def forward(self, x, y, kwargs_arr):

        training = self.models[0].training
        if training:
            for model in self.models:
                model.eval()

        inv_normalize = transforms.Normalize(mean=[-m/s for m, s in zip(self.mean, self.std)], std=[1/s for s in self.std])
        normalize = transforms.Normalize(mean=self.mean, std=self.std)
        x = inv_normalize(x) # x in [0, 1]
                
        x_attacked = x.clone().detach()
        x_attacked.requires_grad_(True)
        
        probs_ensemble = 0
        
        for model, kwargs in zip(self.models, kwargs_arr):
            logits = model(normalize(x_attacked), **kwargs)
            probs_ensemble = probs_ensemble + nn.Softmax()(logits)
            
        probs_ensemble /= len(self.models)
        
        loss = self.loss_fn(torch.log(probs_ensemble), y)
        grad = torch.autograd.grad(
            [loss], [x_attacked], create_graph=False, retain_graph=False
        )[0]
        x_attacked = x_attacked + self.eps * grad.sign()
        x_attacked = self._project(x_attacked)
        x_attacked = normalize(x_attacked)
        x_attacked = x_attacked.detach()
        
        if training:
            for model in self.models:
                model.train()
        return x_attacked, y
