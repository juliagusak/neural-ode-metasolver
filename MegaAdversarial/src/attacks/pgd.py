from .attack import Attack
import torch
import torch.nn as nn
import torchvision.transforms as transforms



class PGD(Attack):
    """
    The standard PGD attack. Assumes (0, 1) normalization.
    """

    def __init__(self, model, eps=None, lr=None, n_iter=None, randomized_start=True, mean=None, std=None):
        super(PGD, self).__init__(model)
        self.eps = eps
        self.lr = lr
        self.n_iter = n_iter
        self.randomized_start = randomized_start
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

        if self.randomized_start:
            x_attacked = (
                self._project(x + torch.zeros_like(x).uniform_(-self.eps, self.eps))
                .clone()
                .detach()
            )
        else:
            x_attacked = x.clone().detach()

        for i in range(self.n_iter):
            x_attacked.requires_grad_(True)
            loss = self.loss_fn(self.model(normalize(x_attacked), **kwargs), y)
            grad = torch.autograd.grad(
                [loss], [x_attacked], create_graph=False, retain_graph=False
            )[0]
            x_attacked = self._clamp(
                x_attacked + self.lr * grad.sign(), x - self.eps, x + self.eps
            )
            x_attacked = self._project(x_attacked)
            if i == self.n_iter - 1:
                x_attacked = normalize(x_attacked)
            x_attacked = x_attacked.detach()

        if training:
            self.model.train()
        return x_attacked, y