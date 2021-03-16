import torch
import torch.nn as nn

from .rk_parametric import RKParametricSolver

def build_ButcherTableau_Midpoint(dtype=None, device=None):
    c = torch.tensor([0., 1/2.],dtype=dtype, device=device)
    w = [torch.tensor([0.,], dtype=dtype, device=device)] + [torch.tensor([1/2., 0.],dtype=dtype, device=device)]
    b = torch.tensor([0., 1.],dtype=dtype, device=device)
    return c, w, b


def build_ButcherTableau_Heun(dtype=None, device=None):
    c = torch.tensor([0., 1.],dtype=dtype, device=device)
    w = [torch.tensor([0.,],dtype=dtype, device=device)] + [torch.tensor([1., 0.],dtype=dtype, device=device)]
    b = torch.tensor([1/2., 1/2.],dtype=dtype, device=device)
    return c, w, b


class RKOrder2Stage2(RKParametricSolver):
    def __init__(self, parameterization='u', u0 = None, v0 = None, dtype=None, device=None, **kwargs):
        super(RKOrder2Stage2, self).__init__(**kwargs)
        self.dtype = dtype
        self.device = device
        
        if parameterization != 'u':
            raise ValueError('Unknown parameterization for RKOrder2Stage2 solver')
        self.parameterization = parameterization
        self.u = nn.Parameter(data = torch.tensor((u0,), dtype=self.dtype, device=self.device))
        self.u0 = torch.tensor((u0,), dtype=self.dtype, device=self.device)
        self.v = None
        self.v0 = None
            
        self.build_ButcherTableau()

        
    def _compute_c(self):
        self.c1 = torch.tensor((0.,),dtype=self.dtype, device=self.device) 
        self.c2 = self.u_.clone()
            
        
    def _compute_b(self):
        self.b2 = 1. / (2 * self.u_)
        self.b1 = 1. - self.b2

        
    def _compute_w(self):
        self.w21 = self.c2
        self.w11, self.w22 = [torch.tensor((0.,),dtype=self.dtype, device=self.device) for _ in range(2)]

        
    def _make_u_valid(self, eps):
        self.u_ = torch.clamp(self.u, eps, 1.)
        
        
    def _make_params_valid(self):
        if self.u.dtype == torch.float64:
            eps = torch.finfo(torch.float32).eps
        elif self.u.dtype == torch.float32:
            eps = torch.finfo(torch.float16).eps
            
        self._make_u_valid(eps)
                
    
    def _get_c(self):
        c = torch.tensor([self.c1, self.c2])
        return c
    
    
    def _get_w(self):
        w = [torch.tensor([self.w11,])] + [
            torch.tensor([self.w21, self.w22])] 
        return w
    
    
    def _get_b(self):
        b = torch.tensor([self.b1, self.b2])
        return b    
    
    
    def _get_t(self, t, dt):
        t0 = t 
        t1 = t + self.c2 * dt
        return (t0, t1)

    
    def _make_step(self, rhs_func, x, t, dt):
        t0, t1 = self._get_t(t, dt)

        k1 = rhs_func(t0, x)
        k2 = rhs_func(t1, x + k1 * self.w21 * dt)

        return (k1 * self.b1 + k2 * self.b2) * dt
    
    
    def freeze_params(self):
        self.u.requires_grad = False
        if self.v is not None:
            self.v.requires_grad = False
            
        self.build_ButcherTableau() # recompute params to set non leaf requires_grad to False

        
    def unfreeze_params(self):
        self.u.requires_grad = True
        if self.v is not None:
            self.v.requires_grad = True
            
        self.build_ButcherTableau() # recompute params to set non leaf requires_grad to True
    
    @property
    def order(self):
        return 2