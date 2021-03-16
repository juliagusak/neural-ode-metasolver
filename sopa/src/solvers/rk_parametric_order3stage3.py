import torch
import torch.nn as nn

from .rk_parametric import RKParametricSolver


class RKOrder3Stage3(RKParametricSolver):
    def __init__(self, parameterization = 'uv', u0 = 1/3., v0 = 2/3., dtype=None, device=None, **kwargs):
        super(RKOrder3Stage3, self).__init__(**kwargs)
        self.dtype = dtype
        self.device = device

        if parameterization != 'uv':
            raise ValueError('Unknown parameterization for RKOrder3Stage3 solver')
        self.parameterization = parameterization
        self.u = nn.Parameter(data = torch.tensor((u0,), dtype=self.dtype, device=self.device))
        self.u0 = torch.tensor((u0,), dtype=self.dtype, device=self.device)

        self.v = nn.Parameter(data = torch.tensor((v0,), dtype=self.dtype, device=self.device))
        self.v0 = torch.tensor((v0,), dtype=self.dtype, device=self.device)
            
        self.build_ButcherTableau()

        
    def _compute_c(self):
        self.c1 = torch.tensor((0.,), dtype=self.dtype, device=self.device) 
        self.c2 = self.u_.clone()
        self.c3 = self.v_.clone()
            
        
    def _compute_b(self):
        v_sub_u = self.v_ - self.u_
            
        self.b2 = (2. - 3. * self.v_) / (6. * self.u_ * (-v_sub_u))
        self.b3 = (2. - 3. * self.u_) / (6. * self.v_ * v_sub_u)
        self.b1 = 1. - self.b2 - self.b3

        
    def _compute_w(self):
        self.w32 = self.v_ * (self.v_ - self.u_) / (self.u_ * (2. - 3. * self.u_))
        self.w31 = self.c3 - self.w32
        self.w21 = self.c2
        
        self.w11, self.w22, self.w33 = [torch.tensor((0.,), dtype=self.dtype, device=self.device) for _ in range(3)]

        
    def _make_u_valid(self, eps):
        self.u_ = torch.clamp(self.u, eps, 1.)
        
        
    def _make_v_valid(self, eps):
        self.v_ = torch.clamp(self.v, eps, 1.) 
        
        
    def _make_params_valid(self):
        if self.u.dtype == torch.float64:
            eps = torch.finfo(torch.float32).eps
        elif self.u.dtype == torch.float32:
            eps = torch.finfo(torch.float16).eps
        
        self._make_u_valid(eps)
        self._make_v_valid(eps)

        if self.u_ == self.v_:
            if self.u_ < 1. - eps:
                self.v_ = self.u_ + eps
            else:
                self.u_ = self.v_ - eps
                
    
    def _get_c(self):
        c = torch.tensor([self.c1, self.c2, self.c3])
        return c
    
    
    def _get_w(self):
        w = [torch.tensor([self.w11,])] + [
            torch.tensor([self.w21, self.w22])] + [
            torch.tensor([self.w31, self.w32, self.w33])]
        return w
    
    
    def _get_b(self):
        b = torch.tensor([self.b1, self.b2, self.b3])
        return b    
    
    
    def _get_t(self, t, dt):
        t0 = t 
        t1 = t + self.c2 * dt
        t2 = t + self.c3 * dt
        
        return (t0, t1, t2)

    
    def _make_step(self, rhs_func, x, t, dt):
        t0, t1, t2 = self._get_t(t, dt)

        k1 = rhs_func(t0, x)
        k2 = rhs_func(t1, x + k1 * self.w21 * dt)
        k3 = rhs_func(t2, x + (k1 * self.w31 + k2 * self.w32) * dt)

        return (k1 * self.b1 + k2 * self.b2 + k3 * self.b3) * dt
    
    
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
        return 3