import torch
import torch.nn as nn

from .rk_parametric import RKParametricSolver


class Euler(RKParametricSolver):
    def __init__(self, parameterization=None, u0 = None, v0 = None, dtype=None, device=None, **kwargs):
        super(Euler, self).__init__(**kwargs)
        self.dtype = dtype
        self.device = device
        
        self.parameterization = None
        
        self.u = None
        self.u0 = None
        self.v = None
        self.v0 = None
            
        self.build_ButcherTableau()

        
    def _compute_c(self):
        self.c1 = torch.tensor((0.,),dtype=self.dtype, device=self.device) 
            
        
    def _compute_b(self):
        self.b1 = torch.tensor((1.,),dtype=self.dtype, device=self.device)

        
    def _compute_w(self):
        self.w11 = torch.tensor((0.,),dtype=self.dtype, device=self.device)

        
    def _make_u_valid(self, eps):
        pass
        
        
    def _make_params_valid(self):
        pass
                
    
    def _get_c(self):
        c = torch.tensor([self.c1,])
        return c
    
    
    def _get_w(self):
        w = [torch.tensor([self.w11,])]
        return w
    
    
    def _get_b(self):
        b = torch.tensor([self.b1,])
        return b    
    
    
    def _get_t(self, t, dt):
        t0 = t 
        return t0

    
    def _make_step(self, rhs_func, x, t, dt):
        t0 = self._get_t(t, dt)
        
        k1 = rhs_func(t0, x)

        return (k1 * self.b1) * dt
    
    
    def freeze_params(self):
        pass

        
    def unfreeze_params(self):
        pass
        
    
    @property
    def order(self):
        return 1
