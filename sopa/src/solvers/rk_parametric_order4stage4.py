import torch
import torch.nn as nn

from .rk_parametric import RKParametricSolver

def build_ButcherTableau_RKStandard(dtype=None, device=None):
    c = torch.tensor([0., 1/2., 1/2., 1.], dtype=dtype, device=device)
    w = [torch.tensor([0.,], dtype=dtype, device=device)] + [torch.tensor([1/2., 0.], dtype=dtype, device=device)] + [torch.tensor(w_i, dtype=dtype, device=device) for w_i in [[0., 1/2., 0.], [0., 0., 1., 0.]]]
    b = torch.tensor([1/6., 1/3., 1/3., 1/6.])
    return c, w, b


def build_ButcherTableau_RK38(dtype=None, device=None):
    c = torch.tensor([0., 1/3., 2/3., 1.], dtype=dtype, device=device)
    w = [torch.tensor([0.,], dtype=dtype, device=device)] + [torch.tensor([1/3., 0.], dtype=dtype, device=device)] + [torch.tensor(w_i, dtype=dtype, device=device) for w_i in [[-1/3., 1., 0.], [1., -1., 1., 0.]]]
    b = torch.tensor([1/8., 3/8., 3/8., 1/8.], dtype=dtype, device=device)
    return c, w, b


class RKOrder4Stage4(RKParametricSolver):
    def __init__(self, parameterization = 'u2', u0 = 1/3., v0 = 2/3., dtype=torch.float64, device='cpu', **kwargs):
        super(RKOrder4Stage4, self).__init__(**kwargs)
        self.dtype = dtype
        self.device = device
        
        self.parameterization = parameterization
        self.u = nn.Parameter(data = torch.tensor((u0,), dtype=self.dtype, device=self.device))
        self.u0 = torch.tensor((u0,), dtype=self.dtype, device=self.device)
        
        if self.parameterization == 'uv':
            self.v = nn.Parameter(data = torch.tensor((v0,), dtype=self.dtype, device=self.device))
            self.v0 = torch.tensor((v0,), dtype=self.dtype, device=self.device)
        else:
            self.v = None
            self.v0 = None
            
        self.build_ButcherTableau()

    
    def _compute_c(self):
        self.c1 = torch.tensor((0.,), dtype=self.dtype, device=self.device) 

        if self.parameterization == 'u1':
            self.c2 = torch.tensor((0.5,), dtype=self.dtype, device=self.device)
            self.c3 = torch.tensor((0.,), dtype=self.dtype, device=self.device)
            
        elif self.parameterization == 'u2':
            self.c2 = torch.tensor((0.5,), dtype=self.dtype, device=self.device)
            self.c3 = torch.tensor((0.5,), dtype=self.dtype, device=self.device)
            
        elif self.parameterization == 'u3':
            self.c2 = torch.tensor((1.,), dtype=self.dtype, device=self.device)
            self.c3 = torch.tensor((0.5,), dtype=self.dtype, device=self.device)
            
        elif self.parameterization == 'uv':
            self.c2 = self.u_.clone() # .clone() is nedeed, because without it self.c2 will be nn.Parameter
            self.c3 = self.v_.clone()

        self.c4 = torch.tensor((1.,), dtype=self.dtype, device=self.device) 


        
    
    def _compute_b(self):
        if self.parameterization == 'u1':
            self.b1 = torch.tensor((1/6.,),dtype=self.dtype, device=self.device) - self.u_
            self.b2 = torch.tensor((2/3.,),dtype=self.dtype, device=self.device)
            self.b3 = self.u_.clone()
            self.b4 = torch.tensor((1/6.,),dtype=self.dtype, device=self.device)
            
        elif self.parameterization == 'u2':
            self.b1 = torch.tensor((1/6.,),dtype=self.dtype, device=self.device)
            self.b2 = torch.tensor((2/3.,),dtype=self.dtype, device=self.device) - self.u_
            self.b3 = self.u_.clone()
            self.b4 = torch.tensor((1/6.,),dtype=self.dtype, device=self.device)
            
        elif self.parameterization == 'u3':
            self.b1 = torch.tensor((1/6.,),dtype=self.dtype, device=self.device)
            self.b2 = torch.tensor((1/6.,),dtype=self.dtype, device=self.device) - self.u_
            self.b3 = torch.tensor((2/3.,),dtype=self.dtype, device=self.device)
            self.b4 = self.u_.clone()
            
        elif self.parameterization == 'uv':
            sub_u = 1. - self.u_
            sub_v = 1. - self.v_
            v_sub_u = self.v_ - self.u_
            
            self.b2 = (2. * self.v_ - 1.) / (12 * self.u_ * sub_u * v_sub_u)
            self.b3 = (1. - 2 * self.u_) / (12 * self.v_ * sub_v * v_sub_u)
            self.b4 = (6. * self.u_ * self.v_ + 3. - 4. * self.u_ - 4. * self.v_) / (12 * sub_u * sub_v)
            self.b1 = 1. - self.b2 - self.b3 - self.b4

    
    def _compute_w(self):
        self.w43 = self.b3 * (1 - self.c3) / self.b4
        
        A00 = self.b3 * self.c3 * self.c2
        A01 = self.b4 * self.c4 * self.c2
        A10 = self.b3
        A11 = self.b4

        B0 = 0.125 - self.b4 * self.c4 * self.c3 * self.w43
        B1 = self.b2 * (1 - self.c2)
        
        ### Find w32, w42 using Cramer's rule
        detA = A00 * A11 - A01 * A10
        detA0 = B0 * A11 - B1 * A01
        detA1 = A00 * B1 - A10 * B0
        
        self.w32 = detA0 / detA
        self.w42 = detA1 / detA
        ###
        
#         ### Find w32, w42 using torch.solve
#         A = torch.cat((A00, A01, A10, A11)).reshape((2,2))
#         B = torch.cat((B0, B1)).reshape((2,1))
#         (self.w32, self.w42), _ = torch.solve(B, A)
#         ###

        self.w41 = self.c4 - (self.w42 + self.w43)
        self.w31 = self.c3 - self.w32
        self.w21 = self.c2
        
        self.w11, self.w22, self.w33, self.w44 = [torch.tensor((0.,),dtype=self.dtype, device = self.device) for _ in range(4)]
        
    
    def _make_u_valid(self, eps):
        if self.v is not None:
            if self.u < 0.5:
                self.u_ = torch.clamp(self.u, eps, 0.5 - eps)
            else:
                self.u_ = torch.clamp(self.u, 0.5 + eps, 1. - eps)
        else:
            self.u_ = torch.clamp(self.u, eps, 1. - eps)
            
    
    def _make_v_valid(self, eps):
        self.v_ = torch.clamp(self.v, eps, 1. - eps) 
            
    
    def _make_params_valid(self):
        if self.u.dtype == torch.float64:
            eps = torch.finfo(torch.float32).eps
        elif self.u.dtype == torch.float32:
            eps = torch.finfo(torch.float16).eps

        self._make_u_valid(eps)
        
        if self.v is not None:
            self._make_v_valid(eps)

            if self.u_ == self.v_:
                if self.u_ < 1. - eps:
                    self.v_ = self.u_ + eps
                else:
                    self.u_ = self.v_ - eps

    
    def _get_c(self):
        c = torch.tensor([self.c1, self.c2, self.c3, self.c4])
        return c
    
    def _get_w(self):
        w = [torch.tensor([self.w11,])] + [
            torch.tensor([self.w21, self.w22])] + [
            torch.tensor([self.w31, self.w32, self.w33])] + [
            torch.tensor([self.w41, self.w42, self.w43, self.w44])] 

        return w
    
    def _get_b(self):
        b = torch.tensor([self.b1, self.b2, self.b3, self.b4])
        return b

    def _get_t(self, t, dt):
        t0 = t 
        t1 = t + self.c2 * dt
        t2 = t + self.c3 * dt
        t3 = t + self.c4 * dt
        
        return (t0, t1, t2, t3)

    
    def _make_step(self, rhs_func, x, t, dt):
        t0, t1, t2, t3 = self._get_t(t, dt)

        k1 = rhs_func(t0, x)
        k2 = rhs_func(t1, x + k1 * self.w21 * dt)
        k3 = rhs_func(t2, x + (k1 * self.w31 + k2 * self.w32) * dt)
        k4 = rhs_func(t3, x + (k1 * self.w41 + k2 * self.w42 + k3 * self.w43) * dt)

        return (k1 * self.b1 + k2 * self.b2 + k3 * self.b3 + k4 * self.b4) * dt
    
    
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
        return 4