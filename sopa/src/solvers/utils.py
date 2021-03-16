import torch
import numpy as np
import copy

from torch.distributions.normal import Normal
from torch.distributions.cauchy import Cauchy

from .rk_parametric_order4stage4 import RKOrder4Stage4
from .rk_parametric_order3stage3 import RKOrder3Stage3
from .rk_parametric_order2stage2 import RKOrder2Stage2
from .euler import Euler

def create_solver(method, parameterization, n_steps, step_size, u0, v0, dtype, device):
    '''
        method: str
        parameterization: str
        n_steps: int
        step_size: Decimal
        u0: Decimal
        v0: Decimal
        dtype: torch dtype
    '''
    if n_steps == -1:
        n_steps = None
        
    if step_size == -1:
        step_size = None
        
    if dtype == torch.float64:
        u0, v0 = map(lambda el : np.float64(el), [u0, v0])
    elif dtype == torch.float32:
        u0, v0 = map(lambda el : np.float32(el), [u0, v0])
        
    if method == 'euler':
        return Euler(n_steps = n_steps,
                     step_size = step_size,
                     parameterization=parameterization,
                     u0 = u0, v0 = v0,
                     dtype =dtype, device = device)       
    elif method == 'rk2':
        return RKOrder2Stage2(n_steps = n_steps,
                              step_size = step_size,
                              parameterization=parameterization,
                              u0 = u0, v0 = v0,
                              dtype =dtype, device = device)
    elif method == 'rk3':
        return RKOrder3Stage3(n_steps = n_steps,
                              step_size = step_size,
                              parameterization=parameterization,
                              u0 = u0, v0 = v0,
                              dtype =dtype, device = device)
    elif method == 'rk4':
        return RKOrder4Stage4(n_steps = n_steps,
                              step_size = step_size,
                              parameterization=parameterization,
                              u0 = u0, v0 = v0,
                              dtype =dtype, device = device)
    
    
def sample_noise(mu, sigma, noise_type='cauchy', size=1, device='cpu', minimize_rk2_error=False):
        if not minimize_rk2_error:
            if noise_type == 'cauchy':
                d = Cauchy(torch.tensor([mu]), torch.tensor([sigma]))
            elif noise_type == 'normal':
                d = Normal(torch.tensor([mu]), torch.tensor([sigma]))
        else:
            if noise_type == 'cauchy':
                d = Cauchy(torch.tensor([2/3.]), torch.tensor([2/3. * sigma]))
            elif noise_type == 'normal':
                d = Normal(torch.tensor([2/3.]), torch.tensor([2/3. * sigma]))
            
        return torch.tensor([d.sample() for _ in range(size)], device=device) 
    
    
def noise_params(mean_u, mean_v=None, std=0.01, bernoulli_p=1.0, noise_type='cauchy', minimize_rk2_error=False):
    ''' Noise solver paramers with Cauchy/Normal noise with probability p
    '''
    d = torch.distributions.Bernoulli(torch.tensor([bernoulli_p], dtype=torch.float32))
    v = None
    device = mean_u.device
    eps = torch.finfo(mean_u.dtype).eps

    if d.sample():
        std = torch.abs(torch.tensor(std, device=device))

        u = sample_noise(mean_u, std, noise_type=noise_type, size=1, device=device, minimize_rk2_error=minimize_rk2_error)
        if u <= mean_u - 2*std or u >= mean_u + 2*std:
            u = mean_u
#             u = min(max(u, mean_u - 2*std,0), mean_u + 2*std, 1.)
        
        if mean_v is not None:
            v = sample_noise(mean_v, std, noise_type=noise_type, size=1, device=device, minimize_rk2_error=minimize_rk2_error)
    else:
        u = mean_u
        if mean_v is not None:
            v = mean_v

    return u, v

def sample_solver_by_noising_params(solver, std=0.01, bernoulli_p=1., noise_type='cauchy', minimize_rk2_error=False):
    new_solver = copy.deepcopy(solver)
    new_solver.u, new_solver.v = noise_params(mean_u=new_solver.u0,
                                              mean_v=new_solver.v0,
                                              std=std,
                                              bernoulli_p=bernoulli_p,
                                              noise_type=noise_type,
                                              minimize_rk2_error=minimize_rk2_error)
    new_solver.build_ButcherTableau()
    print(new_solver.u, new_solver.v)
    return new_solver

def create_solver_ensemble_by_noising_params(solver, ensemble_size=1, kwargs_noise={}):
    solver_ensemble = [solver]
    for _ in range(1, ensemble_size):
        new_solver = sample_solver_by_noising_params(solver, **kwargs_noise)
        solver_ensemble.append(new_solver)
    return solver_ensemble






