import torch
import torch.nn as nn
import abc

class RKParametricSolver(object, metaclass = abc.ABCMeta):
    def __init__(self, n_steps = None, step_size = None, grid_constructor=None):
            
        ## Compute number of grid related args != None
        if sum(1 for _ in filter(None.__ne__, [n_steps, step_size, grid_constructor])) >= 2:
            raise ValueError("n_steps, step_size and grid_constructor are pairwise exclusive arguments.")
        
        ## Initialize time grid
        if n_steps is not None:
            self.grid_constructor = self._grid_constructor_from_n_steps(n_steps)
        elif step_size is not None:
            self.grid_constructor = self._grid_constructor_from_step_size(step_size)
        elif grid_constructor is not None:
            self.grid_constructor = grid_constructor
        else:
            self.grid_constructor = lambda t: t
                
                
    def _grid_constructor_from_step_size(self, step_size):

        def _grid_constructor(t):
            start_time = t[0]
            end_time = t[-1]

            n_steps = torch.ceil((end_time - start_time) / step_size + 1).item()
            t_infer = torch.arange(0, n_steps).to(t) * step_size + start_time
            if t_infer[-1] > t[-1]:
                t_infer[-1] = t[-1]
            return t_infer

        return _grid_constructor
    
    
    def _grid_constructor_from_n_steps(self, n_steps):
        
        def _grid_constructor(t):
            start_time = t[0]
            end_time = t[-1]
            
            t_infer = torch.linspace(start_time, end_time, n_steps + 1).to(t)
            return t_infer
        
        return _grid_constructor
       
        
    @property
    @abc.abstractmethod
    def order(self):
        pass
    
    @abc.abstractmethod
    def freeze_params(self):
        pass
        
    @abc.abstractmethod
    def unfreeze_params(self):
        pass
    
    @abc.abstractmethod
    def _make_params_valid(self):
        pass
    
    
    def build_ButcherTableau(self, return_tableau = False):
        self._make_params_valid()
        self._compute_c()
        self._compute_b()         
        self._compute_w()
        
        if return_tableau:
            return self._collect_ButcherTableau()
        
        
    def _collect_ButcherTableau(self):
        c = self._get_c()
        w = self._get_w()
        b = self._get_b()
        return c, w, b
    
    
    @abc.abstractmethod
    def _make_step(self, rhs_func, x, t, dt):
        pass
    
    def integrate(self, rhs_func, x, t):
#         _assert_increasing(t)
        t = t.type_as(x[0])
        
        time_grid = self.grid_constructor(t)
        
        assert time_grid[0] == t[0] and time_grid[-1] == t[-1]
        time_grid = time_grid.to(x[0])
        
#         print('\ntime_grid (for evaluation):', time_grid)

        solution = [x]
        
        j = 1
        y0 = x # x has shape (batch_size, *x.shape)
        for t0, t1 in zip(time_grid[:-1], time_grid[1:]):
            dy = self._make_step(rhs_func, y0, t0, t1 - t0) # dy has shape (batch_size, *x.shape)
            y1 = y0 + dy

            # interpolate values at intermediate points
            while j < len(t) and t1 >= t[j]:
                solution.append(self._linear_interp(t0, t1, y0, y1, t[j]))
                j += 1
            y0 = y1
        return torch.stack(solution) # has shape (len(t), batch_size, *x.shape)
    
    
    def _linear_interp(self, t0, t1, y0, y1, t):
        if t == t0:
            return y0
        if t == t1:
            return y1
        t0, t1, t = t0.to(y0[0]), t1.to(y0[0]), t.to(y0[0])
        slope = (y1 - y0) / (t1 - t0) # slope has shape (batch_size, *x.shape)
        return y0 + slope * (t - t0) 

    
    def print_is_requires_grad(self):
        print('\nIs requires grad? (RK solver)')
        
        for pname, p in self.__dict__.items():
            if hasattr(p, 'requires_grad'):
                print(pname, p.requires_grad)
