import torch
from torch.optim import Optimizer
import numpy as np

class DGWO(Optimizer):
    def __init__(self, params, lr=0.01, alpha=0.1, beta=0.2, max_iters=100):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if alpha <= 0.0:
            raise ValueError(f"Invalid alpha: {alpha}")
        if beta <= 0.0:
            raise ValueError(f"Invalid beta: {beta}")
        if max_iters <= 0:
            raise ValueError(f"Invalid max_iters: {max_iters}")
        
        defaults = dict(lr=lr, alpha=alpha, beta=beta, max_iters=max_iters)
        super(DGWO, self).__init__(params, defaults)
    
    def step(self, closure=None):
        # Closure is required to reevaluate the model
        if closure is None:
            raise RuntimeError("DGWO requires a closure to reevaluate the model.")
        
        # Initialize state for the first step
        state = self.state
        for group in self.param_groups:
            if len(state) == 0:
                state['best_solution'] = {}
                state['best_fitness'] = float('inf')
                for p in group['params']:
                    state['best_solution'][p] = p.clone()
                    state[p] = {'alpha_pos': p.clone(), 'beta_pos': p.clone(), 'delta_pos': p.clone()}
        
        # Update alpha, beta, and delta positions
        alpha = group['alpha']
        beta = group['beta']
        max_iters = group['max_iters']
        lr = group['lr']
        
        a = 2 - self._step_count * (2 / max_iters)  # Linearly decreasing factor

        # Evaluate fitness for current parameters
        fitness = closure()
        
        # Update best solution if current fitness is better
        if fitness < state['best_fitness']:
            state['best_fitness'] = fitness
            for p in group['params']:
                state['best_solution'][p].copy_(p.data)
        
        # Update the positions of the wolves
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad.data

            # Grey wolf positions updates
            X = p.data
            A = 2 * a * np.random.random() - a
            C = 2 * np.random.random()

            D_alpha = abs(C * state['best_solution'][p] - X)
            X1 = state['best_solution'][p] - A * D_alpha

            D_beta = abs(C * state['best_solution'][p] - X)
            X2 = state['best_solution'][p] - A * D_beta

            D_delta = abs(C * state['best_solution'][p] - X)
            X3 = state['best_solution'][p] - A * D_delta

            # Update the position of the wolf
            p.data = (X1 + X2 + X3) / 3

        self._step_count += 1

DGWO._step_count = 0
