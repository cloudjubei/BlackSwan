import torch
from torch.optim import Optimizer


class DGWO(Optimizer):
    """Grey-Wolf-flavoured gradient optimiser — a usable drop-in `torch.optim.Optimizer`.

    `step()` applies a gradient-descent update (so it works inside SB3's training loop, which calls
    `step()` with NO closure) plus a Grey-Wolf exploration pull toward a tracked leader position. The
    exploration coefficient ``a`` decays linearly to zero over ``max_iters`` steps, after which the update
    is pure gradient descent — giving early metaheuristic exploration while still converging on a convex
    objective. A closure, when supplied, is evaluated and its loss returned (standard torch contract).
    """

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
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            alpha = group["alpha"]
            beta = group["beta"]
            max_iters = group["max_iters"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["leader"] = p.detach().clone()

                # Gradient-descent base step — the term that makes the policy actually learn.
                update = grad.mul(-lr)

                # Decaying Grey-Wolf exploration toward the tracked leader position.
                a = max(0.0, 2.0 - state["step"] * (2.0 / max_iters))
                if a > 0.0:
                    A = (2.0 * torch.rand_like(p) - 1.0) * a
                    C = 2.0 * torch.rand_like(p)
                    D = torch.abs(C * state["leader"] - p)
                    explore = (state["leader"] - A * D) - p
                    update = update + explore.mul(alpha * a * 0.5)

                p.add_(update)

                # The leader drifts toward the latest position (EMA), keeping a moving pack centre.
                state["leader"].mul_(1.0 - beta).add_(p, alpha=beta)
                state["step"] += 1

        return loss
