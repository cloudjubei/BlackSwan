"""DGWO must be a usable drop-in torch optimizer. SB3 constructs it as `optimizer_class(params, lr=...)`
and then calls `optimizer.step()` with NO closure, so `step()` must update from gradients without a closure
and actually minimise a loss. (Before the fix it raised RuntimeError whenever closure was None, and it
ignored `p.grad` entirely, so SB3 training crashed on the first optimisation step.)"""
import torch

from src.model.custom.dgwo import DGWO


def test_step_without_closure_does_not_raise():
    w = torch.nn.Parameter(torch.tensor([5.0]))
    opt = DGWO([w], lr=0.05)
    ((w - 3.0) ** 2).backward()
    opt.step()  # SB3 calls step() with no closure — must not raise


def test_minimises_convex_objective_without_closure():
    torch.manual_seed(0)
    w = torch.nn.Parameter(torch.tensor([5.0]))
    opt = DGWO([w], lr=0.05, max_iters=50)
    for _ in range(500):
        opt.zero_grad()
        ((w - 3.0) ** 2).backward()
        opt.step()
    assert abs(w.item() - 3.0) < 0.1


def test_step_count_is_per_instance_not_shared_on_the_class():
    a = torch.nn.Parameter(torch.tensor([1.0]))
    b = torch.nn.Parameter(torch.tensor([1.0]))
    o1 = DGWO([a], lr=0.01)
    o2 = DGWO([b], lr=0.01)
    a.sum().backward()
    o1.step()
    assert o1.state[a].get("step") == 1
    assert o2.state[b].get("step", 0) == 0


def test_closure_is_still_supported():
    w = torch.nn.Parameter(torch.tensor([5.0]))
    opt = DGWO([w], lr=0.05)

    def closure():
        opt.zero_grad()
        loss = (w - 3.0) ** 2
        loss.backward()
        return loss

    loss = opt.step(closure)
    assert loss is not None
