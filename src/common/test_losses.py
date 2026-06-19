import pytest
import torch

from src.common.losses import R2Loss, SquaredMeanLoss


# ---------------------------------------------------------------------------
# SquaredMeanLoss : mean of ((target - input) / target) ** 2  (squared % error)
# ---------------------------------------------------------------------------


def test_squared_mean_loss_zero_when_input_equals_target():
    # input == target -> every per-element term is 0 -> mean 0.
    loss = SquaredMeanLoss()
    out = loss(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.0, 2.0, 3.0]))
    assert isinstance(out, torch.Tensor)
    assert out.item() == pytest.approx(0.0)


def test_squared_mean_loss_hand_computed_reference():
    # input=[1,2,3], target=[2,2,6]:
    #   ((2-1)/2)^2 = 0.25, ((2-2)/2)^2 = 0.0, ((6-3)/6)^2 = 0.25
    #   mean = (0.25 + 0.0 + 0.25) / 3 = 0.5 / 3
    loss = SquaredMeanLoss()
    out = loss(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([2.0, 2.0, 6.0]))
    assert out.item() == pytest.approx(0.5 / 3.0)


def test_squared_mean_loss_single_element():
    # ((4 - 1) / 4) ** 2 = 0.5625; mean of one element is itself.
    out = SquaredMeanLoss()(torch.tensor([1.0]), torch.tensor([4.0]))
    assert out.item() == pytest.approx(0.5625)


def test_squared_mean_loss_is_scale_invariant_in_target():
    # The error is normalised by target, so scaling input & target together leaves it unchanged.
    loss = SquaredMeanLoss()
    a = loss(torch.tensor([1.0, 2.0]), torch.tensor([2.0, 4.0]))
    b = loss(torch.tensor([10.0, 20.0]), torch.tensor([20.0, 40.0]))
    assert a.item() == pytest.approx(b.item())


def test_squared_mean_loss_negative_targets():
    # ((−2 − (−1)) / −2)^2 = (−1/−2)^2 = 0.25 ; ((−4 − (−2)) / −4)^2 = (−2/−4)^2 = 0.25
    out = SquaredMeanLoss()(torch.tensor([-1.0, -2.0]), torch.tensor([-2.0, -4.0]))
    assert out.item() == pytest.approx(0.25)


def test_squared_mean_loss_returns_tensor_with_grad():
    # Differentiable: a tensor that requires grad flows through and yields a scalar tensor.
    inp = torch.tensor([1.0, 2.0], requires_grad=True)
    tgt = torch.tensor([2.0, 4.0])
    out = SquaredMeanLoss()(inp, tgt)
    assert out.requires_grad
    out.backward()
    assert inp.grad is not None


def test_squared_mean_loss_target_zero_is_finite():
    # The denominator is guarded (target + 1e-8) so a zero target no longer blows up to inf/nan.
    out = SquaredMeanLoss()(torch.tensor([1.0, 2.0]), torch.tensor([0.0, 4.0]))
    assert torch.isfinite(out)


# ---------------------------------------------------------------------------
# R2Loss : R2 = 1 - SS_res / SS_tot, computed in pure torch.
# `target` is ground truth, `input` is the prediction, matching the sklearn
# convention r2_score(y_true=target, y_pred=input).
# ---------------------------------------------------------------------------


def test_r2_loss_perfect_fit_is_one():
    # When the two series match exactly, R2 is 1.0.
    out = R2Loss()(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.0, 2.0, 3.0]))
    assert float(out) == pytest.approx(1.0)


def test_r2_loss_matches_sklearn_reference():
    # R2 matches sklearn with (y_true=target, y_pred=input).
    from sklearn.metrics import r2_score

    inp = torch.tensor([2.5, 0.0, 2.0, 8.0])
    tgt = torch.tensor([3.0, -0.5, 2.0, 7.0])
    out = R2Loss()(inp, tgt)
    assert float(out) == pytest.approx(r2_score(tgt, inp))


def test_r2_loss_argument_order_is_input_then_target():
    # R2 is asymmetric in its args; this pins that R2Loss treats target as ground truth
    # and input as prediction (the sklearn-canonical (y_true, y_pred) = (target, input)).
    from sklearn.metrics import r2_score

    inp = torch.tensor([2.5, 0.0, 2.0, 8.0])
    tgt = torch.tensor([3.0, -0.5, 2.0, 7.0])
    out = R2Loss()(inp, tgt)
    # equals r2_score(y_true=target, y_pred=input), and differs from the swapped order.
    assert float(out) == pytest.approx(r2_score(tgt, inp))
    assert float(out) != pytest.approx(r2_score(inp, tgt))


def test_r2_loss_should_return_tensor():
    # Computed in pure torch, the result is a torch.Tensor that can backprop.
    out = R2Loss()(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.1, 1.9, 3.2]))
    assert isinstance(out, torch.Tensor)


def test_r2_loss_returns_tensor_with_grad():
    # A prediction that requires grad flows through and yields a differentiable scalar tensor.
    inp = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    tgt = torch.tensor([1.1, 1.9, 3.2])
    out = R2Loss()(inp, tgt)
    assert isinstance(out, torch.Tensor)
    assert out.requires_grad
    out.backward()
    assert inp.grad is not None


def test_r2_loss_reduction_argument_is_accepted_but_unused():
    # _Loss base accepts reduction; forward ignores it (R2 has its own normalisation).
    out_mean = R2Loss(reduction="mean")(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.0, 2.0, 3.0]))
    out_sum = R2Loss(reduction="sum")(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.0, 2.0, 3.0]))
    assert float(out_mean) == pytest.approx(float(out_sum))
