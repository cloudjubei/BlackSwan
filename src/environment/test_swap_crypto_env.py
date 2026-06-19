"""Direct tests for ``SwapCryptoEnv``'s overridden ``create_action_space`` / ``take_action``.

``__init__`` is bypassed with ``__new__`` (it would need a real data provider + a full reset); we set
only the few attributes ``take_action`` reads. The swap env has a single non-hold action that flips
the whole portfolio between cash and the asset, applying a transaction fee each way.
"""

from gymnasium import spaces

from src.environment.swap_crypto_env import SwapCryptoEnv


def _env(*, position, balance, price=100.0, fee=0.001):
    e = SwapCryptoEnv.__new__(SwapCryptoEnv)
    e.positions = [position]
    e.balances = [balance]
    e.current_price = price
    e.transaction_fee_multiplier = fee
    e.fees = []
    return e


def test_action_space_is_discrete_two():
    e = SwapCryptoEnv.__new__(SwapCryptoEnv)
    space = e.create_action_space()
    assert isinstance(space, spaces.Discrete)
    assert space.n == 2


def test_hold_action_zero_does_nothing():
    e = _env(position=5.0, balance=0.0)
    made = e.take_action(0)
    assert made is False
    # State untouched by a hold.
    assert e.positions == [5.0]
    assert e.balances == [0.0]
    assert e.fees == []


def test_swap_out_of_long_into_cash():
    # Hold 2 units @ price 100 -> gross 200; fee = 200 * 0.001 = 0.2; balance = 199.8.
    e = _env(position=2.0, balance=0.0, price=100.0, fee=0.001)
    made = e.take_action(1)
    assert made is True
    assert e.positions == [0]
    assert abs(e.balances[-1] - (200.0 - 0.2)) < 1e-9
    assert abs(e.fees[-1] - 0.2) < 1e-9


def test_swap_out_of_cash_into_position():
    # Flat with 199.8 cash @ price 100 -> amount = 1.998; fee = 0.001998 units; position = amount-fee.
    e = _env(position=0.0, balance=199.8, price=100.0, fee=0.001)
    made = e.take_action(1)
    assert made is True
    amount = 199.8 / 100.0
    fee_units = amount * 0.001
    assert abs(e.positions[-1] - (amount - fee_units)) < 1e-12
    assert e.balances[-1] == 0
    # Buy-side fee is recorded in quote currency (fee_units * price).
    assert abs(e.fees[-1] - fee_units * 100.0) < 1e-12


def test_swap_position_takes_precedence_over_balance():
    # When BOTH a position and cash are present, take_action must close the position (long branch),
    # not buy more: positions go to 0 and the position branch's fee is appended.
    e = _env(position=2.0, balance=500.0, price=100.0, fee=0.001)
    made = e.take_action(1)
    assert made is True
    assert e.positions == [0]
    # Long-close branch overwrites the balance with the realized amount (it does NOT add to the 500).
    assert abs(e.balances[-1] - (200.0 - 0.2)) < 1e-9


def test_any_nonzero_action_swaps():
    # The action space is Discrete(2); "swap" is "any action != 0". A model only ever emits 0/1, but
    # the branch is gated on != 0, so a larger value must also swap (characterizes the contract).
    e = _env(position=3.0, balance=0.0, price=50.0)
    made = e.take_action(7)
    assert made is True
    assert e.positions == [0]


def test_round_trip_swap_loses_two_fees():
    # Swap cash->asset then asset->cash; ending cash must be below the start by ~2 fees (round-trip cost).
    e = _env(position=0.0, balance=1000.0, price=100.0, fee=0.001)
    e.take_action(1)  # buy
    # After buy, balance is 0 and a position exists; swap back to cash.
    e.take_action(1)  # sell
    assert e.positions == [0]
    assert e.balances[-1] < 1000.0
    assert len(e.fees) == 2
