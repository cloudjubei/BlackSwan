"""Direct unit tests for the deterministic TechnicalStrategyModel baseline.

get_action reads the current decision bar's indicator value from the data provider's look-ahead-safe
named-feature accessor (env.data_provider.get_feature(step, name)), then applies a buy rule (BUY=1)
and, failing that, a sell rule (SELL=2), defaulting to HOLD=0. Each rule has a 2x2 matrix: price-check
vs amount-check, and direction (down/up). We stub the provider's feature values and assert the exact
action for every branch.
"""

import types

from src.model.technical_strategy_model import TechnicalStrategyModel


# The synthetic indicator columns the buy/sell rules key on.
COLUMNS = ["buy_ind", "sell_ind", "filler"]


def _env(*, last_item, price):
    """A minimal env whose provider returns ``last_item`` values by indicator NAME (aligned to COLUMNS),
    mirroring the real contract: the model reads env.data_provider.get_feature(current_step, name)."""
    feats = dict(zip(COLUMNS, last_item))
    provider = types.SimpleNamespace(get_feature=lambda step, name: feats[name])
    env = types.SimpleNamespace(
        current_step=0,
        get_price=lambda step: price,
        data_provider=provider,
    )
    return env, None


def _model(**tech):
    m = TechnicalStrategyModel.__new__(TechnicalStrategyModel)
    m.technical_config = types.SimpleNamespace(
        buy_indicator="buy_ind",
        buy_amount_threshold=100.0,
        sell_indicator="sell_ind",
        sell_amount_threshold=200.0,
        buy_amount_is_multiplier=False,
        buy_is_price_check=False,
        buy_is_down_check=True,
        sell_amount_is_multiplier=False,
        sell_is_price_check=False,
        sell_is_up_check=True,
    )
    for k, v in tech.items():
        setattr(m.technical_config, k, v)
    return m


# --- get_id ---------------------------------------------------------------

def test_get_id_distinct_indicators():
    cfg = types.SimpleNamespace(
        model_technical=types.SimpleNamespace(
            buy_indicator="rsi10",
            buy_amount_threshold=30.5,
            sell_indicator="rsi15",
            sell_amount_threshold=70.0,
        )
    )
    # '.' becomes '~'; both indicators included because they differ.
    assert TechnicalStrategyModel.get_id(None, cfg) == "rsi10_30~5_rsi15_70~0"


def test_get_id_same_indicator_blanks_sell_name():
    cfg = types.SimpleNamespace(
        model_technical=types.SimpleNamespace(
            buy_indicator="rsi10",
            buy_amount_threshold=30.0,
            sell_indicator="rsi10",
            sell_amount_threshold=70.0,
        )
    )
    # sell_indicator collapses to "" when it equals buy_indicator.
    assert TechnicalStrategyModel.get_id(None, cfg) == "rsi10_30~0__70~0"


# --- buy: amount-check branches ------------------------------------------

def test_buy_amount_down_check_fires_when_indicator_at_or_below_threshold():
    m = _model(buy_is_down_check=True, buy_amount_threshold=100.0)
    env, obs = _env(last_item=[100.0, 0.0, 0.0], price=500.0)
    assert m.get_action(env, obs) == 1


def test_buy_amount_down_check_no_buy_when_above_threshold():
    # indicator 150 > 100 -> buy rule fails; sell indicator 0 (down) below 200 -> no sell either.
    m = _model(buy_is_down_check=True, buy_amount_threshold=100.0, sell_is_up_check=True)
    env, obs = _env(last_item=[150.0, 0.0, 0.0], price=500.0)
    assert m.get_action(env, obs) == 0


def test_buy_amount_up_check_fires_when_indicator_at_or_above_threshold():
    m = _model(buy_is_down_check=False, buy_amount_threshold=100.0)
    env, obs = _env(last_item=[100.0, 0.0, 0.0], price=500.0)
    assert m.get_action(env, obs) == 1


def test_buy_amount_up_check_no_buy_when_below_threshold():
    m = _model(buy_is_down_check=False, buy_amount_threshold=100.0)
    env, obs = _env(last_item=[99.0, 0.0, 0.0], price=500.0)
    assert m.get_action(env, obs) == 0


# --- buy: price-check branches -------------------------------------------

def test_buy_price_down_check_buys_when_price_below_indicator():
    # buy_is_price_check + down: BUY when price <= buy_indicator (a "buy the dip" trigger).
    m = _model(buy_is_price_check=True, buy_is_down_check=True)
    env, obs = _env(last_item=[120.0, 0.0, 0.0], price=110.0)
    assert m.get_action(env, obs) == 1


def test_buy_price_down_check_no_buy_when_price_above_indicator():
    m = _model(buy_is_price_check=True, buy_is_down_check=True, sell_is_up_check=True)
    env, obs = _env(last_item=[120.0, 0.0, 0.0], price=130.0)
    assert m.get_action(env, obs) == 0


def test_buy_price_up_check_buys_when_price_at_or_above_indicator():
    m = _model(buy_is_price_check=True, buy_is_down_check=False)
    env, obs = _env(last_item=[120.0, 0.0, 0.0], price=120.0)
    assert m.get_action(env, obs) == 1


def test_buy_price_up_check_no_buy_when_price_below_indicator():
    m = _model(buy_is_price_check=True, buy_is_down_check=False, sell_is_up_check=True)
    env, obs = _env(last_item=[120.0, 0.0, 0.0], price=119.0)
    assert m.get_action(env, obs) == 0


# --- buy: multiplier ------------------------------------------------------

def test_buy_amount_is_multiplier_scales_indicator():
    # indicator 2 * threshold 100 = 200; up-check 200 >= 100 -> BUY.
    m = _model(buy_amount_is_multiplier=True, buy_is_down_check=False, buy_amount_threshold=100.0)
    env, obs = _env(last_item=[2.0, 0.0, 0.0], price=500.0)
    assert m.get_action(env, obs) == 1


def test_buy_price_check_with_multiplier_scales_against_price():
    # multiplier path: buy_indicator becomes 1.2 * 100 = 120; price-up check price 130 >= 120 -> BUY.
    m = _model(
        buy_amount_is_multiplier=True,
        buy_is_price_check=True,
        buy_is_down_check=False,
        buy_amount_threshold=100.0,
    )
    env, obs = _env(last_item=[1.2, 0.0, 0.0], price=130.0)
    assert m.get_action(env, obs) == 1


# --- sell: amount-check branches -----------------------------------------

def test_sell_amount_up_check_fires_when_indicator_at_or_above_threshold():
    # Force buy to miss; sell amount up-check: indicator 200 >= 200 -> SELL.
    m = _model(buy_is_down_check=False, buy_amount_threshold=999.0, sell_is_up_check=True, sell_amount_threshold=200.0)
    env, obs = _env(last_item=[0.0, 200.0, 0.0], price=500.0)
    assert m.get_action(env, obs) == 2


def test_sell_amount_down_check_fires_when_indicator_at_or_below_threshold():
    m = _model(buy_is_down_check=False, buy_amount_threshold=999.0, sell_is_up_check=False, sell_amount_threshold=200.0)
    env, obs = _env(last_item=[0.0, 200.0, 0.0], price=500.0)
    assert m.get_action(env, obs) == 2


def test_sell_amount_down_check_no_sell_when_above_threshold():
    m = _model(buy_is_down_check=False, buy_amount_threshold=999.0, sell_is_up_check=False, sell_amount_threshold=200.0)
    env, obs = _env(last_item=[0.0, 250.0, 0.0], price=500.0)
    assert m.get_action(env, obs) == 0


# --- sell: price-check branches ------------------------------------------

def test_sell_price_up_check_sells_when_price_at_or_above_indicator():
    # sell price up-check: SELL when price >= sell_indicator (take profit above target).
    m = _model(buy_is_down_check=False, buy_amount_threshold=999.0, sell_is_price_check=True, sell_is_up_check=True)
    env, obs = _env(last_item=[0.0, 120.0, 0.0], price=130.0)
    assert m.get_action(env, obs) == 2


def test_sell_price_up_check_no_sell_when_price_below_indicator():
    m = _model(buy_is_down_check=False, buy_amount_threshold=999.0, sell_is_price_check=True, sell_is_up_check=True)
    env, obs = _env(last_item=[0.0, 120.0, 0.0], price=110.0)
    assert m.get_action(env, obs) == 0


def test_sell_price_down_check_sells_when_price_at_or_below_indicator():
    m = _model(buy_is_down_check=False, buy_amount_threshold=999.0, sell_is_price_check=True, sell_is_up_check=False)
    env, obs = _env(last_item=[0.0, 120.0, 0.0], price=110.0)
    assert m.get_action(env, obs) == 2


def test_sell_price_down_check_no_sell_when_price_above_indicator():
    m = _model(buy_is_down_check=False, buy_amount_threshold=999.0, sell_is_price_check=True, sell_is_up_check=False)
    env, obs = _env(last_item=[0.0, 120.0, 0.0], price=130.0)
    assert m.get_action(env, obs) == 0


def test_sell_amount_is_multiplier_scales_indicator():
    # sell indicator 3 * threshold 200 = 600; up-check 600 >= 200 -> SELL.
    m = _model(
        buy_is_down_check=False,
        buy_amount_threshold=999.0,
        sell_amount_is_multiplier=True,
        sell_is_up_check=True,
        sell_amount_threshold=200.0,
    )
    env, obs = _env(last_item=[0.0, 3.0, 0.0], price=500.0)
    assert m.get_action(env, obs) == 2


# --- precedence + fallthrough --------------------------------------------

def test_buy_takes_precedence_over_sell():
    # Both rules would trigger; BUY is evaluated first so it wins.
    m = _model(
        buy_is_down_check=True,
        buy_amount_threshold=100.0,
        sell_is_up_check=True,
        sell_amount_threshold=200.0,
    )
    env, obs = _env(last_item=[50.0, 300.0, 0.0], price=500.0)
    assert m.get_action(env, obs) == 1


def test_hold_when_neither_rule_triggers():
    m = _model(
        buy_is_down_check=False,
        buy_amount_threshold=1000.0,
        sell_is_up_check=True,
        sell_amount_threshold=1000.0,
    )
    env, obs = _env(last_item=[10.0, 10.0, 0.0], price=500.0)
    assert m.get_action(env, obs) == 0


def test_get_action_reads_feature_at_current_step_via_provider():
    # The model delegates decision-bar selection to the provider: it must query get_feature at the env's
    # CURRENT step (not step 0), so per-step values drive the action. Here step 7's buy indicator is 150
    # (>100 down-check) -> no buy; step 0 would be 10 (<=100) -> a stale buy the model must NOT make.
    m = _model(buy_is_down_check=True, buy_amount_threshold=100.0, sell_is_up_check=True,
               sell_amount_threshold=999.0)
    by_step = {0: {"buy_ind": 10.0, "sell_ind": 0.0}, 7: {"buy_ind": 150.0, "sell_ind": 0.0}}
    provider = types.SimpleNamespace(get_feature=lambda step, name: by_step[step][name])
    env = types.SimpleNamespace(
        current_step=7,
        get_price=lambda step: 500.0,
        data_provider=provider,
    )
    assert m.get_action(env, None) == 0
