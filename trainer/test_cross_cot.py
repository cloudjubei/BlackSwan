"""Direct gating tests for the CROSS-SECTIONAL COT probe — the structurally-different positioning thread.

Every single-asset positioning probe nulled because its tempting cells were BETA (long a rising commodity). This
removes that: each bar it ranks the basket by managed-money positioning extremity and goes LONG the least-crowded
/ SHORT the most-crowded (cross_contrarian), a DOLLAR-NEUTRAL long/short so the common commodity beta cancels. The
novel surface is the cross-sectional ranking; the release-lagged join + trailing index are reused from cot.py.

Guards pinned here: contrarian goes LONG the LEAST-crowded (lowest score) and SHORT the most-crowded; the book is
market-neutral (weights sum to ~0); and no trade fires unless at least 2k commodities have a defined score.
"""

import numpy as np
import pytest

from trainer import cross_cot


def test_cross_row_weights_longs_least_crowded_shorts_most():
    scores = [('A', 10.0), ('B', 90.0), ('C', 50.0), ('D', 20.0), ('E', 80.0), ('F', None)]
    w = cross_cot.cross_row_weights(scores, k=2, invert=False)  # contrarian
    assert w['A'] == pytest.approx(0.5) and w['D'] == pytest.approx(0.5)    # 2 lowest -> long
    assert w['B'] == pytest.approx(-0.5) and w['E'] == pytest.approx(-0.5)  # 2 highest -> short
    assert w['C'] == 0.0 and w['F'] == 0.0                                  # middle / undefined -> flat


def test_cross_row_weights_momentum_flips():
    scores = [('A', 10.0), ('B', 90.0), ('C', 50.0), ('D', 20.0), ('E', 80.0)]
    w = cross_cot.cross_row_weights(scores, k=2, invert=True)  # momentum: ride the crowd
    assert w['A'] == pytest.approx(-0.5) and w['D'] == pytest.approx(-0.5)  # least-crowded -> short
    assert w['B'] == pytest.approx(0.5) and w['E'] == pytest.approx(0.5)    # most-crowded -> long


def test_cross_row_weights_is_market_neutral():
    scores = [('A', 10.0), ('B', 90.0), ('C', 50.0), ('D', 20.0), ('E', 80.0)]
    w = cross_cot.cross_row_weights(scores, k=2, invert=False)
    assert sum(w.values()) == pytest.approx(0.0)


def test_cross_row_weights_no_trade_when_too_few_defined():
    scores = [('A', 10.0), ('B', 90.0), ('C', None), ('D', None)]  # only 2 defined, need 2k=4
    w = cross_cot.cross_row_weights(scores, k=2, invert=False)
    assert all(v == 0.0 for v in w.values())


# --- the run() contract (cross-sectional axis provided directly) -------------------------------------


def _synthetic_aligned(n=400):
    # 6 commodities; oscillating scores so the ranking rotates and it trades; mild independent returns.
    rng = np.random.default_rng(7)
    base = cross_cot._month_start_ms('2020-01')
    ts = [base + i * 86_400_000 for i in range(n)]
    data = {}
    for j, c in enumerate(cross_cot.BASKETS['complex6']):
        score = [50.0 + 45.0 * np.sin(i / 20.0 + j) for i in range(n)]
        ret = list(rng.normal(0.0002, 0.012, n))
        data[c] = {'ret': ret, 'score': score}
    return ts, data


def test_run_emits_the_full_metric_vocabulary(monkeypatch):
    ts, data = _synthetic_aligned(1100)  # span into the 2022 test window
    monkeypatch.setattr(cross_cot, '_aligned', lambda basket, pairs, rl, iw, mh: (ts, data))
    cfg = {
        'basket': 'complex6', 'signal': 'cross_contrarian', 'cross_k': 2, 'cot_index_window': 756,
        'min_history': 126, 'release_lag_days': 4, 'transaction_fee': 0.0005, 'walk_forward_window': '2022', 'seed': 0,
    }
    summary = cross_cot.run(cfg)
    m = summary['metrics']
    for key in ('total_return_pct', 'oos_sharpe', 'signal_expectancy', 'n_trades', 'realized_cost_bps', 'gross_exposure'):
        assert key in m, f'missing metric {key}'
        assert np.isfinite(m[key])
    assert summary['objective'] == m['oos_sharpe']
    assert m['n_trades'] > 0
    assert summary['dataset']['asset'] == 'complex6'


def test_run_refuses_unknown_signal(monkeypatch):
    ts, data = _synthetic_aligned(60)
    monkeypatch.setattr(cross_cot, '_aligned', lambda basket, pairs, rl, iw, mh: (ts, data))
    with pytest.raises(SystemExit):
        cross_cot.run({'signal': 'cross_wat', 'walk_forward_window': '2022'})
