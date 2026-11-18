import math

import numpy as np
import pytest

from streaming_research.policy.accounting import (
    COST_MODE_FIXED_PER_SHARE,
    COST_MODE_PRICE_FRACTION,
    ExecutionAccountingInput,
    compute_effective_cost_per_share,
    compute_execution_accounting,
    last_valid_observation_index,
    resolve_dense_execution_source,
)


def _acct(shares, action, p_obs=100.0, p_exec=101.0, p_mark=103.0, fill_allowed=True, **kw):
    args = dict(
        cash_before=0.0,
        shares_before=shares,
        observation_mark=p_obs,
        execution_price_raw=p_exec,
        next_mark_price=p_mark,
        executable_action=action,
        max_notional=1000.0,
        max_delta_shares=100,
        deadband_u=0.0,
        deadband_shares=0,
        frac_deadband=0.0,
        cost_per_share=0.01,
        slippage_bps=0.0,
        long_only=True,
        fill_allowed=fill_allowed,
        notional_cap_mode="strict_position_cap",
    )
    args.update(kw)
    return compute_execution_accounting(ExecutionAccountingInput(**args))


def test_buy_from_flat_hand_checked():
    out = _acct(0, 1.0)
    assert out.max_shares_exec == 9
    assert out.executed_delta == 9
    assert math.isclose(out.commission_cost, 0.09)
    assert math.isclose(9 * (103.0 - 101.0), 18.0)
    assert math.isclose(out.real_pnl_step, 18.0)
    assert math.isclose(out.real_net_step, 17.91)


def test_noop_from_flat():
    out = _acct(0, 0.0)
    assert out.executed_delta == 0
    assert out.real_pnl_step == 0.0
    assert out.real_cost_step == 0.0


def test_increase_existing_long():
    out = _acct(4, 0.5)
    assert out.executed_delta == 4
    assert out.shares_after == 8
    assert math.isclose(out.commission_cost, 0.04)


def test_partial_sell():
    out = _acct(9, -0.5)
    assert out.executed_delta == -4
    assert out.shares_after == 5
    assert math.isclose(out.commission_cost, 0.04)


def test_full_sell_long_only():
    out = _acct(9, -1.0)
    assert out.executed_delta == -9
    assert out.shares_after == 0
    assert math.isclose(out.commission_cost, 0.09)


def test_existing_inventory_gap_is_not_lost():
    out = _acct(5, 0.0, p_obs=100.0, p_exec=101.0, p_mark=103.0)
    assert out.executed_delta == 0
    assert math.isclose(out.pre_execution_gap_pnl, 5.0)
    assert math.isclose(out.post_execution_pnl, 10.0)
    assert math.isclose(out.real_pnl_step, 15.0)


def test_cost_per_share_and_slippage():
    out = _acct(0, 1.0, slippage_bps=10.0)
    assert out.execution_price_after_slippage > out.execution_price_raw
    assert out.slippage_cost > 0.0
    assert out.commission_cost > 0.0


def test_max_delta_clipping():
    out = _acct(0, 1.0, max_delta_shares=3)
    assert out.executed_delta == 3


def test_deadband_and_frac_deadband():
    assert _acct(0, 0.01, deadband_u=0.02).executed_delta == 0
    assert _acct(0, 0.04, frac_deadband=0.05).executed_delta == 0


def test_no_real_trade_next_second_no_fill():
    prices = np.array([100.0, 101.0, 103.0])
    trades = np.array([1.0, 0.0, 1.0])
    src = resolve_dense_execution_source(prices, 0, "next_close_real_bar", trades=trades)
    assert not src.fill_allowed
    out = _acct(0, 1.0, fill_allowed=src.fill_allowed)
    assert out.executed_delta == 0
    assert out.fill_status == "no_fill"


def test_no_fill_keeps_actual_exposure_and_requested_sell_visible():
    out = _acct(5, -1.0, p_obs=100.0, p_exec=float("nan"), p_mark=100.0, fill_allowed=False)
    assert out.executed_delta == 0
    assert out.shares_after == 5
    assert out.cash_after == out.cash_after
    assert out.real_cost_step == 0.0
    assert out.max_shares_exec > 0
    assert out.max_shares_reference > 0
    assert out.inventory_fraction_before > 0.0
    assert out.inventory_fraction_after > 0.0
    assert out.actual_notional_fraction_after > 0.0
    assert out.requested_action_signed < 0.0
    assert out.requested_delta_shares < 0


def test_pending_mode_is_stateful_not_dense_source():
    prices = np.array([100.0, 101.0, 103.0])
    trades = np.array([1.0, 0.0, 1.0])
    with pytest.raises(RuntimeError, match="stateful"):
        resolve_dense_execution_source(prices, 0, "pending_first_real_bar", trades=trades)


def test_forward_filled_proxy_requires_ack():
    prices = np.array([100.0, 101.0, 103.0])
    with pytest.raises(RuntimeError):
        resolve_dense_execution_source(prices, 0, "next_close_ffill_proxy")
    src = resolve_dense_execution_source(prices, 0, "next_close_ffill_proxy", allow_synthetic_execution_proxy=True)
    assert src.synthetic_execution


def test_legacy_same_close_requires_ack():
    prices = np.array([100.0, 101.0, 103.0])
    with pytest.raises(RuntimeError):
        resolve_dense_execution_source(prices, 0, "legacy_same_close")
    src = resolve_dense_execution_source(prices, 0, "legacy_same_close", allow_legacy_same_close=True)
    assert src.exec_idx == 0


def test_final_valid_step_formula():
    assert last_valid_observation_index(n_rows=4, execution_lag=1, max_reward_horizon=1) == 1
    assert last_valid_observation_index(n_rows=3, execution_lag=1, max_reward_horizon=1) == 0


def test_cost_helper_fixed_legacy_regression():
    cps = compute_effective_cost_per_share(
        fill_price=1.50,
        cost_model=COST_MODE_FIXED_PER_SHARE,
        fixed_cost_per_share=0.01,
        cost_price_fraction=0.005,
    )
    assert math.isclose(cps, 0.01)
    assert math.isclose(100 * cps, 1.00)


def test_cost_helper_price_fraction_low_and_high_price():
    cps_low = compute_effective_cost_per_share(
        fill_price=1.50,
        cost_model=COST_MODE_PRICE_FRACTION,
        fixed_cost_per_share=0.01,
        cost_price_fraction=0.005,
    )
    cps_high = compute_effective_cost_per_share(
        fill_price=15.0,
        cost_model=COST_MODE_PRICE_FRACTION,
        fixed_cost_per_share=0.01,
        cost_price_fraction=0.005,
    )
    assert math.isclose(cps_low, 0.0075)
    assert math.isclose(1000 * cps_low, 7.50)
    assert math.isclose(cps_high, 0.075)
    assert math.isclose(1000 * cps_high, 75.00)


def test_price_fraction_cost_is_constant_notional_ratio():
    rate = 0.005
    for px, qty in [(0.8, 2000), (1.5, 1000), (15.0, 100), (37.25, 23)]:
        cps = compute_effective_cost_per_share(
            fill_price=px,
            cost_model=COST_MODE_PRICE_FRACTION,
            fixed_cost_per_share=0.0,
            cost_price_fraction=rate,
        )
        total_cost = qty * cps
        notional = qty * px
        assert math.isclose(total_cost / notional, rate, rel_tol=1e-12, abs_tol=1e-12)


def test_execution_accounting_price_fraction_no_fill_and_fill_cost():
    nofill = _acct(5, -1.0, p_exec=float("nan"), fill_allowed=False, cost_model=COST_MODE_PRICE_FRACTION, cost_price_fraction=0.005)
    assert nofill.executed_delta == 0
    assert nofill.commission_cost == 0.0
    filled = _acct(0, 1.0, p_exec=1.50, p_mark=1.50, cost_model=COST_MODE_PRICE_FRACTION, cost_price_fraction=0.005)
    assert filled.executed_delta > 0
    assert math.isclose(filled.effective_cost_per_share, filled.execution_price_after_slippage * 0.005)
    assert math.isclose(filled.commission_cost, abs(filled.executed_delta) * filled.effective_cost_per_share)


def test_price_fraction_partial_fills_sum_independently():
    rate = 0.005
    qty1, p1 = 200, 1.5
    qty2, p2 = 800, 15.0
    cps1 = compute_effective_cost_per_share(fill_price=p1, cost_model=COST_MODE_PRICE_FRACTION, fixed_cost_per_share=0.0, cost_price_fraction=rate)
    cps2 = compute_effective_cost_per_share(fill_price=p2, cost_model=COST_MODE_PRICE_FRACTION, fixed_cost_per_share=0.0, cost_price_fraction=rate)
    assert math.isclose(qty1 * cps1 + qty2 * cps2, qty1 * p1 * rate + qty2 * p2 * rate)


def test_cost_helper_bounds():
    assert math.isclose(
        compute_effective_cost_per_share(
            fill_price=1.0,
            cost_model=COST_MODE_PRICE_FRACTION,
            fixed_cost_per_share=0.0,
            cost_price_fraction=0.001,
            cost_min_per_share=0.01,
        ),
        0.01,
    )
    assert math.isclose(
        compute_effective_cost_per_share(
            fill_price=100.0,
            cost_model=COST_MODE_PRICE_FRACTION,
            fixed_cost_per_share=0.0,
            cost_price_fraction=0.01,
            cost_max_per_share=0.25,
        ),
        0.25,
    )
