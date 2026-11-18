#!/usr/bin/env python3
"""
Shared deterministic execution/accounting helpers for causally-correct SAC v4.

The helper is intentionally Torch-free so training, backtest, and tests can call
the same code path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

import numpy as np


EXECUTION_MODES = (
    "first_post_bar_trade_sidecar",
    "next_close_real_bar",
    "next_close_ffill_proxy",
    "pending_first_real_bar",
    "legacy_same_close",
)

COST_MODE_FIXED_PER_SHARE = "fixed_per_share"
COST_MODE_PRICE_FRACTION = "price_fraction"
COST_MODE_INHERIT = "inherit"
COST_MODE_VALUES = (COST_MODE_FIXED_PER_SHARE, COST_MODE_PRICE_FRACTION)
LIQUIDATION_COST_MODE_VALUES = (COST_MODE_INHERIT, COST_MODE_FIXED_PER_SHARE, COST_MODE_PRICE_FRACTION)


@dataclass(frozen=True)
class ExecutionSource:
    mode: str
    obs_idx: int
    next_obs_idx: int
    exec_idx: int
    mark_idx: int
    p_obs: float
    p_exec_raw: float
    p_mark: float
    fill_allowed: bool
    fill_status: str
    synthetic_execution: bool
    execution_price_source: str
    timestamp_bar_start: str = ""
    timestamp_bar_end: str = ""
    timestamp_observation_available: str = ""
    timestamp_execution: str = ""
    timestamp_mark: str = ""


@dataclass(frozen=True)
class ExecutionAccountingInput:
    cash_before: float
    shares_before: int
    observation_mark: float
    execution_price_raw: float
    next_mark_price: float
    executable_action: float
    max_notional: float
    max_delta_shares: int
    deadband_u: float
    deadband_shares: int
    frac_deadband: float
    cost_per_share: float
    cost_model: str = COST_MODE_FIXED_PER_SHARE
    cost_price_fraction: float = 0.005
    cost_min_per_share: float = 0.0
    cost_max_per_share: float = 0.0
    slippage_bps: float = 0.0
    long_only: bool = True
    fill_allowed: bool = True
    notional_cap_mode: str = "strict_position_cap"


@dataclass(frozen=True)
class ExecutionAccountingResult:
    fill_status: str
    target_shares: int
    requested_delta: int
    executed_delta: int
    shares_after: int
    execution_price_raw: float
    execution_price_after_slippage: float
    transaction_notional: float
    commission_cost: float
    effective_cost_per_share: float
    cost_model: str
    cost_price_fraction: float
    slippage_cost: float
    cash_after: float
    equity_before: float
    equity_after: float
    inventory_fraction_before: float
    inventory_fraction_after: float
    max_shares_exec: int
    pre_execution_gap_pnl: float
    post_execution_pnl: float
    real_pnl_step: float
    real_cost_step: float
    real_net_step: float
    requested_action_signed: float = 0.0
    requested_target_shares: int = 0
    requested_delta_shares: int = 0
    max_shares_reference: int = 0
    actual_notional_fraction_after: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _finite_positive(x: float) -> bool:
    return bool(np.isfinite(float(x)) and float(x) > 0.0)


def compute_effective_cost_per_share(
    *,
    fill_price: float,
    cost_model: str,
    fixed_cost_per_share: float,
    cost_price_fraction: float,
    cost_min_per_share: float = 0.0,
    cost_max_per_share: float = 0.0,
) -> float:
    model = str(cost_model or COST_MODE_FIXED_PER_SHARE)
    fixed = float(fixed_cost_per_share)
    frac = float(cost_price_fraction)
    mn = float(cost_min_per_share)
    mx = float(cost_max_per_share)
    if fixed < 0.0 or not np.isfinite(fixed):
        raise ValueError(f"fixed_cost_per_share must be finite and >= 0, got {fixed_cost_per_share}")
    if frac < 0.0 or not np.isfinite(frac):
        raise ValueError(f"cost_price_fraction must be finite and >= 0, got {cost_price_fraction}")
    if mn < 0.0 or not np.isfinite(mn):
        raise ValueError(f"cost_min_per_share must be finite and >= 0, got {cost_min_per_share}")
    if mx < 0.0 or not np.isfinite(mx):
        raise ValueError(f"cost_max_per_share must be finite and >= 0, got {cost_max_per_share}")
    if mn > 0.0 and mx > 0.0 and mx < mn:
        raise ValueError(f"cost_max_per_share must be >= cost_min_per_share when both bounds are enabled: min={mn} max={mx}")
    if model == COST_MODE_FIXED_PER_SHARE:
        cps = fixed
    elif model == COST_MODE_PRICE_FRACTION:
        px = float(fill_price)
        if not _finite_positive(px):
            raise ValueError(f"fill_price must be finite and positive for price_fraction cost model, got {fill_price}")
        cps = px * frac
    else:
        raise ValueError(f"Unknown cost_model={cost_model!r}; expected one of {COST_MODE_VALUES}")
    if mn > 0.0:
        cps = max(cps, mn)
    if mx > 0.0:
        cps = min(cps, mx)
    if not np.isfinite(cps) or cps < 0.0:
        raise ValueError(f"effective cost per share must be finite and nonnegative, got {cps}")
    return float(cps)


def compute_gross_pnl_from_equity_delta(
    *,
    equity_before: float,
    equity_after: float,
    real_cost_step: float,
) -> float:
    """Return gross economic PnL implied by marked equity and explicit costs."""
    eq_before = float(equity_before)
    eq_after = float(equity_after)
    cost = float(real_cost_step)
    if not np.isfinite(eq_before):
        raise ValueError(f"equity_before must be finite, got {equity_before}")
    if not np.isfinite(eq_after):
        raise ValueError(f"equity_after must be finite, got {equity_after}")
    if not np.isfinite(cost):
        raise ValueError(f"real_cost_step must be finite, got {real_cost_step}")
    return float((eq_after - eq_before) + cost)


def _cap_delta(delta: int, cur_shares: int, max_shares: int, max_delta: int, mode: str, long_only: bool) -> int:
    delta = int(np.clip(int(delta), -int(max_delta), int(max_delta)))
    next_shares = int(cur_shares) + delta
    if long_only and next_shares < 0:
        delta = -int(cur_shares)
        next_shares = 0

    if mode == "strict_position_cap":
        if next_shares > int(max_shares):
            delta = int(max_shares) - int(cur_shares)
    elif mode == "buy_only_cap":
        if delta > 0:
            buy_room = max(0, int(max_shares) - int(cur_shares))
            delta = min(delta, buy_room)
        if long_only and int(cur_shares) + delta < 0:
            delta = -int(cur_shares)
    else:
        raise ValueError(f"Unknown notional_cap_mode={mode!r}")
    return int(delta)


def compute_execution_accounting(inp: ExecutionAccountingInput) -> ExecutionAccountingResult:
    p_obs = float(inp.observation_mark)
    p_exec_raw = float(inp.execution_price_raw)
    p_mark = float(inp.next_mark_price)
    action = float(np.clip(float(inp.executable_action), -1.0, 1.0))

    equity_before = float(inp.cash_before) + float(inp.shares_before) * p_obs

    if abs(action) < float(inp.deadband_u):
        action = 0.0

    p_ref = p_mark if _finite_positive(p_mark) else (p_exec_raw if _finite_positive(p_exec_raw) else p_obs)
    max_shares_ref = int(max(1, math.floor(float(inp.max_notional) / max(float(p_ref), 1e-12))))
    requested_delta_ref = int(round(action * max_shares_ref))
    if float(inp.frac_deadband) > 0.0:
        band_shares_ref = int(round(float(inp.frac_deadband) * max_shares_ref))
        if abs(requested_delta_ref) <= band_shares_ref:
            requested_delta_ref = 0
    requested_target_delta_ref = _cap_delta(
        requested_delta_ref,
        int(inp.shares_before),
        max_shares_ref,
        int(inp.max_delta_shares),
        str(inp.notional_cap_mode),
        bool(inp.long_only),
    )
    if abs(requested_target_delta_ref) < int(inp.deadband_shares):
        requested_target_delta_ref = 0
    requested_target_shares_ref = int(inp.shares_before) + int(requested_target_delta_ref)
    if bool(inp.long_only):
        requested_target_shares_ref = max(0, requested_target_shares_ref)

    fill_allowed = bool(inp.fill_allowed and _finite_positive(p_exec_raw))
    if not fill_allowed:
        shares_after = int(inp.shares_before)
        cash_after = float(inp.cash_before)
        equity_after = cash_after + float(shares_after) * p_mark
        real_net = equity_after - equity_before
        return ExecutionAccountingResult(
            fill_status="no_fill",
            target_shares=int(requested_target_shares_ref),
            requested_delta=int(requested_delta_ref),
            executed_delta=0,
            shares_after=shares_after,
            execution_price_raw=p_exec_raw,
            execution_price_after_slippage=p_exec_raw,
            transaction_notional=0.0,
            commission_cost=0.0,
            effective_cost_per_share=0.0,
            cost_model=str(inp.cost_model or COST_MODE_FIXED_PER_SHARE),
            cost_price_fraction=float(inp.cost_price_fraction),
            slippage_cost=0.0,
            cash_after=cash_after,
            equity_before=equity_before,
            equity_after=equity_after,
            inventory_fraction_before=float(np.clip(float(inp.shares_before) / float(max_shares_ref), 0.0, 1.0)),
            inventory_fraction_after=float(np.clip(float(shares_after) / float(max_shares_ref), 0.0, 1.0)),
            max_shares_exec=max_shares_ref,
            pre_execution_gap_pnl=float(inp.shares_before) * (p_exec_raw - p_obs) if np.isfinite(p_exec_raw) else 0.0,
            post_execution_pnl=float(inp.shares_before) * (p_mark - p_exec_raw) if np.isfinite(p_exec_raw) else real_net,
            real_pnl_step=real_net,
            real_cost_step=0.0,
            real_net_step=real_net,
            requested_action_signed=float(action),
            requested_target_shares=int(requested_target_shares_ref),
            requested_delta_shares=int(requested_delta_ref),
            max_shares_reference=int(max_shares_ref),
            actual_notional_fraction_after=float((shares_after * p_ref) / max(float(inp.max_notional), 1e-12)),
        )

    slip_mult = 1.0
    if action > 0:
        slip_mult += float(inp.slippage_bps) / 10000.0
    elif action < 0:
        slip_mult -= float(inp.slippage_bps) / 10000.0
    p_exec = float(p_exec_raw) * slip_mult
    if not _finite_positive(p_exec):
        raise ValueError(f"Invalid execution price after slippage: {p_exec}")

    max_shares_exec = int(max(1, math.floor(float(inp.max_notional) / max(p_exec, 1e-12))))
    requested_delta = int(round(action * max_shares_exec))
    if float(inp.frac_deadband) > 0.0:
        band_shares = int(round(float(inp.frac_deadband) * max_shares_exec))
        if abs(requested_delta) <= band_shares:
            requested_delta = 0

    executed_delta = _cap_delta(
        requested_delta,
        int(inp.shares_before),
        max_shares_exec,
        int(inp.max_delta_shares),
        str(inp.notional_cap_mode),
        bool(inp.long_only),
    )
    if abs(executed_delta) < int(inp.deadband_shares):
        executed_delta = 0

    shares_after = int(inp.shares_before) + int(executed_delta)
    if bool(inp.long_only):
        shares_after = max(0, shares_after)

    target_shares = int(shares_after)
    transaction_notional = float(executed_delta) * p_exec
    effective_cost_per_share = (
        compute_effective_cost_per_share(
            fill_price=p_exec,
            cost_model=str(inp.cost_model),
            fixed_cost_per_share=float(inp.cost_per_share),
            cost_price_fraction=float(inp.cost_price_fraction),
            cost_min_per_share=float(inp.cost_min_per_share),
            cost_max_per_share=float(inp.cost_max_per_share),
        )
        if int(executed_delta) != 0
        else 0.0
    )
    commission_cost = float(effective_cost_per_share) * abs(int(executed_delta))
    slippage_cost = abs(float(executed_delta) * (p_exec - p_exec_raw))
    cash_after = float(inp.cash_before) - transaction_notional - commission_cost
    equity_after = cash_after + float(shares_after) * p_mark

    pre_gap = float(inp.shares_before) * (p_exec_raw - p_obs)
    post_pnl = float(shares_after) * (p_mark - p_exec_raw)
    real_pnl = pre_gap + post_pnl
    real_cost = commission_cost + slippage_cost
    real_net = equity_after - equity_before

    return ExecutionAccountingResult(
        fill_status="filled" if executed_delta != 0 else "no_trade",
        target_shares=target_shares,
        requested_delta=int(requested_delta),
        executed_delta=int(executed_delta),
        shares_after=int(shares_after),
        execution_price_raw=p_exec_raw,
        execution_price_after_slippage=p_exec,
        transaction_notional=float(transaction_notional),
        commission_cost=float(commission_cost),
        effective_cost_per_share=float(effective_cost_per_share),
        cost_model=str(inp.cost_model or COST_MODE_FIXED_PER_SHARE),
        cost_price_fraction=float(inp.cost_price_fraction),
        slippage_cost=float(slippage_cost),
        cash_after=float(cash_after),
        equity_before=float(equity_before),
        equity_after=float(equity_after),
        inventory_fraction_before=float(np.clip(float(inp.shares_before) / float(max_shares_exec), 0.0, 1.0)),
        inventory_fraction_after=float(np.clip(float(shares_after) / float(max_shares_exec), 0.0, 1.0)),
        max_shares_exec=int(max_shares_exec),
        pre_execution_gap_pnl=float(pre_gap),
        post_execution_pnl=float(post_pnl),
        real_pnl_step=float(real_pnl),
        real_cost_step=float(real_cost),
        real_net_step=float(real_net),
        requested_action_signed=float(action),
        requested_target_shares=int(target_shares),
        requested_delta_shares=int(requested_delta),
        max_shares_reference=int(max_shares_exec),
        actual_notional_fraction_after=float((shares_after * p_mark) / max(float(inp.max_notional), 1e-12)),
    )


def resolve_dense_execution_source(
    prices: np.ndarray,
    obs_idx: int,
    mode: str,
    trades: Optional[np.ndarray] = None,
    allow_synthetic_execution_proxy: bool = False,
    allow_legacy_same_close: bool = False,
) -> ExecutionSource:
    mode = str(mode)
    if mode not in EXECUTION_MODES:
        raise ValueError(f"Unsupported execution_mode={mode!r}; expected one of {EXECUTION_MODES}")
    n = int(len(prices))
    obs_idx = int(obs_idx)
    if obs_idx < 0 or obs_idx >= n:
        raise IndexError(f"obs_idx out of range: {obs_idx}")

    if mode == "legacy_same_close":
        if not bool(allow_legacy_same_close):
            raise RuntimeError("legacy_same_close requires --allow_legacy_same_close 1")
        exec_idx = obs_idx
        mark_idx = min(n - 1, obs_idx + 1)
        return ExecutionSource(
            mode=mode,
            obs_idx=obs_idx,
            next_obs_idx=min(n - 1, obs_idx + 1),
            exec_idx=exec_idx,
            mark_idx=mark_idx,
            p_obs=float(prices[obs_idx]),
            p_exec_raw=float(prices[exec_idx]),
            p_mark=float(prices[mark_idx]),
            fill_allowed=True,
            fill_status="legacy_same_close",
            synthetic_execution=False,
            execution_price_source="close[obs_idx]",
        )

    exec_idx = obs_idx + 1
    mark_idx = obs_idx + 2
    if exec_idx >= n or mark_idx >= n:
        raise IndexError(f"not enough rows for v4 execution: obs_idx={obs_idx} n={n}")

    fill_allowed = True
    fill_status = "fillable"
    synthetic = False
    source = "close[obs_idx+1]"

    if mode == "next_close_real_bar":
        if trades is None:
            raise RuntimeError("next_close_real_bar requires a real-trade count array")
        fill_allowed = bool(float(trades[exec_idx]) > 0.0)
        fill_status = "fillable" if fill_allowed else "no_real_trade"
        source = "close[obs_idx+1] gated by trades[obs_idx+1] > 0"
    elif mode == "next_close_ffill_proxy":
        if not bool(allow_synthetic_execution_proxy):
            raise RuntimeError("next_close_ffill_proxy requires --allow_synthetic_execution_proxy 1")
        synthetic = True
        source = "close[obs_idx+1] synthetic ffill proxy"
    elif mode == "first_post_bar_trade_sidecar":
        raise RuntimeError("sidecar resolution must be supplied by the caller before accounting")
    elif mode == "pending_first_real_bar":
        raise RuntimeError("pending_first_real_bar is stateful and must be resolved by the environment")

    return ExecutionSource(
        mode=mode,
        obs_idx=obs_idx,
        next_obs_idx=obs_idx + 1,
        exec_idx=exec_idx,
        mark_idx=mark_idx,
        p_obs=float(prices[obs_idx]),
        p_exec_raw=float(prices[exec_idx]),
        p_mark=float(prices[mark_idx]),
        fill_allowed=fill_allowed,
        fill_status=fill_status,
        synthetic_execution=synthetic,
        execution_price_source=source,
    )


def reward_endpoint_index(exec_anchor_idx: int, horizon: int) -> int:
    return int(exec_anchor_idx) + int(horizon)


def last_valid_observation_index(n_rows: int, execution_lag: int, max_reward_horizon: int, require_post_exec_mark: bool = True) -> int:
    extra_mark = 1 if bool(require_post_exec_mark) else 0
    max_need = max(int(max_reward_horizon), int(extra_mark))
    return int(n_rows) - 1 - int(execution_lag) - int(max_need)
