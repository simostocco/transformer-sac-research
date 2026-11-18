"""Extracted research definitions; see docs/source_map.json for provenance."""
from __future__ import annotations
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import deque
import numpy as np
import pandas as pd
from streaming_research.policy.accounting import COST_MODE_FIXED_PER_SHARE, COST_MODE_INHERIT, COST_MODE_VALUES, LIQUIDATION_COST_MODE_VALUES, ExecutionAccountingInput, ExecutionSource, compute_execution_accounting, compute_effective_cost_per_share, compute_gross_pnl_from_equity_delta, last_valid_observation_index, resolve_dense_execution_source, reward_endpoint_index

def apply_signed_action_bias(action_raw_signed: float, action_bias: float) -> float:
    return float(np.clip(float(action_raw_signed) + float(action_bias), -1.0, 1.0))


def safe_div(a: float, b: float) -> float:
    return float(a / b) if abs(b) > 1e-12 else 0.0


OBSERVATION_TIMING_MODES = ("legacy_same_row", "previous_completed_bar")


TEMPORAL_SEMANTICS_LEGACY = "V4_LEGACY_SAME_ROW_AFTER_BAR_CLOSE_V1"


def fracdiff_weights(d: float, n_lags: int, min_weight: float = 1e-6) -> np.ndarray:
    w = np.empty(int(n_lags), dtype=np.float64)
    w[0] = 1.0
    for k in range(1, int(n_lags)):
        w[k] = w[k - 1] * ((k - 1) - d) / k
    if min_weight is not None and min_weight > 0:
        keep = np.where(np.abs(w) >= float(min_weight))[0]
        if keep.size > 0:
            w = w[: int(keep[-1]) + 1]
    return w.astype(np.float32)


def fracdiff_series(x: np.ndarray, d: float, n_lags: int = 60, min_weight: float = 1e-6) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    T = x.shape[0]
    w = fracdiff_weights(d=float(d), n_lags=int(n_lags), min_weight=float(min_weight))
    L = len(w)
    y = np.full(T, np.nan, dtype=np.float32)
    for t in range(L - 1, T):
        y[t] = float(np.dot(w, x[t - L + 1 : t + 1][::-1]))
    return y


def qcol(prefix: str, kind: str, N: int, q: float) -> str:
    qs = str(float(q)).replace(".", "p")
    return f"{prefix}{kind}_{int(N)}_q{qs}"


def pick_state_cols(df: pd.DataFrame, use_ctx_q: bool = True) -> List[str]:
    cols = [c for c in df.columns if c.startswith("loc_") and "_q" in c]
    if use_ctx_q:
        cols += [c for c in df.columns if c.startswith("ctx_") and "_q" in c]
    return sorted(cols)


def pick_embedding_cols(df: pd.DataFrame, use_ctx: bool = True) -> List[str]:
    cols = []
    cols += sorted([c for c in df.columns if c.startswith("loc_emb_")])
    if use_ctx:
        cols += sorted([c for c in df.columns if c.startswith("ctx_emb_")])
    return cols


class EpisodeCache:
    def __init__(self, max_items: int = 2):
        self.max_items = int(max_items)
        self._cache: Dict[str, Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        self._order: List[str] = []

    def get(self, key: str):
        if key in self._cache:
            self._order.remove(key)
            self._order.append(key)
            return self._cache[key]
        return None

    def put(self, key: str, value):
        if key in self._cache:
            self._order.remove(key)
        self._cache[key] = value
        self._order.append(key)
        while len(self._order) > self.max_items:
            old = self._order.pop(0)
            self._cache.pop(old, None)


@dataclass
class EnvCfg:
    price_col: str = "close"
    use_embeddings: bool = True

    max_notional: float = 20000.0
    max_delta_shares: int = 2000
    notional_cap_mode: str = "strict_position_cap"
    deadband_shares: int = 100
    cost_per_share: float = 0.005
    liquidation_cost_per_share: Optional[float] = None
    cost_model: str = COST_MODE_FIXED_PER_SHARE
    cost_price_fraction: float = 0.005
    cost_min_per_share: float = 0.0
    cost_max_per_share: float = 0.0
    liquidation_cost_model: str = COST_MODE_INHERIT

    start_mode: str = "random"
    segment_len: int = 0
    warm_start_flat: bool = True

    pos_frac_clip: float = 1.5

    reward_scale: float = 1.0
    lam_dact: float = 0.0
    lam_turnover: float = 0.0
    lam_inv: float = 0.0
    deadband_u: float = 0.0

    stack_len: int = 32
    use_ctx_embeddings: bool = True
    edge_windows: Tuple[int, int] = (10, 60)
    rollW: int = 5
    include_close_feat: bool = True

    frac_deadband: float = 0.05
    cost_amort_steps: int = 1

    reward_horizons: Tuple[int, ...] = (1, 10, 30)
    reward_weights: Tuple[float, ...] = (1.0, 0.5, 0.25)
    reward_kappa: float = 0.0
    flat_bonus: float = 0.0

    lam_hold_age: float = 0.0
    hold_age_scale: int = 300
    lam_stale_pos: float = 0.0
    lam_stale_edge: float = 0.0

    use_regime_filter: bool = True
    regime_lookback: int = 300
    regime_ret_thresh: float = 0.05
    regime_flat_action: float = 0.0
    regime_use_abs: bool = False

    premarket_only_replay: bool = False
    premarket_start_hhmm: int = 400
    premarket_end_hhmm: int = 930
    timestamp_col: str = "datetime_utc"
    market_tz: str = "America/New_York"
    train_time_filter: bool = False
    train_start_hhmm: int = 700
    train_end_hhmm: int = 1100
    train_market_tz: str = "America/New_York"

    # V4_CAUSAL_TIMING
    environment_version: str = "signedexec_longonly_v4"
    accounting_version: str = "post_observation_execution_v1"
    bar_label_convention: str = "start_labeled_completed_1s_bar"
    observation_availability: str = "after_bar_close"
    observation_timing: str = "legacy_same_row"
    feature_lag_bars: int = 0
    temporal_semantics_version: str = TEMPORAL_SEMANTICS_LEGACY
    execution_mode: str = "next_close_real_bar"
    execution_price_source: str = "close[obs_idx+1]"
    execution_sidecar: str = ""
    execution_price_col: str = "execution_price"
    execution_timestamp_col: str = "execution_timestamp_utc"
    execution_bar_start_col: str = "datetime_utc"
    execution_symbol_col: str = "symbol"
    max_execution_lag_seconds: float = 1.0
    missing_execution_policy: str = "no_fill"
    allow_synthetic_execution_proxy: bool = False
    allow_legacy_same_close: bool = False
    slippage_bps: float = 0.0
    bar_seconds: int = 1
    reward_timing_version: str = "post_exec_elapsed_horizon_v1"
    synthetic_execution_proxy: bool = False
    pending_order_semantics_version: str = ""
    pending_timeout_seconds: float = 900.0
    pending_max_participation_rate: float = 0.05
    pending_replacement_policy: str = "latest_target_wins"
    pending_observation_fields: Tuple[str, ...] = (
        "pending_exists",
        "pending_side",
        "pending_remaining_fraction",
        "pending_target_fraction",
        "pending_age_fraction",
    )
    replay_action_semantics: str = "executed_u"


class ParquetTradingEnv:
    def __init__(self, parquet_paths: List[str], cfg: EnvCfg, device: str = "cpu"):
        self.paths = list(parquet_paths)
        if not self.paths:
            raise RuntimeError("No parquet paths provided.")

        self.cfg = cfg
        self.device = device
        self._cache = EpisodeCache(max_items=2)

        self.Xs: Optional[np.ndarray] = None
        self.Xe: Optional[np.ndarray] = None
        self.prices: Optional[np.ndarray] = None
        self.trade_counts: Optional[np.ndarray] = None
        self.trade_volume: Optional[np.ndarray] = None
        self.sidecar_price: Optional[np.ndarray] = None
        self.sidecar_ts: Optional[np.ndarray] = None
        self.cash: float = 0.0
        self.timestamps: Optional[np.ndarray] = None
        self.t = 0
        self.t_end = 0
        self.shares = 0
        self.s_dim = None

        self.prev_exposure = 0.0
        self.prev_trade_action = 0.0
        self.prev_u = 0.0
        self.last_exec_delta = 0
        self.stack = None
        self._cost_queue = deque()
        self.current_path: Optional[str] = None

        self.hold_steps = 0
        self.prev_edge = 0.0
        self.pending_target_shares = 0
        self.pending_remaining_delta = 0
        self.pending_created_ts: Optional[pd.Timestamp] = None
        self.pending_eligible_ts: Optional[pd.Timestamp] = None
        self.pending_last_update_ts: Optional[pd.Timestamp] = None
        self.pending_replaced = False
        self.pending_expired = False
        self.pending_expiry_reason = ""
        self.pending_fill_fraction_last_step = 0.0

        self.base_xs_dim = None
        self.nonemb_dim = None
        self.stack_dim = None
        self.emb_dim = None
        self.prev_u_idx_in_nonemb = None
        self.prev_u_idx_in_obs = None
        self.edge_feat_index = None
        self.premarket_mask: Optional[np.ndarray] = None
        self.regime_flags: Optional[np.ndarray] = None
        self._episode_meta: Dict[str, Dict] = {}

    def _regime_filter_on(self) -> bool:
        if not bool(getattr(self.cfg, "use_regime_filter", False)):
            return True
        if self.regime_flags is None:
            return True
        if self.t < 0 or self.t >= len(self.regime_flags):
            return False
        return bool(self.regime_flags[self._market_feature_idx(self.t)] > 0.5)

    def _is_premarket_now(self) -> bool:
        if self.premarket_mask is None:
            return False
        if self.t < 0 or self.t >= len(self.premarket_mask):
            return False
        return bool(self.premarket_mask[self.t] > 0.5)

    def _nonemb_feat(self) -> np.ndarray:
        assert self.Xs is not None and self.prices is not None
        market_idx = self._market_feature_idx(self.t)
        p = float(self.prices[market_idx])

        max_shares_t = int(max(1, math.floor(self.cfg.max_notional / max(p, 1e-12))))

        exposure01 = float(self.shares) / float(max_shares_t)
        last_exec = float(self.last_exec_delta) / float(max_shares_t)

        pos_frac = (float(self.shares) * p) / max(self.cfg.max_notional, 1e-12)
        pos_frac = float(np.clip(pos_frac, 0.0, self.cfg.pos_frac_clip))

        prev_exposure = float(np.clip(self.prev_exposure, 0.0, 1.0))
        hold_feat = float(self.hold_steps) / max(float(self.cfg.hold_age_scale), 1.0)
        hold_feat = float(np.clip(hold_feat, 0.0, 1.0))

        if self.cfg.include_close_feat:
            p_feat = float(p / max(self.cfg.max_notional, 1e-12))
            extra = np.array([pos_frac, exposure01, last_exec, prev_exposure, p_feat, hold_feat], np.float32)
        else:
            extra = np.array([pos_frac, exposure01, last_exec, prev_exposure, hold_feat], np.float32)

        if self._pending_enabled():
            extra = np.concatenate([extra, self._pending_obs_fields(p)], axis=0).astype(np.float32)

        return np.concatenate([self.Xs[market_idx], extra]).astype(np.float32)

    def _feature_lag_bars(self) -> int:
        timing = str(getattr(self.cfg, "observation_timing", "legacy_same_row"))
        if timing == "legacy_same_row":
            return 0
        if timing == "previous_completed_bar":
            return max(1, int(getattr(self.cfg, "feature_lag_bars", 1)))
        raise ValueError(f"unknown observation_timing={timing!r}; expected one of {OBSERVATION_TIMING_MODES}")

    def _market_feature_idx(self, decision_idx: Optional[int] = None) -> int:
        idx = int(self.t if decision_idx is None else decision_idx)
        lag = int(self._feature_lag_bars())
        return int(max(0, idx - lag))

    def _pending_enabled(self) -> bool:
        return str(getattr(self.cfg, "execution_mode", "")) == "pending_first_real_bar"

    def _timestamp_at(self, idx: int) -> Optional[pd.Timestamp]:
        if self.timestamps is None or idx < 0 or idx >= len(self.timestamps):
            return None
        return pd.Timestamp(self.timestamps[idx])

    def _pending_age_seconds(self, idx: Optional[int] = None) -> float:
        if self.pending_created_ts is None:
            return 0.0
        now = self._timestamp_at(int(self.t if idx is None else idx))
        if now is None:
            return 0.0
        return max(0.0, float((now - self.pending_created_ts).total_seconds()))

    def _pending_obs_fields(self, p: float) -> np.ndarray:
        max_shares_t = int(max(1, math.floor(float(self.cfg.max_notional) / max(float(p), 1e-12))))
        exists = 1.0 if int(self.pending_remaining_delta) != 0 else 0.0
        side = float(np.sign(int(self.pending_remaining_delta))) if exists else 0.0
        remaining_frac = float(np.clip(int(self.pending_remaining_delta) / max(1, max_shares_t), -1.0, 1.0))
        target_frac = float(np.clip(int(self.pending_target_shares) / max(1, max_shares_t), 0.0, 1.0))
        timeout = max(1e-9, float(getattr(self.cfg, "pending_timeout_seconds", 900.0)))
        age_frac = float(np.clip(self._pending_age_seconds() / timeout, 0.0, 1.0)) if exists else 0.0
        return np.array([exists, side, remaining_frac, target_frac, age_frac], dtype=np.float32)

    def _load_episode_arrays(self, path: str):
        cached = self._cache.get(path)
        if cached is not None:
            return cached

        df_raw = pd.read_parquet(path)

        if self.cfg.price_col not in df_raw.columns:
            raise ValueError(f"{path}: missing price_col={self.cfg.price_col}")

        ts_col = getattr(self.cfg, "timestamp_col", "datetime_utc")
        train_tz = str(getattr(self.cfg, "train_market_tz", "America/New_York"))
        if ts_col not in df_raw.columns:
            raise ValueError(f"{path}: missing timestamp column '{ts_col}' needed for time diagnostics")

        ts_raw = pd.to_datetime(df_raw[ts_col], errors="coerce", utc=True)
        if ts_raw.isna().all():
            raise ValueError(f"{path}: timestamp column '{ts_col}' could not be parsed")

        raw_rows = int(len(df_raw))
        raw_ts_min_utc = ts_raw.min()
        raw_ts_max_utc = ts_raw.max()

        close_raw = pd.to_numeric(df_raw[self.cfg.price_col], errors="coerce").to_numpy(np.float64)

        lb = int(getattr(self.cfg, "regime_lookback", 300))
        thr = float(getattr(self.cfg, "regime_ret_thresh", 0.05))
        use_abs = bool(getattr(self.cfg, "regime_use_abs", False))
        use_regime = bool(getattr(self.cfg, "use_regime_filter", False))

        if use_regime:
            regime_flag_raw = np.zeros(len(close_raw), dtype=np.float32)
            for t in range(lb, len(close_raw)):
                p_now = close_raw[t]
                p_past = close_raw[t - lb]
                if np.isfinite(p_now) and np.isfinite(p_past) and p_past > 1e-12:
                    ret = (p_now / p_past) - 1.0
                    metric = abs(ret) if use_abs else ret
                    regime_flag_raw[t] = 1.0 if metric >= thr else 0.0
        else:
            regime_flag_raw = np.ones(len(close_raw), dtype=np.float32)

        if "rl_valid" in df_raw.columns:
            keep_rl = df_raw["rl_valid"].fillna(False).to_numpy(bool)
            df = df_raw.loc[keep_rl].reset_index(drop=True)
            regime_flag = regime_flag_raw[keep_rl]
        else:
            df = df_raw.reset_index(drop=True)
            regime_flag = regime_flag_raw.copy()

        rows_after_rl_valid = int(len(df))
        market_tz = getattr(self.cfg, "market_tz", "America/New_York")

        ts = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
        if ts.isna().all():
            raise ValueError(f"{path}: timestamp column '{ts_col}' could not be parsed")

        ts_local = ts.dt.tz_convert(market_tz)
        hhmm = ts_local.dt.hour * 100 + ts_local.dt.minute
        premarket_mask = (
            (hhmm >= int(self.cfg.premarket_start_hhmm)) &
            (hhmm < int(self.cfg.premarket_end_hhmm))
        ).to_numpy(np.float32)

        if bool(getattr(self.cfg, "train_time_filter", False)):
            ts_train_local = ts.dt.tz_convert(train_tz)
            train_hhmm = ts_train_local.dt.hour * 100 + ts_train_local.dt.minute

            start_hhmm = int(getattr(self.cfg, "train_start_hhmm", 700))
            end_hhmm = int(getattr(self.cfg, "train_end_hhmm", 1100))

            if start_hhmm <= end_hhmm:
                keep_time = (train_hhmm >= start_hhmm) & (train_hhmm < end_hhmm)
            else:
                keep_time = (train_hhmm >= start_hhmm) | (train_hhmm < end_hhmm)

            keep_time_np = keep_time.to_numpy(bool)
            df = df.loc[keep_time_np].reset_index(drop=True)
            regime_flag = regime_flag[keep_time_np]
            premarket_mask = premarket_mask[keep_time_np]
            ts = ts.loc[keep_time].reset_index(drop=True)

            if len(df) < 3:
                raise ValueError(
                    f"{path}: train_time_filter kept only {len(df)} rows "
                    f"for window {start_hhmm}-{end_hhmm} {train_tz}"
                )

        rows_after_time_filter = int(len(df))
        sidecar_price = np.full(len(df), np.nan, dtype=np.float64)
        sidecar_ts = np.array([""] * len(df), dtype=object)
        if str(getattr(self.cfg, "execution_mode", "")) == "first_post_bar_trade_sidecar":
            sidecar_path = str(getattr(self.cfg, "execution_sidecar", "")).strip()
            if sidecar_path:
                sc_path = Path(sidecar_path)
                if not sc_path.exists():
                    raise FileNotFoundError(f"--execution_sidecar not found: {sidecar_path}")
                sc = pd.read_parquet(sc_path) if sc_path.suffix.lower() == ".parquet" else pd.read_csv(sc_path)
                sym_col = str(getattr(self.cfg, "execution_symbol_col", "symbol"))
                bar_col = str(getattr(self.cfg, "execution_bar_start_col", "datetime_utc"))
                ets_col = str(getattr(self.cfg, "execution_timestamp_col", "execution_timestamp_utc"))
                px_col = str(getattr(self.cfg, "execution_price_col", "execution_price"))
                missing = [c for c in (sym_col, bar_col, ets_col, px_col) if c not in sc.columns]
                if missing:
                    raise ValueError(f"{sidecar_path}: missing sidecar columns {missing}")
                symbol_val = str(df["symbol"].iloc[0]).upper() if "symbol" in df.columns and len(df) else ""
                sc = sc.copy()
                sc[sym_col] = sc[sym_col].astype(str).str.upper()
                if symbol_val:
                    sc = sc.loc[sc[sym_col] == symbol_val].copy()
                sc[bar_col] = pd.to_datetime(sc[bar_col], utc=True, errors="coerce")
                sc[ets_col] = pd.to_datetime(sc[ets_col], utc=True, errors="coerce")
                sc[px_col] = pd.to_numeric(sc[px_col], errors="coerce")
                bar_end = sc[bar_col] + pd.to_timedelta(int(getattr(self.cfg, "bar_seconds", 1)), unit="s")
                max_end = bar_end + pd.to_timedelta(float(getattr(self.cfg, "max_execution_lag_seconds", 1.0)), unit="s")
                valid = (
                    sc[bar_col].notna()
                    & sc[ets_col].notna()
                    & np.isfinite(sc[px_col].to_numpy(dtype=np.float64))
                    & (sc[px_col] > 0.0)
                    & (sc[ets_col] >= bar_end)
                    & (sc[ets_col] < max_end)
                )
                sc = sc.loc[valid].sort_values([bar_col, ets_col], kind="mergesort")
                sc = sc.drop_duplicates(subset=[bar_col], keep="first")
                tmp = pd.DataFrame({"_row": np.arange(len(df)), "_bar": ts.reset_index(drop=True)})
                merged = tmp.merge(sc[[bar_col, ets_col, px_col]], left_on="_bar", right_on=bar_col, how="left")
                sidecar_price = pd.to_numeric(merged[px_col], errors="coerce").to_numpy(dtype=np.float64)
                sidecar_ts = merged[ets_col].astype(str).replace("NaT", "").to_numpy(dtype=object)
        timestamps = ts.to_numpy()

        state_cols = pick_state_cols(df, use_ctx_q=False)
        if not state_cols:
            raise ValueError(f"{path}: no loc_/ctx_ quantile columns found")

        emb_cols = pick_embedding_cols(df, use_ctx=self.cfg.use_ctx_embeddings) if self.cfg.use_embeddings else []

        Xs_quant = df[state_cols].to_numpy(dtype=np.float32, copy=False)
        base_quant_dim = Xs_quant.shape[1]

        N1, N2 = self.cfg.edge_windows
        rollW = int(self.cfg.rollW)

        mfe1 = pd.to_numeric(df[qcol("loc_", "mfe", N1, 0.9)], errors="coerce").to_numpy(np.float32)
        mae1 = pd.to_numeric(df[qcol("loc_", "mae", N1, 0.1)], errors="coerce").to_numpy(np.float32)
        mfe2 = pd.to_numeric(df[qcol("loc_", "mfe", N2, 0.9)], errors="coerce").to_numpy(np.float32)
        mae2 = pd.to_numeric(df[qcol("loc_", "mae", N2, 0.1)], errors="coerce").to_numpy(np.float32)

        edge1 = (mfe1 - np.abs(mae1)).astype(np.float32)
        edge2 = (mfe2 - np.abs(mae2)).astype(np.float32)
        ediff = (edge2 - edge1).astype(np.float32)

        n_lags = 60
        min_w = 1e-6

        f_edge1_d06 = fracdiff_series(edge1, d=0.6, n_lags=n_lags, min_weight=min_w)
        f_edge1_d04 = fracdiff_series(edge1, d=0.4, n_lags=n_lags, min_weight=min_w)
        f_edge2_d06 = fracdiff_series(edge2, d=0.6, n_lags=n_lags, min_weight=min_w)
        f_edge2_d04 = fracdiff_series(edge2, d=0.4, n_lags=n_lags, min_weight=min_w)
        f_ediff_d06 = fracdiff_series(ediff, d=0.6, n_lags=n_lags, min_weight=min_w)
        f_ediff_d04 = fracdiff_series(ediff, d=0.4, n_lags=n_lags, min_weight=min_w)

        def d1(x: np.ndarray) -> np.ndarray:
            x = x.astype(np.float32, copy=False)
            dx = np.empty_like(x, dtype=np.float32)
            dx[0] = 0.0
            dx[1:] = x[1:] - x[:-1]
            return dx

        d_edge1 = d1(edge1)
        d_edge2 = d1(edge2)
        d_ediff = d1(ediff)

        dd_edge1 = d1(d_edge1)
        dd_edge2 = d1(d_edge2)
        dd_ediff = d1(d_ediff)

        def roll_mean_nan(x: np.ndarray, W: int) -> np.ndarray:
            x = x.astype(np.float32, copy=False)
            if W <= 1:
                return x
            s = pd.Series(x)
            return s.rolling(window=W, min_periods=W).mean().to_numpy(np.float32)

        roll_d_edge1 = roll_mean_nan(d_edge1, rollW)
        roll_d_edge2 = roll_mean_nan(d_edge2, rollW)
        roll_d_ediff = roll_mean_nan(d_ediff, rollW)

        roll_f_edge1_d06 = roll_mean_nan(f_edge1_d06, rollW)
        roll_f_edge1_d04 = roll_mean_nan(f_edge1_d04, rollW)
        roll_f_edge2_d06 = roll_mean_nan(f_edge2_d06, rollW)
        roll_f_edge2_d04 = roll_mean_nan(f_edge2_d04, rollW)
        roll_f_ediff_d06 = roll_mean_nan(f_ediff_d06, rollW)
        roll_f_ediff_d04 = roll_mean_nan(f_ediff_d04, rollW)

        volW = 60
        vol_eps = 1e-4
        vol_mode = "diff"

        def rolling_std(x: np.ndarray, W: int) -> np.ndarray:
            return pd.Series(x).rolling(window=W, min_periods=W).std(ddof=0).to_numpy(np.float32)

        def safe_vol_from_signal(sig: np.ndarray) -> np.ndarray:
            sig = sig.astype(np.float32, copy=False)
            if vol_mode == "diff":
                incr = np.full_like(sig, np.nan, dtype=np.float32)
                incr[1:] = sig[1:] - sig[:-1]
                vol = rolling_std(incr, volW)
            else:
                vol = rolling_std(sig, volW)
            return np.maximum(vol, vol_eps).astype(np.float32)

        vol_e1 = safe_vol_from_signal(edge1)
        vol_e2 = safe_vol_from_signal(edge2)
        vol_ed = safe_vol_from_signal(ediff)

        d_edge1_n = d_edge1 / vol_e1
        d_edge2_n = d_edge2 / vol_e2
        d_ediff_n = d_ediff / vol_ed

        f_edge1_d06_n = f_edge1_d06 / vol_e1
        f_edge1_d04_n = f_edge1_d04 / vol_e1
        f_edge2_d06_n = f_edge2_d06 / vol_e2
        f_edge2_d04_n = f_edge2_d04 / vol_e2
        f_ediff_d06_n = f_ediff_d06 / vol_ed
        f_ediff_d04_n = f_ediff_d04 / vol_ed

        roll_d_edge1_n = roll_d_edge1 / vol_e1
        roll_d_edge2_n = roll_d_edge2 / vol_e2
        roll_d_ediff_n = roll_d_ediff / vol_ed

        roll_f_edge1_d06_n = roll_f_edge1_d06 / vol_e1
        roll_f_edge1_d04_n = roll_f_edge1_d04 / vol_e1
        roll_f_edge2_d06_n = roll_f_edge2_d06 / vol_e2
        roll_f_edge2_d04_n = roll_f_edge2_d04 / vol_e2
        roll_f_ediff_d06_n = roll_f_ediff_d06 / vol_ed
        roll_f_ediff_d04_n = roll_f_ediff_d04 / vol_ed

        Xeng_orig = np.stack(
            [
                edge1, edge2, ediff,
                d_edge1_n, d_edge2_n, d_ediff_n,
                dd_edge1, dd_edge2, dd_ediff,
                f_edge1_d06_n, f_edge1_d04_n,
                f_edge2_d06_n, f_edge2_d04_n,
                f_ediff_d06_n, f_ediff_d04_n,
                roll_d_edge1_n, roll_d_edge2_n, roll_d_ediff_n,
                roll_f_edge1_d06_n, roll_f_edge1_d04_n,
                roll_f_edge2_d06_n, roll_f_edge2_d04_n,
                roll_f_ediff_d06_n, roll_f_ediff_d04_n,
            ],
            axis=1,
        ).astype(np.float32)

        prices = df[self.cfg.price_col].to_numpy(dtype=np.float64, copy=False)
        if "trades" in df.columns:
            trade_counts = pd.to_numeric(df["trades"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64, copy=False)
        elif "n_trades" in df.columns:
            trade_counts = pd.to_numeric(df["n_trades"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64, copy=False)
        else:
            trade_counts = np.full(len(df), np.nan, dtype=np.float64)
        volume_col = next((c for c in ("real_trade_volume", "volume", "size", "cum_volume") if c in df.columns), None)
        if volume_col is not None:
            trade_volume = pd.to_numeric(df[volume_col], errors="coerce").to_numpy(dtype=np.float64, copy=False)
        else:
            trade_volume = np.full(len(df), np.nan, dtype=np.float64)
        price_f32 = prices.astype(np.float32, copy=False)
        ret = np.empty_like(price_f32)
        ret[0] = 0.0
        ret[1:] = price_f32[1:] - price_f32[:-1]

        def roll_mean(x: np.ndarray, W: int) -> np.ndarray:
            return pd.Series(x).rolling(window=W, min_periods=W).mean().to_numpy(np.float32)

        def roll_abs_sum(x: np.ndarray, W: int) -> np.ndarray:
            return pd.Series(np.abs(x)).rolling(window=W, min_periods=W).sum().to_numpy(np.float32)

        def roll_up_ratio(x: np.ndarray, W: int) -> np.ndarray:
            return pd.Series((x > 0).astype(np.float32)).rolling(window=W, min_periods=W).mean().to_numpy(np.float32)

        def roll_std_ret(x: np.ndarray, W: int) -> np.ndarray:
            return pd.Series(x).rolling(window=W, min_periods=W).std(ddof=0).to_numpy(np.float32)

        def efficiency(price_arr: np.ndarray, returns_arr: np.ndarray, window: int) -> np.ndarray:
            move = np.abs(price_arr - np.roll(price_arr, window))
            path = roll_abs_sum(returns_arr, window)
            eff = move / np.maximum(path, 1e-6)
            eff[:window] = np.nan
            return eff.astype(np.float32)

        drift10 = roll_mean(ret, 10)
        drift30 = roll_mean(ret, 30)
        drift60 = roll_mean(ret, 60)

        vol10 = roll_std_ret(ret, 10)
        vol30 = roll_std_ret(ret, 30)
        vol60 = roll_std_ret(ret, 60)

        drift10_n = drift10 / np.maximum(vol10, 1e-6)
        drift30_n = drift30 / np.maximum(vol30, 1e-6)
        drift60_n = drift60 / np.maximum(vol60, 1e-6)

        up10 = roll_up_ratio(ret, 10)
        up30 = roll_up_ratio(ret, 30)
        up60 = roll_up_ratio(ret, 60)

        eff10 = efficiency(price_f32, ret, 10)
        eff30 = efficiency(price_f32, ret, 30)
        eff60 = efficiency(price_f32, ret, 60)

        vol_ratio_10_60 = vol10 / np.maximum(vol60, 1e-6)

        Xeng_trend = np.stack(
            [
                drift10, drift30, drift60,
                drift10_n, drift30_n, drift60_n,
                up10, up30, up60,
                eff10, eff30, eff60,
                vol10, vol30, vol60,
                vol_ratio_10_60,
            ],
            axis=1,
        ).astype(np.float32)

        Xs = np.concatenate([Xs_quant, Xeng_orig, Xeng_trend], axis=1).astype(np.float32, copy=False)
        Xe = df[emb_cols].to_numpy(dtype=np.float32, copy=False) if emb_cols else None

        m = np.isfinite(Xs).all(axis=1)
        if Xe is not None:
            m = m & np.isfinite(Xe).all(axis=1)
        m = m & np.isfinite(prices)

        Xs = Xs[m]
        Xe = Xe[m] if Xe is not None else None
        prices = prices[m]
        trade_counts = trade_counts[m]
        trade_volume = trade_volume[m]
        sidecar_price = sidecar_price[m]
        sidecar_ts = sidecar_ts[m]
        premarket_mask = premarket_mask[m]
        regime_flag = regime_flag[m]
        timestamps = timestamps[m]

        self.base_xs_dim = Xs.shape[1]
        self.edge_feat_index = base_quant_dim + 2

        ts_filtered = pd.Series(pd.to_datetime(timestamps, utc=True))
        if len(ts_filtered) > 0:
            filt_min = ts_filtered.dt.tz_convert(train_tz).min()
            filt_max = ts_filtered.dt.tz_convert(train_tz).max()
        else:
            filt_min = pd.NaT
            filt_max = pd.NaT

        self._episode_meta[path] = {
            "raw_rows": raw_rows,
            "rows_after_rl_valid": rows_after_rl_valid,
            "rows_after_time_filter": rows_after_time_filter,
            "rows_after_finite": int(len(prices)),
            "raw_ts_min_utc": raw_ts_min_utc,
            "raw_ts_max_utc": raw_ts_max_utc,
            "filtered_ts_min_train_tz": filt_min,
            "filtered_ts_max_train_tz": filt_max,
        }

        self._cache.put(path, (Xs, Xe, prices, premarket_mask, regime_flag, timestamps, trade_counts, trade_volume, sidecar_price, sidecar_ts))
        return Xs, Xe, prices, premarket_mask, regime_flag, timestamps, trade_counts, trade_volume, sidecar_price, sidecar_ts

    def obs_dim(self) -> int:
        if self.s_dim is not None:
            return int(self.s_dim)

        Xs, Xe, _, _, _, _, _, _, _, _ = self._load_episode_arrays(self.paths[0])
        extra_dim = 6 if self.cfg.include_close_feat else 5
        if str(getattr(self.cfg, "execution_mode", "")) == "pending_first_real_bar":
            extra_dim += 5
        nonemb_dim = Xs.shape[1] + extra_dim
        stack_dim = int(self.cfg.stack_len) * nonemb_dim
        emb_dim = 0 if Xe is None else Xe.shape[1]
        d = stack_dim + emb_dim

        self.s_dim = int(d)
        self.nonemb_dim = int(nonemb_dim)
        self.stack_dim = int(stack_dim)
        self.emb_dim = int(emb_dim)

        self.prev_u_idx_in_nonemb = Xs.shape[1] + 3
        self.prev_u_idx_in_obs = (int(self.cfg.stack_len) - 1) * nonemb_dim + self.prev_u_idx_in_nonemb
        return int(d)

    def reset(self, path: Optional[str] = None) -> np.ndarray:
        if path is None:
            path = self.paths[0]
        self.current_path = path

        Xs, Xe, prices, premarket_mask, regime_flags, timestamps, trade_counts, trade_volume, sidecar_price, sidecar_ts = self._load_episode_arrays(path)
        meta = self._episode_meta.get(path, {})
        print(
            f"[reset] episode_file={Path(path).name} rows={len(prices)} "
            f"regime_frac={float(regime_flags.mean()):.3f} "
            f"premarket_frac={float(premarket_mask.mean()):.3f} "
            f"train_time_filter={int(bool(getattr(self.cfg, 'train_time_filter', False)))} "
            f"train_window={int(getattr(self.cfg, 'train_start_hhmm', 700)):04d}-"
            f"{int(getattr(self.cfg, 'train_end_hhmm', 1100)):04d} "
            f"train_market_tz={getattr(self.cfg, 'train_market_tz', 'America/New_York')} "
            f"timestamp_col={getattr(self.cfg, 'timestamp_col', 'datetime_utc')} "
            f"raw_ts_utc={meta.get('raw_ts_min_utc')}..{meta.get('raw_ts_max_utc')} "
            f"filtered_ts_train_tz={meta.get('filtered_ts_min_train_tz')}..{meta.get('filtered_ts_max_train_tz')} "
            f"rows_raw={meta.get('raw_rows')} rows_rl_valid={meta.get('rows_after_rl_valid')} "
            f"rows_time={meta.get('rows_after_time_filter')} rows_finite={meta.get('rows_after_finite')}",
            flush=True,
        )
        n = len(prices)
        max_reward_H = max([1] + [int(h) for h in tuple(getattr(self.cfg, "reward_horizons", (1,)))])
        last_valid_obs = last_valid_observation_index(
            n_rows=n,
            execution_lag=0 if str(getattr(self.cfg, "execution_mode", "")) == "legacy_same_close" else 1,
            max_reward_horizon=max_reward_H,
            require_post_exec_mark=True,
        )
        if last_valid_obs < 0:
            raise ValueError(f"{path}: not enough rows")

        if self.cfg.segment_len and self.cfg.segment_len > 2:
            seg = int(self.cfg.segment_len)
            t0 = 0 if self.cfg.start_mode == "begin" else np.random.randint(0, max(1, min(last_valid_obs, n - seg - 1) + 1))
            self.t = int(t0)
            self.t_end = min(last_valid_obs + 1, t0 + seg)
        else:
            self.t = 0 if self.cfg.start_mode == "begin" else np.random.randint(0, max(1, last_valid_obs + 1))
            self.t_end = last_valid_obs + 1

        self.Xs, self.Xe, self.prices, self.timestamps = Xs, Xe, prices, timestamps
        self.trade_counts = trade_counts
        self.trade_volume = trade_volume
        self.sidecar_price = sidecar_price
        self.sidecar_ts = sidecar_ts
        self.premarket_mask = premarket_mask
        self.regime_flags = regime_flags
        self.shares = 0
        self.cash = 0.0
        self.prev_exposure = 0.0
        self.prev_trade_action = 0.0
        self.prev_u = 0.0
        self.last_exec_delta = 0
        self.hold_steps = 0
        self.prev_edge = 0.0
        self.pending_target_shares = 0
        self.pending_remaining_delta = 0
        self.pending_created_ts = None
        self.pending_eligible_ts = None
        self.pending_last_update_ts = None
        self.pending_replaced = False
        self.pending_expired = False
        self.pending_expiry_reason = ""
        self.pending_fill_fraction_last_step = 0.0
        self.stack = deque(maxlen=int(self.cfg.stack_len))
        self._cost_queue.clear()
        return self._get_obs()

    def _current_edge_value(self) -> float:
        if self.Xs is None or self.edge_feat_index is None:
            return 0.0
        if self.edge_feat_index >= self.Xs.shape[1]:
            return 0.0
        return float(self.Xs[self._market_feature_idx(self.t), self.edge_feat_index])

    def _get_obs(self) -> np.ndarray:
        nonemb = self._nonemb_feat()

        if self.stack is None:
            self.stack = deque(maxlen=int(self.cfg.stack_len))

        if len(self.stack) == 0:
            K = int(self.cfg.stack_len)
            t0 = self.t
            start = max(0, t0 - (K - 1))
            first = None
            for tt in range(start, t0 + 1):
                old_t = self.t
                self.t = tt
                v = self._nonemb_feat()
                self.t = old_t
                if first is None:
                    first = v
                self.stack.append(v)
            while len(self.stack) < K:
                self.stack.appendleft(first)
        else:
            self.stack.append(nonemb)

        stack_flat = np.concatenate(list(self.stack), axis=0).astype(np.float32)

        emb_parts = []
        if self.Xe is not None:
            emb_parts.append(self.Xe[self._market_feature_idx(self.t)].astype(np.float32))

        if emb_parts:
            return np.concatenate([stack_flat] + emb_parts).astype(np.float32)
        return stack_flat

    def _requested_delta_from_action(self, action: float, shares: int, price: float) -> Tuple[int, int, int]:
        max_shares_t = int(max(1, math.floor(float(self.cfg.max_notional) / max(float(price), 1e-12))))
        action = float(np.clip(float(action), -1.0, 1.0))
        if abs(action) < float(getattr(self.cfg, "deadband_u", 0.0)):
            action = 0.0
        requested_delta = int(round(action * max_shares_t))
        frac_deadband = float(getattr(self.cfg, "frac_deadband", 0.0))
        if frac_deadband > 0.0:
            band_shares = int(round(frac_deadband * max_shares_t))
            if abs(requested_delta) <= band_shares:
                requested_delta = 0
        requested_delta = int(np.clip(requested_delta, -int(self.cfg.max_delta_shares), int(self.cfg.max_delta_shares)))
        next_shares = int(shares) + requested_delta
        if next_shares < 0:
            requested_delta = -int(shares)
            next_shares = 0
        mode = str(getattr(self.cfg, "notional_cap_mode", "strict_position_cap"))
        if mode == "strict_position_cap":
            if next_shares > max_shares_t:
                requested_delta = max_shares_t - int(shares)
        elif mode == "buy_only_cap":
            if requested_delta > 0:
                requested_delta = min(requested_delta, max(0, max_shares_t - int(shares)))
            if int(shares) + requested_delta < 0:
                requested_delta = -int(shares)
        else:
            raise ValueError(f"Unknown notional_cap_mode={mode!r}")
        if abs(requested_delta) < int(self.cfg.deadband_shares):
            requested_delta = 0
        target_shares = max(0, int(shares) + int(requested_delta))
        return int(requested_delta), int(target_shares), int(max_shares_t)

    def _replace_pending_target(self, target_shares: int, bar_ts: Optional[pd.Timestamp], eligible_ts: Optional[pd.Timestamp]) -> Dict[str, Any]:
        old_target = int(self.pending_target_shares)
        old_remaining = int(self.pending_remaining_delta)
        new_remaining = int(target_shares) - int(self.shares)
        if abs(new_remaining) < int(self.cfg.deadband_shares):
            new_remaining = 0
        replaced = bool(old_remaining != new_remaining or old_target != int(target_shares))
        reason = "unchanged"
        if new_remaining == 0:
            reason = "cancel_flat_or_deadband" if old_remaining != 0 else "no_pending"
            self.pending_target_shares = int(self.shares)
            self.pending_remaining_delta = 0
            self.pending_created_ts = None
            self.pending_eligible_ts = None
        else:
            reason = "replace_latest_target" if old_remaining != 0 else "new_pending"
            self.pending_target_shares = int(target_shares)
            self.pending_remaining_delta = int(new_remaining)
            if replaced or self.pending_created_ts is None:
                self.pending_created_ts = bar_ts
                self.pending_eligible_ts = eligible_ts
            self.pending_last_update_ts = bar_ts
        self.pending_replaced = bool(replaced)
        self.pending_expired = False
        self.pending_expiry_reason = ""
        return {
            "pending_replaced": int(replaced),
            "pending_replacement_reason": reason,
            "pending_old_target_shares": int(old_target),
            "pending_new_target_shares": int(self.pending_target_shares),
            "pending_old_remaining_shares": int(old_remaining),
            "pending_new_remaining_shares": int(self.pending_remaining_delta),
        }

    def _execute_pending_at_bar(self, idx: int, p_mark: float) -> Dict[str, Any]:
        remaining_before = int(self.pending_remaining_delta)
        age_before = self._pending_age_seconds(idx)
        out = {
            "filled_delta_shares": 0,
            "fill_allowed": 0,
            "fill_price": float("nan"),
            "fill_source": "none",
            "fill_status": "no_pending" if remaining_before == 0 else "pending_no_real_trade",
            "trade_cost": 0.0,
            "effective_cost_per_share": 0.0,
            "cost_model": str(getattr(self.cfg, "cost_model", COST_MODE_FIXED_PER_SHARE)),
            "cost_price_fraction": float(getattr(self.cfg, "cost_price_fraction", 0.005)),
            "cash_before": float(self.cash),
            "cash_after": float(self.cash),
            "shares_before": int(self.shares),
            "shares_after": int(self.shares),
            "pending_remaining_before": int(remaining_before),
            "pending_remaining_after": int(remaining_before),
            "pending_age_before": float(age_before),
            "pending_age_after": float(age_before),
            "pending_expired": 0,
            "expiry_reason": "",
            "liquidity_confidence": "none",
            "real_trade_volume": float("nan"),
        }
        if remaining_before == 0:
            return out
        timeout = float(getattr(self.cfg, "pending_timeout_seconds", 900.0))
        if age_before >= timeout:
            self.pending_remaining_delta = 0
            self.pending_target_shares = int(self.shares)
            self.pending_expired = True
            self.pending_expiry_reason = "timeout"
            out.update({"fill_status": "expired_timeout", "pending_remaining_after": 0, "pending_expired": 1, "expiry_reason": "timeout"})
            return out
        real_count = float(self.trade_counts[idx]) if self.trade_counts is not None and 0 <= idx < len(self.trade_counts) else 0.0
        if not (np.isfinite(real_count) and real_count > 0.0):
            return out
        volume = float(self.trade_volume[idx]) if self.trade_volume is not None and 0 <= idx < len(self.trade_volume) else float("nan")
        participation = float(getattr(self.cfg, "pending_max_participation_rate", 0.05))
        if np.isfinite(volume) and volume > 0.0:
            max_fill = int(max(1, math.floor(volume * max(0.0, participation))))
            liquidity_confidence = "volume_participation"
        else:
            max_fill = abs(remaining_before)
            liquidity_confidence = "unknown_volume_full_remaining_fallback"
        max_fill = min(max_fill, abs(remaining_before), int(self.cfg.max_delta_shares))
        fill_delta = int(np.sign(remaining_before) * max_fill)
        if int(self.shares) + fill_delta < 0:
            fill_delta = -int(self.shares)
        p_fill = float(p_mark)
        if not (np.isfinite(p_fill) and p_fill > 0.0) or fill_delta == 0:
            return out
        slip_mult = 1.0 + (float(getattr(self.cfg, "slippage_bps", 0.0)) / 10000.0 if fill_delta > 0 else -float(getattr(self.cfg, "slippage_bps", 0.0)) / 10000.0)
        p_exec = float(p_fill * slip_mult)
        effective_cps = effective_trade_cost_per_share(self.cfg, p_exec)
        slippage_cost = float(abs(fill_delta) * abs(p_exec - p_fill))
        commission_cost = float(abs(fill_delta) * effective_cps)
        cost = float(commission_cost + slippage_cost)
        cash_before = float(self.cash)
        shares_before = int(self.shares)
        self.cash = float(self.cash - fill_delta * p_exec - commission_cost)
        self.shares = int(self.shares + fill_delta)
        self.pending_remaining_delta = int(remaining_before - fill_delta)
        if self.pending_remaining_delta == 0:
            self.pending_target_shares = int(self.shares)
            self.pending_created_ts = None
            self.pending_eligible_ts = None
            status = "filled_full"
        else:
            status = "filled_partial"
        self.pending_last_update_ts = self._timestamp_at(idx)
        self.pending_fill_fraction_last_step = float(abs(fill_delta) / max(1, abs(remaining_before)))
        out.update({
            "filled_delta_shares": int(fill_delta),
            "fill_allowed": 1,
            "fill_price": float(p_exec),
            "fill_source": "first_real_bar_close_approximation",
            "fill_status": status,
            "trade_cost": float(cost),
            "effective_cost_per_share": float(effective_cps),
            "cost_model": str(getattr(self.cfg, "cost_model", COST_MODE_FIXED_PER_SHARE)),
            "cost_price_fraction": float(getattr(self.cfg, "cost_price_fraction", 0.005)),
            "cash_before": float(cash_before),
            "cash_after": float(self.cash),
            "shares_before": int(shares_before),
            "shares_after": int(self.shares),
            "pending_remaining_after": int(self.pending_remaining_delta),
            "pending_age_after": float(self._pending_age_seconds(idx)),
            "liquidity_confidence": liquidity_confidence,
            "real_trade_volume": float(volume),
        })
        return out

    def _step_pending_first_real_bar(self, u_signed: float, return_info: bool = False):
        assert self.Xs is not None and self.prices is not None
        obs_idx = int(self.t)
        exec_idx = min(obs_idx + 1, len(self.prices) - 1)
        p_obs = float(self.prices[obs_idx])
        p_mark = float(self.prices[exec_idx])
        raw_u = float(np.clip(float(u_signed), -1.0, 1.0))
        intent_u = raw_u
        regime_on = self._regime_filter_on()
        if bool(getattr(self.cfg, "use_regime_filter", False)) and not regime_on:
            intent_u = float(np.clip(float(getattr(self.cfg, "regime_flat_action", 0.0)), -1.0, 1.0))
        is_premarket = self._is_premarket_now()
        shares_pre = int(self.shares)
        cash_pre = float(self.cash)
        pending_before = int(self.pending_remaining_delta)
        pending_age_before = float(self._pending_age_seconds(obs_idx))
        pending_created_ts_before = self.pending_created_ts
        max_shares_obs = int(max(1, math.floor(float(self.cfg.max_notional) / max(p_obs, 1e-12))))
        exposure_pre = float(np.clip(shares_pre / max(1, max_shares_obs), 0.0, 1.0))

        fill = self._execute_pending_at_bar(exec_idx, p_mark)
        filled_delta = int(fill["filled_delta_shares"])
        shares_after_fill = int(self.shares)
        cash_after_fill = float(self.cash)
        executed_u = float(np.clip(filled_delta / max(1, max_shares_obs), -1.0, 1.0))

        requested_delta, requested_target, _ = self._requested_delta_from_action(intent_u, int(self.shares), p_obs)
        bar_ts = self._timestamp_at(obs_idx)
        eligible_ts = None if bar_ts is None else bar_ts + pd.Timedelta(seconds=int(getattr(self.cfg, "bar_seconds", 1)))
        replacement = self._replace_pending_target(requested_target, bar_ts, eligible_ts)

        next_hold_steps = 0 if shares_after_fill == 0 or shares_after_fill != shares_pre else self.hold_steps + 1
        horizons = tuple(int(h) for h in self.cfg.reward_horizons)
        weights = tuple(float(w) for w in self.cfg.reward_weights)
        if len(horizons) != len(weights):
            raise ValueError("reward_horizons and reward_weights must have same length")
        pnl = 0.0
        for H, W in zip(horizons, weights):
            t_future = min(exec_idx + int(H), len(self.prices) - 1)
            pnl += float(W) * float(shares_after_fill) * float(self.prices[t_future] - p_mark)
        raw_cost = float(fill["trade_cost"])
        cost_paid = raw_cost
        pos_frac = float(np.clip((shares_after_fill * p_mark) / max(self.cfg.max_notional, 1e-12), 0.0, self.cfg.pos_frac_clip))
        turnover_pen = float(self.cfg.lam_turnover) * (abs(filled_delta) / max(1, max_shares_obs))
        dact_pen = float(self.cfg.lam_dact) * abs(executed_u - float(self.prev_trade_action))
        inv_pen = float(self.cfg.lam_inv) * (pos_frac ** 2)
        edge_now = float(self._current_edge_value())
        edge_prev = float(self.prev_edge)
        edge_decay = max(0.0, edge_prev - edge_now)
        age_pen = float(getattr(self.cfg, "lam_hold_age", 0.0)) * (float(next_hold_steps) / max(1.0, float(self.cfg.hold_age_scale))) * pos_frac if shares_after_fill > 0 else 0.0
        stale_pos_pen = float(getattr(self.cfg, "lam_stale_pos", 0.0)) * pos_frac if shares_after_fill > 0 and shares_after_fill == shares_pre else 0.0
        stale_edge_pen = float(getattr(self.cfg, "lam_stale_edge", 0.0)) * pos_frac * edge_decay if shares_after_fill > 0 else 0.0
        net = float(pnl - cost_paid)
        reward_base = (net / max(self.cfg.max_notional, 1e-12)) * float(self.cfg.reward_scale)
        reward = reward_base - turnover_pen - dact_pen - inv_pen - age_pen - stale_pos_pen - stale_edge_pen

        self.last_exec_delta = int(filled_delta)
        self.prev_exposure = float(np.clip(self.shares / max(1, max_shares_obs), 0.0, 1.0))
        self.prev_trade_action = float(executed_u)
        self.prev_u = float(executed_u)
        self.hold_steps = int(next_hold_steps)
        self.prev_edge = float(edge_now)
        self.t += 1
        done = self.t >= self.t_end

        liq_exec = 0
        liq_cost = 0.0
        liquidation_price = float("nan")
        expiry_reason = str(fill.get("expiry_reason", ""))
        shares_post_preliq = int(self.shares)
        if done:
            if int(self.pending_remaining_delta) != 0:
                self.pending_remaining_delta = 0
                self.pending_target_shares = int(self.shares)
                self.pending_expired = True
                self.pending_expiry_reason = "terminal"
                expiry_reason = "terminal"
            if self.shares != 0:
                liq_exec = -int(self.shares)
                liquidation_price = float(p_mark)
                liq_cps = effective_liquidation_cost_per_share(self.cfg, liquidation_price)
                liq_cost = liq_cps * abs(liq_exec)
                self.cash += float(self.shares) * liquidation_price - liq_cost
                self.shares = 0
                self.prev_exposure = 0.0
                self.last_exec_delta = liq_exec
        total_cost_step = float(cost_paid + liq_cost)
        equity_before_step = float(cash_pre + shares_pre * p_obs)
        equity_after_step = float(self.cash + int(self.shares) * p_mark)
        real_net_step = float(equity_after_step - equity_before_step)
        real_pnl_step = compute_gross_pnl_from_equity_delta(
            equity_before=equity_before_step,
            equity_after=equity_after_step,
            real_cost_step=total_cost_step,
        )
        obs = self._get_obs()

        info = {}
        if return_info:
            exposure_final = float(np.clip(int(self.shares) / max(1, max_shares_obs), 0.0, 1.0))
            ts0 = self._timestamp_at(obs_idx)
            ts1 = self._timestamp_at(exec_idx)
            info = {
                "t": obs_idx, "t_next": self.t, "t_decision": obs_idx, "t_pnl": exec_idx,
                "p": p_obs, "p_next": p_mark, "dp": float(p_mark - p_obs),
                "shares_pre": int(shares_pre), "shares_post": int(shares_post_preliq),
                "shares_post_preliq": int(shares_post_preliq), "shares_final": int(self.shares),
                "shares_for_pnl": int(shares_after_fill), "target_shares": int(requested_target),
                "exec": int(filled_delta), "liq_exec": int(liq_exec), "policy_exec_delta": int(filled_delta),
                "total_exec_delta_for_accounting": int(filled_delta + liq_exec),
                "desired_delta": int(requested_delta), "exec_delta_pre_cap": int(requested_delta),
                "max_shares_t": int(max_shares_obs), "notional_cap_mode": str(self.cfg.notional_cap_mode),
                "cur_shares": int(shares_pre), "buy_room": int(max(0, max_shares_obs - shares_pre)),
                "position_notional_before": float(shares_pre * p_obs), "position_notional_after": float(shares_post_preliq * p_mark),
                "above_max_notional_before": bool(shares_pre * p_obs > float(self.cfg.max_notional)),
                "above_max_notional_after": bool(shares_post_preliq * p_mark > float(self.cfg.max_notional)),
                "blocked_by_notional_cap": False,
                "intent_u": float(intent_u), "raw_u": float(raw_u), "u": float(intent_u),
                "executed_u": float(executed_u), "executed_u_policy": float(executed_u),
                "execution_clip_error": float(intent_u - executed_u),
                "exposure_pre": float(exposure_pre), "exposure_final": float(exposure_final),
                "hold_steps": int(next_hold_steps), "edge_now": float(edge_now), "edge_prev": float(edge_prev), "edge_decay": float(edge_decay),
                "pnl": float(pnl), "cost_paid": float(cost_paid), "raw_cost": float(raw_cost),
                "effective_cost_per_share": float(fill.get("effective_cost_per_share", 0.0)),
                "cost_model": str(getattr(self.cfg, "cost_model", COST_MODE_FIXED_PER_SHARE)),
                "cost_price_fraction": float(getattr(self.cfg, "cost_price_fraction", 0.005)),
                "remaining_cost_flush": 0.0, "liq_cost": float(liq_cost), "cost": float(total_cost_step), "net": float(net),
                "trade_cost_step": float(cost_paid), "liquidation_cost_step": float(liq_cost),
                "real_pnl_step": float(real_pnl_step), "real_cost_step": float(total_cost_step), "real_net_step": float(real_net_step),
                "reward_total": float(reward), "reward_base": float(reward_base),
                "flat_bonus_term": 0.0, "turnover_pen": float(turnover_pen), "dact_pen": float(dact_pen), "inv_pen": float(inv_pen),
                "age_pen": float(age_pen), "stale_pos_pen": float(stale_pos_pen), "stale_edge_pen": float(stale_edge_pen),
                "invalid_sell_attempt": bool(shares_pre <= 0 and intent_u < 0 and requested_delta < 0),
                "invalid_overbuy_attempt": bool(intent_u > 0 and requested_delta > max(0, max_shares_obs - shares_pre)),
                "overbuy_excess_shares": int(max(0, requested_delta - max(0, max_shares_obs - shares_pre))),
                "regime_on": int(regime_on), "premarket_on": int(is_premarket), "done": bool(done),
                "timestamp_decision": str(ts0 or ""), "timestamp_pnl": str(ts1 or ""),
                "timestamp_bar_start": str(ts0 or ""), "timestamp_execution": str(ts1 or ""), "timestamp_mark": str(ts1 or ""),
                "p_obs": float(p_obs), "p_exec_raw": float(fill["fill_price"] if int(fill["fill_allowed"]) else p_mark),
                "p_exec_after_slippage": float(fill["fill_price"] if int(fill["fill_allowed"]) else p_mark), "p_mark": float(p_mark),
                "execution_price_source": str(fill["fill_source"]), "fill_allowed": int(fill["fill_allowed"]),
                "fill_status": str(fill["fill_status"]), "synthetic_execution": 0,
                "cash_before": float(cash_pre), "cash_after": float(self.cash),
                "shares_before": int(shares_pre), "shares_after": int(self.shares),
                "inventory_fraction_before": float(exposure_pre), "inventory_fraction_after": float(exposure_final),
                "transaction_notional": float(abs(filled_delta) * p_mark),
                "commission_cost": float(abs(filled_delta) * float(fill.get("effective_cost_per_share", 0.0))),
                "slippage_cost": float(max(0.0, raw_cost - abs(filled_delta) * float(fill.get("effective_cost_per_share", 0.0)))),
                "equity_before": float(equity_before_step),
                "equity_after": float(equity_after_step),
                "requested_action_signed": float(intent_u), "requested_target_shares": int(requested_target),
                "requested_delta_shares": int(requested_delta), "executed_delta_shares": int(filled_delta),
                "executed_action_signed": float(executed_u),
                "pending_exists_before": int(pending_before != 0), "pending_side_before": int(np.sign(pending_before)) if pending_before else 0,
                "pending_remaining_before": int(pending_before), "pending_age_before": float(pending_age_before),
                "pending_remaining_after": int(self.pending_remaining_delta), "pending_age_after": float(self._pending_age_seconds()),
                "pending_created_ts": str(pending_created_ts_before or self.pending_created_ts or ""),
                "pending_eligible_ts": str(self.pending_eligible_ts or ""), "pending_last_update_ts": str(self.pending_last_update_ts or ""),
                "pending_timeout_seconds": float(getattr(self.cfg, "pending_timeout_seconds", 900.0)),
                "pending_replaced": int(replacement["pending_replaced"]),
                "pending_replacement_reason": str(replacement["pending_replacement_reason"]),
                "pending_expired": int(bool(self.pending_expired or fill.get("pending_expired", 0))),
                "expiry_reason": str(expiry_reason or self.pending_expiry_reason),
                "actual_shares_after": int(self.shares),
                "actual_inventory_fraction_after": float(exposure_final),
                "actual_notional_fraction_after": float((int(self.shares) * p_mark) / max(float(self.cfg.max_notional), 1e-12)),
                "fill_price": float(fill["fill_price"]), "fill_source": str(fill["fill_source"]),
                "real_trade_volume": float(fill.get("real_trade_volume", float("nan"))),
            }
        return obs, reward, done, filled_delta, pnl, total_cost_step, net, info

    def step(self, u_signed: float, return_info: bool = False):
        assert self.Xs is not None and self.prices is not None

        obs_idx = int(self.t)
        mode_exec = str(getattr(self.cfg, "execution_mode", "next_close_real_bar"))
        if mode_exec == "pending_first_real_bar":
            return self._step_pending_first_real_bar(u_signed, return_info=return_info)
        raw_u = float(np.clip(float(u_signed), -1.0, 1.0))
        intent_u = float(raw_u)
        regime_on = self._regime_filter_on()
        if bool(getattr(self.cfg, "use_regime_filter", False)) and not regime_on:
            intent_u = float(np.clip(float(getattr(self.cfg, "regime_flat_action", 0.0)), -1.0, 1.0))
        is_premarket = self._is_premarket_now()

        if mode_exec == "first_post_bar_trade_sidecar":
            exec_idx = obs_idx + 1
            mark_idx = obs_idx + 2
            if mark_idx >= len(self.prices):
                raise IndexError(f"not enough rows for sidecar v4 execution: obs_idx={obs_idx}")
            side_px = float(self.sidecar_price[obs_idx]) if self.sidecar_price is not None else float("nan")
            fill_allowed = bool(np.isfinite(side_px) and side_px > 0.0)
            exec_source = ExecutionSource(
                mode=mode_exec,
                obs_idx=obs_idx,
                next_obs_idx=obs_idx + 1,
                exec_idx=exec_idx,
                mark_idx=mark_idx,
                p_obs=float(self.prices[obs_idx]),
                p_exec_raw=side_px if fill_allowed else float(self.prices[exec_idx]),
                p_mark=float(self.prices[mark_idx]),
                fill_allowed=fill_allowed,
                fill_status="fillable" if fill_allowed else str(getattr(self.cfg, "missing_execution_policy", "no_fill")),
                synthetic_execution=False,
                execution_price_source="first_post_bar_trade_sidecar",
                timestamp_execution=str(self.sidecar_ts[obs_idx]) if self.sidecar_ts is not None else "",
            )
        else:
            exec_source = resolve_dense_execution_source(
                self.prices,
                obs_idx=obs_idx,
                mode=mode_exec,
                trades=self.trade_counts,
                allow_synthetic_execution_proxy=bool(getattr(self.cfg, "allow_synthetic_execution_proxy", False)),
                allow_legacy_same_close=bool(getattr(self.cfg, "allow_legacy_same_close", False)),
            )

        shares_pre = int(self.shares)
        p_obs = float(exec_source.p_obs)
        p_exec = float(exec_source.p_exec_raw)
        p_mark = float(exec_source.p_mark)
        acc = compute_execution_accounting(
            ExecutionAccountingInput(
                cash_before=float(self.cash),
                shares_before=shares_pre,
                observation_mark=p_obs,
                execution_price_raw=p_exec,
                next_mark_price=p_mark,
                executable_action=float(intent_u),
                max_notional=float(self.cfg.max_notional),
                max_delta_shares=int(self.cfg.max_delta_shares),
                deadband_u=float(getattr(self.cfg, "deadband_u", 0.0)),
                deadband_shares=int(self.cfg.deadband_shares),
                frac_deadband=float(getattr(self.cfg, "frac_deadband", 0.0)),
                cost_per_share=float(self.cfg.cost_per_share),
                cost_model=str(getattr(self.cfg, "cost_model", COST_MODE_FIXED_PER_SHARE)),
                cost_price_fraction=float(getattr(self.cfg, "cost_price_fraction", 0.005)),
                cost_min_per_share=float(getattr(self.cfg, "cost_min_per_share", 0.0)),
                cost_max_per_share=float(getattr(self.cfg, "cost_max_per_share", 0.0)),
                slippage_bps=float(getattr(self.cfg, "slippage_bps", 0.0)),
                long_only=True,
                fill_allowed=bool(exec_source.fill_allowed),
                notional_cap_mode=str(getattr(self.cfg, "notional_cap_mode", "strict_position_cap")),
            )
        )
        max_shares_t = int(acc.max_shares_exec)
        mode = str(getattr(self.cfg, "notional_cap_mode", "strict_position_cap"))
        desired_delta = int(acc.requested_delta)
        exec_delta_pre_cap = int(np.clip(desired_delta, -self.cfg.max_delta_shares, self.cfg.max_delta_shares))
        policy_exec_delta = int(acc.executed_delta)
        exec_delta = int(acc.executed_delta)
        next_shares = int(acc.shares_after)
        executed_u = float(np.clip(policy_exec_delta / max(1, max_shares_t), -1.0, 1.0))
        execution_clip_error = float(intent_u - executed_u)
        buy_room = max(0, max_shares_t - shares_pre)
        invalid_sell_attempt = bool(shares_pre <= 0 and intent_u < 0.0 and desired_delta < 0)
        invalid_overbuy_attempt = bool(intent_u > 0.0 and desired_delta > buy_room)
        overbuy_excess_shares = int(max(0, desired_delta - buy_room)) if invalid_overbuy_attempt else 0
        blocked_by_notional_cap = bool(exec_delta_pre_cap != policy_exec_delta)
        position_notional_before = float(shares_pre * p_obs)
        position_notional_after = float(next_shares * float(acc.execution_price_after_slippage))
        above_max_notional_before = bool(position_notional_before > float(self.cfg.max_notional))
        above_max_notional_after = bool(position_notional_after > float(self.cfg.max_notional))

        next_hold_steps = 0 if next_shares == 0 or next_shares != shares_pre else self.hold_steps + 1
        horizons = tuple(int(h) for h in self.cfg.reward_horizons)
        weights = tuple(float(w) for w in self.cfg.reward_weights)
        if len(horizons) != len(weights):
            raise ValueError("reward_horizons and reward_weights must have same length")
        exec_anchor_idx = int(exec_source.exec_idx)
        reward_end_indices = [reward_endpoint_index(exec_anchor_idx, H) for H in horizons]
        if reward_end_indices and max(reward_end_indices) >= len(self.prices):
            raise IndexError(f"reward endpoint exceeds prices: obs_idx={obs_idx} endpoints={reward_end_indices}")

        pnl = 0.0
        for H, W in zip(horizons, weights):
            t_future = reward_endpoint_index(exec_anchor_idx, H)
            pnl += W * float(next_shares) * float(self.prices[t_future] - p_exec)
        kappa = float(getattr(self.cfg, "reward_kappa", 0.0))
        if kappa != 0.0:
            Hk = horizons[-1]
            t_future = reward_endpoint_index(exec_anchor_idx, Hk)
            pnl += kappa * float(next_shares) * np.sign(float(self.prices[t_future] - p_exec))

        raw_cost = float(acc.commission_cost + acc.slippage_cost)
        H_cost = int(getattr(self.cfg, "cost_amort_steps", 0) or 0)
        if raw_cost > 0.0:
            if H_cost > 1:
                pay = raw_cost / float(H_cost)
                for _ in range(H_cost):
                    self._cost_queue.append(pay)
            else:
                self._cost_queue.append(raw_cost)
        cost_paid = float(self._cost_queue.popleft()) if len(self._cost_queue) else 0.0

        flat_bonus = float(getattr(self.cfg, "flat_bonus", 0.0))
        flat_bonus_term = float(flat_bonus if (abs(executed_u) < 0.02) else 0.0)
        turnover_pen = float(self.cfg.lam_turnover) * (abs(exec_delta) / max(1, max_shares_t))
        prev_a = float(self.prev_trade_action)
        a_change = abs(executed_u - prev_a)
        dact_pen = float(self.cfg.lam_dact) * a_change
        pos_frac = (float(next_shares) * float(acc.execution_price_after_slippage)) / max(self.cfg.max_notional, 1e-12)
        pos_frac = float(np.clip(pos_frac, 0.0, self.cfg.pos_frac_clip))
        inv_pen = float(self.cfg.lam_inv) * (pos_frac ** 2)
        age_scale = max(1.0, float(getattr(self.cfg, "hold_age_scale", 300)))
        lam_hold_age = float(getattr(self.cfg, "lam_hold_age", 0.0))
        age_pen = lam_hold_age * (float(next_hold_steps) / age_scale) * pos_frac if lam_hold_age > 0.0 and next_shares > 0 else 0.0
        lam_stale_pos = float(getattr(self.cfg, "lam_stale_pos", 0.0))
        same_pos = (next_shares > 0 and next_shares == shares_pre)
        stale_pos_pen = lam_stale_pos * pos_frac if lam_stale_pos > 0.0 and same_pos else 0.0
        edge_now = float(self._current_edge_value())
        edge_prev = float(self.prev_edge)
        edge_decay = max(0.0, edge_prev - edge_now)
        lam_stale_edge = float(getattr(self.cfg, "lam_stale_edge", 0.0))
        stale_edge_pen = lam_stale_edge * pos_frac * edge_decay if lam_stale_edge > 0.0 and next_shares > 0 else 0.0

        exposure_pre = safe_div(shares_pre, max_shares_t)
        exposure_post_preliq = safe_div(next_shares, max_shares_t)
        self.cash = float(acc.cash_after)
        self.shares = int(next_shares)
        self.last_exec_delta = int(exec_delta)
        self.prev_exposure = float(np.clip(self.shares / max(1, max_shares_t), 0.0, 1.0))
        self.prev_trade_action = float(executed_u)
        self.prev_u = float(executed_u)
        self.hold_steps = int(next_hold_steps)
        self.prev_edge = float(edge_now)
        self.t += 1
        done = self.t >= self.t_end

        remaining_cost_flush = float(sum(self._cost_queue)) if done and len(self._cost_queue) else 0.0
        if remaining_cost_flush:
            self._cost_queue.clear()
        liq_cost = 0.0
        liq_exec = 0
        liq_turn_pen = 0.0
        liquidation_price = float("nan")
        if done and self.shares != 0:
            liq_exec = -int(self.shares)
            liquidation_price = p_mark
            liq_cps = effective_liquidation_cost_per_share(self.cfg, liquidation_price)
            liq_cost = liq_cps * abs(liq_exec)
            liq_turn_pen = float(self.cfg.lam_turnover) * (abs(liq_exec) / max(1, max_shares_t))
            self.cash += float(self.shares) * liquidation_price - liq_cost
            self.shares = 0
            self.prev_exposure = 0.0
            self.last_exec_delta = liq_exec

        total_cost_step = float(cost_paid + remaining_cost_flush + liq_cost)
        total_turn_pen = float(turnover_pen + liq_turn_pen)
        net = float(pnl - total_cost_step)
        reward_base = (net / max(self.cfg.max_notional, 1e-12)) * float(self.cfg.reward_scale)
        reward = reward_base + flat_bonus_term - total_turn_pen - dact_pen - inv_pen - age_pen - stale_pos_pen - stale_edge_pen

        obs = self._get_obs()
        exposure_final = safe_div(self.shares, max_shares_t)

        info = {}
        if return_info:
            def _ts(idx: int) -> str:
                if self.timestamps is None or idx < 0 or idx >= len(self.timestamps):
                    return ""
                return pd.Timestamp(self.timestamps[idx]).isoformat()
            timestamp_decision = _ts(obs_idx)
            timestamp_pnl = _ts(exec_source.mark_idx)
            bar_end = ""
            if self.timestamps is not None:
                bar_end = (pd.Timestamp(self.timestamps[obs_idx]) + pd.Timedelta(seconds=int(getattr(self.cfg, "bar_seconds", 1)))).isoformat()
            info = {
                "t": obs_idx,
                "t_next": int(exec_source.next_obs_idx),
                "t_decision": obs_idx,
                "t_pnl": int(exec_source.mark_idx),
                "timestamp_decision": timestamp_decision,
                "timestamp_pnl": timestamp_pnl,
                "p": p_obs,
                "p_next": p_mark,
                "dp": float(p_mark - p_exec),

                "shares_pre": shares_pre,
                "exec": int(exec_delta),
                "shares_post": int(next_shares),
                "shares_for_pnl": int(next_shares),
                "target_shares": int(next_shares),
                "shares_post_preliq": int(next_shares),
                "shares_final": int(self.shares),
                "liq_exec": int(liq_exec),
                "policy_exec_delta": int(policy_exec_delta),
                "total_exec_delta_for_accounting": int(policy_exec_delta + liq_exec),
                "max_shares_t": int(max_shares_t),
                "notional_cap_mode": str(mode),
                "cur_shares": int(shares_pre),
                "buy_room": int(buy_room),
                "position_notional_before": float(position_notional_before),
                "position_notional_after": float(position_notional_after),
                "above_max_notional_before": bool(above_max_notional_before),
                "above_max_notional_after": bool(above_max_notional_after),
                "blocked_by_notional_cap": bool(blocked_by_notional_cap),
                "desired_delta": int(desired_delta),
                "exec_delta_pre_cap": int(exec_delta_pre_cap),
                "exec_delta": int(policy_exec_delta),
                "execution_clip_error": float(execution_clip_error),
                "invalid_sell_attempt": bool(invalid_sell_attempt),
                "invalid_overbuy_attempt": bool(invalid_overbuy_attempt),
                "overbuy_excess_shares": int(overbuy_excess_shares),

                "exposure_pre": float(exposure_pre),
                "exposure_post_preliq": float(exposure_post_preliq),
                "exposure_final": float(exposure_final),

                "u_signed": float(intent_u),
                "raw_u": float(raw_u),
                "intent_u": float(intent_u),
                "executed_u": float(executed_u),
                "executed_u_policy": float(executed_u),
                "prev_u_signed": float(prev_a),
                "u_change": float(a_change),

                "pnl": float(pnl),
                "cost": float(total_cost_step),
                "cost_paid": float(cost_paid),
                "raw_cost": float(raw_cost),
                "effective_cost_per_share": float(acc.effective_cost_per_share),
                "cost_model": str(getattr(self.cfg, "cost_model", COST_MODE_FIXED_PER_SHARE)),
                "cost_price_fraction": float(getattr(self.cfg, "cost_price_fraction", 0.005)),
                "remaining_cost_flush": float(remaining_cost_flush),
                "liq_cost": float(liq_cost),
                "net": float(net),

                "reward": float(reward),
                "reward_total": float(reward),
                "reward_base": float(reward_base),
                "reward_base_pnl_scaled": float((pnl / max(self.cfg.max_notional, 1e-12)) * float(self.cfg.reward_scale)),
                "reward_base_cost_scaled": float((total_cost_step / max(self.cfg.max_notional, 1e-12)) * float(self.cfg.reward_scale)),
                "flat_bonus_term": float(flat_bonus_term),
                "turnover_pen": float(total_turn_pen),
                "dact_pen": float(dact_pen),
                "inv_pen": float(inv_pen),
                "age_pen": float(age_pen),
                "stale_pos_pen": float(stale_pos_pen),
                "stale_edge_pen": float(stale_edge_pen),

                "hold_steps": int(next_hold_steps),
                "edge_now": float(edge_now),
                "edge_prev": float(edge_prev),
                "edge_decay": float(edge_decay),

                "regime_on": int(regime_on),
                "premarket_on": int(is_premarket),
                "done": bool(done),
                "obs_idx": int(obs_idx),
                "next_obs_idx": int(exec_source.next_obs_idx),
                "exec_idx": int(exec_source.exec_idx),
                "mark_idx": int(exec_source.mark_idx),
                "reward_end_indices": ",".join(str(int(x)) for x in reward_end_indices),
                "timestamp_bar_start": str(timestamp_decision),
                "timestamp_bar_end": str(bar_end),
                "timestamp_observation_available": str(bar_end),
                "timestamp_execution": str(exec_source.timestamp_execution or _ts(exec_source.exec_idx)),
                "timestamp_mark": str(_ts(exec_source.mark_idx)),
                "p_obs": float(p_obs),
                "p_exec_raw": float(acc.execution_price_raw),
                "p_exec_after_slippage": float(acc.execution_price_after_slippage),
                "p_mark": float(p_mark),
                "execution_price_source": str(exec_source.execution_price_source),
                "fill_allowed": int(bool(exec_source.fill_allowed)),
                "fill_status": str(exec_source.fill_status if exec_delta == 0 and not exec_source.fill_allowed else acc.fill_status),
                "synthetic_execution": int(bool(exec_source.synthetic_execution)),
                "cash_before": float(acc.equity_before - shares_pre * p_obs),
                "cash_after": float(self.cash),
                "shares_before": int(shares_pre),
                "shares_after": int(self.shares),
                "inventory_fraction_before": float(acc.inventory_fraction_before),
                "inventory_fraction_after": float(acc.inventory_fraction_after),
                "transaction_notional": float(acc.transaction_notional),
                "commission_cost": float(acc.commission_cost),
                "executed_notional": float(abs(exec_delta) * acc.execution_price_after_slippage),
                "slippage_cost": float(acc.slippage_cost),
                "pre_execution_gap_pnl": float(acc.pre_execution_gap_pnl),
                "post_execution_pnl": float(acc.post_execution_pnl),
                "equity_before": float(acc.equity_before),
                "equity_after": float(acc.equity_after),
                "real_pnl_step": float(acc.real_pnl_step),
                "real_cost_step": float(acc.real_cost_step + liq_cost),
                "real_net_step": float(acc.real_net_step - liq_cost),
                "trade_cost_step": float(acc.commission_cost + acc.slippage_cost),
                "liquidation_cost_step": float(liq_cost),
                "execution_mode": str(mode_exec),
                "timing_version": str(getattr(self.cfg, "reward_timing_version", "post_exec_elapsed_horizon_v1")),
                "liquidation_price": float(liquidation_price),
            }

        return obs, reward, done, exec_delta, pnl, total_cost_step, net, info


def resolve_liquidation_cost_model(env_cfg: EnvCfg) -> str:
    mode = str(getattr(env_cfg, "liquidation_cost_model", COST_MODE_INHERIT) or COST_MODE_INHERIT)
    if mode == COST_MODE_INHERIT:
        return str(getattr(env_cfg, "cost_model", COST_MODE_FIXED_PER_SHARE) or COST_MODE_FIXED_PER_SHARE)
    if mode not in COST_MODE_VALUES:
        raise ValueError(f"Unknown liquidation_cost_model={mode!r}; expected one of {LIQUIDATION_COST_MODE_VALUES}")
    return mode


def effective_trade_cost_per_share(env_cfg: EnvCfg, fill_price: float) -> float:
    return compute_effective_cost_per_share(
        fill_price=float(fill_price),
        cost_model=str(getattr(env_cfg, "cost_model", COST_MODE_FIXED_PER_SHARE)),
        fixed_cost_per_share=float(getattr(env_cfg, "cost_per_share", 0.0)),
        cost_price_fraction=float(getattr(env_cfg, "cost_price_fraction", 0.005)),
        cost_min_per_share=float(getattr(env_cfg, "cost_min_per_share", 0.0)),
        cost_max_per_share=float(getattr(env_cfg, "cost_max_per_share", 0.0)),
    )


def effective_liquidation_cost_per_share(env_cfg: EnvCfg, fill_price: float) -> float:
    fixed = float(
        getattr(env_cfg, "cost_per_share", 0.0)
        if getattr(env_cfg, "liquidation_cost_per_share", None) is None
        else getattr(env_cfg, "liquidation_cost_per_share")
    )
    return compute_effective_cost_per_share(
        fill_price=float(fill_price),
        cost_model=resolve_liquidation_cost_model(env_cfg),
        fixed_cost_per_share=fixed,
        cost_price_fraction=float(getattr(env_cfg, "cost_price_fraction", 0.005)),
        cost_min_per_share=float(getattr(env_cfg, "cost_min_per_share", 0.0)),
        cost_max_per_share=float(getattr(env_cfg, "cost_max_per_share", 0.0)),
    )


def compute_real_accounting_step(
    shares_post_preliq: int,
    dp_1step: float,
    policy_exec_delta: int,
    liq_exec: int,
    cost_per_share: float,
    liquidation_cost_per_share: float,
    fill_price: Optional[float] = None,
    liquidation_fill_price: Optional[float] = None,
    cost_model: str = COST_MODE_FIXED_PER_SHARE,
    cost_price_fraction: float = 0.005,
    cost_min_per_share: float = 0.0,
    cost_max_per_share: float = 0.0,
    liquidation_cost_model: str = COST_MODE_INHERIT,
) -> Dict[str, float]:
    real_pnl_step = float(shares_post_preliq) * float(dp_1step)
    trade_cps = (
        compute_effective_cost_per_share(
            fill_price=float(fill_price),
            cost_model=str(cost_model),
            fixed_cost_per_share=float(cost_per_share),
            cost_price_fraction=float(cost_price_fraction),
            cost_min_per_share=float(cost_min_per_share),
            cost_max_per_share=float(cost_max_per_share),
        )
        if int(policy_exec_delta) != 0
        else 0.0
    )
    liq_model = str(cost_model if str(liquidation_cost_model) == COST_MODE_INHERIT else liquidation_cost_model)
    liq_cps = (
        compute_effective_cost_per_share(
            fill_price=float(liquidation_fill_price),
            cost_model=liq_model,
            fixed_cost_per_share=float(liquidation_cost_per_share),
            cost_price_fraction=float(cost_price_fraction),
            cost_min_per_share=float(cost_min_per_share),
            cost_max_per_share=float(cost_max_per_share),
        )
        if int(liq_exec) != 0
        else 0.0
    )
    trade_cost_step = float(trade_cps) * abs(int(policy_exec_delta))
    liquidation_cost_step = float(liq_cps) * abs(int(liq_exec))
    real_cost_step = trade_cost_step + liquidation_cost_step
    real_net_step = real_pnl_step - real_cost_step
    return {
        "real_pnl_step": float(real_pnl_step),
        "trade_cost_step": float(trade_cost_step),
        "liquidation_cost_step": float(liquidation_cost_step),
        "real_cost_step": float(real_cost_step),
        "real_net_step": float(real_net_step),
    }
