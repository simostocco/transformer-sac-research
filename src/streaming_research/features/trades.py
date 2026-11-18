"""Extracted research definitions; see docs/source_map.json for provenance."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np
import pandas as pd
import json
from zoneinfo import ZoneInfo

NS_PER_S = 1_000_000_000


@dataclass
class BasicSecondAggConfig:
    cadence_s: int = 1
    session_start_ns: Optional[int] = None
    session_end_ns: Optional[int] = None


@dataclass
class FeatureCfg:
    W_entropy: int = 30
    W_persist: int = 30
    W_slope: int = 30
    W_z: int = 300
    eps: float = 1e-12


NY_TZ = ZoneInfo("America/New_York")


def add_tod_sincos_ny(df: pd.DataFrame, ts_col: str = "ts_sec") -> pd.DataFrame:
    """
    Adds tod_sin/tod_cos computed from NY-local time-of-day (DST-safe).
    Expects df[ts_col] in ns epoch.
    """
    dt_utc = pd.to_datetime(df[ts_col], unit="ns", utc=True)
    dt_ny = dt_utc.dt.tz_convert(NY_TZ)

    sec_mid = (dt_ny.dt.hour * 3600 + dt_ny.dt.minute * 60 + dt_ny.dt.second).astype(np.float64)
    ang = 2.0 * np.pi * (sec_mid / 86400.0)

    df["tod_sin"] = np.sin(ang).astype(np.float32)
    df["tod_cos"] = np.cos(ang).astype(np.float32)
    return df


def kmeans_feature_cols(cfg: FeatureCfg) -> list[str]:
    W1 = cfg.W_entropy
    W2 = cfg.W_persist
    W3 = cfg.W_slope
    return [
            #f"vol_log1p_z_{cfg.W_z}s",
            #f"abs_ofi_ratio_z_{cfg.W_z}s",
            f"ofi_ratio_std_{W2}s",
            f"abs_ofi_ratio_mean_{W2}s",
            f"abs_ofi_persist_sharpe_{W2}s",
            f"abs_cum_ofi_norm_slope_{W3}s",
            f"cum_ofi_norm_r2_{W3}s",
    ]


def kmeans_return_feature_cols() -> list[str]:
    return [
        "dlogret_1",
        "dlogret_3",
        #"dlogret_10",
        "ema_ret_fast",
        "ema_ret_impulse",
        "ret_jerk_1",
    ]


def load_scaler_json(path: str) -> tuple[list[str], np.ndarray, np.ndarray, dict]:
    with open(path, "r") as f:
        blob = json.load(f)
    cols = blob["cols"]
    mean = np.array(blob["mean"], dtype=np.float32)
    std = np.array(blob["std"], dtype=np.float32)
    meta = blob.get("meta", {})
    return cols, mean, std, meta


def standardize(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean[None, :]) / std[None, :]


def _second_floor_ns(x_ns: np.ndarray) -> np.ndarray:
    return (x_ns // NS_PER_S) * NS_PER_S


def _session_grid_ns(start_ns: int, end_ns: int, cadence_s: int = 1) -> np.ndarray:
    start_s = start_ns // NS_PER_S
    end_s   = end_ns   // NS_PER_S
    secs = np.arange(start_s, end_s + 1, cadence_s, dtype=np.int64)
    return secs * NS_PER_S


def _normalize_side_to_code(side) -> str:
    # Databento trades aggressor side often 'A'/'B'/'N' or enum with .value
    if side is None:
        return "N"
    if hasattr(side, "value"):
        code = str(side.value)
    else:
        code = str(side)
    code = code.strip().upper()
    code = code[:1] if code and code[0] in ("A", "B", "N") else "N"
    return code


def _binary_entropy(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def rolling_entropy_from_counts(buy: np.ndarray, sell: np.ndarray, W: int, eps: float = 1e-12) -> np.ndarray:
    """
    Rolling entropy using window sums of buy/sell counts (or volumes).
    Returns NaN until window is full.
    """
    buy = buy.astype(np.float64)
    sell = sell.astype(np.float64)

    cb = np.cumsum(buy)
    cs = np.cumsum(sell)

    out = np.full_like(buy, np.nan, dtype=np.float64)
    for t in range(W - 1, len(buy)):
        sb = cb[t] - (cb[t - W] if t >= W else 0.0)
        ss = cs[t] - (cs[t - W] if t >= W else 0.0)
        tot = sb + ss
        if tot <= eps:
            out[t] = np.nan
        else:
            p = sb / tot
            out[t] = _binary_entropy(p, eps=eps)
    return out


def rolling_sum(x: np.ndarray, W: int) -> np.ndarray:
    c = np.cumsum(np.nan_to_num(x, nan=0.0))
    out = np.full_like(x, np.nan, dtype=np.float64)
    for t in range(W - 1, len(x)):
        out[t] = c[t] - (c[t - W] if t >= W else 0.0)
    return out


def rolling_mean(x: np.ndarray, W: int) -> np.ndarray:
    return rolling_sum(x, W) / float(W)


def rolling_std(x: np.ndarray, W: int, eps: float = 1e-12) -> np.ndarray:
    x0 = np.nan_to_num(x, nan=0.0)
    c1 = np.cumsum(x0)
    c2 = np.cumsum(x0 * x0)
    out = np.full_like(x0, np.nan, dtype=np.float64)
    for t in range(W - 1, len(x0)):
        s1 = c1[t] - (c1[t - W] if t >= W else 0.0)
        s2 = c2[t] - (c2[t - W] if t >= W else 0.0)
        m1 = s1 / W
        m2 = s2 / W
        var = max(m2 - m1 * m1, 0.0)
        out[t] = np.sqrt(var + eps)
    return out


def rolling_z(x: np.ndarray, W: int, eps: float = 1e-12) -> np.ndarray:
    mu = rolling_mean(x, W)
    sd = rolling_std(x, W, eps=eps)
    return (x - mu) / (sd + eps)


def rolling_linreg_slope_r2(y: np.ndarray, W: int, eps: float = 1e-12) -> Tuple[np.ndarray, np.ndarray]:
    """
    Causal rolling OLS slope and R^2 of y vs time index 0..W-1.
    NaN until W-1.
    """
    t = np.arange(W, dtype=np.float64)
    t_mean = (W - 1) / 2.0
    Sxx = np.sum((t - t_mean) ** 2) + eps

    slope = np.full_like(y, np.nan, dtype=np.float64)
    r2    = np.full_like(y, np.nan, dtype=np.float64)

    y0 = y.astype(np.float64)

    for end in range(W - 1, len(y0)):
        start = end - W + 1
        yw = y0[start:end + 1]
        if not np.isfinite(yw).all():
            continue
        y_mean = yw.mean()
        Syy = np.sum((yw - y_mean) ** 2) + eps
        Sxy = np.sum((t - t_mean) * (yw - y_mean))
        b = Sxy / Sxx
        slope[end] = b

        y_hat = y_mean + b * (t - t_mean)
        sse = np.sum((yw - y_hat) ** 2)
        r2[end] = 1.0 - (sse / Syy)

    return slope, r2


def transformer_feature_cols(feat_cfg: FeatureCfg) -> list[str]:
    """
    Columns you want to feed to the Transformer as inputs.
    Keep tod_sin/tod_cos unscaled via skip list, but still included in this list.
    """
    W1 = feat_cfg.W_entropy
    W2 = feat_cfg.W_persist
    W3 = feat_cfg.W_slope
    Wz = feat_cfg.W_z

    cols = [
        "buy_trades_1s",
        "sell_trades_1s",
        "buy_vol_1s",
        "sell_vol_1s",

        # time encodings
        "tod_sin",
        "tod_cos",

        # minimal price context (avoid return-stack redundancy)
        "dlogret_1",
        "dlogret_3",
        "vol_log1p",

        # order flow (signed)
        "ofi_1s",
        "ofi_ratio_1s",
        "ema_ofi_fast",
        "ema_ofi_slow",
        "ema_ofi_impulse",

        # NEW: trade pressure (signed)
        "imb_vol_1s",
        "imb_trades_1s",
        "log_buy_sell_vol_1s",
        "log_buy_sell_trades_1s",
        "imb_impulse",

        # entropy / persist families (still useful)
        f"dir_entropy_count_{W1}s",
        f"dir_persist_count_{W1}s",
        f"dir_entropy_vol_{W1}s",
        f"dir_persist_vol_{W1}s",

        f"ofi_ratio_mean_{W2}s",
        f"ofi_ratio_std_{W2}s",
        f"ofi_persist_sharpe_{W2}s",

        f"cum_ofi_norm_slope_{W3}s",
        f"cum_ofi_norm_r2_{W3}s",

        # NEW: normalized activity (replace raw volume/trades)
        "vol_norm",
        "trades_norm",
    ]
    return cols


def trades_records_to_dataframe(records: List[object], symbol: str) -> pd.DataFrame:
    """
    Convert trades records to DataFrame with:
      ts_event (ns), price, size, side ('A'/'B'/'N'), symbol
    """
    rows = []
    for r in records:
        ts = getattr(r, "ts_event", getattr(r, "ts", None))
        px = getattr(r, "price", getattr(r, "px", None))
        sz = getattr(r, "size", getattr(r, "qty", None))
        side = getattr(r, "side", getattr(r, "aggressor_side", None))
        if ts is None or px is None or sz is None:
            continue
        code = _normalize_side_to_code(side)
        rows.append((int(ts), float(px), float(sz), code))
    df = pd.DataFrame(rows, columns=["ts_event", "price", "size", "side"])
    if df.empty:
        return df
    df["symbol"] = symbol
    df = df.sort_values("ts_event", kind="stable").reset_index(drop=True)
    return df


def aggregate_trades_1s(df_tr: pd.DataFrame, cfg: BasicSecondAggConfig) -> pd.DataFrame:
    """
    Input trades: columns [ts_event(ns), price(float), size(float), side('A','B','N')]
    Output per-second:
      ts_sec, close, volume, trades, ofi_1s,
      buy_vol_1s, sell_vol_1s, buy_trades_1s, sell_trades_1s
    """
    if cfg.cadence_s != 1:
        raise ValueError("aggregate_trades_1s requires cadence_s=1")
    if df_tr is None or df_tr.empty:
        return pd.DataFrame(columns=[
            "ts_sec","close","volume","trades","ofi_1s",
            "buy_vol_1s","sell_vol_1s","buy_trades_1s","sell_trades_1s"
        ])

    df = df_tr.sort_values("ts_event", kind="stable").copy()

    ts_ns = df["ts_event"].to_numpy(np.int64)
    px = pd.to_numeric(df["price"], errors="coerce").to_numpy(np.float64)
    sz = pd.to_numeric(df["size"], errors="coerce").to_numpy(np.float64)
    side = df["side"].astype(object).to_numpy()

    sess_start_ns = cfg.session_start_ns or int(ts_ns[0])
    sess_end_ns   = cfg.session_end_ns   or int(ts_ns[-1])

    keep = (ts_ns >= sess_start_ns) & (ts_ns <= sess_end_ns)
    ts_ns = ts_ns[keep]; px = px[keep]; sz = sz[keep]; side = side[keep]

    grid = _session_grid_ns(sess_start_ns, sess_end_ns, cfg.cadence_s)
    G = grid.size
    if G == 0:
        return pd.DataFrame(columns=[
            "ts_sec","close","volume","trades","ofi_1s",
            "buy_vol_1s","sell_vol_1s","buy_trades_1s","sell_trades_1s"
        ])

    cad_ns = cfg.cadence_s * NS_PER_S
    start_ns = int(grid[0])

    sec_ns = _second_floor_ns(ts_ns)
    off = sec_ns - start_ns
    mask = (off >= 0) & (off % cad_ns == 0)
    sec_ns = sec_ns[mask]; px = px[mask]; sz = sz[mask]; side = side[mask]

    idx = ((sec_ns - start_ns) // cad_ns).astype(np.int64)

    close = np.full(G, np.nan, np.float64)
    vol   = np.zeros(G, np.float64)
    ntr   = np.zeros(G, np.int32)
    ofi   = np.zeros(G, np.float64)

    buy_vol  = np.zeros(G, np.float64)
    sell_vol = np.zeros(G, np.float64)
    buy_n    = np.zeros(G, np.int32)
    sell_n   = np.zeros(G, np.int32)

    for k in range(idx.size):
        b = idx[k]
        if b < 0 or b >= G:
            continue

        ntr[b] += 1
        vol[b] += sz[k]
        close[b] = px[k]

        s = side[k]
        if hasattr(s, "value"):
            s = str(s.value)
        elif isinstance(s, (bytes, bytearray)):
            s = s.decode(errors="ignore")
        else:
            s = str(s)
        s = s.strip().upper()[:1] if s else "N"

        if s == "B":
            buy_vol[b] += sz[k]
            buy_n[b] += 1
            ofi[b] += sz[k]
        elif s == "A":
            sell_vol[b] += sz[k]
            sell_n[b] += 1
            ofi[b] -= sz[k]

    # forward-fill close after first trade
    last_close = np.nan
    for i in range(G):
        if np.isnan(close[i]):
            close[i] = last_close
        else:
            last_close = close[i]

    return pd.DataFrame({
        "ts_sec": grid,
        "close": close,
        "volume": vol,
        "trades": ntr,
        "ofi_1s": ofi,
        "buy_vol_1s": buy_vol,
        "sell_vol_1s": sell_vol,
        "buy_trades_1s": buy_n,
        "sell_trades_1s": sell_n,
    })


def build_features_1s(sec: pd.DataFrame, cfg: FeatureCfg) -> pd.DataFrame:
    df = sec.sort_values("ts_sec").reset_index(drop=True).copy()

    # --- TOD encoding in NY local time (DST-safe) ---
    df = add_tod_sincos_ny(df, ts_col="ts_sec")
    df["datetime_utc"] = pd.to_datetime(df["ts_sec"], unit="ns", utc=True)  # keep if you like

    close = pd.to_numeric(df["close"], errors="coerce").to_numpy(np.float64)
    close = np.where(close <= 1e-12, np.nan, close)
    logp = np.log(close)
    dlogret_1 = np.diff(logp, prepend=logp[0])
    dlogret_1[~np.isfinite(dlogret_1)] = 0.0
    df["dlogret_1"] = dlogret_1.astype(np.float32)

    vol = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    df["vol_log1p"] = np.log1p(np.clip(vol, 0.0, None)).astype(np.float32)

    ofi = pd.to_numeric(df["ofi_1s"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    df["ofi_1s"] = ofi.astype(np.float32)
    ofi_ratio = ofi / (vol + cfg.eps)
    df["ofi_ratio_1s"] = ofi_ratio.astype(np.float32)

    # ===========================
    # Reactive drift features (causal)
    # ===========================
    # 3s / 10s log-return (causal, uses past only)
    # NOTE: this is "past drift", NOT forward targets.
    def lagged_logret(logp_arr: np.ndarray, k: int) -> np.ndarray:
        out = np.zeros_like(logp_arr, dtype=np.float64)
        out[:k] = 0.0
        out[k:] = logp_arr[k:] - logp_arr[:-k]
        out[~np.isfinite(out)] = 0.0
        return out

    dlogret_3  = lagged_logret(logp, 3)
    dlogret_10 = lagged_logret(logp, 10)

    df["dlogret_3"]  = dlogret_3.astype(np.float32)
    df["dlogret_10"] = dlogret_10.astype(np.float32)

    # EMA drift on 1s returns: fast + slow + "impulse"
    # fast reacts to spikes; slow captures regime; impulse captures sudden change.
    ema_fast = ema(dlogret_1, alpha=0.30)   # ~3s-ish memory
    ema_slow = ema(dlogret_1, alpha=0.05)   # ~20s-ish memory
    df["ema_ret_fast"] = ema_fast.astype(np.float32)
    df["ema_ret_slow"] = ema_slow.astype(np.float32)
    df["ema_ret_impulse"] = (ema_fast - ema_slow).astype(np.float32)

    # "Jerk" = change in return (how fast it accelerates)
    jerk = np.diff(dlogret_1.astype(np.float64), prepend=float(dlogret_1[0]))
    jerk[~np.isfinite(jerk)] = 0.0
    df["ret_jerk_1"] = jerk.astype(np.float32)

    # Signed OFI drift (causal): EMA of ofi_ratio_1s
    ofi_ema_fast = ema(ofi_ratio, alpha=0.30)
    ofi_ema_slow = ema(ofi_ratio, alpha=0.05)
    df["ema_ofi_fast"] = ofi_ema_fast.astype(np.float32)
    df["ema_ofi_slow"] = ofi_ema_slow.astype(np.float32)
    df["ema_ofi_impulse"] = (ofi_ema_fast - ofi_ema_slow).astype(np.float32)

    # (1) Entropy + persistence (EXACT using buy/sell counts & vols from aggregation)
    W = cfg.W_entropy
    buyN  = pd.to_numeric(df["buy_trades_1s"], errors="coerce").fillna(0).to_numpy(np.float64)
    sellN = pd.to_numeric(df["sell_trades_1s"], errors="coerce").fillna(0).to_numpy(np.float64)
    buyV  = pd.to_numeric(df["buy_vol_1s"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    sellV = pd.to_numeric(df["sell_vol_1s"], errors="coerce").fillna(0.0).to_numpy(np.float64)

    ent_count = rolling_entropy_from_counts(buyN, sellN, W=W, eps=cfg.eps)
    ent_vol   = rolling_entropy_from_counts(buyV, sellV, W=W, eps=cfg.eps)

    df[f"dir_entropy_count_{W}s"] = ent_count.astype(np.float32)
    df[f"dir_persist_count_{W}s"] = (1.0 - ent_count).astype(np.float32)
    df[f"dir_entropy_vol_{W}s"]   = ent_vol.astype(np.float32)
    df[f"dir_persist_vol_{W}s"]   = (1.0 - ent_vol).astype(np.float32)

    # (2) Persistence (Sharpe-like) of normalized OFI
    mu = rolling_mean(ofi_ratio, cfg.W_persist)
    sd = rolling_std(ofi_ratio, cfg.W_persist, eps=cfg.eps)
    df[f"ofi_ratio_mean_{cfg.W_persist}s"] = mu.astype(np.float32)
    df[f"ofi_ratio_std_{cfg.W_persist}s"]  = sd.astype(np.float32)
    df[f"ofi_persist_sharpe_{cfg.W_persist}s"] = (mu / (sd + cfg.eps)).astype(np.float32)

    # (3) Cumulative delta steepness: slope + R^2 (on cum_ofi_norm)
    cum_ofi = np.cumsum(ofi)
    cum_vol = np.cumsum(vol)
    cum_ofi_norm = cum_ofi / (cum_vol + cfg.eps)
    slope, r2 = rolling_linreg_slope_r2(cum_ofi_norm, cfg.W_slope, eps=cfg.eps)
    df[f"cum_ofi_norm_slope_{cfg.W_slope}s"] = slope.astype(np.float32)
    df[f"cum_ofi_norm_r2_{cfg.W_slope}s"]    = r2.astype(np.float32)

    # z-score baselines (so model/cluster can see "unusual")
    df[f"vol_log1p_z_{cfg.W_z}s"] = rolling_z(df["vol_log1p"].to_numpy(np.float64), cfg.W_z, eps=cfg.eps).astype(np.float32)
    df[f"abs_ofi_ratio_z_{cfg.W_z}s"] = rolling_z(np.abs(ofi_ratio), cfg.W_z, eps=cfg.eps).astype(np.float32)

    df[f"abs_ofi_ratio_mean_{cfg.W_persist}s"] = np.abs(mu).astype(np.float32)
    df[f"abs_ofi_persist_sharpe_{cfg.W_persist}s"] = np.abs((mu / (sd + cfg.eps))).astype(np.float32)
    
    df[f"abs_cum_ofi_norm_slope_{cfg.W_slope}s"] = np.abs(
        df[f"cum_ofi_norm_slope_{cfg.W_slope}s"].to_numpy(np.float64)
    ).astype(np.float32)
    # ---- final sanitation for rolling features ----
    entropy_cols = [
        f"dir_entropy_count_{cfg.W_entropy}s",
        f"dir_persist_count_{cfg.W_entropy}s",
        f"dir_entropy_vol_{cfg.W_entropy}s",
        f"dir_persist_vol_{cfg.W_entropy}s",
    ]
    
    df[entropy_cols] = df[entropy_cols].fillna(0.0)
    # ==========================================================
    # NEW: trade-pressure features (signed, directional)
    # ==========================================================
    denomV = (buyV + sellV + cfg.eps)
    denomN = (buyN + sellN + cfg.eps)

    # Signed imbalance (volume + trade count)
    df["imb_vol_1s"] = ((buyV - sellV) / denomV).astype(np.float32)
    df["imb_trades_1s"] = ((buyN - sellN) / denomN).astype(np.float32)

    # Log-difference (robust alternative to ratio)
    df["log_buy_sell_vol_1s"] = (np.log(buyV + cfg.eps) - np.log(sellV + cfg.eps)).astype(np.float32)
    df["log_buy_sell_trades_1s"] = (np.log(buyN + cfg.eps) - np.log(sellN + cfg.eps)).astype(np.float32)

    # Imbalance impulse (detect "takeoff starts")
    imb_vol = df["imb_vol_1s"].to_numpy(np.float64)
    imb_fast = ema(imb_vol, alpha=0.30)   # ~3s-ish
    imb_slow = ema(imb_vol, alpha=0.05)   # ~20s-ish
    df["imb_impulse"] = (imb_fast - imb_slow).astype(np.float32)

    # ==========================================================
    # NEW: normalized activity (removes "busy=volatile" shortcut)
    # ==========================================================
    # Normalize by slow EMA so activity only matters when unusual
    vol_raw = df["volume"].to_numpy(np.float64)
    tr_raw  = df["trades"].to_numpy(np.float64)

    vol_ema = ema(vol_raw, alpha=0.03)     # ~100s-ish memory (tune)
    tr_ema  = ema(tr_raw,  alpha=0.03)

    df["vol_norm"] = (vol_raw / (vol_ema + cfg.eps)).astype(np.float32)
    df["trades_norm"] = (tr_raw / (tr_ema + cfg.eps)).astype(np.float32)

    return df


def ema(x: np.ndarray, alpha: float) -> np.ndarray:
    """
    Simple causal EMA. alpha in (0,1]. Larger alpha = more reactive.
    """
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x, dtype=np.float64)
    s = 0.0
    init = False
    for i in range(len(x)):
        xi = x[i]
        if not np.isfinite(xi):
            xi = 0.0
        if not init:
            s = xi
            init = True
        else:
            s = alpha * xi + (1.0 - alpha) * s
        out[i] = s
    return out
