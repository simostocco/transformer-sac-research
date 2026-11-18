"""Extracted research definitions; see docs/source_map.json for provenance."""
from __future__ import annotations
import math
from typing import List, Tuple, Optional, Dict
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import torch
import torch.nn as nn
import torch.nn.functional as F

def standardize_np(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean[None, :]) / std[None, :]


def compute_future_targets_mfe_mae_premium(
    close: np.ndarray,
    windows: List[int],
    premium_ref: int,
    premium_windows: List[int],
) -> Tuple[np.ndarray, List[Tuple[str,int]]]:
    """
    Returns:
      Y: (T, H) float32 where H = 2*len(windows) + len(premium_windows)
      spec: list of (kind, N) for each column in Y
        kind in {"mfe","mae","prem"}.

    prem_N = MFE_N - MFE_premium_ref
    """
    close = close.astype(np.float64)
    close = np.where(close > 1e-12, close, np.nan)
    logp = np.log(close)
    T = len(logp)

    windows = sorted(set(int(w) for w in windows if int(w) > 0))
    prem_ref = int(premium_ref)
    prem_windows = [int(w) for w in premium_windows if int(w) > 0]

    # compute MFE/MAE for each window into dicts
    mfe_map: Dict[int, np.ndarray] = {}
    mae_map: Dict[int, np.ndarray] = {}

    lp1 = logp[1:]  # (T-1,)
    for N in windows:
        if (T - 1) < N:
            mfe = np.full(T, np.nan, dtype=np.float32)
            mae = np.full(T, np.nan, dtype=np.float32)
            mfe_map[N] = mfe
            mae_map[N] = mae
            continue

        win = sliding_window_view(lp1, window_shape=N)  # (T-N, N)
        fut_max = np.nanmax(win, axis=1)  # (T-N,)
        fut_min = np.nanmin(win, axis=1)  # (T-N,)
        base = logp[: T - N]              # (T-N,)
        mfe = np.full(T, np.nan, dtype=np.float32)
        mae = np.full(T, np.nan, dtype=np.float32)
        mfe[: T - N] = (fut_max - base).astype(np.float32)
        mae[: T - N] = (fut_min - base).astype(np.float32)
        mfe_map[N] = mfe
        mae_map[N] = mae

    # build output
    spec: List[Tuple[str,int]] = []
    cols: List[np.ndarray] = []

    for N in windows:
        spec.append(("mfe", N)); cols.append(mfe_map[N])
        spec.append(("mae", N)); cols.append(mae_map[N])

    # premium targets (only if ref exists in mfe_map)
    if prem_windows:
        if prem_ref not in mfe_map:
            raise ValueError(f"premium_ref={prem_ref} not in windows={windows}")
        mfe_ref = mfe_map[prem_ref]
        for N in prem_windows:
            if N not in mfe_map:
                raise ValueError(f"premium window N={N} not in windows={windows}")
            prem = mfe_map[N] - mfe_ref
            spec.append(("prem", N)); cols.append(prem.astype(np.float32))

    Y = np.stack(cols, axis=1).astype(np.float32)  # (T,H)
    return Y, spec


def weighted_quantile_pinball_loss(
    y_true: torch.Tensor,         # (B,H)
    y_pred_q: torch.Tensor,       # (B,H,Q)
    w: torch.Tensor,              # (B,)
    quantiles: List[float],
    target_weights: Optional[torch.Tensor] = None,  # (H,) on same device/dtype
) -> torch.Tensor:
    qs = torch.tensor(quantiles, device=y_true.device, dtype=y_true.dtype).view(1, 1, -1)  # (1,1,Q)
    e = y_true.unsqueeze(-1) - y_pred_q  # (B,H,Q)
    loss = torch.maximum(qs * e, (qs - 1.0) * e)  # (B,H,Q)

    # average over quantiles first -> (B,H)
    loss_bh = loss.mean(dim=-1)

    # apply per-target weights across H
    if target_weights is not None:
        tw = target_weights.to(device=y_true.device, dtype=y_true.dtype).view(1, -1)  # (1,H)
        tw = torch.clamp(tw, 0.0, 100.0)
        denom = tw.sum() + 1e-8
        loss_b = (loss_bh * tw).sum(dim=1) / denom  # (B,)
    else:
        loss_b = loss_bh.mean(dim=1)  # (B,)

    # apply sample weights
    w = torch.clamp(w, 0.0, 100.0)
    return (loss_b * w).sum() / (w.sum() + 1e-8)


def monotonicity_penalty(
    q_pred: torch.Tensor,            # (B,H,Q)
    target_spec: List[Tuple[str,int]],
) -> torch.Tensor:
    """
    Penalize violations of monotonicity across windows:
      - MFE should be non-decreasing with horizon (longer window >= shorter window)
      - MAE (min return) should be non-increasing with horizon (longer window <= shorter window)
    Applied across all quantiles, averaged.
    Premium targets are ignored.
    """
    # collect indices for MFE and MAE in order of increasing N
    mfe = [(i, N) for i, (k, N) in enumerate(target_spec) if k == "mfe"]
    mae = [(i, N) for i, (k, N) in enumerate(target_spec) if k == "mae"]

    mfe = sorted(mfe, key=lambda x: x[1])
    mae = sorted(mae, key=lambda x: x[1])

    pen = q_pred.new_tensor(0.0)

    # MFE: enforce pred(N2) >= pred(N1) for N2 > N1
    for (i1, _), (i2, _) in zip(mfe[:-1], mfe[1:]):
        short = q_pred[:, i1, :]  # (B,Q)
        long  = q_pred[:, i2, :]
        pen = pen + F.relu(short - long).mean()

    # MAE: enforce pred(N2) <= pred(N1) for N2 > N1  (more negative allowed)
    for (i1, _), (i2, _) in zip(mae[:-1], mae[1:]):
        short = q_pred[:, i1, :]
        long  = q_pred[:, i2, :]
        pen = pen + F.relu(long - short).mean()

    # average by number of constraints so scaling stays stable
    n_cons = max((len(mfe) - 1) + (len(mae) - 1), 1)
    return pen / float(n_cons)


class RelativePositionBias(nn.Module):
    """
    Learned bias added to attention logits: (H, S, S) via buckets of relative distance.
    This is very explicit "relative position" information.

    Implementation: map relative positions to buckets, embed per-head.
    """
    def __init__(self, n_heads: int, n_buckets: int = 32, max_distance: int = 512):
        super().__init__()
        self.n_heads = n_heads
        self.n_buckets = n_buckets
        self.max_distance = max_distance
        self.emb = nn.Embedding(n_buckets, n_heads)

    def _bucket(self, rel: torch.Tensor) -> torch.Tensor:
        # rel: (S,S) where rel = j - i (key - query)
        # We only need causal direction, but keep general.
        n = self.n_buckets
        maxd = self.max_distance
        # split buckets: exact for small |rel|, log for larger
        half = n // 2
        abs_rel = rel.abs()
        small = abs_rel < half
        # log buckets for large distances
        log_rel = (torch.log(abs_rel.float() / half + 1e-6) / math.log(maxd / half)) * (n - half)
        log_rel = log_rel.long().clamp(0, n - half - 1) + half
        b = torch.where(small, abs_rel.long().clamp(0, half - 1), log_rel)
        return b.clamp(0, n - 1)

    def forward(self, S: int, device: torch.device) -> torch.Tensor:
        i = torch.arange(S, device=device)
        j = torch.arange(S, device=device)
        rel = j[None, :] - i[:, None]  # (S,S)
        b = self._bucket(rel)          # (S,S)
        bias = self.emb(b)             # (S,S,H)
        return bias.permute(2, 0, 1)   # (H,S,S)


class ReturnDeltaMLPBiasLocal(nn.Module):
    """
    Local (banded) return-delta bias: only for |i-j| <= K.
    Produces bias of shape (B,H,S,S) but constructed efficiently by filling bands.
    """
    def __init__(self, n_heads: int, hidden: int = 32, K: int = 30):
        super().__init__()
        self.n_heads = int(n_heads)
        self.K = int(K)
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.n_heads),
        )

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """
        r: (B,S)
        returns: (B,H,S,S) but only non-zero on bands |i-j|<=K
        """
        B, S = r.shape
        device = r.device
        dtype = r.dtype

        bias = torch.zeros((B, self.n_heads, S, S), device=device, dtype=dtype)

        # Fill bands for offsets d = 0..K
        for d in range(0, self.K + 1):
            # j = i + d
            i = torch.arange(0, S - d, device=device)
            j = i + d

            delta = (r[:, j] - r[:, i]).unsqueeze(-1)  # (B, S-d, 1)
            out = self.mlp(delta.reshape(-1, 1)).reshape(B, S - d, self.n_heads)  # (B,S-d,H)
            out = out.permute(0, 2, 1)  # (B,H,S-d)

            # upper diagonal
            bias[:, :, i, j] += out

            # lower diagonal (j = i - d), skip d=0
            if d > 0:
                bias[:, :, j, i] += -out  # antisymmetric (optional but sensible)

        return bias


class RoPEMultiheadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_len: int,
        attn_dropout: float = 0.1,
        theta: float = 10000.0,
        use_rel_bias: bool = False,
        rel_bias_buckets: int = 32,
        rel_bias_maxdist: int = 512,
        use_ret_delta_bias: bool = True,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert self.head_dim % 2 == 0, "RoPE needs even head_dim"

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(attn_dropout)
        # So you always add it and always require r.But right now it’s not tied to --pairwise_rel_bias.
        self.use_rel_bias = bool(use_rel_bias)
        self.rel_bias = RelativePositionBias(n_heads, rel_bias_buckets, rel_bias_maxdist) if self.use_rel_bias else None

        self.ret_delta_bias = ReturnDeltaMLPBiasLocal(n_heads=n_heads, hidden=32, K=30)

        freqs_cis = self._precompute_freqs_cis(self.head_dim, max_len, theta=theta)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        causal_mask = torch.triu(torch.ones(max_len, max_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", causal_mask, persistent=False)

        self.use_ret_delta_bias = bool(use_ret_delta_bias)
        self.ret_delta_bias = ReturnDeltaMLPBiasLocal(
            n_heads=n_heads,
            hidden=32,
            K=30
        ) if self.use_ret_delta_bias else None

    @staticmethod
    def _precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(end, dtype=torch.float32)
        freqs = torch.outer(t, freqs)
        return torch.polar(torch.ones_like(freqs), freqs)

    @staticmethod
    def _apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        B, S, H, D = x.shape
        x_complex = torch.view_as_complex(x.float().reshape(B, S, H, D // 2, 2))
        freqs = freqs_cis.view(1, S, 1, D // 2)
        out = torch.view_as_real(x_complex * freqs).reshape(B, S, H, D)
        return out.type_as(x)

    def forward(self, x, r=None, w_tok=None):
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, S, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(B, S, self.n_heads, self.head_dim)

        freqs_cis = self.freqs_cis[:S].to(device=x.device)
        q = self._apply_rope(q, freqs_cis)
        k = self._apply_rope(k, freqs_cis)

        q = q.transpose(1, 2)  # (B,H,S,D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # ---- build additive attn_mask for SDPA: (B,H,S,S), dtype matches q ----
        attn_mask = None
        scale = 1.0  # SDPA applies scaling internally

        # Start with causal mask (upper triangle masked)
        mask = self.causal_mask[:S, :S].to(device=x.device)  # (S,S) bool
        base = torch.zeros((S, S), device=x.device, dtype=q.dtype)
        base = base.masked_fill(mask, torch.finfo(q.dtype).min)  # -inf on masked

        attn_mask = base.view(1, 1, S, S).expand(B, self.n_heads, S, S)  # (B,H,S,S)

        # Token bias (keys dimension)
        if w_tok is not None:
            wt = w_tok.to(dtype=q.dtype, device=x.device)
            wt = wt / (wt.mean(dim=1, keepdim=True) + 1e-8)
            tok_bias = torch.log(torch.clamp(wt, 0.5, 3.0))  # (B,S)
            attn_mask = attn_mask + tok_bias[:, None, None, :]  # add to keys

        # Return-delta bias (already (B,H,S,S))
        if self.use_ret_delta_bias:
            if r is None:
                raise ValueError("Return-delta bias enabled but r was not provided")
            attn_mask = attn_mask + self.ret_delta_bias(r.to(device=x.device, dtype=q.dtype))

        # Learned relpos bias (H,S,S) -> (1,H,S,S)
        if self.use_rel_bias:
            attn_mask = attn_mask + self.rel_bias(S, x.device).to(dtype=q.dtype).unsqueeze(0)

        attn_mask = attn_mask.contiguous()
        # ---- SDPA (FlashAttention / mem-efficient kernels when available) ----
        # dropout_p is only applied during training
        drop_p = self.attn_drop.p if self.training else 0.0

        # Prefer flash if available
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
            ctx = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=drop_p,
                is_causal=False,   # we already encoded causality in attn_mask
            )  # (B,H,S,D)

        ctx = ctx.transpose(1, 2).reshape(B, S, self.d_model)
        return self.out_proj(ctx)


class RoPEBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_len: int, dropout: float = 0.1, ffn_mult: int = 4, use_rel_bias: bool = False, use_ret_delta_bias: bool = True):
        super().__init__()
        self.attn = RoPEMultiheadAttention(
            d_model, n_heads, max_len=max_len, attn_dropout=dropout,
            use_rel_bias=use_rel_bias,
            use_ret_delta_bias=use_ret_delta_bias,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        hidden = ffn_mult * d_model
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, r=None, w_tok=None):
        x = self.norm1(x + self.drop(self.attn(x, r=r, w_tok=w_tok)))
        x = self.norm2(x + self.ffn(x))
        return x


class HFTQuantileCPCTransformer(nn.Module):
    def __init__(
        self,
        x_dim: int,
        n_kmeans1: int,
        n_kmeans2: int,
        windows: List[int],
        neutral_id: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        max_len: int = 600,
        ffn_mult: int = 4,
        horizons: List[int] = None,
        quantiles: List[float] = None,
        use_cpc: bool = False,
        cpc_horizons: List[int] = None,
        cpc_dim: int = 128,
        relpos_features: bool = False,
        use_rel_bias: bool = False,
        extreme_labels: List[int] = None,
        attn_extreme_boost: float = 1.5,
        use_ret_delta_bias: bool = True,
    ):
        super().__init__()
        self.horizons = list(horizons or [10,20,30,40,50,60,70,80,90,100,110,120])
        self.quantiles = list(sorted(quantiles or [0.1,0.25,0.5,0.75,0.9]))
        self.use_cpc = bool(use_cpc)
        self.cpc_horizons = list(cpc_horizons or [10,30,60])
        self.relpos_features = bool(relpos_features)
        self.neutral_id = int(neutral_id)
        
        extra = 2 if self.relpos_features else 0
        self.x_proj = nn.Linear(x_dim + extra, d_model)

        self.k1_dim = 16
        self.k2_dim = 16

        self.k1_emb = nn.Embedding(n_kmeans1 + 1, self.k1_dim)
        self.k2_emb = nn.Embedding(n_kmeans2 + 1, self.k2_dim)

        self.k_proj = nn.Linear(self.k1_dim + self.k2_dim, d_model)

        self.norm0 = nn.LayerNorm(d_model)
        self.drop0 = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            RoPEBlock(d_model, n_heads, max_len=max_len, dropout=dropout, ffn_mult=ffn_mult, use_rel_bias=use_rel_bias, use_ret_delta_bias=use_ret_delta_bias)
            for _ in range(n_layers)
        ])
        self.normf = nn.LayerNorm(d_model)

        self.use_edge_head = True
        self.windows = windows
        self.n_windows = len(windows)

        self.edge_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, self.n_windows))
        # ---- auxiliary direction head: predict sign(dlogret_3) ----
        self.dir_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 2))   # logits for classes {0,1}

        H = len(self.horizons)
        Q = len(self.quantiles)
        self.q_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, H * Q),
        )

        if self.use_cpc:
            self.cpc_proj = nn.Linear(d_model, cpc_dim)
            self.cpc_pred = nn.ModuleList([nn.Linear(d_model, cpc_dim) for _ in self.cpc_horizons])

        self.attn_extreme_boost = float(attn_extreme_boost)

        imp = torch.tensor(extreme_labels if extreme_labels else [], dtype=torch.long)
        self.register_buffer("k1_important_labels", imp, persistent=False)

    def forward(self, x_num, k1_label, k2_label, r):
        if self.k1_important_labels.numel() == 0:
            w_tok = None
        else:
            # (B,S,L) compare each token label with each important label
            is_imp = (k1_label.unsqueeze(-1) == self.k1_important_labels.view(1, 1, -1)).any(dim=-1)  # (B,S)
            w_tok = 1.0 + is_imp.float() * self.attn_extreme_boost  # (B,S)
        
        #if not hasattr(self, "_dbg_tok"):
        #    self._dbg_tok = True
        #    if w_tok is None:
        #        print("[tok-bias] disabled (no extreme_labels)")
        #    else:
        #        print("[tok-bias] boost=", float(self.attn_extreme_boost),
        #              "labels=", self.k1_important_labels.tolist(),
        #              "w_tok stats:", float(w_tok.min()), float(w_tok.mean()), float(w_tok.max()))
        
        e1 = self.k1_emb(k1_label)     # (B,S,16)
        e2 = self.k2_emb(k2_label)     # (B,S,16)

        k_emb = torch.cat([e1, e2], dim=-1)
        k_emb = self.k_proj(k_emb)     # (B,S,d_model)
        # x_num: (B,S,D)
        B, S, D = x_num.shape
        r = r.squeeze(-1)

        if self.relpos_features:
            # explicit "where am I in this window"
            t = torch.arange(S, device=x_num.device).float()
            idx01 = (t / max(S - 1, 1)).view(1, S, 1).expand(B, -1, -1)            # 0..1
            dist_to_end = ((S - 1 - t) / max(S - 1, 1)).view(1, S, 1).expand(B, -1, -1)  # 1..0
            x_num = torch.cat([x_num, idx01, dist_to_end], dim=-1)

        h = self.x_proj(x_num)
        h = h + k_emb
        h = self.drop0(self.norm0(h))

        for blk in self.blocks:
            h = blk(h, r=r, w_tok=w_tok)

        h = self.normf(h)
        h_last = h[:, -1, :]
        dir_logits = self.dir_head(h_last)   # (B,2)
        edge = self.edge_head(h_last)

        out = self.q_head(h_last)
        Hn = len(self.horizons)
        Qn = len(self.quantiles)
        q = out.view(B, Hn, Qn)

        ret = {"q": q, "edge": edge, "dir_logits": dir_logits, "h": h}

        if self.use_cpc:
            z = self.cpc_proj(h)
            preds = [pred_layer(h_last) for pred_layer in self.cpc_pred]
            ret["z"] = z
            ret["cpc_preds"] = torch.stack(preds, dim=1)
        
        return ret
