"""Extracted research definitions; see docs/source_map.json for provenance."""
from __future__ import annotations
import numpy as np

def pool_embedding(h: np.ndarray, mode: str) -> np.ndarray:
    # h: (B,S,D)
    if mode == "last":
        return h[:, -1, :]
    if mode == "mean":
        return h.mean(axis=1)
    if mode == "mean16":
        K = 16
        if h.shape[1] >= K:
            return h[:, -K:, :].mean(axis=1)
        return h.mean(axis=1)
    raise ValueError(f"Unknown emb_pool={mode}")


def fmt_qcol(prefix: str, kind: str, N: int, q: float) -> str:
    # e.g. loc_mfe_60_q0p5
    qs = str(float(q)).replace(".", "p")
    return f"{prefix}{kind}_{int(N)}_q{qs}"
