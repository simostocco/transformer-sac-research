"""Extracted research definitions; see docs/source_map.json for provenance."""
from __future__ import annotations
import time
from typing import Any, Dict, List, Sequence, Tuple
import numpy as np
import pandas as pd

class LocalTransformerRunner:
    def __init__(self, model, metadata: dict, *, device: str = "cpu", emb_pool: str = "mean16"):
        import torch
        from streaming_research.forecasting.transformer import standardize_np
        from streaming_research.forecasting.inference import fmt_qcol, pool_embedding
        self.torch = torch
        self.device = torch.device(device)
        self.standardize_np = standardize_np
        self.fmt_qcol, self.pool_embedding = fmt_qcol, pool_embedding
        self.feat_cols = list(metadata["feat_cols"])
        self.mean = np.asarray(metadata["mean"], dtype=np.float32)
        self.std = np.asarray(metadata["std"], dtype=np.float32)
        if (self.mean.shape != (len(self.feat_cols),) or self.std.shape != self.mean.shape
                or not np.isfinite(self.mean).all() or not np.isfinite(self.std).all()
                or (self.std <= 0).any()):
            raise ValueError("invalid frozen feature scaler")
        self.seq_len = int(metadata["seq_len"])
        minimum = max((int(m.K) for m in model.modules() if hasattr(m, "K")), default=1)
        if self.seq_len < minimum or len(set(self.feat_cols)) != len(self.feat_cols):
            raise ValueError("invalid sequence length or duplicate feature columns")
        self.quantiles = [float(q) for q in metadata["quantiles"]]
        self.target_spec = [tuple(t) for t in metadata["target_spec"]]
        self.neutral_id = int(metadata["neutral_id"])
        self.neutral_ret_id = int(metadata["neutral_ret_id"])
        if (self.quantiles != list(model.quantiles) or len(self.target_spec) != len(model.horizons)
                or model.x_proj.in_features != len(self.feat_cols) + 1 + (2 if model.relpos_features else 0)
                or self.neutral_id != model.neutral_id
                or self.neutral_ret_id != model.k2_emb.num_embeddings - 1):
            raise ValueError("model and metadata contracts disagree")
        labels = [fmt_qcol("loc_", kind, int(n), q) for kind, n in self.target_spec for q in self.quantiles]
        if len(labels) != len(set(labels)):
            raise ValueError("duplicate forecast output labels")
        self.d_model = int(model.normf.normalized_shape[0])
        self.emb_pool = str(emb_pool)
        self.model = model.to(self.device).eval()
        self.model.requires_grad_(False)

    def warmup(self, df: pd.DataFrame, count: int = 3) -> None:
        if len(df) < self.seq_len:
            return
        for _ in range(max(0, int(count))):
            self.infer_last(df.iloc[: self.seq_len].copy())

    def infer_last(self, df: pd.DataFrame) -> Dict[str, Any]:
        vals, _ = self.infer_batch([df])
        return vals[0]

    def _materialize_output(self, q: np.ndarray, emb: np.ndarray) -> Dict[str, Any]:
        if not np.isfinite(q).all() or not np.isfinite(emb).all():
            raise ValueError("nonfinite Transformer output")
        vals: Dict[str, Any] = {}
        for h_i, (kind, N) in enumerate(self.target_spec):
            for q_i, qv in enumerate(self.quantiles):
                vals[self.fmt_qcol("loc_", kind, int(N), float(qv))] = float(q[h_i, q_i])
        for j, v in enumerate(emb):
            vals[f"loc_emb_{j:03d}"] = float(v)
        vals["loc_valid"] = True
        vals["ctx_skipped"] = True
        vals["ctx_valid"] = True
        vals["rl_valid"] = True
        return vals

    def infer_batch(self, dfs: Sequence[pd.DataFrame]) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        torch = self.torch
        if not dfs:
            return [], {"preprocess_ms": 0.0, "forward_ms": 0.0, "materialize_ms": 0.0, "total_ms": 0.0}
        t0 = time.perf_counter_ns()
        xb_parts: List[np.ndarray] = []
        k1_parts: List[np.ndarray] = []
        k2_parts: List[np.ndarray] = []
        r_parts: List[np.ndarray] = []
        for df in dfs:
            if len(df) < self.seq_len:
                raise ValueError(f"need at least seq_len={self.seq_len} rows, got {len(df)}")
            for c in self.feat_cols + ["close", "kmeans_k4", "kmeans_ret_k3"]:
                if c not in df.columns:
                    raise ValueError(f"missing transformer input column {c}")
            tail = df.iloc[-self.seq_len :].copy()
            close = pd.to_numeric(tail["close"], errors="coerce").to_numpy(np.float32)
            X = tail[self.feat_cols].to_numpy(np.float32)
            X = self.standardize_np(X, self.mean, self.std).astype(np.float32)
            logp = np.log(np.clip(close.astype(np.float32), 1e-12, None))
            pos_from_start = (logp - logp[0]).reshape(-1, 1).astype(np.float32)
            xb_parts.append(np.concatenate([X, pos_from_start], axis=1).astype(np.float32))
            k1_raw = pd.to_numeric(tail["kmeans_k4"], errors="coerce").fillna(-1).to_numpy(np.int64)
            k2_raw = pd.to_numeric(tail["kmeans_ret_k3"], errors="coerce").fillna(-1).to_numpy(np.int64)
            k1_parts.append(np.where(k1_raw < 0, self.neutral_id, k1_raw).astype(np.int64))
            k2_parts.append(np.where(k2_raw < 0, self.neutral_ret_id, k2_raw).astype(np.int64))
            r_parts.append(pos_from_start.astype(np.float32))
        x_batch = np.stack(xb_parts, axis=0)
        k1 = np.stack(k1_parts, axis=0)
        k2 = np.stack(k2_parts, axis=0)
        r = np.stack(r_parts, axis=0)
        x_t = torch.from_numpy(x_batch).to(self.device)
        k1_t = torch.from_numpy(k1).to(self.device)
        k2_t = torch.from_numpy(k2).to(self.device)
        r_t = torch.from_numpy(r).to(self.device)
        cuda_forward_ms = float("nan")
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter_ns()
        with torch.no_grad():
            start_ev = end_ev = None
            if self.device.type == "cuda":
                start_ev = torch.cuda.Event(enable_timing=True)
                end_ev = torch.cuda.Event(enable_timing=True)
                start_ev.record()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(self.device.type == "cuda")):
                out = self.model(x_t, k1_t, k2_t, r_t)
            if self.device.type == "cuda" and end_ev is not None and start_ev is not None:
                end_ev.record()
            if self.device.type == "cuda":
                torch.cuda.synchronize()
                if start_ev is not None and end_ev is not None:
                    cuda_forward_ms = float(start_ev.elapsed_time(end_ev))
            t2 = time.perf_counter_ns()
            q = out["q"].detach().to(torch.float32).cpu().numpy()
            h = out["h"].detach().to(torch.float32).cpu().numpy()
        embs = self.pool_embedding(h, mode=self.emb_pool).astype(np.float32)
        vals = [self._materialize_output(q[i], embs[i]) for i in range(len(dfs))]
        t3 = time.perf_counter_ns()
        return vals, {
            "preprocess_ms": (t1 - t0) / 1_000_000.0,
            "forward_ms": (t2 - t1) / 1_000_000.0,
            "forward_cuda_ms": cuda_forward_ms,
            "materialize_ms": (t3 - t2) / 1_000_000.0,
            "total_ms": (t3 - t0) / 1_000_000.0,
        }
