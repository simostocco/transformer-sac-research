"""Extracted research definitions; see docs/source_map.json for provenance."""
from __future__ import annotations
from dataclasses import dataclass
import math
import time
from typing import Any, Mapping, Sequence
import numpy as np
import pandas as pd
from streaming_research.runtime.transformer_runner import LocalTransformerRunner

@dataclass(frozen=True)
class PreparedTransformerInputWindow:
    """Epoch-owned, read-only FP32 inputs, detached from the CPU writer's ring."""

    x: np.ndarray
    k1: np.ndarray
    k2: np.ndarray
    relative: np.ndarray

    def copy_into(self, x: np.ndarray, k1: np.ndarray, k2: np.ndarray,
                  relative: np.ndarray, log_scratch: np.ndarray) -> None:
        # Relative prices were computed by the same ring copy at preparation.
        # No persistent runner host/device buffer is owned by this window.
        for source, destination in zip(
                (self.x, self.k1, self.k2, self.relative), (x, k1, k2, relative)):
            if source.shape != destination.shape or source.dtype != destination.dtype:
                raise ValueError(
                    f"prepared Transformer window invariant: source={source.shape}/{source.dtype} "
                    f"destination={destination.shape}/{destination.dtype}")
            destination[:] = source


class RollingTransformerInputState:
    """Single CPU-writer input history; snapshot before asynchronous inference."""

    def __init__(self, runner: "FastLocalTransformerRunner"):
        self.runner = runner
        self.seq_len = int(runner.seq_len)
        self.feature_width = len(runner.feat_cols)
        self.features = np.empty((self.seq_len, self.feature_width), dtype=np.float32)
        self.log_close = np.empty(self.seq_len, dtype=np.float32)
        self.k1 = np.empty(self.seq_len, dtype=np.int64)
        self.k2 = np.empty(self.seq_len, dtype=np.int64)
        self._raw_scratch = np.empty(self.feature_width, dtype=np.float32)
        self.cursor = 0
        self.count = 0

    @staticmethod
    def _numeric(value: Any, dtype: Any) -> Any:
        try:
            return dtype(value)
        except (TypeError, ValueError):
            return dtype(np.nan)

    @staticmethod
    def _label(value: Any, neutral: int) -> int:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return int(neutral)
        return int(neutral) if not np.isfinite(numeric) or numeric < 0 else int(numeric)

    def append_row(self, row: Mapping[str, Any]) -> None:
        for index, column in enumerate(self.runner.feat_cols):
            self._raw_scratch[index] = self._numeric(row.get(column), np.float32)
        # This is the same float32 elementwise expression as standardize_np,
        # moved from every 120-row window to the single newly arriving row.
        np.subtract(self._raw_scratch, self.runner.mean, out=self.features[self.cursor])
        np.divide(self.features[self.cursor], self.runner.std, out=self.features[self.cursor])
        close = self._numeric(row.get("close"), np.float32)
        self.log_close[self.cursor] = np.log(np.float32(max(float(close), 1e-12)))
        self.k1[self.cursor] = self._label(row.get("kmeans_k4"), self.runner.neutral_id)
        self.k2[self.cursor] = self._label(row.get("kmeans_ret_k3"), self.runner.neutral_ret_id)
        self.cursor = (self.cursor + 1) % self.seq_len
        self.count = min(self.seq_len, self.count + 1)


    def load_rows(self, rows: Sequence[Mapping[str, Any]]) -> None:
        self.cursor = 0
        self.count = 0
        for row in rows[-self.seq_len:]:
            self.append_row(row)

    def snapshot(self) -> PreparedTransformerInputWindow:
        """Freeze this epoch while its CPU owner is still the sole ring reader/writer."""
        expected = ((self.seq_len, self.feature_width), (self.seq_len,),
                    (self.seq_len,), (self.seq_len,))
        actual = tuple(value.shape for value in
                       (self.features, self.log_close, self.k1, self.k2))
        if (self.count != self.seq_len or not 0 <= self.cursor < self.seq_len
                or actual != expected):
            raise ValueError(
                f"rolling Transformer window invariant: count={self.count} "
                f"cursor={self.cursor} seq_len={self.seq_len} shapes={actual}")
        x = np.empty((self.seq_len, self.feature_width + 1), dtype=np.float32)
        k1 = np.empty(self.seq_len, dtype=np.int64)
        k2 = np.empty(self.seq_len, dtype=np.int64)
        relative = np.empty((self.seq_len, 1), dtype=np.float32)
        self.copy_into(x, k1, k2, relative, np.empty(self.seq_len, dtype=np.float32))
        for value in (x, k1, k2, relative):
            value.setflags(write=False)
        return PreparedTransformerInputWindow(x, k1, k2, relative)

    def _copy_ring(self, source: np.ndarray, destination: np.ndarray) -> None:
        if self.count < self.seq_len:
            raise ValueError(f"need at least seq_len={self.seq_len} rows, got {self.count}")
        split = self.seq_len - self.cursor
        destination[:split] = source[self.cursor:]
        if self.cursor:
            destination[split:] = source[:self.cursor]

    def copy_into(self, x: np.ndarray, k1: np.ndarray, k2: np.ndarray,
                  relative: np.ndarray, log_scratch: np.ndarray) -> None:
        self._copy_ring(self.features, x[:, :self.feature_width])
        self._copy_ring(self.log_close, log_scratch)
        np.subtract(log_scratch, log_scratch[0], out=relative[:, 0])
        x[:, self.feature_width] = relative[:, 0]
        self._copy_ring(self.k1, k1)
        self._copy_ring(self.k2, k2)



class FastLocalTransformerRunner(LocalTransformerRunner):
    """Buffered inference wrapper; equivalence is qualified by input and precision."""

    def __init__(self, model, metadata: dict, *, device: str,
                 emb_pool: str, benchmark_timing: bool = False,
                 max_batch_size: int = 64, precision: str = "bf16"):
        super().__init__(model, metadata, device=device, emb_pool=emb_pool)
        self.benchmark_timing = bool(benchmark_timing)
        self.precision = str(precision).lower()
        if self.precision not in {"bf16", "fp32"}:
            raise ValueError(f"unsupported transformer precision={self.precision!r}")
        self._device_buffers: tuple[Any, Any, Any, Any] | None = None
        self._device_buffer_signature: tuple[Any, ...] | None = None
        self.max_batch_size = max(64, int(max_batch_size))
        self._host_buffers: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
        self._host_tensors: tuple[Any, Any, Any, Any] | None = None
        self._log_scratch: np.ndarray | None = None
        self.last_prepare_breakdown_ms: dict[str, float] = {}

    def new_input_state(self) -> RollingTransformerInputState:
        return RollingTransformerInputState(self)

    def _ensure_host_buffers(self, active: int) -> None:
        if active > self.max_batch_size:
            raise ValueError(f"batch size {active} exceeds prepared maximum {self.max_batch_size}")
        if self._host_buffers is not None:
            return
        shape = (self.max_batch_size, self.seq_len)
        tensor_shapes = ((*shape, len(self.feat_cols) + 1), shape, shape, (*shape, 1))
        tensor_dtypes = (self.torch.float32, self.torch.int64,
                         self.torch.int64, self.torch.float32)
        try:
            self._host_tensors = tuple(self.torch.empty(item_shape, dtype=dtype, pin_memory=True)
                                       for item_shape, dtype in zip(tensor_shapes, tensor_dtypes))
        except RuntimeError:
            self._host_tensors = tuple(self.torch.empty(item_shape, dtype=dtype)
                                       for item_shape, dtype in zip(tensor_shapes, tensor_dtypes))
        self._host_buffers = tuple(tensor.numpy() for tensor in self._host_tensors)  # type: ignore[assignment]
        self._log_scratch = np.empty(shape, dtype=np.float32)

    def prepare_states(
        self, states: Sequence[RollingTransformerInputState | PreparedTransformerInputWindow]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Assemble active rows into persistent contiguous BMAX host storage."""
        active = len(states)
        self._ensure_host_buffers(active)
        assert self._host_buffers is not None and self._log_scratch is not None
        started = time.perf_counter_ns()
        for index, state in enumerate(states):
            state.copy_into(self._host_buffers[0][index], self._host_buffers[1][index],
                            self._host_buffers[2][index], self._host_buffers[3][index],
                            self._log_scratch[index])
        completed = time.perf_counter_ns()
        self.last_prepare_breakdown_ms = {
            "ring_copy_scale_ready_k1_k2_relative_price_ms": (completed - started) / 1_000_000.0,
            "numpy_allocation_ms": 0.0,
            "dataframe_ms": 0.0,
        }
        return tuple(value[:active] for value in self._host_buffers)  # type: ignore[return-value]

    def prepare_arrays(
        self, dfs: Sequence[pd.DataFrame]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build the exact frozen-runner inputs as contiguous CPU arrays."""
        if not dfs:
            return (
                np.empty((0, self.seq_len, len(self.feat_cols) + 1), dtype=np.float32),
                np.empty((0, self.seq_len), dtype=np.int64),
                np.empty((0, self.seq_len), dtype=np.int64),
                np.empty((0, self.seq_len, 1), dtype=np.float32),
            )
        xb_parts: list[np.ndarray] = []
        k1_parts: list[np.ndarray] = []
        k2_parts: list[np.ndarray] = []
        r_parts: list[np.ndarray] = []
        required = self.feat_cols + ["close", "kmeans_k4", "kmeans_ret_k3"]
        for df in dfs:
            if len(df) < self.seq_len:
                raise ValueError(f"need at least seq_len={self.seq_len} rows, got {len(df)}")
            for column in required:
                if column not in df.columns:
                    raise ValueError(f"missing transformer input column {column}")
            tail = df.iloc[-self.seq_len:].copy()
            close = pd.to_numeric(tail["close"], errors="coerce").to_numpy(np.float32)
            values = tail[self.feat_cols].to_numpy(np.float32)
            values = self.standardize_np(values, self.mean, self.std).astype(np.float32)
            logp = np.log(np.clip(close.astype(np.float32), 1e-12, None))
            position = (logp - logp[0]).reshape(-1, 1).astype(np.float32)
            xb_parts.append(np.concatenate([values, position], axis=1).astype(np.float32))
            k1_raw = pd.to_numeric(tail["kmeans_k4"], errors="coerce").fillna(-1).to_numpy(np.int64)
            k2_raw = pd.to_numeric(tail["kmeans_ret_k3"], errors="coerce").fillna(-1).to_numpy(np.int64)
            k1_parts.append(np.where(k1_raw < 0, self.neutral_id, k1_raw).astype(np.int64))
            k2_parts.append(np.where(k2_raw < 0, self.neutral_ret_id, k2_raw).astype(np.int64))
            r_parts.append(position)
        return tuple(np.ascontiguousarray(np.stack(parts, axis=0))
                     for parts in (xb_parts, k1_parts, k2_parts, r_parts))

    def _device_inputs(self, arrays: Sequence[np.ndarray]) -> tuple[Any, Any, Any, Any]:
        torch = self.torch
        uses_preallocated_host = (self._host_buffers is not None and self._host_tensors is not None
                                  and all(np.shares_memory(value, backing)
                                          for value, backing in zip(arrays, self._host_buffers)))
        if uses_preallocated_host:
            active = len(arrays[0])
            cpu = tuple(tensor[:active] for tensor in self._host_tensors)
        else:
            cpu = tuple(torch.from_numpy(np.ascontiguousarray(value)) for value in arrays)
        if self.device.type != "cuda":
            return cpu  # type: ignore[return-value]
        signature = tuple((tuple(tensor.shape[1:]), tensor.dtype) for tensor in cpu)
        if self._device_buffers is None or signature != self._device_buffer_signature:
            self._device_buffers = tuple(torch.empty((self.max_batch_size, *tensor.shape[1:]),
                                                       dtype=tensor.dtype, device=self.device)
                                         for tensor in cpu)
            self._device_buffer_signature = signature
        for destination, source in zip(self._device_buffers, cpu):
            destination[:len(source)].copy_(source, non_blocking=True)
        return tuple(value[:len(cpu[0])] for value in self._device_buffers)  # type: ignore[return-value]

    def infer_states(self, states: Sequence[RollingTransformerInputState | PreparedTransformerInputWindow]):
        started = time.perf_counter_ns()
        arrays = self.prepare_states(states)
        prepared = time.perf_counter_ns()
        values, timing = self.infer_arrays(*arrays)
        prep_ms = (prepared - started) / 1_000_000.0
        timing["preprocess_ms"] += prep_ms
        timing["total_ms"] += prep_ms
        return values, timing

    def _compact_cpu_outputs(self, output: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        """Transfer q plus only the hidden rows required by the frozen pooling rule."""
        q = output["q"].detach().to(self.torch.float32)
        h = output["h"].detach().to(self.torch.float32)
        if self.emb_pool == "last":
            selected = h[:, -1:, :]
        elif self.emb_pool == "mean16":
            selected = h[:, -min(16, int(h.shape[1])):, :]
        elif self.emb_pool == "mean":
            selected = h
        else:
            raise ValueError(f"Unknown emb_pool={self.emb_pool}")
        q_width = int(q[0].numel())
        packed = self.torch.cat((q.reshape(len(q), -1), selected.reshape(len(q), -1)), dim=1)
        host = packed.cpu().numpy()
        q_np = host[:, :q_width].reshape(tuple(q.shape))
        h_np = host[:, q_width:].reshape(tuple(selected.shape))
        return q_np, self.pool_embedding(h_np, mode=self.emb_pool).astype(np.float32)

    def infer_arrays(
        self, x: np.ndarray, k1: np.ndarray, k2: np.ndarray, r: np.ndarray
    ) -> tuple[list[dict[str, Any]], dict[str, float]]:
        """Infer from a persisted exact input batch without rebuilding upstream state."""
        logical_batch_size = len(x)
        if logical_batch_size == 0:
            return [], {"preprocess_ms": 0.0, "forward_ms": 0.0,
                        "forward_cuda_ms": float("nan"), "materialize_ms": 0.0,
                        "total_ms": 0.0, "logical_batch_size": 0,
                        "numerical_batch_size": 0, "padded_slots": 0}
        arrays = (x, k1, k2, r)
        if any(len(value) != logical_batch_size for value in arrays):
            raise ValueError("transformer input batch cardinality mismatch")
        numerical_batch_size = logical_batch_size
        if (self.device.type == "cuda" and self.precision == "fp32"
                and logical_batch_size < 20):
            # B1..B20 is one numerical family: duplicate the last valid sample
            # only inside this model call. The caller still owns exactly N rows.
            numerical_batch_size = 20
            padding = 20 - logical_batch_size
            arrays = tuple(np.concatenate((value, np.repeat(value[-1:], padding, axis=0)), axis=0)
                           for value in arrays)
        torch = self.torch
        t0 = time.perf_counter_ns()
        tensors = self._device_inputs(arrays)
        t1 = time.perf_counter_ns()
        start_event = end_event = None
        with torch.inference_mode():
            if self.benchmark_timing and self.device.type == "cuda":
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=(self.device.type == "cuda" and self.precision == "bf16"),
            ):
                output = self.model(*tensors)
            if (len(output["q"]) != numerical_batch_size
                    or len(output["h"]) != numerical_batch_size):
                raise ValueError("transformer numerical output cardinality mismatch")
            if end_event is not None:
                end_event.record()
            submitted_ns = time.perf_counter_ns()
            # Dummy numerical slots never cross the output/materialization
            # boundary and therefore cannot acquire logical ownership.
            q, embedding = self._compact_cpu_outputs({
                "q": output["q"][:logical_batch_size],
                "h": output["h"][:logical_batch_size],
            })
        materialized_ns = time.perf_counter_ns()
        cuda_ms = float("nan")
        if start_event is not None and end_event is not None:
            # The blocking CPU result materialization above completed the same
            # stream, so elapsed_time needs no extra event/global synchronize.
            cuda_ms = float(start_event.elapsed_time(end_event))
        forward_ms = cuda_ms if math.isfinite(cuda_ms) else (submitted_ns - t1) / 1_000_000.0
        values = [self._materialize_output(q[row], embedding[row])
                  for row in range(logical_batch_size)]
        completed_ns = time.perf_counter_ns()
        return values, {
            "preprocess_ms": (t1 - t0) / 1_000_000.0,
            "forward_ms": forward_ms,
            "forward_cuda_ms": cuda_ms,
            "materialize_ms": (completed_ns - submitted_ns) / 1_000_000.0,
            "total_ms": (completed_ns - t0) / 1_000_000.0,
            "logical_batch_size": logical_batch_size,
            "numerical_batch_size": numerical_batch_size,
            "padded_slots": numerical_batch_size - logical_batch_size,
            # Benchmark-only absolute boundaries; numeric keys preserve existing aggregation.
            "_trace_preprocess_start_ns": t0,
            "_trace_gpu_start_ns": t1,
            "_trace_gpu_submit_end_ns": submitted_ns,
            "_trace_output_copy_end_ns": materialized_ns,
            "_trace_output_rows_end_ns": completed_ns,
        }

    def infer_batch(self, dfs: Sequence[pd.DataFrame]):
        if not dfs:
            return self.infer_arrays(*self.prepare_arrays(dfs))
        started = time.perf_counter_ns()
        arrays = self.prepare_arrays(dfs)
        prepared = time.perf_counter_ns()
        values, timing = self.infer_arrays(*arrays)
        timing["preprocess_ms"] += (prepared - started) / 1_000_000.0
        timing["total_ms"] += (prepared - started) / 1_000_000.0
        return values, timing
