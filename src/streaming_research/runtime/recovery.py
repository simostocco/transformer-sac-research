"""Qualification lifecycle around the native service's validated source health."""
import math
import json
import logging


NS = 10**9
# stage1::SourceGate bits accepted only during bounded native recovery.
RECOVERABLE_SOURCE_GATES = 2 | 8 | 16 | 128  # NotCurrentBuffer | Feed/TradeTimeInvalid | Recovering
# One vendor clock unit is admissible; larger bounded leads quarantine inference.
FEED_FUTURE_TOLERANCE_MS = 1.0
# One quiet-feed update interval bounds quarantine, not admissible timestamps.
CLOCK_RECOVERY_LEAD_MS = 25.0


class QualificationGuard:
    def __init__(self, opening_wall_ns, closing_wall_ns, started_wall_ns,
                 started_monotonic_ns, acquisition_seconds=1800, recovery_seconds=10):
        if (closing_wall_ns <= opening_wall_ns or not math.isfinite(acquisition_seconds)
                or not math.isfinite(recovery_seconds) or acquisition_seconds <= 0 or recovery_seconds <= 0):
            raise ValueError("invalid qualification window")
        self.opening_wall_ns = opening_wall_ns
        self.closing_wall_ns = closing_wall_ns
        self.acquisition_deadline_ns = (started_monotonic_ns
                                        + max(0, opening_wall_ns - started_wall_ns)
                                        + acquisition_seconds * NS)
        self.phase = "PRESTART"
        self.active_wall_ns = None
        self.active_monotonic_ns = None
        self.recovery_deadline_ns = None
        self.recovery_timeout_ns = recovery_seconds * NS
        self.first_cause = None
        self.last_health_ns = None
        self.source_backlog = 0
        self.inference_backlog = 0

    def fail(self, reason):
        if self.first_cause is None:
            self.first_cause = reason
        self.phase = "FAILED"
        return self.first_cause

    def check(self, health, wall_ns, monotonic_ns):
        if self.first_cause is not None:
            return self.first_cause

        if health is not None:
            if health["state"] == "FAILED":
                return self.fail(health["first_cause"])
            for key in ("deadline_misses", "overflow", "gpu_errors", "invalid_inputs"):
                if health[key]:
                    return self.fail(key.upper())
            if (health["scheduler"] != "context_microbatch" or health["broker"] != "DISABLED"
                    or health["paper_mutations"] or health["orders"] or health["full_trace"]):
                return self.fail("SHADOW_ISOLATION_OR_SCHEDULER_CHANGED")
            if (health["source_depth"] >= 1_000_000
                    or health["inference_depth"] >= health["context_pending_limit"]):
                return self.fail("QUEUE_AT_CAPACITY")

        if wall_ns >= self.closing_wall_ns:
            self.phase = "COMPLETE"
            return None
        if wall_ns < self.opening_wall_ns:
            if self.active_monotonic_ns is not None:
                return self.fail("SESSION_CLOCK_REWOUND")
            self.phase = "PRESTART"
            return None

        if self.recovery_deadline_ns is not None and monotonic_ns >= self.recovery_deadline_ns:
            # One bounded timeout receipt in the existing controller stderr log.
            diagnostics = (health or {}).get('recovery_diagnostics', {})
            self.timeout_diagnostics = dict(
                cause='SOURCE_RECOVERY_TIMEOUT', monotonic_ns=monotonic_ns,
                recovery_deadline_ns=self.recovery_deadline_ns,
                native_recovery=diagnostics,
                health={k: (health or {}).get(k) for k in (
                    'monotonic_ns', 'source_gate_mask', 'source_fresh', 'source_recovering',
                    'source_lag_ms', 'trade_lag_ms', 'sdk_mode', 'source_depth', 'first_cause')})
            logging.getLogger(__name__).warning('RECOVERY_TIMEOUT_DIAGNOSTICS %s',
                                              json.dumps(self.timeout_diagnostics))
            return self.fail("SOURCE_RECOVERY_TIMEOUT")

        if health is not None:
            # Native freshness already includes initialization, timestamps, mode,
            # source generation and the two-second stabilization interval.
            sample_age_ms = (monotonic_ns - health["monotonic_ns"]) / 1_000_000
            data_valid = (health["real_source"] and health["frontier_acquired"]
                          and health.get("source_initialized", True)
                          and health["wall_ns"] >= self.opening_wall_ns
                          and sample_age_ms >= 0
                          and health.get("feed_future_tolerance_ns") == 1_000_000
                          and monotonic_ns <= health.get("source_valid_until_monotonic_ns", -1)
                          and -CLOCK_RECOVERY_LEAD_MS <= health["source_lag_ms"] <= 5000 - sample_age_ms
                          and -CLOCK_RECOVERY_LEAD_MS <= health["trade_lag_ms"] <= 5000 - sample_age_ms)
            clocks_admissible = (health["source_lag_ms"] >= -FEED_FUTURE_TOLERANCE_MS
                                 and health["trade_lag_ms"] >= 0)
            ready = (data_valid and clocks_admissible
                     and 2 <= health["sdk_mode"] <= 6
                     and health["source_fresh"] and health.get("source_gate_mask", 0) == 0)
            if self.active_monotonic_ns is not None and not ready:
                time_gates = health.get("source_gate_mask", 0) & (8 | 16)
                bounded_clock_lead = (time_gates
                                      and (not time_gates & 8 or health["source_lag_ms"] < -FEED_FUTURE_TOLERANCE_MS)
                                      and (not time_gates & 16 or health["trade_lag_ms"] < 0)
                                      and health.get("blocker") == "SOURCE_TIMESTAMPS_NOT_FRESH")
                recoverable = (data_valid and 1 <= health["sdk_mode"] <= 6
                               and (health.get("source_gate_mask", 0) & ~RECOVERABLE_SOURCE_GATES) == 0
                               and (not time_gates or bounded_clock_lead)
                               and (bounded_clock_lead or (clocks_admissible and health.get("blocker") in (
                                   "RECOVERY_STABILIZING", "SOURCE_BUFFER_CATCHUP"))))
                if not recoverable:
                    return self.fail("SOURCE_FRESHNESS_LOST")
                if self.recovery_deadline_ns is None:
                    self.recovery_deadline_ns = monotonic_ns + self.recovery_timeout_ns
                self.phase = "RECOVERING"
            if ready:
                self.phase = "ACTIVE"
                self.recovery_deadline_ns = None
                if self.active_monotonic_ns is None:
                    self.active_wall_ns = wall_ns
                    self.active_monotonic_ns = monotonic_ns

        if self.phase not in ("ACTIVE", "RECOVERING"):
            self.phase = "CATCHING_UP"
            if monotonic_ns >= self.acquisition_deadline_ns:
                return self.fail("STARTUP_TIMEOUT")
            return None

        if health is not None and health["monotonic_ns"] != self.last_health_ns:
            self.last_health_ns = health["monotonic_ns"]
            self.source_backlog = self.source_backlog + 1 if health["oldest_pending_ms"] > 500 else 0
            self.inference_backlog = (self.inference_backlog + 1
                                      if 5 * health["inference_depth"] >= 4 * health["context_pending_limit"]
                                      else 0)
            if self.source_backlog >= 3 or self.inference_backlog >= 3:
                return self.fail("SUSTAINED_QUEUE_BACKLOG")
        return None
