"""Offline qualification activation and unchanged active-session safety limits."""
import unittest

from streaming_research.runtime.recovery import NS, QualificationGuard


class QualificationGuardTests(unittest.TestCase):
    def setUp(self):
        self.guard = QualificationGuard(100 * NS, 1000 * NS, 90 * NS, 10 * NS)

    def sample(self, wall=100, **changes):
        health = dict(wall_ns=wall * NS, monotonic_ns=(wall - 80) * NS,
                      state="RUNNING", first_cause="NONE", real_source=True,
                      frontier_acquired=True, source_fresh=True,
                      feed_future_tolerance_ns=1_000_000,
                      source_valid_until_monotonic_ns=(wall - 75) * NS,
                      source_lag_ms=300, trade_lag_ms=400, sdk_mode=6,
                      deadline_misses=0, overflow=0, gpu_errors=0, invalid_inputs=0,
                      scheduler="context_microbatch", broker="DISABLED",
                      paper_mutations=0, orders=0, full_trace=False,
                      source_depth=0, inference_depth=0, context_pending_limit=400,
                      oldest_pending_ms=0)
        health.update(changes)
        return health

    def check(self, wall, **changes):
        return self.guard.check(self.sample(wall, **changes), wall * NS, (wall - 80) * NS)

    def test_prestart_fresh_then_stale_does_not_arm(self):
        self.assertIsNone(self.check(90))
        self.assertIsNone(self.check(96, source_fresh=False, trade_lag_ms=6000))
        self.assertEqual(self.guard.phase, "PRESTART")
        self.assertIsNone(self.guard.active_wall_ns)

    def test_session_open_does_not_imply_source_ready(self):
        for changes in (dict(source_fresh=False, trade_lag_ms=120_000, sdk_mode=1),
                        dict(real_source=False), dict(frontier_acquired=False),
                        dict(sdk_mode=1),
                        dict(trade_lag_ms=5001), dict(source_lag_ms=-25.001)):
            with self.subTest(changes=changes):
                guard = QualificationGuard(100 * NS, 3000 * NS, 90 * NS, 10 * NS)
                self.assertIsNone(guard.check(self.sample(101, **changes), 101 * NS, 21 * NS))
                self.assertEqual(guard.phase, "CATCHING_UP")

    def test_prestart_health_cannot_activate_at_open(self):
        self.assertIsNone(self.guard.check(self.sample(99), 100 * NS, 20 * NS))
        self.assertEqual(self.guard.phase, "CATCHING_UP")
        self.assertIsNone(self.check(101))
        self.assertEqual(self.guard.phase, "ACTIVE")

    def test_catchup_recovery_then_genuine_stale_failure(self):
        self.assertIsNone(self.check(100, source_fresh=False, trade_lag_ms=100_000, sdk_mode=1))
        self.assertIsNone(self.check(102, source_fresh=False, frontier_acquired=False))
        self.assertIsNone(self.check(104, state="DEGRADED", first_cause="NONE"))
        self.assertEqual(self.guard.phase, "ACTIVE")
        self.assertEqual(self.guard.active_wall_ns, 104 * NS)
        self.assertEqual(self.check(110, source_fresh=False, trade_lag_ms=6000), "SOURCE_FRESHNESS_LOST")
        self.assertEqual(self.check(111), "SOURCE_FRESHNESS_LOST")

    def test_partial_session_starts_on_valid_live_readiness(self):
        self.guard = QualificationGuard(100 * NS, 3000 * NS, 500 * NS, 10 * NS)
        self.assertIsNone(self.guard.check(self.sample(501, sdk_mode=4, monotonic_ns=11 * NS,
                                                       source_valid_until_monotonic_ns=16 * NS),
                                          501 * NS, 11 * NS))
        self.assertEqual(self.guard.phase, "ACTIVE")
        self.assertEqual(self.guard.active_wall_ns, 501 * NS)

    def test_feed_future_tolerance_keeps_trade_and_stale_limits(self):
        for lag in (-0.003884, -0.100136, -0.272749, -0.782948, -1.0):
            self.assertIsNone(self.check(100, source_lag_ms=lag))
        self.assertEqual(self.guard.phase, "ACTIVE")
        self.assertEqual(self.check(101, source_lag_ms=-25.000001), "SOURCE_FRESHNESS_LOST")
        self.setUp()
        self.assertIsNone(self.check(100))
        self.assertEqual(self.check(101, trade_lag_ms=-0.000001), "SOURCE_FRESHNESS_LOST")

    def test_five_second_limit_is_unchanged(self):
        self.assertIsNone(self.check(100, source_lag_ms=5000, trade_lag_ms=5000))
        self.assertEqual(self.check(101, trade_lag_ms=5000.001), "SOURCE_FRESHNESS_LOST")

    def test_nov17_terminal_sample_no_longer_stops_controller(self):
        self.assertIsNone(self.check(100))
        # Exact terminal feed/trade lag; native now accepts this sub-millisecond lead.
        self.assertIsNone(self.check(101, source_lag_ms=-0.272749, trade_lag_ms=2.727251))
        self.assertEqual(self.guard.phase, "ACTIVE")
        self.assertGreater(0.272749, 0.25)  # Previous guard's fatal comparison.

    def test_larger_transient_lead_quarantines_then_recovers(self):
        self.assertIsNone(self.check(100))
        self.assertIsNone(self.check(101, source_lag_ms=-2, source_fresh=False,
                                     source_gate_mask=8, blocker="SOURCE_TIMESTAMPS_NOT_FRESH"))
        self.assertEqual(self.guard.phase, "RECOVERING")
        deadline = self.guard.recovery_deadline_ns
        self.assertIsNone(self.recovery(102))
        self.assertEqual(self.guard.recovery_deadline_ns, deadline)
        self.assertIsNone(self.check(104))
        self.assertEqual(self.guard.phase, "ACTIVE")

    def test_repeated_leads_do_not_extend_recovery_timeout(self):
        self.assertIsNone(self.check(100))
        for wall in range(101, 111):
            self.assertIsNone(self.check(wall, source_lag_ms=-25, source_fresh=False,
                                        source_gate_mask=8, blocker="SOURCE_TIMESTAMPS_NOT_FRESH"))
        with self.assertLogs('streaming_research.runtime.recovery', level='WARNING'):
            self.assertEqual(self.check(111), "SOURCE_RECOVERY_TIMEOUT")

    def test_large_future_or_trade_future_is_fatal(self):
        for changes in (dict(source_lag_ms=-25.000001), dict(source_lag_ms=-1000),
                        dict(source_lag_ms=-2, trade_lag_ms=-25.001),
                        dict(source_lag_ms=-2, source_gate_mask=8 | 16)):
            with self.subTest(changes=changes):
                self.setUp()
                self.assertIsNone(self.check(100))
                self.assertEqual(self.check(101, source_fresh=False, blocker="SOURCE_TIMESTAMPS_NOT_FRESH",
                                            **changes), "SOURCE_FRESHNESS_LOST")

    def test_transient_future_trade_gates_inference_without_immediate_stop(self):
        self.assertIsNone(self.check(100))
        self.assertIsNone(self.check(101, source_lag_ms=-0.072917, trade_lag_ms=-0.072917,
                                     source_fresh=False, source_gate_mask=16,
                                     blocker="SOURCE_TIMESTAMPS_NOT_FRESH"))
        self.assertEqual(self.guard.phase, "RECOVERING")
        self.assertIsNone(self.recovery(102))
        self.assertIsNone(self.check(104))
        self.assertEqual(self.guard.phase, "ACTIVE")

    def test_cached_health_uses_monotonic_age_during_wall_jump(self):
        self.assertIsNone(self.check(100))
        h = self.sample(101, source_lag_ms=0, trade_lag_ms=0)
        self.assertIsNone(self.guard.check(h, 100 * NS, 22 * NS))
        self.assertEqual(self.guard.check(h, 100 * NS, 27 * NS), "SOURCE_FRESHNESS_LOST")

    def test_native_deadline_and_policy_are_required(self):
        for changes in (dict(source_valid_until_monotonic_ns=20 * NS),
                        dict(feed_future_tolerance_ns=250000),
                        dict(monotonic_ns=22 * NS)):
            with self.subTest(changes=changes):
                self.setUp()
                self.assertIsNone(self.check(100))
                self.assertEqual(self.check(101, **changes), "SOURCE_FRESHNESS_LOST")

    def test_multiple_recovery_cycles(self):
        self.assertIsNone(self.check(100))
        for wall in (101, 106, 111):
            self.assertIsNone(self.recovery(wall))
            self.assertIsNone(self.check(wall + 3))
            self.assertEqual(self.guard.phase, "ACTIVE")

    def test_native_loss_or_invalid_source_is_fatal_after_activation(self):
        for changes in (dict(source_fresh=False), dict(real_source=False),
                        dict(frontier_acquired=False), dict(source_lag_ms=5001),
                        dict(trade_lag_ms=float("nan"))):
            with self.subTest(changes=changes):
                self.setUp()
                self.assertIsNone(self.check(100))
                self.assertEqual(self.check(101, **changes), "SOURCE_FRESHNESS_LOST")

    def test_startup_timeout_starts_no_earlier_than_open(self):
        self.guard = QualificationGuard(100 * NS, 3000 * NS, 0, 0, acquisition_seconds=30)
        self.assertIsNone(self.guard.check(None, 99 * NS, 99 * NS))
        self.assertIsNone(self.guard.check(None, 129 * NS, 129 * NS))
        self.assertEqual(self.guard.check(None, 130 * NS, 130 * NS), "STARTUP_TIMEOUT")

    def test_late_start_receives_bounded_catchup(self):
        self.guard = QualificationGuard(100 * NS, 3000 * NS, 500 * NS, 10 * NS, acquisition_seconds=30)
        self.assertIsNone(self.guard.check(None, 529 * NS, 39 * NS))
        self.assertEqual(self.guard.check(None, 530 * NS, 40 * NS), "STARTUP_TIMEOUT")

    def test_integrity_errors_are_fatal_in_every_phase(self):
        for wall in (90, 100, 101):
            for field in ("deadline_misses", "overflow", "gpu_errors", "invalid_inputs"):
                with self.subTest(wall=wall, field=field):
                    self.setUp()
                    self.assertEqual(self.check(wall, **{field: 1}), field.upper())

    def test_native_failure_and_isolation_are_never_deferred(self):
        self.assertEqual(self.check(90, state="FAILED", first_cause="SOURCE_REWIND"), "SOURCE_REWIND")
        self.setUp()
        self.assertEqual(self.check(90, orders=1), "SHADOW_ISOLATION_OR_SCHEDULER_CHANGED")

    def test_queue_capacity_is_fatal_before_activation(self):
        self.assertEqual(self.check(90, source_depth=1_000_000), "QUEUE_AT_CAPACITY")
        self.setUp()
        self.assertEqual(self.check(90, inference_depth=400), "QUEUE_AT_CAPACITY")

    def test_active_queue_health_requires_distinct_samples(self):
        self.assertIsNone(self.check(100, oldest_pending_ms=501))
        self.assertIsNone(self.check(100, oldest_pending_ms=501))
        self.assertIsNone(self.check(101, oldest_pending_ms=501))
        self.assertEqual(self.check(102, oldest_pending_ms=501), "SUSTAINED_QUEUE_BACKLOG")
        self.setUp()
        self.assertIsNone(self.check(100, inference_depth=320))
        self.assertIsNone(self.check(101, inference_depth=320))
        self.assertEqual(self.check(102, inference_depth=320), "SUSTAINED_QUEUE_BACKLOG")

    def test_close_finishes_without_mistaking_shutdown_for_staleness(self):
        self.assertIsNone(self.check(100))
        self.assertIsNone(self.check(1000, source_fresh=False))
        self.assertEqual(self.guard.phase, "COMPLETE")

    def test_active_guard_cannot_disarm_on_clock_rewind(self):
        self.assertIsNone(self.check(100))
        self.assertEqual(self.check(99), "SESSION_CLOCK_REWOUND")

    def recovery(self, wall, **changes):
        values = dict(source_fresh=False, source_gate_mask=128,
                      blocker="RECOVERY_STABILIZING")
        values.update(changes)
        return self.check(wall, **values)

    def test_native_recovery_pauses_then_resumes_without_resetting_window(self):
        self.assertIsNone(self.check(100))
        self.assertIsNone(self.recovery(101))
        self.assertEqual(self.guard.phase, "RECOVERING")
        self.assertEqual(self.guard.recovery_deadline_ns, 31 * NS)
        self.assertIsNone(self.recovery(102))
        self.assertIsNone(self.check(103))
        self.assertEqual(self.guard.phase, "ACTIVE")
        self.assertIsNone(self.guard.recovery_deadline_ns)
        self.assertEqual(self.guard.active_wall_ns, 100 * NS)

    def test_recovery_timeout_is_bounded_and_cannot_be_extended(self):
        self.assertIsNone(self.check(100))
        self.assertIsNone(self.recovery(101))
        for wall in range(102, 111):
            self.assertIsNone(self.recovery(wall))
        self.assertEqual(self.recovery(111), "SOURCE_RECOVERY_TIMEOUT")
        self.assertEqual(self.check(112), "SOURCE_RECOVERY_TIMEOUT")

    def test_recovery_with_missing_health_still_times_out(self):
        self.assertIsNone(self.check(100))
        self.assertIsNone(self.recovery(101))
        self.assertEqual(self.guard.check(None, 111 * NS, 31 * NS), "SOURCE_RECOVERY_TIMEOUT")

    def test_real_staleness_and_invalid_clocks_fail_during_recovery(self):
        for changes in (dict(trade_lag_ms=5001), dict(source_lag_ms=5001),
                        dict(source_lag_ms=-25.001), dict(real_source=False),
                        dict(source_initialized=False), dict(source_gate_mask=8),
                        dict(frontier_acquired=False)):
            with self.subTest(changes=changes):
                self.setUp()
                self.assertIsNone(self.check(100))
                self.assertIsNone(self.recovery(101))
                self.assertEqual(self.recovery(102, **changes), "SOURCE_FRESHNESS_LOST")

    def test_cached_health_cannot_hide_five_second_expiry(self):
        self.assertIsNone(self.check(100))
        h = self.sample(101, source_lag_ms=0, trade_lag_ms=0)
        self.assertEqual(self.guard.check(h, 107 * NS, 27 * NS), "SOURCE_FRESHNESS_LOST")

    def test_recovery_preserves_integrity_and_queue_limits(self):
        self.assertIsNone(self.check(100))
        self.assertIsNone(self.recovery(101, oldest_pending_ms=501))
        self.assertIsNone(self.recovery(102, oldest_pending_ms=501))
        self.assertEqual(self.recovery(103, oldest_pending_ms=501), "SUSTAINED_QUEUE_BACKLOG")
        self.setUp()
        self.assertIsNone(self.check(100))
        self.assertIsNone(self.recovery(101))
        self.assertEqual(self.recovery(102, overflow=1), "OVERFLOW")

    def test_known_buffer_catchup_can_recover_with_current_data(self):
        self.assertIsNone(self.check(100))
        self.assertIsNone(self.recovery(101, sdk_mode=1, source_gate_mask=2,
                                       blocker="SOURCE_BUFFER_CATCHUP"))
        self.assertEqual(self.guard.phase, "RECOVERING")
        self.assertIsNone(self.check(103, sdk_mode=4))
        self.assertEqual(self.guard.phase, "ACTIVE")

    def test_unknown_native_invalid_state_is_fail_closed(self):
        self.assertIsNone(self.check(100))
        self.assertEqual(self.recovery(101, blocker="SOURCE_NOT_FRESH"), "SOURCE_FRESHNESS_LOST")

    def test_timeout_retains_native_reset_diagnostics(self):
        self.assertIsNone(self.check(100))
        self.assertIsNone(self.recovery(101))
        diagnostics = dict(timer_resets=40, last_invalid=dict(reason_mask=8, monotonic_ns=30 * NS))
        with self.assertLogs('streaming_research.runtime.recovery', level='WARNING') as captured:
            self.assertEqual(self.recovery(111, recovery_diagnostics=diagnostics), 'SOURCE_RECOVERY_TIMEOUT')
        self.assertEqual(self.guard.timeout_diagnostics['native_recovery'], diagnostics)
        self.assertIn('RECOVERY_TIMEOUT_DIAGNOSTICS', captured.output[0])
        self.assertEqual(self.guard.first_cause, 'SOURCE_RECOVERY_TIMEOUT')

    def test_invalid_timeouts_cannot_disable_the_bound(self):
        for name in ("acquisition_seconds", "recovery_seconds"):
            for value in (0, -1, float("inf"), float("nan")):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    QualificationGuard(100 * NS, 1000 * NS, 90 * NS, 10 * NS, **{name: value})


if __name__ == "__main__":
    unittest.main()
