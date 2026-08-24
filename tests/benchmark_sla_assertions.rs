/* holosphere/tests/benchmark_sla_assertions.rs */
//! Benchmark SLA Assertions Unit Tests

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SlaGate {
    pub target_us: f64,
    pub max_acceptable_us: f64,
    pub min_recall_pct: f64,
}

impl SlaGate {
    pub fn evaluate_latency(&self, measured_us: f64) -> bool {
        measured_us <= self.target_us
    }

    pub fn evaluate_recall(&self, measured_recall_pct: f64) -> bool {
        measured_recall_pct >= self.min_recall_pct
    }

    pub fn evaluate_admission(&self, recall_pct: f64, exact_lat_us: f64, cand_lat_us: f64) -> bool {
        let speedup = exact_lat_us / cand_lat_us.max(1e-6);
        recall_pct >= self.min_recall_pct && speedup >= 2.0
    }
}

#[test]
fn test_cold_attach_sla_detects_violation() {
    let cold_attach_sla = SlaGate {
        target_us: 10_000.0, // 10 ms
        max_acceptable_us: 10_000.0,
        min_recall_pct: 100.0,
    };

    // 475 ms should definitely fail a 10 ms target
    let measured_475ms_us = 475_531.0;
    assert!(
        !cold_attach_sla.evaluate_latency(measured_475ms_us),
        "475ms must fail 10ms SLA"
    );

    // 5 ms should pass
    let measured_5ms_us = 5_000.0;
    assert!(
        cold_attach_sla.evaluate_latency(measured_5ms_us),
        "5ms must pass 10ms SLA"
    );
}

#[test]
fn test_warm_p50_sub_millisecond_sla() {
    let sub_ms_sla = SlaGate {
        target_us: 1_000.0, // 1 ms
        max_acceptable_us: 1_000.0,
        min_recall_pct: 99.0,
    };

    // 1.31 ms must fail sub-millisecond target
    let measured_1310_us = 1_310.0;
    assert!(
        !sub_ms_sla.evaluate_latency(measured_1310_us),
        "1.31ms must fail <1ms SLA"
    );

    // 400 µs must pass
    let measured_400_us = 400.0;
    assert!(
        sub_ms_sla.evaluate_latency(measured_400_us),
        "400us must pass <1ms SLA"
    );
}

#[test]
fn test_approximate_admission_gate() {
    let gate = SlaGate {
        target_us: 10_000.0,
        max_acceptable_us: 10_000.0,
        min_recall_pct: 99.0,
    };

    // 99.8% recall, 1.24x speedup -> MUST FAIL (requires >= 2.0x)
    assert!(
        !gate.evaluate_admission(99.8, 58.2, 46.9),
        "1.24x speedup must not pass 2x admission gate"
    );

    // 99.1% recall, 2.5x speedup -> MUST PASS
    assert!(
        gate.evaluate_admission(99.1, 58.2, 23.0),
        "99.1% recall with 2.5x speedup must pass admission gate"
    );

    // 94.0% recall, 5.0x speedup -> MUST FAIL (failed recall survival gate)
    assert!(
        !gate.evaluate_admission(94.0, 58.2, 10.0),
        "94% recall must fail admission gate regardless of speedup"
    );
}

#[test]
fn test_speedup_vs_latency_ratio_arithmetic() {
    let exact_us: f64 = 483.6;
    let candidate_us: f64 = 47.0;

    let latency_ratio: f64 = candidate_us / exact_us;
    let speedup: f64 = exact_us / candidate_us;

    assert!((latency_ratio - 0.09718).abs() < 1e-4);
    assert!((speedup - 10.289).abs() < 1e-2);
    assert_eq!(speedup > 1.0, latency_ratio < 1.0);
}
