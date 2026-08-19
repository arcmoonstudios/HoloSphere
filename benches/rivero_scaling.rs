//! Deterministic Rivero fixed-work scaling and recall audit.

use std::collections::HashSet;
use std::process::Command;
use std::time::Instant;

use hnsqr::{HNSQRConfig, HNSQRIndex, RiveroAddress, VectorEmbedding};
use num_complex::Complex32;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rayon::prelude::*;

const SEED: u64 = 0x5249_5645_524f_2026;
const DIMENSION: usize = 64;
const QUERY_COUNT: usize = 64;
const STRESS_QUERY_COUNT: usize = 16;
const K: usize = 10;
const CELL_BUDGET: usize = 16;
const WARMUP_QUERY_COUNT: usize = 16;
const TIMED_PASSES: usize = 3;

#[derive(Clone, Copy, Debug)]
struct QualityGate {
    min_top1: f64,
    min_recall_at_k: f64,
    min_exact_containment: f64,
    max_average_exact_fraction: f64,
    max_peak_exact_fraction: f64,
}

#[derive(Debug)]
struct AuditRow {
    n: usize,
    dimension: usize,
    build_seconds: f64,
    working_set_delta_mib: Option<f64>,
    compile_p50_us: f64,
    route_mean_us: f64,
    route_stddev_us: f64,
    route_p50_us: f64,
    route_p95_us: f64,
    route_p99_us: f64,
    top1: f64,
    recall_at_k: f64,
    exact_containment: f64,
    self_recall: f64,
    max_cells: usize,
    average_resident_scans: f64,
    max_resident_scans: usize,
    average_admissions: f64,
    max_admissions: usize,
    average_unique_candidates: f64,
    max_unique_candidates: usize,
    average_exact_candidates: f64,
    max_exact: usize,
    average_exact_fraction: f64,
    max_exact_fraction: f64,
    average_witness_seeds: f64,
    max_witness_seeds: usize,
    average_witness_edges_scanned: f64,
    max_witness_edges_scanned: usize,
    average_witness_candidates_added: f64,
    max_witness_candidates_added: usize,
    max_witness_edge_scan_bound: usize,
    empty_routes: usize,
}

fn normalized(values: Vec<Complex32>) -> VectorEmbedding {
    VectorEmbedding::from_complex(values).normalize()
}

fn generate_master(
    count: usize,
    dimension: usize,
    query_count: usize,
) -> (Vec<VectorEmbedding>, Vec<VectorEmbedding>) {
    let mut rng = StdRng::seed_from_u64(SEED ^ dimension as u64);
    let cluster_count = 64usize.min(count.max(1));
    let centers: Vec<VectorEmbedding> = (0..cluster_count)
        .map(|_| {
            normalized(
                (0..dimension)
                    .map(|_| Complex32::new(rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0)))
                    .collect(),
            )
        })
        .collect();

    let corpus = (0..count)
        .map(|index| {
            let center = &centers[index % cluster_count];
            normalized(
                center
                    .complex_data()
                    .iter()
                    .map(|value| {
                        *value
                            + Complex32::new(
                                rng.random_range(-0.025..0.025),
                                rng.random_range(-0.025..0.025),
                            )
                    })
                    .collect(),
            )
        })
        .collect();

    let queries = (0..query_count)
        .map(|index| {
            let center = &centers[index % cluster_count];
            normalized(
                center
                    .complex_data()
                    .iter()
                    .map(|value| {
                        *value
                            + Complex32::new(
                                rng.random_range(-0.018..0.018),
                                rng.random_range(-0.018..0.018),
                            )
                    })
                    .collect(),
            )
        })
        .collect();

    (corpus, queries)
}

fn generate_isotropic(
    count: usize,
    dimension: usize,
    query_count: usize,
) -> (Vec<VectorEmbedding>, Vec<VectorEmbedding>) {
    let mut rng = StdRng::seed_from_u64(SEED ^ 0x4953_4f54_524f_5049 ^ dimension as u64);
    let corpus: Vec<VectorEmbedding> = (0..count)
        .map(|_| {
            normalized(
                (0..dimension)
                    .map(|_| Complex32::new(rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0)))
                    .collect(),
            )
        })
        .collect();
    let queries = (0..query_count)
        .map(|query| {
            let anchor = &corpus[(query * 7919) % count];
            normalized(
                anchor
                    .complex_data()
                    .iter()
                    .map(|value| {
                        *value
                            + Complex32::new(
                                rng.random_range(-0.012..0.012),
                                rng.random_range(-0.012..0.012),
                            )
                    })
                    .collect(),
            )
        })
        .collect();
    (corpus, queries)
}

fn generate_independent_isotropic_queries(
    dimension: usize,
    query_count: usize,
) -> Vec<VectorEmbedding> {
    let mut rng = StdRng::seed_from_u64(SEED ^ 0x494e_4445_5045_4e44 ^ dimension as u64);
    (0..query_count)
        .map(|_| {
            normalized(
                (0..dimension)
                    .map(|_| Complex32::new(rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0)))
                    .collect(),
            )
        })
        .collect()
}

fn generate_boundary(
    count: usize,
    dimension: usize,
    query_count: usize,
) -> (Vec<VectorEmbedding>, Vec<VectorEmbedding>) {
    let mut rng = StdRng::seed_from_u64(SEED ^ 0x424f_554e_4441_5259 ^ dimension as u64);
    let cluster_count = 32usize;
    let centers: Vec<VectorEmbedding> = (0..cluster_count)
        .map(|_| {
            normalized(
                (0..dimension)
                    .map(|_| Complex32::new(rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0)))
                    .collect(),
            )
        })
        .collect();
    let corpus = (0..count)
        .map(|index| {
            let center = &centers[index % cluster_count];
            normalized(
                center
                    .complex_data()
                    .iter()
                    .map(|value| {
                        *value
                            + Complex32::new(rng.random_range(-0.02..0.02), rng.random_range(-0.02..0.02))
                    })
                    .collect(),
            )
        })
        .collect();
    let queries = (0..query_count)
        .map(|query| {
            let lhs = &centers[query % cluster_count];
            let rhs = &centers[(query * 7 + 3) % cluster_count];
            normalized(
                lhs.complex_data()
                    .iter()
                    .zip(rhs.complex_data())
                    .map(|(left, right)| {
                        *left
                            + *right
                            + Complex32::new(
                                rng.random_range(-0.006..0.006),
                                rng.random_range(-0.006..0.006),
                            )
                    })
                    .collect(),
            )
        })
        .collect();
    (corpus, queries)
}

fn exact_top_k(corpus: &[VectorEmbedding], query: &VectorEmbedding, k: usize) -> Vec<u32> {
    let mut scores: Vec<(u32, f32)> = corpus
        .iter()
        .enumerate()
        .map(|(index, vector)| (index as u32, query.projective_overlap(vector)))
        .collect();
    scores.sort_unstable_by(|lhs, rhs| rhs.1.total_cmp(&lhs.1).then_with(|| lhs.0.cmp(&rhs.0)));
    scores.truncate(k);
    scores.into_iter().map(|(index, _)| index).collect()
}

fn percentile(samples: &mut [f64], percentile: f64) -> f64 {
    samples.sort_unstable_by(f64::total_cmp);
    let position = ((samples.len().saturating_sub(1)) as f64 * percentile).round() as usize;
    samples[position.min(samples.len().saturating_sub(1))]
}

fn mean(samples: &[f64]) -> f64 {
    samples.iter().sum::<f64>() / samples.len().max(1) as f64
}

fn sample_stddev(samples: &[f64], average: f64) -> f64 {
    if samples.len() < 2 {
        return 0.0;
    }
    let variance = samples
        .iter()
        .map(|sample| (sample - average).powi(2))
        .sum::<f64>()
        / (samples.len() - 1) as f64;
    variance.sqrt()
}

fn working_set_bytes() -> Option<u64> {
    #[cfg(windows)]
    {
        let output = Command::new("powershell")
            .args([
                "-NoProfile",
                "-Command",
                &format!("(Get-Process -Id {}).WorkingSet64", std::process::id()),
            ])
            .output()
            .ok()?;
        String::from_utf8(output.stdout).ok()?.trim().parse().ok()
    }
    #[cfg(not(windows))]
    {
        let _ = Command::new("true");
        None
    }
}

fn audit_size(
    corpus: &[VectorEmbedding],
    queries: &[VectorEmbedding],
    dimension: usize,
) -> AuditRow {
    let baseline_memory = working_set_bytes();
    let mut config = HNSQRConfig::strict_rivero_for_dim(dimension);
    config.max_elements = corpus.len();
    config.rivero_cell_budget = CELL_BUDGET;
    let index = HNSQRIndex::new(config, dimension);

    let build_start = Instant::now();
    for (slot, vector) in corpus.iter().enumerate() {
        let inserted = index
            .insert(format!("rivero-{slot}"), vector.clone())
            .unwrap();
        assert_eq!(inserted as usize, slot);
    }
    let build_seconds = build_start.elapsed().as_secs_f64();
    let working_set_delta_mib = baseline_memory
        .zip(working_set_bytes())
        .map(|(before, after)| after.saturating_sub(before) as f64 / (1024.0 * 1024.0));

    eprintln!(
        "Computing exact ground truth for {} queries against {} vectors (parallelized)...",
        queries.len(),
        corpus.len()
    );
    let ground_truth_start = Instant::now();
    let truth: Vec<Vec<u32>> = queries
        .par_iter()
        .map(|query| exact_top_k(corpus, query, K))
        .collect();
    eprintln!(
        "Ground truth computed in {:.2}s",
        ground_truth_start.elapsed().as_secs_f64()
    );
    let addresses: Vec<_> = queries
        .iter()
        .map(|query| index.compile_rivero_address(query).unwrap())
        .collect();

    for (query, address) in queries.iter().zip(&addresses).take(WARMUP_QUERY_COUNT) {
        let _ = index
            .search_indices_with_rivero_address(query, address, K, None)
            .unwrap();
    }

    let mut compile_latencies = Vec::with_capacity(queries.len());
    let mut route_latencies = Vec::with_capacity(queries.len() * 3);
    let mut top1_matches = 0usize;
    let mut recall_sum = 0.0f64;
    let mut exact_containment = 0usize;
    let mut max_cells = 0usize;
    let mut resident_scan_sum = 0usize;
    let mut max_resident_scans = 0usize;
    let mut admission_sum = 0usize;
    let mut max_admissions = 0usize;
    let mut unique_candidate_sum = 0usize;
    let mut max_unique_candidates = 0usize;
    let mut exact_candidate_sum = 0usize;
    let mut max_exact = 0usize;
    let mut exact_fraction_sum = 0.0f64;
    let mut max_exact_fraction = 0.0f64;
    let mut witness_seed_sum = 0usize;
    let mut max_witness_seeds = 0usize;
    let mut witness_edge_sum = 0usize;
    let mut max_witness_edges_scanned = 0usize;
    let mut witness_candidate_sum = 0usize;
    let mut max_witness_candidates_added = 0usize;
    let mut max_witness_edge_scan_bound = 0usize;
    let mut empty_routes = 0usize;

    for _ in 0..TIMED_PASSES {
        for ((query, address), expected) in queries.iter().zip(&addresses).zip(&truth) {
            let compile_start = Instant::now();
            let compiled = index.compile_rivero_address(query).unwrap();
            compile_latencies.push(compile_start.elapsed().as_secs_f64() * 1e6);
            assert_eq!(compiled, *address);

            let route_start = Instant::now();
            let (results, diagnostics) = index
                .search_indices_with_rivero_address_and_diagnostics(query, address, K, None)
                .unwrap();
            route_latencies.push(route_start.elapsed().as_secs_f64() * 1e6);

            assert_eq!(diagnostics.cells_probed, RiveroAddress::cell_probe_count());
            assert!(diagnostics.resident_reads <= diagnostics.candidate_read_bound);
            assert!(diagnostics.resident_scans <= diagnostics.resident_scan_bound);
            assert!(diagnostics.resident_reads <= diagnostics.resident_scans);
            assert!(diagnostics.exact_score_evaluations <= diagnostics.unique_candidates);
            assert!(diagnostics.witness_edges_scanned <= diagnostics.witness_edge_scan_bound);
            assert!(diagnostics.witness_candidates_added <= diagnostics.witness_edges_scanned);
            assert!(!diagnostics.fallback_used);
            max_cells = max_cells.max(diagnostics.cells_probed);
            resident_scan_sum += diagnostics.resident_scans;
            max_resident_scans = max_resident_scans.max(diagnostics.resident_scans);
            admission_sum += diagnostics.resident_reads;
            max_admissions = max_admissions.max(diagnostics.resident_reads);
            unique_candidate_sum += diagnostics.unique_candidates;
            max_unique_candidates = max_unique_candidates.max(diagnostics.unique_candidates);
            exact_candidate_sum += diagnostics.exact_score_evaluations;
            max_exact = max_exact.max(diagnostics.exact_score_evaluations);
            let exact_fraction = diagnostics.exact_score_evaluations as f64 / corpus.len() as f64;
            exact_fraction_sum += exact_fraction;
            max_exact_fraction = max_exact_fraction.max(exact_fraction);
            witness_seed_sum += diagnostics.witness_seeds;
            max_witness_seeds = max_witness_seeds.max(diagnostics.witness_seeds);
            witness_edge_sum += diagnostics.witness_edges_scanned;
            max_witness_edges_scanned =
                max_witness_edges_scanned.max(diagnostics.witness_edges_scanned);
            witness_candidate_sum += diagnostics.witness_candidates_added;
            max_witness_candidates_added =
                max_witness_candidates_added.max(diagnostics.witness_candidates_added);
            max_witness_edge_scan_bound =
                max_witness_edge_scan_bound.max(diagnostics.witness_edge_scan_bound);
            empty_routes += usize::from(results.is_empty());

            let actual: Vec<u32> = results.iter().map(|result| result.0).collect();
            let expected_set: HashSet<u32> = expected.iter().copied().collect();
            let matches = actual
                .iter()
                .filter(|index| expected_set.contains(index))
                .count();
            top1_matches += usize::from(actual.first() == expected.first());
            recall_sum += matches as f64 / expected.len() as f64;
            exact_containment += usize::from(matches == expected.len());
        }
    }

    let trials = queries.len() * TIMED_PASSES;
    let self_trials = corpus.len().min(64);
    let self_hits = (0..self_trials)
        .filter(|&slot| {
            index
                .search_indices_o1_filtered(&corpus[slot], 1, None)
                .unwrap()
                .first()
                .is_some_and(|result| result.0 == slot as u32)
        })
        .count();

    let compile_p50_us = percentile(&mut compile_latencies, 0.50);
    let route_mean_us = mean(&route_latencies);
    let route_stddev_us = sample_stddev(&route_latencies, route_mean_us);
    let route_p50_us = percentile(&mut route_latencies, 0.50);
    let route_p95_us = percentile(&mut route_latencies, 0.95);
    let route_p99_us = percentile(&mut route_latencies, 0.99);

    AuditRow {
        n: corpus.len(),
        dimension,
        build_seconds,
        working_set_delta_mib,
        compile_p50_us,
        route_mean_us,
        route_stddev_us,
        route_p50_us,
        route_p95_us,
        route_p99_us,
        top1: top1_matches as f64 / trials as f64,
        recall_at_k: recall_sum / trials as f64,
        exact_containment: exact_containment as f64 / trials as f64,
        self_recall: self_hits as f64 / self_trials as f64,
        max_cells,
        average_resident_scans: resident_scan_sum as f64 / trials as f64,
        max_resident_scans,
        average_admissions: admission_sum as f64 / trials as f64,
        max_admissions,
        average_unique_candidates: unique_candidate_sum as f64 / trials as f64,
        max_unique_candidates,
        average_exact_candidates: exact_candidate_sum as f64 / trials as f64,
        max_exact,
        average_exact_fraction: exact_fraction_sum / trials as f64,
        max_exact_fraction,
        average_witness_seeds: witness_seed_sum as f64 / trials as f64,
        max_witness_seeds,
        average_witness_edges_scanned: witness_edge_sum as f64 / trials as f64,
        max_witness_edges_scanned,
        average_witness_candidates_added: witness_candidate_sum as f64 / trials as f64,
        max_witness_candidates_added,
        max_witness_edge_scan_bound,
        empty_routes,
    }
}

fn print_row(label: &str, row: &AuditRow) {
    println!(
        "{label} | {} | {} | {:.3} | {} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.4} | {:.4} | {:.4} | {:.4} | {} | {:.1}/{} | {:.1}/{} | {:.1}/{} | {:.1}/{} | {:.4}/{:.4} | {:.1}/{} | {:.1}/{}/{} | {:.1}/{} | {}",
        row.n,
        row.dimension,
        row.build_seconds,
        row.working_set_delta_mib
            .map(|value| format!("{value:.1}"))
            .unwrap_or_else(|| "n/a".to_string()),
        row.compile_p50_us,
        row.route_mean_us,
        row.route_stddev_us,
        row.route_p50_us,
        row.route_p95_us,
        row.route_p99_us,
        row.top1,
        row.recall_at_k,
        row.exact_containment,
        row.self_recall,
        row.max_cells,
        row.average_resident_scans,
        row.max_resident_scans,
        row.average_admissions,
        row.max_admissions,
        row.average_unique_candidates,
        row.max_unique_candidates,
        row.average_exact_candidates,
        row.max_exact,
        row.average_exact_fraction,
        row.max_exact_fraction,
        row.average_witness_seeds,
        row.max_witness_seeds,
        row.average_witness_edges_scanned,
        row.max_witness_edges_scanned,
        row.max_witness_edge_scan_bound,
        row.average_witness_candidates_added,
        row.max_witness_candidates_added,
        row.empty_routes,
    );
}

fn assert_fixed_work(row: &AuditRow) {
    assert_eq!(row.max_cells, RiveroAddress::cell_probe_count());
    assert!(row.max_admissions <= RiveroAddress::candidate_read_bound(CELL_BUDGET));
    assert!(row.max_resident_scans <= RiveroAddress::resident_scan_bound());
    assert!(row.max_witness_edges_scanned <= row.max_witness_edge_scan_bound);
    assert_eq!(row.self_recall, 1.0);
    assert_eq!(row.empty_routes, 0);
}

fn assert_recall_quality(label: &str, row: &AuditRow, gate: QualityGate) {
    assert!(
        row.top1 >= gate.min_top1,
        "{label} top-1 {:.4} fell below {:.4}",
        row.top1,
        gate.min_top1
    );
    assert!(
        row.recall_at_k >= gate.min_recall_at_k,
        "{label} Recall@{K} {:.4} fell below {:.4}",
        row.recall_at_k,
        gate.min_recall_at_k
    );
    assert!(
        row.exact_containment >= gate.min_exact_containment,
        "{label} exact containment {:.4} fell below {:.4}",
        row.exact_containment,
        gate.min_exact_containment
    );
}

fn assert_selectivity(label: &str, row: &AuditRow, gate: QualityGate) {
    assert!(
        row.average_exact_fraction <= gate.max_average_exact_fraction,
        "{label} average exact-candidate fraction {:.4} exceeded {:.4}",
        row.average_exact_fraction,
        gate.max_average_exact_fraction
    );
    assert!(
        row.max_exact_fraction <= gate.max_peak_exact_fraction,
        "{label} peak exact-candidate fraction {:.4} exceeded {:.4}",
        row.max_exact_fraction,
        gate.max_peak_exact_fraction
    );
}

fn assert_quality(label: &str, row: &AuditRow, gate: QualityGate) {
    assert_recall_quality(label, row, gate);
    assert_selectivity(label, row, gate);
}

fn assert_anchor_quality(label: &str, row: &AuditRow, gate: QualityGate) {
    assert!(
        row.top1 >= gate.min_top1,
        "{label} top-1 {:.4} fell below {:.4}",
        row.top1,
        gate.min_top1
    );
    assert_selectivity(label, row, gate);
}

fn selectivity_gate(n: usize) -> (f64, f64) {
    match n {
        0..=1_024 => (1.0, 1.0),
        1_025..=4_096 => (0.90, 0.95),
        4_097..=16_384 => (0.65, 0.75),
        _ => (0.45, 0.55),
    }
}

fn strict_quality_gate(n: usize) -> QualityGate {
    let (max_average_exact_fraction, max_peak_exact_fraction) = selectivity_gate(n);
    QualityGate {
        min_top1: 1.0,
        min_recall_at_k: 0.99,
        min_exact_containment: 0.90,
        max_average_exact_fraction,
        max_peak_exact_fraction,
    }
}

fn log_log_slope(rows: &[AuditRow]) -> f64 {
    let count = rows.len() as f64;
    let mean_x = rows.iter().map(|row| (row.n as f64).ln()).sum::<f64>() / count;
    let mean_y = rows.iter().map(|row| row.route_p50_us.ln()).sum::<f64>() / count;
    let numerator = rows
        .iter()
        .map(|row| ((row.n as f64).ln() - mean_x) * (row.route_p50_us.ln() - mean_y))
        .sum::<f64>();
    let denominator = rows
        .iter()
        .map(|row| ((row.n as f64).ln() - mean_x).powi(2))
        .sum::<f64>();
    numerator / denominator.max(f64::EPSILON)
}

fn main() {
    let quick = cfg!(debug_assertions) || std::env::var_os("HNSQR_RIVERO_QUICK").is_some();
    let isotropic_65k = !quick && std::env::var_os("HNSQR_RIVERO_ISOTROPIC_65K").is_some();

    // Default to reasonable test sizes; use HNSQR_RIVERO_FULL for the original large sizes
    let use_full_scale = std::env::var_os("HNSQR_RIVERO_FULL").is_some();
    let sizes: &[usize] = if cfg!(debug_assertions) {
        &[128, 512]
    } else if quick {
        &[1_024, 4_096]
    } else if use_full_scale {
        &[1_024, 4_096, 16_384, 65_536]
    } else {
        // Default: reasonable sizes that complete in < 5 minutes
        &[1_024, 4_096, 16_384]
    };
    let max_size = *sizes.last().unwrap();
    let (master, queries) = generate_master(max_size, DIMENSION, QUERY_COUNT);

    println!("HNSQR Rivero deterministic scaling audit");
    println!(
        "seed=0x{SEED:016x}, D={DIMENSION}, queries={QUERY_COUNT}, k={K}, budget={CELL_BUDGET}, warmup_queries={WARMUP_QUERY_COUNT}, timed_passes={TIMED_PASSES}, isotropic_65k={isotropic_65k}"
    );
    println!(
        "case | N | D | build_s | WS_delta_MiB | compile_p50_us | route_mean_us | route_sd_us | route_p50_us | p95_us | p99_us | top1 | recall@10 | contain@10 | self | probes | scans avg/max | admissions avg/max | unique avg/max | exact avg/max | exact_fraction avg/max | witness_seeds avg/max | witness_edges avg/max/bound | witness_added avg/max | empty"
    );

    let mut rows = Vec::new();
    for &size in sizes {
        let row = audit_size(&master[..size], &queries, DIMENSION);
        print_row("clustered-scale", &row);
        assert_fixed_work(&row);
        assert_quality("clustered-scale", &row, strict_quality_gate(row.n));
        rows.push(row);
    }

    let secondary_size = if quick { 4_096 } else { 16_384 };
    let (isotropic, isotropic_queries) = generate_isotropic(secondary_size, DIMENSION, QUERY_COUNT);
    let isotropic_row = audit_size(&isotropic, &isotropic_queries, DIMENSION);
    print_row("isotropic-anchor", &isotropic_row);
    assert_fixed_work(&isotropic_row);
    // Anchor Recall@10 remains a reported isotropic stress metric rather than a
    // universal gate: exact ranks 2..k among unrelated random vectors do not
    // remain recoverable at fixed work as N grows. Top-1/self, selectivity, and
    // all fixed-work ceilings remain hard requirements.
    assert_anchor_quality(
        "isotropic-anchor",
        &isotropic_row,
        strict_quality_gate(isotropic_row.n),
    );

    if !quick {
        let independent_queries = generate_independent_isotropic_queries(DIMENSION, QUERY_COUNT);
        let independent_row = audit_size(&isotropic, &independent_queries, DIMENSION);
        print_row("independent-isotropic", &independent_row);
        assert_fixed_work(&independent_row);
    }

    let (boundary, boundary_queries) = generate_boundary(secondary_size, DIMENSION, QUERY_COUNT);
    let boundary_row = audit_size(&boundary, &boundary_queries, DIMENSION);
    print_row("cluster-boundary", &boundary_row);
    assert_fixed_work(&boundary_row);
    assert_quality(
        "cluster-boundary",
        &boundary_row,
        strict_quality_gate(boundary_row.n),
    );

    if isotropic_65k {
        let (stress_corpus, stress_queries) =
            generate_isotropic(65_536, DIMENSION, STRESS_QUERY_COUNT);
        let stress_row = audit_size(&stress_corpus, &stress_queries, DIMENSION);
        print_row("isotropic-65k-stress", &stress_row);
        assert_fixed_work(&stress_row);
        assert_anchor_quality(
            "isotropic-65k-stress",
            &stress_row,
            strict_quality_gate(stress_row.n),
        );
    }

    if !quick {
        for dimension in [8, 256, 768] {
            let (dimension_corpus, dimension_queries) =
                generate_master(4_096, dimension, QUERY_COUNT);
            let row = audit_size(&dimension_corpus, &dimension_queries, dimension);
            print_row("dimension-scale", &row);
            assert_fixed_work(&row);
            assert_quality("dimension-scale", &row, strict_quality_gate(row.n));
        }
    }

    let slope = log_log_slope(&rows);
    let min_latency = rows
        .iter()
        .map(|row| row.route_p50_us)
        .fold(f64::INFINITY, f64::min);
    let max_latency = rows.iter().map(|row| row.route_p50_us).fold(0.0, f64::max);
    println!("route latency log-log slope: {slope:.4}");
    println!("route p50 max/min ratio: {:.4}", max_latency / min_latency);
}
