use hnsqr::bench_support as common;

use std::time::Instant;

use common::load_real_dataset_corpus;
use hnsqr::rivero::RiveroProfile;

fn percentile(mut latencies: Vec<f64>, p: f64) -> f64 {
    if latencies.is_empty() {
        return 0.0;
    }
    latencies.sort_by(|a, b| a.total_cmp(b));
    let idx = ((latencies.len() as f64 - 1.0) * (p / 100.0)).round() as usize;
    latencies[idx]
}

struct DimCrossoverResult {
    real_dim: usize,
    complex_dim: usize,
    measured_crossover: usize,
}

fn measure_point(n: usize, complex_dim: usize, num_queries: usize) -> (f64, f64) {
    let dataset =
        load_real_dataset_corpus(n, num_queries, complex_dim * 2, common::DEFAULT_BENCH_SEED);

    let actual_complex_dim = dataset.complex_dim;
    let index = common::open_prebuilt_index(
        &format!("dim_crossover_n{n}_d{actual_complex_dim}"),
        &dataset.folded_corpus,
        actual_complex_dim,
        RiveroProfile::Balanced,
    );

    // 1. Exact Scan
    let mut exact_lats = Vec::with_capacity(num_queries);
    for q in &dataset.folded_queries {
        let t0 = Instant::now();
        let _ = index.search_indices_exact(q, 10, None).unwrap();
        exact_lats.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    let exact_p50 = percentile(exact_lats, 50.0);

    // 2. Fast Rivero
    let mut rivero_lats = Vec::with_capacity(num_queries);
    let fast_cfg = RiveroProfile::Fast.config();
    for q in &dataset.folded_queries {
        let t0 = Instant::now();
        let _ = index
            .search_indices_o1_with_config(q, 10, None, &fast_cfg)
            .unwrap();
        rivero_lats.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    let rivero_p50 = percentile(rivero_lats, 50.0);

    (exact_p50, rivero_p50)
}

fn find_crossover(complex_dim: usize) -> Option<(usize, f64, f64)> {
    let num_queries = 16;
    // Compute the real-dim for this complex_dim tier and check dataset capacity.
    let real_dim = complex_dim * 2;
    let available = common::corpus_available_count(real_dim);

    let all_test_points: Vec<usize> = match complex_dim {
        16..=32 => vec![25_000, 40_000, 50_000, 60_000, 75_000],
        33..=64 => vec![12_000, 20_000, 26_000, 32_000, 40_000],
        65..=128 => vec![6_000, 10_000, 14_000, 18_000, 24_000],
        129..=192 => vec![4_000, 7_000, 10_000, 13_000, 16_000],
        193..=256 => vec![3_000, 5_000, 7_500, 10_000, 13_000],
        257..=384 => vec![2_000, 3_500, 5_000, 7_000, 9_000],
        385..=512 => vec![1_500, 2_800, 4_000, 5_500, 7_500],
        513..=768 => vec![1_200, 2_000, 3_000, 4_200, 5_500],
        769..=1024 => vec![1_000, 1_600, 2_400, 3_400, 4_500],
        1025..=1536 => vec![800, 1_300, 1_900, 2_600, 3_500],
        _ => vec![600, 1_000, 1_500, 2_100, 2_800],
    };

    // Clamp to what the dataset file actually contains.
    let test_points: Vec<usize> = all_test_points
        .into_iter()
        .filter(|&n| n <= available)
        .collect();

    if test_points.len() < 2 {
        // Not enough probe points to bracket a crossover — skip this dimension tier.
        return None;
    }

    let mut points: Vec<(usize, f64, f64)> = Vec::new();
    for &n in &test_points {
        let (exact_p50, rivero_p50) = measure_point(n, complex_dim, num_queries);
        points.push((n, exact_p50, rivero_p50));
    }

    // Find crossover point by linear interpolation
    for i in 0..points.len() - 1 {
        let (n1, e1, r1) = points[i];
        let (n2, e2, r2) = points[i + 1];
        let diff1 = e1 - r1;
        let diff2 = e2 - r2;

        if (diff1 <= 0.0 && diff2 >= 0.0) || (diff1 >= 0.0 && diff2 <= 0.0) {
            let t = if (diff2 - diff1).abs() > 1e-9 {
                (-diff1) / (diff2 - diff1)
            } else {
                0.5
            };
            let n_cross = (n1 as f64 + t * (n2 as f64 - n1 as f64)).round() as usize;
            let e_cross = e1 + t * (e2 - e1);
            let r_cross = r1 + t * (r2 - r1);
            return Some((n_cross, e_cross, r_cross));
        }
    }

    let (n_last, e_last, r_last) = points.last().copied().unwrap();
    Some((n_last, e_last, r_last))
}

fn fit_model(results: &[DimCrossoverResult]) -> (f64, f64, f64) {
    let mut best_a = 0.0f64;
    let mut best_b = 0.0f64;
    let mut best_p = 1.0f64;
    let mut best_loss = f64::MAX;

    for a_step in 0..=80 {
        let a = a_step as f64 * 50.0; // A in 0..4000
        for p_step in 70..=130 {
            let p = p_step as f64 * 0.01; // p in 0.70..1.30

            // Solve for optimal B analytically: B = sum((N - A) * d^-p) / sum(d^-2p)
            let mut num = 0.0f64;
            let mut denom = 0.0f64;
            for r in results {
                let d = r.complex_dim as f64;
                let weight = d.powf(-p);
                num += (r.measured_crossover as f64 - a) * weight;
                denom += weight * weight;
            }

            if denom > 1e-12 {
                let b = (num / denom).max(0.0);
                let mut loss = 0.0f64;
                for r in results {
                    let d = r.complex_dim as f64;
                    let pred = a + b / d.powf(p);
                    let rel_err =
                        (pred - r.measured_crossover as f64) / (r.measured_crossover as f64);
                    loss += rel_err * rel_err;
                }

                if loss < best_loss {
                    best_loss = loss;
                    best_a = a;
                    best_b = b;
                    best_p = p;
                }
            }
        }
    }

    (best_a, best_b, best_p)
}

fn main() {
    println!(
        "╔══════════════════════════════════════════════════════════════════════════════════════╗"
    );
    println!(
        "║ HNSQR HIGH-DIMENSIONAL CROSSOVER EMPIRICAL CHARACTERIZATION (D=32..4096 REAL)       ║"
    );
    println!(
        "╚══════════════════════════════════════════════════════════════════════════════════════╝\n"
    );

    let dims = if cfg!(debug_assertions) {
        vec![(64, 32), (128, 64)]
    } else {
        vec![
            (64, 32),
            (128, 64),
            (256, 128),
            (384, 192),
            (512, 256),
            (768, 384),
            (1024, 512),
            (1536, 768),
            (2048, 1024),
            (3072, 1536),
            (4096, 2048),
        ]
    };

    println!("  ┌────────┬─────────┬───────────────────┬────────────────────────────────┐");
    println!("  │ Real D │ Cmplx D │ Empirical Cross N │ Crossover Latency (Exact/Riv)  │");
    println!("  ├────────┼─────────┼───────────────────┼────────────────────────────────┤");

    let mut results: Vec<DimCrossoverResult> = Vec::new();

    for &(real_d, complex_d) in &dims {
        match find_crossover(complex_d) {
            Some((n_cross, e_lat, r_lat)) => {
                results.push(DimCrossoverResult {
                    real_dim: real_d,
                    complex_dim: complex_d,
                    measured_crossover: n_cross,
                });
                println!(
                    "  │ {:>6} │ {:>7} │ {:>15} N │ {:>6.3} ms  /  {:>6.3} ms    │",
                    real_d, complex_d, n_cross, e_lat, r_lat
                );
            }
            None => {
                let available = common::corpus_available_count(real_d);
                println!(
                    "  │ {:>6} │ {:>7} │ {:>15} │ {:>31} │",
                    real_d,
                    complex_d,
                    "SKIPPED",
                    format!("dataset only has {available} vectors")
                );
            }
        }
    }
    println!("  └────────┴─────────┴───────────────────┴────────────────────────────────┘\n");

    // Fit Mathematical Model: N_cross(D) = A + B / D^p
    let (a, b, p) = fit_model(&results);

    println!(
        "════════════════════════════════════════════════════════════════════════════════════════"
    );
    println!(" CALIBRATED MATHEMATICAL CROSSOVER MODEL: N_cross(D) = A + B / D^p");
    println!(
        "════════════════════════════════════════════════════════════════════════════════════════"
    );
    println!("  • Theoretical Floor (A)   : {:.1}", a);
    println!("  • Scale Coefficient (B)   : {:.1}", b);
    println!("  • Dimension Exponent (p)  : {:.3}", p);
    println!(
        "  • Formula                 : N_cross(D_complex) = {:.1} + {:.1} / (D_complex^{:.3})\n",
        a, b, p
    );

    println!("  ┌────────┬─────────┬──────────────┬──────────────────┬──────────────┐");
    println!("  │ Real D │ Cmplx D │ Measured N   │ Model Prediction │ Rel Error    │");
    println!("  ├────────┼─────────┼──────────────┼──────────────────┼──────────────┤");

    for r in &results {
        let d = r.complex_dim as f64;
        let pred = a + b / d.powf(p);
        let rel_err =
            ((pred - r.measured_crossover as f64).abs() / r.measured_crossover as f64) * 100.0;
        println!(
            "  │ {:>6} │ {:>7} │ {:>10} N │ {:>14.0} N │ {:>10.2}% │",
            r.real_dim, r.complex_dim, r.measured_crossover, pred, rel_err
        );
    }
    println!("  └────────┴─────────┴──────────────┴──────────────────┴──────────────┘\n");
}
