//! Gate A2: Rivero Address Capacity, Funnel Semantics & Set Accounting Benchmark
//!
//! Evaluates the 9-point matrix across D_real in [384, 1536, 4096] and geometries
//! (24F Global, 64F Global, 64F MultiLane) with full set-level materialization:
//!   - |GT ∩ raw|: True Raw Territory Recall
//!   - |GT ∩ vote|: Vote-Selected Recall (< candidate_cap)
//!   - |GT ∩ witness|: Post-Witness Recall
//!   - |GT ∩ (raw - vote)|: GT neighbors dropped by cap truncation
//!   - |GT ∩ (witness - raw)|: GT neighbors discovered outside raw territory by witnesses

use hnsqr::bench_support as common;

use std::collections::HashSet;
use std::time::Instant;

use hnsqr::rivero::bulk::RiveroBulkBuilder;
use hnsqr::rivero::{
    LaneAssignment, RiveroAddressConfig, RiveroCompiler, RiveroConfig, RiveroProjectionMode,
};
use hnsqr::{DistanceFunction, NodeIndex, VectorEmbedding};
use rayon::prelude::*;

#[derive(Debug, Clone)]
struct RunConfig {
    name: String,
    d_real: usize,
    foundations: u8,
    projection: RiveroProjectionMode,
}

impl RunConfig {
    fn c_addr(&self) -> f64 {
        self.d_real as f64 / (8.0 * self.foundations as f64)
    }

    fn address_config(&self) -> RiveroAddressConfig {
        RiveroAddressConfig {
            foundations: self.foundations,
            projection: self.projection,
            geometry: hnsqr::rivero::VectorGeometry::Real,
        }
    }
}

#[derive(Debug, Clone, Default)]
struct MetricSummary {
    c_addr: f64,
    raw_recall_10: f64,
    vote_recall_10: f64,
    witness_recall_10: f64,
    gt_dropped_by_cap: f64,
    gt_escaped_via_witness: f64,
    raw_unique_cands: f64,
    gt_raw_mean_rank: f64,
    total_lat_p50_us: f64,
}

#[inline]
fn cosine_sim(a: &VectorEmbedding, b: &VectorEmbedding) -> f32 {
    1.0 - a.cosine_distance(b)
}

fn percentile(latencies: &mut [f64], p: f64) -> f64 {
    if latencies.is_empty() {
        return 0.0;
    }
    latencies.sort_by(|a, b| a.total_cmp(b));
    let idx = ((latencies.len() as f64 - 1.0) * (p / 100.0)).round() as usize;
    latencies[idx.min(latencies.len() - 1)]
}

/// Evaluates a single configuration with strict set materialization and assertion checks.
fn evaluate_geometry(
    cfg: &RunConfig,
    corpus: &[VectorEmbedding],
    queries: &[VectorEmbedding],
    k: usize,
    candidate_cap: usize,
) -> MetricSummary {
    let num_queries = queries.len();
    let complex_dim = cfg.d_real / 2;

    // 1. Build territory index and witnesses using RiveroBulkBuilder with custom RiveroAddressConfig
    let rivero_cfg = RiveroConfig {
        foundations: cfg.foundations as usize,
        simhash_query_probes: 32,
        cell_capacity: 64,
        affinity_elites: 24,
        cell_budget: 16,
        query_candidate_cap: candidate_cap,
    };

    let builder = RiveroBulkBuilder::new(rivero_cfg)
        .with_address_config(cfg.address_config())
        .with_distance_function(DistanceFunction::Cosine);

    let built = builder.build(corpus).expect("Bulk build must succeed");

    let territory = &built.territory;
    let witnesses = &built.witnesses;
    let compiler = RiveroCompiler::with_config(complex_dim, cfg.address_config());

    // 2. Precompute Ground Truth Top-K for all queries
    let gt_top_k: Vec<Vec<NodeIndex>> = queries
        .par_iter()
        .map(|q| {
            let mut scores: Vec<(NodeIndex, f32)> = corpus
                .iter()
                .enumerate()
                .map(|(idx, doc)| {
                    let sim = cosine_sim(q, doc);
                    (idx as NodeIndex, sim)
                })
                .collect();
            scores.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            scores.truncate(k);
            scores.into_iter().map(|(idx, _)| idx).collect()
        })
        .collect();

    // 3. Set-Level Diagnostics Accumulators
    let mut raw_recalls = Vec::with_capacity(num_queries);
    let mut vote_recalls = Vec::with_capacity(num_queries);
    let mut witness_recalls = Vec::with_capacity(num_queries);
    let mut gt_dropped_by_cap_vec = Vec::with_capacity(num_queries);
    let mut gt_escaped_via_witness_vec = Vec::with_capacity(num_queries);

    let mut raw_cands_counts = Vec::with_capacity(num_queries);
    let mut gt_voted_ranks = Vec::new();
    let mut total_times = Vec::with_capacity(num_queries);

    for (q_idx, query) in queries.iter().enumerate() {
        let gt = &gt_top_k[q_idx];

        let t_start = Instant::now();

        // Stage 1: Compile Address
        let addr = compiler.compile(query.complex_data());

        // Stage 2: Materialize Raw Territory and Vote-Selected Sets
        let mut raw_set: HashSet<NodeIndex> = HashSet::new();
        let mut vote_selected_set: HashSet<NodeIndex> = HashSet::new();
        let mut selected_candidates: Vec<NodeIndex> = Vec::new();
        let mut voted_slots_ranked: Vec<NodeIndex> = Vec::new();

        territory.with_voted_candidates_config(&addr, &rivero_cfg, |cands, voted, _diag| {
            for v in voted {
                raw_set.insert(v.slot);
                voted_slots_ranked.push(v.slot);
            }
            for &c in cands {
                vote_selected_set.insert(c);
            }
            selected_candidates.extend_from_slice(cands);
        });

        // Verify Set Consistency & Hard Assertions
        let raw_hits = gt.iter().filter(|id| raw_set.contains(id)).count();
        let vote_hits = gt
            .iter()
            .filter(|id| vote_selected_set.contains(id))
            .count();

        let vote_rank_survival_count = gt
            .iter()
            .filter(|&&id| {
                voted_slots_ranked
                    .iter()
                    .position(|&slot| slot == id)
                    .is_some_and(|rank| rank < candidate_cap)
            })
            .count();

        assert_eq!(
            vote_hits, vote_rank_survival_count,
            "Contradiction A detected: Vote recall and GT vote-rank survival diverged!"
        );

        raw_recalls.push(raw_hits as f64 / k as f64);
        vote_recalls.push(vote_hits as f64 / k as f64);

        // Track GT dropped by candidate cap: |GT ∩ (raw - vote)|
        let dropped_by_cap = gt
            .iter()
            .filter(|id| raw_set.contains(id) && !vote_selected_set.contains(id))
            .count();
        gt_dropped_by_cap_vec.push(dropped_by_cap as f64 / k as f64);

        // Stage 3: Witness Expansion
        let mut post_witness_set: HashSet<NodeIndex> = vote_selected_set.clone();

        // Exact rank top seeds for witness expansion
        let mut seed_scores: Vec<(NodeIndex, f32)> = selected_candidates
            .iter()
            .take(48)
            .map(|&idx| (idx, cosine_sim(query, &corpus[idx as usize])))
            .collect();
        seed_scores.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

        for &(seed, _) in seed_scores.iter().take(32) {
            let seed_witnesses = &witnesses[seed as usize];
            for w in seed_witnesses.iter() {
                post_witness_set.insert(w.index);
            }
        }

        // Trace GT in Post-Witness Candidates
        let witness_hits = gt.iter().filter(|id| post_witness_set.contains(id)).count();
        witness_recalls.push(witness_hits as f64 / k as f64);

        // Track GT discovered by witnesses OUTSIDE raw territory: |GT ∩ (witness - raw)|
        let escaped_via_witness = gt
            .iter()
            .filter(|id| post_witness_set.contains(id) && !raw_set.contains(id))
            .count();
        gt_escaped_via_witness_vec.push(escaped_via_witness as f64 / k as f64);

        // Record Candidate Universe Counts
        raw_cands_counts.push(raw_set.len());

        // Stage 4: Exact Rerank over post_witness_set
        let mut final_ranked: Vec<(NodeIndex, f32)> = post_witness_set
            .iter()
            .map(|&idx| (idx, cosine_sim(query, &corpus[idx as usize])))
            .collect();
        final_ranked.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        final_ranked.truncate(k);

        let total_us = t_start.elapsed().as_secs_f64() * 1_000_000.0;
        total_times.push(total_us);

        // Compute exact rank in the full voted list for all GT
        for id in gt {
            if let Some(pos) = voted_slots_ranked.iter().position(|x| x == id) {
                gt_voted_ranks.push(pos as f64);
            } else {
                gt_voted_ranks.push(raw_set.len() as f64 + 500.0);
            }
        }
    }

    let raw_recall_avg = raw_recalls.iter().sum::<f64>() / num_queries as f64;
    let vote_recall_avg = vote_recalls.iter().sum::<f64>() / num_queries as f64;
    let witness_recall_avg = witness_recalls.iter().sum::<f64>() / num_queries as f64;
    let dropped_by_cap_avg = gt_dropped_by_cap_vec.iter().sum::<f64>() / num_queries as f64;
    let escaped_via_witness_avg =
        gt_escaped_via_witness_vec.iter().sum::<f64>() / num_queries as f64;

    let gt_mean_rank = if !gt_voted_ranks.is_empty() {
        gt_voted_ranks.iter().sum::<f64>() / gt_voted_ranks.len() as f64
    } else {
        0.0
    };

    let mut total_times_copy = total_times.clone();
    MetricSummary {
        c_addr: cfg.c_addr(),
        raw_recall_10: raw_recall_avg * 100.0,
        vote_recall_10: vote_recall_avg * 100.0,
        witness_recall_10: witness_recall_avg * 100.0,
        gt_dropped_by_cap: dropped_by_cap_avg * 100.0,
        gt_escaped_via_witness: escaped_via_witness_avg * 100.0,
        raw_unique_cands: raw_cands_counts.iter().sum::<usize>() as f64 / num_queries as f64,
        gt_raw_mean_rank: gt_mean_rank,
        total_lat_p50_us: percentile(&mut total_times_copy, 50.0),
    }
}

fn print_header(title: &str) {
    println!(
        "\n╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗"
    );
    println!("║ {:^129} ║", title);
    println!(
        "╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝"
    );
}

fn print_table_header() {
    println!(
        "┌───────────────────────┬───────┬────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┐"
    );
    println!(
        "│ Configuration         │ Dreal │ C_addr │ Raw R@10 │ Vote R@10│ Wit R@10 │ Cap-Drop │ Wit-Esc  │ Raw Cands│ Mean Rk  │ P50 Lat  │"
    );
    println!(
        "├───────────────────────┼───────┼────────┼──────────┼──────────┼──────────┼──────────┼──────────┼──────────┼──────────┼──────────┤"
    );
}

fn print_table_row(name: &str, d_real: usize, m: &MetricSummary) {
    println!(
        "│ {:<21} │ {:>5} │ {:>6.2} │ {:>7.2}% │ {:>7.2}% │ {:>7.2}% │ {:>7.2}% │ {:>7.2}% │ {:>8.0} │ {:>8.1} │ {:>6.1} µs │",
        name,
        d_real,
        m.c_addr,
        m.raw_recall_10,
        m.vote_recall_10,
        m.witness_recall_10,
        m.gt_dropped_by_cap,
        m.gt_escaped_via_witness,
        m.raw_unique_cands,
        m.gt_raw_mean_rank,
        m.total_lat_p50_us
    );
}

fn print_table_footer() {
    println!(
        "└───────────────────────┴───────┴────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘"
    );
}

fn main() {
    println!(
        "╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗"
    );
    println!(
        "║           HNSQR GATE A2: RIVERO CAPACITY CLOSURE & SET-LEVEL FUNNEL ACCOUNTING                                            ║"
    );
    println!(
        "╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝"
    );

    let is_fast_smoke = cfg!(debug_assertions);
    let n = if is_fast_smoke { 1_000 } else { 10_000 };
    let q = if is_fast_smoke { 16 } else { 64 };
    let k = 10;
    let candidate_cap = 2048;

    let dimensions = if is_fast_smoke {
        vec![384, 1536]
    } else {
        vec![384, 1536, 4096]
    };

    print_header(&format!(
        "GATE A2 MATRIX: (N = {}, Queries = {}, K = {}, Candidate Cap = {})",
        n, q, k, candidate_cap
    ));
    print_table_header();

    for &d_real in &dimensions {
        let corpus_data =
            common::generate_realistic_text_corpus(n, q, d_real, 0x5a5a_0000 + d_real as u64);
        let corpus = &corpus_data.folded_corpus;
        let queries = &corpus_data.folded_queries;

        let geometries = vec![
            RunConfig {
                name: "24F Global (Baseline)".to_string(),
                d_real,
                foundations: 24,
                projection: RiveroProjectionMode::GlobalMix,
            },
            RunConfig {
                name: "64F GlobalMix".to_string(),
                d_real,
                foundations: 64,
                projection: RiveroProjectionMode::GlobalMix,
            },
            RunConfig {
                name: "64F / MultiLane".to_string(),
                d_real,
                foundations: 64,
                projection: RiveroProjectionMode::MultiLane {
                    lanes: if d_real >= 4096 { 8 } else { 4 },
                    assignment: LaneAssignment::Hashed,
                },
            },
        ];

        for cfg in geometries {
            let res = evaluate_geometry(&cfg, corpus, queries, k, candidate_cap);
            print_table_row(&cfg.name, d_real, &res);
        }
    }
    print_table_footer();

    println!("\n[SET-LEVEL FUNNEL DEFINITIONS]");
    println!(
        "  • Raw R@10:       |GT ∩ raw| / K (All candidates admitted to any territory cell pre-cap)"
    );
    println!(
        "  • Vote R@10:      |GT ∩ vote| / K (Candidates surviving vote rank < candidate_cap)"
    );
    println!(
        "  • Wit R@10:       |GT ∩ witness| / K (Final candidate pool after 2-hop witness expansion)"
    );
    println!(
        "  • Cap-Drop:       |GT ∩ (raw - vote)| / K (GT present in raw territory but pruned by the 2048 cap)"
    );
    println!(
        "  • Wit-Esc:        |GT ∩ (witness - raw)| / K (GT discovered by witnesses that territory NEVER admitted)\n"
    );
}
