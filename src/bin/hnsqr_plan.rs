/* hnsqr/src/bin/hnsqr_plan.rs */
//!▫~•◦-------------------------------‣
//! # HNSQR Plan: Production Infrastructure Sizing CLI
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Sizes hardware, RAM, NVMe throughput, learner counts, and shard topologies
//! based on corpus cardinality, dimensionality, write rate, and query SLA.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use hnsqr::capacity::planner::{CapacityPlanner, CapacityRequirements};

fn main() {
    println!("╔═════════════════════════════════════════════════════════════════════════════╗");
    println!("║                    HNSQR CLOUD CAPACITY SIZING PLANNER                      ║");
    println!("╚═════════════════════════════════════════════════════════════════════════════╝");

    let req = CapacityRequirements {
        total_vectors: 10_000_000,
        dimension: 1536,
        target_query_qps: 5_000,
        target_write_qps: 500,
        replication_factor: 3,
    };

    let plan = CapacityPlanner::compute_plan(&req);

    println!("\n📋 DEPLOYMENT TARGET:");
    println!("   • Total Vectors:       {:>12}", "10,000,000");
    println!("   • Dimension:           {:>12}", "1,536D (Complex)");
    println!("   • Target Query QPS:    {:>12}", "5,000 QPS");
    println!("   • Ingestion Rate:      {:>12}", "500 writes/sec");
    println!("   • Replication Factor:  {:>12}", "3x (Quorum)");

    println!("\n💻 SIZED INFRASTRUCTURE RECOMMENDATION:");
    println!("   • Total Vector Storage: {:>10.2} GB", plan.total_vector_storage_gb);
    println!("   • Hot Index Memory:     {:>10.2} GB", plan.total_index_memory_gb);
    println!("   • Recommended RAM:      {:>10.2} GB (95% CI: [{:.2} - {:.2}] GB)", plan.recommended_ram_gb, plan.recommended_ram_ci_low_gb, plan.recommended_ram_ci_high_gb);
    println!("   • Recommended NVMe BW:  {:>10.2} MB/s", plan.recommended_nvme_bandwidth_mbps);
    println!("   • Recommended Shards:   {:>10}", plan.recommended_shards);
    println!("   • Recommended Learners: {:>10}", plan.recommended_learners);
    println!("   • Expected p99 Latency: {:>10.2} ms", plan.expected_p99_latency_ms);

    println!("\n✨ SIZING ANALYSIS COMPLETE.\n");
}
