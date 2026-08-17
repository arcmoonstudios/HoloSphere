/* hnsqr/src/planner.rs */
//!▫~•◦-------------------------------‣
//! # Universal Proof-Carrying Retrieval Planner
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Evaluates multidimensional retrieval costs over corpus size, dimension, filter cardinality,
//! residency, and hardware capabilities to select the optimal execution path under a declarative
//! correctness contract.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use serde::{Deserialize, Serialize};
use std::time::Duration;

use crate::lutz::SemanticRerankPlan;
use crate::rivero::RiveroProfile;

/// Declarative user correctness contract.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, Default)]
pub enum RetrievalContract {
    /// Exhaustive ground-truth scan across all candidates.
    Exact,
    /// Bounded candidate universe with mathematical Cauchy-Schwarz proof certificate.
    #[default]
    Certified,
    /// Statistical target recall guarantee (e.g., 0.995).
    HighRecall(f32),
    /// Peak throughput within a strict latency ceiling.
    Budget(Duration),
}

/// Query modality for multidimensional search.
#[derive(Clone, Debug, PartialEq)]
pub enum QueryModality {
    Dense,
    Sparse,
    MultiVector,
    Hybrid {
        dense_weight: f32,
        sparse_weight: f32,
    },
}

/// Concrete execution plan produced by the Universal Planner.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ExecutionPlan {
    /// Pure exact scan (for small effective corpora $N_{\text{eff}} < N_{\text{cross}}$).
    ExactScan { effective_n: usize },
    /// Corpus-Global Cauchy-Schwarz certified retrieval with Rivero seeding.
    LutzGlobalCertified { initial_seed_cap: usize },
    /// Rivero routing + Semantic Reranker.
    RiveroRetrieval {
        profile: RiveroProfile,
        rerank_plan: SemanticRerankPlan,
        candidate_cap: usize,
    },
    /// Sparse BM25 / Block-Max WAND lexical search.
    SparseLexical { use_wand: bool },
    /// Multi-vector late-interaction MaxSim.
    MultiVectorMaxSim,
    /// Reciprocal Rank Fusion hybrid search.
    HybridFusion { rrf_k: f32 },
}

/// Detailed Execution Proof returned with query responses.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ExecutionProof {
    pub plan_name: String,
    pub effective_corpus_size: usize,
    pub candidates_routed: usize,
    pub exact_evaluations: usize,
    pub work_eliminated_pct: f32,
    pub certified: bool,
    pub latency_us: f32,
}

/// Universal Cost-Based Query Planner.
pub struct UniversalPlanner;

impl UniversalPlanner {
    /// Computes the exact-scan crossover threshold:
    /// $$N_{\text{cross}}(D_{\text{complex}}) = 850.0 + \frac{1,416,904.5}{D_{\text{complex}}^{0.860}}$$
    #[inline(always)]
    pub fn compute_crossover(complex_dim: usize) -> usize {
        let d = complex_dim.max(8) as f64;
        let d_pow = d.powf(0.860);
        let n_cross = 850.0 + (1_416_904.5 / d_pow);
        n_cross.round().clamp(100.0, 1_000_000.0) as usize
    }

    /// Plans the optimal execution path given corpus state and query constraints.
    pub fn plan(
        total_vectors: usize,
        complex_dim: usize,
        filter_cardinality: Option<usize>,
        contract: RetrievalContract,
        is_mmap_cold: bool,
    ) -> ExecutionPlan {
        let effective_n = filter_cardinality.unwrap_or(total_vectors);
        let crossover = Self::compute_crossover(complex_dim);

        // 1. If contract is strictly Exact or effective corpus is below crossover threshold -> ExactScan
        if matches!(contract, RetrievalContract::Exact) || effective_n < crossover {
            return ExecutionPlan::ExactScan { effective_n };
        }

        // 2. If contract is Certified -> LutzGlobalCertified
        if matches!(contract, RetrievalContract::Certified) {
            return ExecutionPlan::LutzGlobalCertified {
                initial_seed_cap: 2048,
            };
        }

        // 3. Select Rivero Profile based on contract
        let (profile, candidate_cap) = match contract {
            RetrievalContract::Exact => (RiveroProfile::Strict, 1024),
            RetrievalContract::Certified => (RiveroProfile::Fast, 512),
            RetrievalContract::HighRecall(r) if r >= 0.999 => (RiveroProfile::Strict, 1024),
            RetrievalContract::HighRecall(r) if r >= 0.99 => (RiveroProfile::Balanced, 768),
            RetrievalContract::HighRecall(_) => (RiveroProfile::Fast, 512),
            RetrievalContract::Budget(dur) if dur.as_micros() < 500 => (RiveroProfile::Fast, 256),
            RetrievalContract::Budget(_) => (RiveroProfile::Fast, 512),
        };

        // 4. Select Semantic Reranking Policy based on residency
        let rerank_plan = if is_mmap_cold {
            SemanticRerankPlan::LutzFastScan
        } else {
            SemanticRerankPlan::ExactSimd
        };

        ExecutionPlan::RiveroRetrieval {
            profile,
            rerank_plan,
            candidate_cap,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_universal_planner_decisions() {
        // Small corpus (N=500 at 1536D/768 complex) -> ExactScan
        let plan_small =
            UniversalPlanner::plan(500, 768, None, RetrievalContract::Certified, false);
        assert!(matches!(
            plan_small,
            ExecutionPlan::ExactScan { effective_n: 500 }
        ));

        // Large corpus (N=100,000 at 1536D) with Certified contract -> LutzGlobalCertified
        let plan_large_certified =
            UniversalPlanner::plan(100_000, 768, None, RetrievalContract::Certified, false);
        assert!(matches!(
            plan_large_certified,
            ExecutionPlan::LutzGlobalCertified {
                initial_seed_cap: 2048
            }
        ));

        // Large corpus (N=100,000 at 1536D) with HighRecall contract in cold mmap -> Rivero + LutzFastScan
        let plan_large_cold = UniversalPlanner::plan(
            100_000,
            768,
            None,
            RetrievalContract::HighRecall(0.99),
            true,
        );
        assert!(matches!(
            plan_large_cold,
            ExecutionPlan::RiveroRetrieval {
                profile: RiveroProfile::Balanced,
                rerank_plan: SemanticRerankPlan::LutzFastScan,
                ..
            }
        ));

        // Large corpus with restrictive filter (tenant_id matches only 200 items) -> Filtered ExactScan
        let plan_filtered =
            UniversalPlanner::plan(100_000, 768, Some(200), RetrievalContract::Certified, false);
        assert!(matches!(
            plan_filtered,
            ExecutionPlan::ExactScan { effective_n: 200 }
        ));
    }
}
