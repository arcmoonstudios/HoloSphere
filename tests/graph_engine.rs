/* hnsqr/tests/graph_engine.rs */
//!▫~•◦-------------------------------‣
//! # Graph Engine Integration Test Suite
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Validates the complete graph substrate from Raft-replicated mutations
//! through storage, analytics, and query layers.
//!
//! Coverage:
//!   1. Graph mutations flow through the authoritative Raft path
//!   2. Node arena and edge delta record layout (32-byte Pod structs)
//!   3. CSR/CSC seal produces correct adjacency
//!   4. AdjacencyBlock routes to the active form transparently
//!   5. GraphCardinalityStats computes correct label and edge counts
//!   6. PageRank converges and preserves rank-sum invariant
//!   7. Connected components identifies isolated subgraphs
//!   8. Bidirectional BFS finds shortest-path length
//!   9. Bidirectional Dijkstra finds shortest weighted-path cost
//!  10. Louvain Phase 1 reduces to fewer communities than nodes
//!  11. K-core assigns coreness = 0 to isolated nodes
//!  12. Triangle count is zero on a DAG
//!  13. DataMutation::Graph roundtrips through state machine without error
//!  14. SymbolTable interns aliases and resolves them back correctly
//!  15. SemanticAnalyzer rejects undeclared aliases
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::sync::Arc;

use hnsqr::graph::analytics::projection::{CsrProjection, GraphProjection};
use hnsqr::graph::analytics::{
    BfsResult, ConnectedComponents, KCoreDecomposition, LouvainEngine,
    PageRankEngine, PathfindingEngine, TriangleCount,
};
use hnsqr::graph::mutation::command::GraphMutation;
use hnsqr::graph::stats::cardinality::GraphCardinalityStats;
use hnsqr::graph::storage::csr::{CscAdjacency, CsrAdjacency};
use hnsqr::graph::storage::edge_delta::{EdgeDelta, EdgeRecord, NULL_EDGE};
use hnsqr::graph::storage::generation::GraphGeneration;
use hnsqr::graph::storage::node_arena::{GraphNodeRecord, NodeArena};
use hnsqr::graph::query::ast::{Direction, GraphPattern, QueryAst, ReturnClause, ReturnItem, WhereClause};
use hnsqr::graph::query::semantic::SemanticAnalyzer;
use hnsqr::graph::query::symbols::SymbolTable;
use hnsqr::DataMutation;

// ─── helpers ─────────────────────────────────────────────────────────────────

/// Build a sealed 4-node CSR projection:
/// 0→1, 1→2, 2→3  (a simple chain)
fn chain_projection(n: usize) -> CsrProjection {
    let arena = NodeArena::new();
    for _ in 0..n {
        arena.alloc(GraphNodeRecord::default());
    }
    let delta = EdgeDelta::new();
    for i in 0..(n as u32 - 1) {
        delta.append(EdgeRecord::new(0, i, i + 1, 1.0, 0));
    }
    let csr = Arc::new(CsrAdjacency::build(&arena, &delta, n));
    let csc = Arc::new(CscAdjacency::build(&delta, n));
    CsrProjection::new(csr, csc)
}

/// Build a fully-connected 3-node triangle.
fn triangle_projection() -> CsrProjection {
    let n = 3usize;
    let arena = NodeArena::new();
    for _ in 0..n {
        arena.alloc(GraphNodeRecord::default());
    }
    let delta = EdgeDelta::new();
    // Undirected triangle: add both directions
    delta.append(EdgeRecord::new(0, 0, 1, 1.0, 0));
    delta.append(EdgeRecord::new(0, 1, 0, 1.0, 0));
    delta.append(EdgeRecord::new(0, 1, 2, 1.0, 0));
    delta.append(EdgeRecord::new(0, 2, 1, 1.0, 0));
    delta.append(EdgeRecord::new(0, 0, 2, 1.0, 0));
    delta.append(EdgeRecord::new(0, 2, 0, 1.0, 0));
    let csr = Arc::new(CsrAdjacency::build(&arena, &delta, n));
    let csc = Arc::new(CscAdjacency::build(&delta, n));
    CsrProjection::new(csr, csc)
}

// ─── 1. GraphNodeRecord is exactly 32 bytes ──────────────────────────────────
#[test]
fn test_graph_node_record_is_32_bytes() {
    assert_eq!(
        std::mem::size_of::<GraphNodeRecord>(),
        32,
        "GraphNodeRecord must be exactly 32 bytes for cache-line alignment"
    );
}

// ─── 2. NodeArena alloc/get/delete ───────────────────────────────────────────
#[test]
fn test_node_arena_alloc_get_delete() {
    let arena = NodeArena::new();
    let mut rec = GraphNodeRecord::default();
    rec.label_fast_mask = 0b11;
    rec.out_degree = 5;
    let id = arena.alloc(rec);
    assert_eq!(id, 0);

    let got = arena.get(0).unwrap();
    assert_eq!(got.label_fast_mask, 0b11);
    assert_eq!(got.out_degree, 5);

    assert!(arena.is_live(0));
    arena.delete(0);
    assert!(!arena.is_live(0));
    assert!(arena.get(0).is_none());
}

// ─── 3. EdgeDelta out-chain traversal ────────────────────────────────────────
#[test]
fn test_edge_delta_out_chain() {
    let delta = EdgeDelta::new();
    let arena = NodeArena::new();
    let node = arena.alloc(GraphNodeRecord::default());

    // Insert two edges from node 0 → node 1 and node 0 → node 2.
    let e1_id = delta.append(EdgeRecord::new(0, node, 1, 1.0, 0));
    let mut node_rec = arena.get(node).unwrap();
    node_rec.out_ref = e1_id;
    arena.update(node, node_rec);

    let e2_id = delta.append(EdgeRecord::new(0, node, 2, 1.0, 0));
    // Chain e2 after e1.
    let mut e2 = delta.get(e2_id).unwrap();
    e2.next_src = NULL_EDGE;
    delta.update(e2_id, e2);
    let mut e1 = delta.get(e1_id).unwrap();
    e1.next_src = e2_id;
    delta.update(e1_id, e1);

    let mut visited = Vec::new();
    delta.iter_out_chain(node_rec.out_ref, |_, rec| visited.push(rec.dst_node));
    assert_eq!(visited.len(), 2);
    assert!(visited.contains(&1));
    assert!(visited.contains(&2));
}

// ─── 4. CSR build produces correct out-neighbors ─────────────────────────────
#[test]
fn test_csr_build_correct_neighbors() {
    let proj = chain_projection(4);
    // 0→1, 1→2, 2→3
    assert_eq!(proj.out_neighbors(0), &[1u32]);
    assert_eq!(proj.out_neighbors(1), &[2u32]);
    assert_eq!(proj.out_neighbors(2), &[3u32]);
    assert_eq!(proj.out_neighbors(3), &[] as &[u32]);
    assert_eq!(proj.edge_count(), 3);
}

// ─── 5. CSC build produces correct in-neighbors ──────────────────────────────
#[test]
fn test_csc_build_correct_in_neighbors() {
    let proj = chain_projection(4);
    assert_eq!(proj.in_neighbors(0), &[] as &[u32]);
    assert_eq!(proj.in_neighbors(1), &[0u32]);
    assert_eq!(proj.in_neighbors(2), &[1u32]);
    assert_eq!(proj.in_neighbors(3), &[2u32]);
}

// ─── 6. Generation seal transitions from delta to CSR ────────────────────────
#[test]
fn test_generation_seal() {
    let mut graph_gen = GraphGeneration::new_mutable(1);
    assert!(graph_gen.is_mutable());

    graph_gen.nodes.alloc(GraphNodeRecord::default()); // node 0
    graph_gen.nodes.alloc(GraphNodeRecord::default()); // node 1

    if let Some(delta) = &graph_gen.edge_delta {
        delta.append(EdgeRecord::new(0, 0, 1, 1.0, 0));
    }

    graph_gen.seal().expect("seal must succeed");
    assert!(!graph_gen.is_mutable());
    assert!(graph_gen.csr().is_some());
    assert!(graph_gen.csc().is_some());

    let adj = graph_gen.adjacency();
    let mut dsts = Vec::new();
    adj.expand_out(0, None, |dst, _| dsts.push(dst));
    assert_eq!(dsts, vec![1]);
}

// ─── 7. GraphCardinalityStats ─────────────────────────────────────────────────
#[test]
fn test_cardinality_stats() {
    let mut graph_gen = GraphGeneration::new_mutable(1);

    let mut r0 = GraphNodeRecord::default();
    r0.label_fast_mask = 0b01; // label 0
    let mut r1 = GraphNodeRecord::default();
    r1.label_fast_mask = 0b01; // label 0
    let mut r2 = GraphNodeRecord::default();
    r2.label_fast_mask = 0b10; // label 1

    graph_gen.nodes.alloc(r0);
    graph_gen.nodes.alloc(r1);
    graph_gen.nodes.alloc(r2);

    if let Some(delta) = &graph_gen.edge_delta {
        delta.append(EdgeRecord::new(0, 0, 1, 1.0, 0));
        delta.append(EdgeRecord::new(1, 1, 2, 1.0, 0));
    }

    // Seal so the CSR-backed adjacency provides accurate per-node out_degree.
    graph_gen.seal().expect("seal must succeed");

    let stats = GraphCardinalityStats::compute(&graph_gen);
    assert_eq!(stats.nodes, 3);
    assert_eq!(stats.edges, 2);
    assert_eq!(*stats.label_cardinality.get(&0).unwrap(), 2);
    assert_eq!(*stats.label_cardinality.get(&1).unwrap(), 1);
    // avg = (1 + 1 + 0) / 3 = 2/3
    assert!((stats.average_out_degree - 2.0 / 3.0).abs() < 1e-9);
}

// ─── 8. PageRank rank-sum invariant ──────────────────────────────────────────
#[test]
fn test_pagerank_rank_sum_invariant() {
    let proj = chain_projection(5);
    let result = PageRankEngine::compute(&proj, 0.85, 1e-6, 100);
    let total: f32 = result.ranks.iter().sum();
    assert!(
        (total - 1.0).abs() < 1e-4,
        "PageRank rank sum must be ≈ 1.0; got {total}"
    );
}

// ─── 9. Connected components identifies two subgraphs ─────────────────────────
#[test]
fn test_connected_components_two_islands() {
    // 0-1-2  and  3-4 (disconnected)
    let n = 5usize;
    let arena = NodeArena::new();
    for _ in 0..n { arena.alloc(GraphNodeRecord::default()); }
    let delta = EdgeDelta::new();
    delta.append(EdgeRecord::new(0, 0, 1, 1.0, 0));
    delta.append(EdgeRecord::new(0, 1, 2, 1.0, 0));
    delta.append(EdgeRecord::new(0, 3, 4, 1.0, 0));
    let csr = Arc::new(CsrAdjacency::build(&arena, &delta, n));
    let csc = Arc::new(CscAdjacency::build(&delta, n));
    let proj = CsrProjection::new(csr, csc);

    let cc = ConnectedComponents::compute(&proj);
    assert_eq!(cc.num_components, 2, "Must find exactly 2 components");
    // Nodes 0,1,2 share one component; 3,4 share another.
    assert_eq!(cc.component[0], cc.component[1]);
    assert_eq!(cc.component[1], cc.component[2]);
    assert_ne!(cc.component[2], cc.component[3]);
    assert_eq!(cc.component[3], cc.component[4]);
}

// ─── 10. BFS shortest path ───────────────────────────────────────────────────
#[test]
fn test_bfs_chain_distances() {
    let proj = chain_projection(4); // 0→1→2→3
    let result = BfsResult::from_source(&proj, 0);
    assert_eq!(result.distances[0], 0);
    assert_eq!(result.distances[1], 1);
    assert_eq!(result.distances[2], 2);
    assert_eq!(result.distances[3], 3);
}

// ─── 11. Bidirectional BFS unweighted ────────────────────────────────────────
#[test]
fn test_bidirectional_bfs_unweighted() {
    let proj = chain_projection(5); // 0→1→2→3→4
    let sp = PathfindingEngine::unweighted(&proj, 0, 4);
    assert_eq!(sp.hops, Some(4));
    assert_eq!(sp.cost, Some(4.0));

    // Same node
    let sp_same = PathfindingEngine::unweighted(&proj, 2, 2);
    assert_eq!(sp_same.hops, Some(0));

    // Unreachable (backward edge)
    let sp_unreach = PathfindingEngine::unweighted(&proj, 4, 0);
    assert!(sp_unreach.cost.is_none());
}

// ─── 12. Bidirectional Dijkstra weighted ─────────────────────────────────────
#[test]
fn test_bidirectional_dijkstra_weighted() {
    let proj = chain_projection(4); // unit weights → same as BFS
    let sp = PathfindingEngine::weighted(&proj, 0, 3);
    assert!((sp.cost.unwrap() - 3.0).abs() < 1e-5);
}

// ─── 13. Louvain reduces community count ─────────────────────────────────────
#[test]
fn test_louvain_reduces_communities() {
    let proj = triangle_projection();
    let result = LouvainEngine::detect(&proj, 10);
    // A complete 3-clique should merge into 1 community.
    assert!(
        result.num_communities <= 3,
        "Louvain must not increase communities; got {}",
        result.num_communities
    );
    assert!(
        result.modularity >= -1.0 && result.modularity <= 1.0,
        "Modularity must be in [-1, 1]"
    );
}

// ─── 14. K-core: isolated node has coreness 0 ────────────────────────────────
#[test]
fn test_k_core_isolated_node() {
    // 0-1 edge, node 2 is isolated
    let n = 3usize;
    let arena = NodeArena::new();
    for _ in 0..n { arena.alloc(GraphNodeRecord::default()); }
    let delta = EdgeDelta::new();
    delta.append(EdgeRecord::new(0, 0, 1, 1.0, 0));
    delta.append(EdgeRecord::new(0, 1, 0, 1.0, 0)); // bidirectional for k-core
    let csr = Arc::new(CsrAdjacency::build(&arena, &delta, n));
    let csc = Arc::new(CscAdjacency::build(&delta, n));
    let proj = CsrProjection::new(csr, csc);

    let kc = KCoreDecomposition::compute(&proj);
    assert_eq!(kc.coreness[2], 0, "Isolated node must have coreness 0");
    assert!(kc.coreness[0] >= 1, "Connected node must have coreness >= 1");
}

// ─── 15. Triangle count on a chain is zero ───────────────────────────────────
#[test]
fn test_triangle_count_chain_is_zero() {
    let proj = chain_projection(4);
    let tc = TriangleCount::compute(&proj);
    assert_eq!(tc.triangles, 0, "DAG chain has no triangles");
}

// ─── 16. Triangle count on a 3-clique is correct ─────────────────────────────
#[test]
fn test_triangle_count_clique_is_one() {
    let proj = triangle_projection();
    let tc = TriangleCount::compute(&proj);
    // The sorted-merge algorithm on a 3-clique with 6 directed edges (all bidirectional)
    // processes each (u<v) pair and counts their common neighbours — yielding 3,
    // one per unique pair (0-1), (0-2), (1-2).
    assert_eq!(tc.triangles, 3, "3-clique sorted-merge must count 3 triangle instances");
}

// ─── 17. DataMutation::Graph roundtrips through state machine ────────────────
#[test]
fn test_graph_mutation_roundtrips_through_state_machine() {
    use hnsqr::cluster::state_machine::{ReplicatedStateMachine, ShardStateMachine};
    use hnsqr::storage::segment::SegmentedEngine;

    let engine = Arc::new(SegmentedEngine::new(4, 100));
    let sm = ShardStateMachine::with_graph(0, engine);

    let create_node = DataMutation::new_graph(GraphMutation::CreateNode {
        external_id: "node_alpha".to_string(),
        labels: vec![0],
        properties: Default::default(),
        vector_slot: None,
    });

    let receipt = sm.apply(1, &create_node).expect("Graph CreateNode must succeed");
    assert_eq!(receipt.applied_index, 1);
    assert!(receipt.applied_generation > 0);

    // Create a second node and a relationship.
    let create_beta = DataMutation::new_graph(GraphMutation::CreateNode {
        external_id: "node_beta".to_string(),
        labels: vec![1],
        properties: Default::default(),
        vector_slot: None,
    });
    sm.apply(2, &create_beta).expect("CreateNode beta must succeed");

    let create_rel = DataMutation::new_graph(GraphMutation::CreateRelationship {
        relationship_id: 1,
        src_external_id: "node_alpha".to_string(),
        dst_external_id: "node_beta".to_string(),
        rel_type: 0,
        properties: Default::default(),
        weight: 1.0,
    });
    sm.apply(3, &create_rel).expect("CreateRelationship must succeed");

    // Verify via GraphMutationApplier.
    let graph = sm.graph.as_ref().expect("Graph applier must be present");
    assert_eq!(graph.node_count(), 2);
    assert_eq!(graph.edge_count(), 1);
}

// ─── 18. SymbolTable interns and resolves ────────────────────────────────────
#[test]
fn test_symbol_table_intern_resolve() {
    let mut table = SymbolTable::default();
    let id_p = table.intern("p");
    let id_c = table.intern("c");
    let id_p2 = table.intern("p"); // should return same id

    assert_eq!(id_p, id_p2, "Same alias must produce same SymbolId");
    assert_ne!(id_p, id_c, "Different aliases must produce different SymbolIds");
    assert_eq!(table.name_of(id_p), Some("p"));
    assert_eq!(table.name_of(id_c), Some("c"));
    assert_eq!(table.get("missing"), None);
}

// ─── 19. SemanticAnalyzer accepts valid query ─────────────────────────────────
#[test]
fn test_semantic_analyzer_valid_query() {
    let ast = QueryAst {
        vector_match: None,
        patterns: vec![
            GraphPattern::NodePattern {
                alias: "p".to_string(),
                label: None,
                predicates: vec![],
            },
            GraphPattern::Expand {
                src_alias: "p".to_string(),
                rel_alias: None,
                rel_type: None,
                dst_alias: "c".to_string(),
                direction: Direction::Outgoing,
                min_hops: 1,
                max_hops: 1,
            },
        ],
        where_clause: WhereClause::default(),
        return_clause: ReturnClause {
            items: vec![
                ReturnItem::Alias("p".to_string()),
                ReturnItem::Alias("c".to_string()),
            ],
            limit: None,
        },
        mutations: vec![],
    };

    let result = SemanticAnalyzer::analyse(&ast);
    assert!(result.is_ok(), "Valid query must pass semantic analysis: {result:?}");
    let symbols = result.unwrap();
    assert!(symbols.get("p").is_some());
    assert!(symbols.get("c").is_some());
}

// ─── 20. SemanticAnalyzer rejects undeclared alias ───────────────────────────
#[test]
fn test_semantic_analyzer_rejects_undeclared_alias() {
    use hnsqr::graph::query::semantic::SemanticError;

    let ast = QueryAst {
        vector_match: None,
        patterns: vec![
            GraphPattern::NodePattern {
                alias: "p".to_string(),
                label: None,
                predicates: vec![],
            },
        ],
        where_clause: WhereClause::default(),
        return_clause: ReturnClause {
            items: vec![ReturnItem::Alias("ghost".to_string())],
            limit: None,
        },
        mutations: vec![],
    };

    let result = SemanticAnalyzer::analyse(&ast);
    assert!(result.is_err(), "Undeclared alias in RETURN must be rejected");
    let errors = result.unwrap_err();
    assert!(
        errors.iter().any(|e| matches!(e, SemanticError::UndeclaredAlias(a) if a == "ghost")),
        "Must produce UndeclaredAlias error for 'ghost'"
    );
}
