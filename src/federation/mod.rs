/* hnsqr/src/federation/mod.rs */
//!▫~•◦-------------------------------‣
//! # HNSQR Geo-Distributed Federation Subsystem
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Coordinates federated exact Top-K proof aggregation across sovereign clusters.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

pub mod cluster;

pub use cluster::{
    ClusterProofResponse, ClusterRegionId, FederatedProofCoordinator, FederatedProofStatus,
    FederatedQueryResult,
};
