/* holosphere/src/graph/mutation/mod.rs */
//!▫~•◦-------------------------------‣
//! # Graph Mutation — Raft-Replicated Graph Command Model
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Graph topology changes flow through the same authoritative Raft pipeline
//! as vector mutations.  `GraphMutation` variants are carried inside
//! `DataMutation::Graph` in the cluster state machine.
//!
//! **There is no direct-write production path bypassing Raft.**
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

pub mod apply;
pub mod command;

pub use apply::GraphMutationApplier;
pub use command::{GraphMutation, GraphProperties, RelationshipId};
