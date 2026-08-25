/* holosphere/src/metadata/mod.rs */
//!▫~•◦-------------------------------‣
//! # Metadata Indexing, Inverted Stores & Cardinality Governance Subsystem
//!▫~•◦-------------------------------------------------------------------‣
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

pub mod cardinality;
pub mod geo;
pub mod index;
pub mod store;

pub use cardinality::{
    CardinalityBudget, CardinalityGuard, PostingRepresentation, TenantCardinalityTracker,
};
pub use geo::{BoundingBox2D, GeoPoint, GeoPolygon};
pub use index::{FilterExpr, MetadataInvertedIndex, MetadataValue};
pub use store::{MetadataQuotaConfig, MetadataStore, QuotaTracker};
