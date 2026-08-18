/* hnsqr/src/graph/catalog/mod.rs */
//!▫~•◦-------------------------------‣
//! # Graph Catalog — Label Registry, Relationship-Type Registry, Property Schema
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Interned identifiers for node labels, relationship types, and property keys.
//! All catalog IDs are compact `u32` values that map one-to-one into the
//! [`GraphNodeRecord::label_fast_mask`] bitmask (slots 0–63) or the overflow
//! bitmap/dictionary (slots ≥ 64).
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

pub mod labels;
pub mod properties;
pub mod relationships;

pub use labels::{LabelCatalog, LabelId, LabelResolution};
pub use properties::{PropertyKey, PropertyKeyCatalog};
pub use relationships::{RelTypeCatalog, RelTypeId, RelTypeResolution};
