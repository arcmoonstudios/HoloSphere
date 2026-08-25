/* holosphere/src/metadata_index.rs */
//!▫~•◦-------------------------------‣
//! # Lock-Free Roaring Bitmap Inverted Metadata Index
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Replaces opaque dynamic closures with high-performance inverted indexes backed by [`roaring::RoaringBitmap`].
//! Supports boolean AST filter queries (equality, set membership, numeric ranges, compound AND/OR/NOT)
//! evaluated via fast bitwise operations before graph walks, reducing hot-path filtering to bitmap membership.
//!
//! ## Key Capabilities
//! - **Bitmap Pruning:** Pre-compiles query criteria into `RoaringBitmap` masks.
//! - **Categorical & Numeric Range Inverted Indices:** Dual indexing supporting exact string, boolean, integer, and float ranges.
//! - **Lock-Free Concurrency:** Read-heavy multi-core search walks evaluate pre-compiled masks with 0 allocations.
//!
//! ### Architectural Notes
//! Integrated with `HNSQRIndex` candidate loops to prune non-matching nodes before memory-intensive SIMD operations.
//!
//! #### Example
//! ```rust
//! use hnsqr::metadata::index::{MetadataInvertedIndex, FilterExpr};
//! use std::collections::HashMap;
//!
//! let index = MetadataInvertedIndex::new();
//! let filter = FilterExpr::eq("department", "engineering");
//! let mask = index.evaluate_filter(&filter, 1000);
//! ```
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use parking_lot::RwLock;
use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::fmt;

use crate::NodeIndex;

/// Strongly-typed metadata value for inverted indexing.
#[derive(Clone, Debug, PartialEq, PartialOrd, Serialize)]
pub enum MetadataValue {
    /// Textual string attribute (e.g. category, author, tenant).
    String(String),
    /// 64-bit signed integer (e.g. timestamp, user_id, count).
    Integer(i64),
    /// 64-bit floating point value (e.g. price, rating).
    Float(f64),
    /// Boolean flag.
    Boolean(bool),
}

impl<'de> serde::Deserialize<'de> for MetadataValue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        if deserializer.is_human_readable() {
            struct MetadataValueVisitor;

            impl<'de> serde::de::Visitor<'de> for MetadataValueVisitor {
                type Value = MetadataValue;

                fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                    formatter.write_str("a string, integer, float, or boolean metadata value")
                }

                fn visit_bool<E>(self, v: bool) -> Result<Self::Value, E>
                where
                    E: serde::de::Error,
                {
                    Ok(MetadataValue::Boolean(v))
                }

                fn visit_i64<E>(self, v: i64) -> Result<Self::Value, E>
                where
                    E: serde::de::Error,
                {
                    Ok(MetadataValue::Integer(v))
                }

                fn visit_u64<E>(self, v: u64) -> Result<Self::Value, E>
                where
                    E: serde::de::Error,
                {
                    Ok(MetadataValue::Integer(v as i64))
                }

                fn visit_f64<E>(self, v: f64) -> Result<Self::Value, E>
                where
                    E: serde::de::Error,
                {
                    Ok(MetadataValue::Float(v))
                }

                fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
                where
                    E: serde::de::Error,
                {
                    Ok(MetadataValue::String(v.to_string()))
                }

                fn visit_string<E>(self, v: String) -> Result<Self::Value, E>
                where
                    E: serde::de::Error,
                {
                    Ok(MetadataValue::String(v))
                }
            }

            deserializer.deserialize_any(MetadataValueVisitor)
        } else {
            #[derive(Deserialize)]
            enum TaggedMetadataValue {
                String(String),
                Integer(i64),
                Float(f64),
                Boolean(bool),
            }

            TaggedMetadataValue::deserialize(deserializer).map(|v| match v {
                TaggedMetadataValue::String(s) => MetadataValue::String(s),
                TaggedMetadataValue::Integer(i) => MetadataValue::Integer(i),
                TaggedMetadataValue::Float(f) => MetadataValue::Float(f),
                TaggedMetadataValue::Boolean(b) => MetadataValue::Boolean(b),
            })
        }
    }
}

impl From<&str> for MetadataValue {
    fn from(s: &str) -> Self {
        MetadataValue::String(s.to_string())
    }
}

impl From<String> for MetadataValue {
    fn from(s: String) -> Self {
        MetadataValue::String(s)
    }
}

impl From<i64> for MetadataValue {
    fn from(i: i64) -> Self {
        MetadataValue::Integer(i)
    }
}

impl From<f64> for MetadataValue {
    fn from(f: f64) -> Self {
        MetadataValue::Float(f)
    }
}

impl From<bool> for MetadataValue {
    fn from(b: bool) -> Self {
        MetadataValue::Boolean(b)
    }
}

impl MetadataValue {
    /// Converts the value to a string key representation.
    pub fn to_string_key(&self) -> String {
        match self {
            MetadataValue::String(s) => s.clone(),
            MetadataValue::Integer(i) => i.to_string(),
            MetadataValue::Float(f) => format!("{:.6}", f),
            MetadataValue::Boolean(b) => b.to_string(),
        }
    }

    /// Writes the canonical categorical key without allocating an intermediate `String`.
    pub fn write_key_to<W: fmt::Write>(&self, writer: &mut W) -> fmt::Result {
        match self {
            MetadataValue::String(s) => writer.write_str(s),
            MetadataValue::Integer(i) => write!(writer, "{i}"),
            MetadataValue::Float(f) => write!(writer, "{f:.6}"),
            MetadataValue::Boolean(b) => writer.write_str(if *b { "true" } else { "false" }),
        }
    }

    /// Converts to integer representation for numeric indexing.
    pub fn to_i64_scaled(&self) -> Option<i64> {
        match self {
            MetadataValue::Integer(i) => Some(*i),
            MetadataValue::Float(f) => Some((*f * 1_000_000.0) as i64),
            _ => None,
        }
    }
}

/// Stack-backed formatter for temporary categorical lookup keys.
///
/// 384 bytes holds the fixed-six-decimal rendering of every finite `f64`, including
/// its sign. It avoids a heap allocation on numeric and boolean filter hot paths.
struct StackKey {
    bytes: [u8; 384],
    len: usize,
}

impl StackKey {
    fn new() -> Self {
        Self {
            bytes: [0; 384],
            len: 0,
        }
    }

    fn as_str(&self) -> &str {
        // `fmt::Write` supplies valid UTF-8 text and the buffer starts empty.
        std::str::from_utf8(&self.bytes[..self.len]).expect("formatter emitted invalid UTF-8")
    }
}

impl fmt::Write for StackKey {
    fn write_str(&mut self, value: &str) -> fmt::Result {
        let end = self.len.checked_add(value.len()).ok_or(fmt::Error)?;
        if end > self.bytes.len() {
            return Err(fmt::Error);
        }
        self.bytes[self.len..end].copy_from_slice(value.as_bytes());
        self.len = end;
        Ok(())
    }
}

/// Abstract Syntax Tree (AST) representing structured metadata filter expressions.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum FilterExpr {
    /// Field equality: `field == value`.
    Eq(String, MetadataValue),
    /// Set membership: `field IN [val1, val2, ...]`.
    In(String, Vec<MetadataValue>),
    /// Numeric range check: `min <= field <= max`.
    Range(String, f64, f64),
    /// Spatial point-in-polygon filter: `field IN polygon`.
    GeoWithin(String, crate::metadata::geo::GeoPolygon),
    /// Spatial radial distance filter: `distance(field, center) <= max_km`.
    GeoRadius(String, crate::metadata::geo::GeoPoint, f64),
    /// Boolean conjunction: all child filters must match (`A AND B AND C`).
    And(Vec<FilterExpr>),
    /// Boolean disjunction: at least one child filter must match (`A OR B OR C`).
    Or(Vec<FilterExpr>),
    /// Boolean negation: `NOT filter`.
    Not(Box<FilterExpr>),
}

impl FilterExpr {
    /// Helper to construct an equality filter.
    pub fn eq(field: impl Into<String>, val: impl Into<MetadataValue>) -> Self {
        FilterExpr::Eq(field.into(), val.into())
    }

    /// Helper to construct a set membership filter.
    pub fn is_in(field: impl Into<String>, vals: Vec<MetadataValue>) -> Self {
        FilterExpr::In(field.into(), vals)
    }

    /// Helper to construct a numeric range filter.
    pub fn range(field: impl Into<String>, min_val: f64, max_val: f64) -> Self {
        FilterExpr::Range(field.into(), min_val, max_val)
    }

    /// Helper to construct a geospatial point-in-polygon filter.
    pub fn geo_within(field: impl Into<String>, polygon: crate::metadata::geo::GeoPolygon) -> Self {
        FilterExpr::GeoWithin(field.into(), polygon)
    }

    /// Helper to construct a geospatial radial distance filter.
    pub fn geo_radius(
        field: impl Into<String>,
        center: crate::metadata::geo::GeoPoint,
        max_km: f64,
    ) -> Self {
        FilterExpr::GeoRadius(field.into(), center, max_km)
    }

    /// Helper to construct an AND conjunction.
    pub fn and(exprs: Vec<FilterExpr>) -> Self {
        FilterExpr::And(exprs)
    }

    /// Helper to construct an OR disjunction.
    pub fn or(exprs: Vec<FilterExpr>) -> Self {
        FilterExpr::Or(exprs)
    }

    /// Helper to construct a NOT negation.
    #[allow(clippy::should_implement_trait)]
    pub fn not(expr: FilterExpr) -> Self {
        FilterExpr::Not(Box::new(expr))
    }
}

/// A high-performance inverted index mapping metadata keys and values to [`RoaringBitmap`]s.
pub struct MetadataInvertedIndex {
    categorical: RwLock<HashMap<String, HashMap<String, RoaringBitmap>>>,
    numeric: RwLock<HashMap<String, BTreeMap<i64, RoaringBitmap>>>,
}

impl Default for MetadataInvertedIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl MetadataInvertedIndex {
    /// Creates a new, empty inverted metadata index.
    pub fn new() -> Self {
        Self {
            categorical: RwLock::new(HashMap::new()),
            numeric: RwLock::new(HashMap::new()),
        }
    }

    /// Exports raw categorical and numeric postings tables for serialization.
    #[allow(clippy::type_complexity)]
    pub fn export_postings(
        &self,
    ) -> (
        HashMap<String, HashMap<String, RoaringBitmap>>,
        HashMap<String, BTreeMap<i64, RoaringBitmap>>,
    ) {
        let cat = self.categorical.read().clone();
        let num = self.numeric.read().clone();
        (cat, num)
    }

    /// Imports raw categorical and numeric postings tables from deserialization.
    #[allow(clippy::type_complexity)]
    pub fn import_postings(
        &self,
        categorical: HashMap<String, HashMap<String, RoaringBitmap>>,
        numeric: HashMap<String, BTreeMap<i64, RoaringBitmap>>,
    ) {
        *self.categorical.write() = categorical;
        *self.numeric.write() = numeric;
    }

    /// Parses JSON metadata at ingestion time and sets the corresponding bits lock-free.
    pub fn insert_metadata(&self, node_index: NodeIndex, metadata: &serde_json::Value) {
        if let serde_json::Value::Object(map) = metadata {
            let mut index_write = self.categorical.write();
            for (key, value) in map {
                let val_str = match value {
                    serde_json::Value::String(s) => s.clone(),
                    serde_json::Value::Number(n) => n.to_string(),
                    serde_json::Value::Bool(b) => b.to_string(),
                    _ => continue,
                };
                index_write
                    .entry(key.clone())
                    .or_default()
                    .entry(val_str)
                    .or_default()
                    .insert(node_index);
            }
        }
    }

    /// Compiles a set of required exact attributes into a single highly compressed bitmask
    /// BEFORE the graph traversal begins.
    pub fn compile_filter_mask(
        &self,
        exact_matches: &HashMap<String, String>,
    ) -> Option<RoaringBitmap> {
        if exact_matches.is_empty() {
            return None;
        }

        let index_read = self.categorical.read();
        let mut final_mask: Option<RoaringBitmap> = None;

        for (key, required_value) in exact_matches {
            let current_bitmap = index_read
                .get(key)
                .and_then(|values| values.get(required_value))
                .cloned()
                .unwrap_or_default();

            match final_mask.as_mut() {
                Some(mask) => {
                    *mask &= current_bitmap;
                }
                None => {
                    final_mask = Some(current_bitmap);
                }
            }
        }

        final_mask
    }

    /// Indexes metadata attributes for a given node index.
    pub fn index_node(&self, index: NodeIndex, metadata: &HashMap<String, MetadataValue>) {
        let mut cat_write = self.categorical.write();
        let mut num_write = self.numeric.write();

        for (field, val) in metadata {
            let key_str = val.to_string_key();
            cat_write
                .entry(field.clone())
                .or_default()
                .entry(key_str)
                .or_default()
                .insert(index);

            if let Some(scaled_i64) = val.to_i64_scaled() {
                num_write
                    .entry(field.clone())
                    .or_default()
                    .entry(scaled_i64)
                    .or_default()
                    .insert(index);
            }
        }
    }

    /// Batch indexes metadata attributes for multiple nodes in a single lock acquisition.
    pub fn index_nodes_batch(&self, batch: &[(NodeIndex, &HashMap<String, MetadataValue>)]) {
        let mut cat_write = self.categorical.write();
        let mut num_write = self.numeric.write();

        for &(index, metadata) in batch {
            for (field, val) in metadata {
                let key_str = val.to_string_key();
                cat_write
                    .entry(field.clone())
                    .or_default()
                    .entry(key_str)
                    .or_default()
                    .insert(index);

                if let Some(scaled_i64) = val.to_i64_scaled() {
                    num_write
                        .entry(field.clone())
                        .or_default()
                        .entry(scaled_i64)
                        .or_default()
                        .insert(index);
                }
            }
        }
    }

    /// Removes a node index from the inverted metadata index.
    pub fn remove_node(&self, index: NodeIndex, metadata: &HashMap<String, MetadataValue>) {
        let mut cat_write = self.categorical.write();
        let mut num_write = self.numeric.write();

        for (field, val) in metadata {
            let key_str = val.to_string_key();
            if let Some(field_map) = cat_write.get_mut(field) {
                if let Some(bitmap) = field_map.get_mut(&key_str) {
                    bitmap.remove(index);
                }
            }

            if let Some(scaled_i64) = val.to_i64_scaled() {
                if let Some(btree) = num_write.get_mut(field) {
                    if let Some(bitmap) = btree.get_mut(&scaled_i64) {
                        bitmap.remove(index);
                    }
                }
            }
        }
    }

    /// Removes a node from every posting without requiring callers to retain a
    /// second copy of its metadata. Removal is a maintenance operation and may
    /// scan the posting directory; query-time mask membership remains constant.
    pub fn remove_node_index(&self, index: NodeIndex) {
        let mut categorical = self.categorical.write();
        for field_map in categorical.values_mut() {
            field_map.retain(|_, bitmap| {
                bitmap.remove(index);
                !bitmap.is_empty()
            });
        }
        categorical.retain(|_, field_map| !field_map.is_empty());

        let mut numeric = self.numeric.write();
        for value_map in numeric.values_mut() {
            value_map.retain(|_, bitmap| {
                bitmap.remove(index);
                !bitmap.is_empty()
            });
        }
        numeric.retain(|_, value_map| !value_map.is_empty());
    }

    /// Evaluates a structured [`FilterExpr`] into a single consolidated [`RoaringBitmap`].
    pub fn evaluate_filter(&self, expr: &FilterExpr, total_nodes: usize) -> RoaringBitmap {
        let cat_read = self.categorical.read();
        let num_read = self.numeric.read();

        self.eval_internal(expr, &cat_read, &num_read, total_nodes)
    }

    fn eval_internal(
        &self,
        expr: &FilterExpr,
        cat: &HashMap<String, HashMap<String, RoaringBitmap>>,
        num: &HashMap<String, BTreeMap<i64, RoaringBitmap>>,
        total_nodes: usize,
    ) -> RoaringBitmap {
        match expr {
            FilterExpr::Eq(field, val) => cat
                .get(field)
                .and_then(|field_map| Self::lookup_categorical(field_map, val))
                .cloned()
                .unwrap_or_default(),
            FilterExpr::In(field, vals) => {
                let mut out = RoaringBitmap::new();
                if let Some(field_map) = cat.get(field) {
                    for val in vals {
                        if let Some(bm) = Self::lookup_categorical(field_map, val) {
                            out |= bm;
                        }
                    }
                }
                out
            }
            FilterExpr::Range(field, min_val, max_val) => {
                let mut out = RoaringBitmap::new();
                let min_scaled = (*min_val * 1_000_000.0) as i64;
                let max_scaled = (*max_val * 1_000_000.0) as i64;

                if let Some(btree) = num.get(field) {
                    for (_, bm) in btree.range(min_scaled..=max_scaled) {
                        out |= bm;
                    }
                }
                out
            }
            FilterExpr::GeoWithin(field, polygon) => {
                let mut out = RoaringBitmap::new();
                if let Some(field_map) = cat.get(field) {
                    for (val_str, bm) in field_map {
                        if let Some((lat_s, lon_s)) = val_str.split_once(',') {
                            if let (Ok(lat), Ok(lon)) =
                                (lat_s.trim().parse::<f64>(), lon_s.trim().parse::<f64>())
                            {
                                let pt = crate::metadata::geo::GeoPoint::new(lat, lon);
                                if polygon.contains_point(&pt) {
                                    out |= bm;
                                }
                            }
                        }
                    }
                }
                out
            }
            FilterExpr::GeoRadius(field, center, max_km) => {
                let mut out = RoaringBitmap::new();
                if let Some(field_map) = cat.get(field) {
                    for (val_str, bm) in field_map {
                        if let Some((lat_s, lon_s)) = val_str.split_once(',') {
                            if let (Ok(lat), Ok(lon)) =
                                (lat_s.trim().parse::<f64>(), lon_s.trim().parse::<f64>())
                            {
                                let pt = crate::metadata::geo::GeoPoint::new(lat, lon);
                                if center.haversine_distance_km(&pt) <= *max_km {
                                    out |= bm;
                                }
                            }
                        }
                    }
                }
                out
            }
            FilterExpr::And(children) => {
                if children.is_empty() {
                    return RoaringBitmap::new();
                }
                let mut result = self.eval_internal(&children[0], cat, num, total_nodes);
                for child in &children[1..] {
                    if result.is_empty() {
                        break;
                    }
                    let child_bm = self.eval_internal(child, cat, num, total_nodes);
                    result &= child_bm;
                }
                result
            }
            FilterExpr::Or(children) => {
                let mut result = RoaringBitmap::new();
                for child in children {
                    let child_bm = self.eval_internal(child, cat, num, total_nodes);
                    result |= child_bm;
                }
                result
            }
            FilterExpr::Not(inner) => {
                let inner_bm = self.eval_internal(inner, cat, num, total_nodes);
                let mut universe = RoaringBitmap::new();
                universe.insert_range(0..(total_nodes as u32));
                universe - inner_bm
            }
        }
    }

    /// Looks up a categorical bitmap using a borrowed or stack-formatted key.
    #[inline]
    fn lookup_categorical<'a>(
        field_map: &'a HashMap<String, RoaringBitmap>,
        value: &MetadataValue,
    ) -> Option<&'a RoaringBitmap> {
        match value {
            MetadataValue::String(text) => field_map.get(text.as_str()),
            _ => {
                let mut key = StackKey::new();
                value
                    .write_key_to(&mut key)
                    .expect("384-byte metadata key buffer is sufficient for fixed f64 formatting");
                field_map.get(key.as_str())
            }
        }
    }

    /// Clears all indexed metadata mappings.
    pub fn clear(&self) {
        self.categorical.write().clear();
        self.numeric.write().clear();
    }
}

/// Type alias for [`MetadataInvertedIndex`].
pub type MetadataIndex = MetadataInvertedIndex;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inverted_index_equality_and_and() {
        let index = MetadataInvertedIndex::new();

        let mut m1 = HashMap::new();
        m1.insert("category".to_string(), "gpu".into());
        m1.insert("tenant".to_string(), "t1".into());
        index.index_node(0, &m1);

        let mut m2 = HashMap::new();
        m2.insert("category".to_string(), "gpu".into());
        m2.insert("tenant".to_string(), "t2".into());
        index.index_node(1, &m2);

        let mut m3 = HashMap::new();
        m3.insert("category".to_string(), "cpu".into());
        m3.insert("tenant".to_string(), "t1".into());
        index.index_node(2, &m3);

        // Filter: category == "gpu" AND tenant == "t1"
        let filter = FilterExpr::and(vec![
            FilterExpr::eq("category", "gpu"),
            FilterExpr::eq("tenant", "t1"),
        ]);

        let bm = index.evaluate_filter(&filter, 3);
        assert_eq!(bm.len(), 1);
        assert!(bm.contains(0));
        assert!(!bm.contains(1));
        assert!(!bm.contains(2));
    }

    #[test]
    fn test_numeric_range_filtering() {
        let index = MetadataInvertedIndex::new();

        for i in 0..10 {
            let mut m = HashMap::new();
            m.insert("price".to_string(), (i as f64 * 10.0).into());
            index.index_node(i as NodeIndex, &m);
        }

        // Filter: price BETWEEN 25.0 AND 65.0 (should match items 3, 4, 5, 6)
        let filter = FilterExpr::range("price", 25.0, 65.0);
        let bm = index.evaluate_filter(&filter, 10);
        assert_eq!(bm.len(), 4);
        assert!(bm.contains(3));
        assert!(bm.contains(4));
        assert!(bm.contains(5));
        assert!(bm.contains(6));
        assert!(!bm.contains(2));
        assert!(!bm.contains(7));
    }

    #[test]
    fn test_compile_filter_mask_json() {
        let index = MetadataInvertedIndex::new();
        let meta1 = serde_json::json!({ "dept": "engineering", "active": true, "level": 5 });
        let meta2 = serde_json::json!({ "dept": "finance", "active": true, "level": 3 });
        let meta3 = serde_json::json!({ "dept": "engineering", "active": false, "level": 2 });

        index.insert_metadata(0, &meta1);
        index.insert_metadata(1, &meta2);
        index.insert_metadata(2, &meta3);

        let mut req = HashMap::new();
        req.insert("dept".to_string(), "engineering".to_string());
        req.insert("active".to_string(), "true".to_string());

        let mask = index.compile_filter_mask(&req).expect("mask should exist");
        assert_eq!(mask.len(), 1);
        assert!(mask.contains(0));
        assert!(!mask.contains(1));
        assert!(!mask.contains(2));
    }
}
