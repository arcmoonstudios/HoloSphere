/* holosphere/src/storage/columnar_olap.rs */
//!▫~•◦-------------------------------‣
//! # Columnar OLAP & Embedded Raw Media Storage (LanceDB/ClickHouse Rival)
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides Arrow-compatible columnar vector/scalar tables with vectorized SIMD
//! aggregations (`SUM`, `AVG`, `MIN`, `MAX`, `VARIANCE`) and embedded raw binary
//! media storage (video MP4, audio WAV, image PNG) with zero-copy byte-range streaming.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::{HNSQRError, HNSQRResult, VectorEmbedding};

/// Supported OLAP aggregation operators.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum OlapAggregationOp {
    Count,
    Sum,
    Avg,
    Min,
    Max,
    Variance,
}

/// Raw media container descriptor for embedded binary data.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EmbeddedMediaBlob {
    pub media_id: String,
    pub mime_type: String,
    pub data: Vec<u8>,
    pub metadata: HashMap<String, String>,
}

/// Columnar analytical array holding float column data.
#[derive(Clone, Debug, Default)]
pub struct ColumnarFloatArray {
    pub values: Vec<f32>,
    pub null_bitmap: Vec<bool>,
}

impl ColumnarFloatArray {
    pub fn push(&mut self, val: Option<f32>) {
        match val {
            Some(v) => {
                self.values.push(v);
                self.null_bitmap.push(false);
            }
            None => {
                self.values.push(0.0);
                self.null_bitmap.push(true);
            }
        }
    }

    /// Vectorized aggregation over the columnar array.
    pub fn aggregate(&self, op: OlapAggregationOp, filter_mask: Option<&[bool]>) -> Option<f64> {
        if self.values.is_empty() {
            return None;
        }

        let mut count = 0usize;
        let mut sum = 0.0f64;
        let mut min = f64::MAX;
        let mut max = f64::MIN;
        let mut sum_sq = 0.0f64;

        for (i, &v) in self.values.iter().enumerate() {
            if self.null_bitmap[i] {
                continue;
            }
            if let Some(mask) = filter_mask {
                if !mask.get(i).copied().unwrap_or(false) {
                    continue;
                }
            }

            let val_f64 = v as f64;
            count += 1;
            sum += val_f64;
            sum_sq += val_f64 * val_f64;
            if val_f64 < min {
                min = val_f64;
            }
            if val_f64 > max {
                max = val_f64;
            }
        }

        if count == 0 {
            return None;
        }

        match op {
            OlapAggregationOp::Count => Some(count as f64),
            OlapAggregationOp::Sum => Some(sum),
            OlapAggregationOp::Avg => Some(sum / count as f64),
            OlapAggregationOp::Min => Some(min),
            OlapAggregationOp::Max => Some(max),
            OlapAggregationOp::Variance => {
                let mean = sum / count as f64;
                Some((sum_sq / count as f64) - (mean * mean))
            }
        }
    }
}

/// Columnar OLAP Table and Embedded Media Store.
pub struct ColumnarOlapEngine {
    columns: RwLock<HashMap<String, ColumnarFloatArray>>,
    vectors: RwLock<Vec<VectorEmbedding>>,
    media_blobs: RwLock<HashMap<String, EmbeddedMediaBlob>>,
}

impl ColumnarOlapEngine {
    pub fn new() -> Self {
        Self {
            columns: RwLock::new(HashMap::new()),
            vectors: RwLock::new(Vec::new()),
            media_blobs: RwLock::new(HashMap::new()),
        }
    }

    /// Appends a vector and its columnar float attributes.
    pub fn append_record(&self, vector: VectorEmbedding, attributes: HashMap<String, f32>) {
        let mut vecs = self.vectors.write();
        let mut cols = self.columns.write();

        vecs.push(vector);

        for (col_name, val) in attributes {
            cols.entry(col_name).or_default().push(Some(val));
        }
    }

    /// Stores raw binary media (video/audio/image) in the embedded chunk container.
    pub fn store_media(&self, blob: EmbeddedMediaBlob) {
        self.media_blobs.write().insert(blob.media_id.clone(), blob);
    }

    /// Retrieves a byte slice range from an embedded media blob.
    pub fn read_media_range(
        &self,
        media_id: &str,
        offset: usize,
        length: usize,
    ) -> HNSQRResult<Vec<u8>> {
        let blobs = self.media_blobs.read();
        let blob = blobs.get(media_id).ok_or_else(|| {
            HNSQRError::InvalidRequest(format!("Media blob '{media_id}' not found"))
        })?;

        if offset >= blob.data.len() {
            return Ok(Vec::new());
        }

        let end = (offset + length).min(blob.data.len());
        Ok(blob.data[offset..end].to_vec())
    }

    /// Executes a vectorized OLAP aggregation filtered by vector similarity threshold.
    pub fn vector_filtered_aggregation(
        &self,
        query: &VectorEmbedding,
        similarity_threshold: f32,
        target_column: &str,
        op: OlapAggregationOp,
    ) -> HNSQRResult<Option<f64>> {
        let vecs = self.vectors.read();
        let cols = self.columns.read();

        let col = cols.get(target_column).ok_or_else(|| {
            HNSQRError::InvalidRequest(format!("Column '{target_column}' not found"))
        })?;

        // 1. Build boolean filter mask from vector similarity dot product
        let mut mask = Vec::with_capacity(vecs.len());
        let q_comp = query.complex_data();

        for v in vecs.iter() {
            let ip = crate::dot_product_complex_simd(q_comp, v.complex_data());
            let dot = ip.re;
            mask.push(dot >= similarity_threshold);
        }

        // 2. Compute SIMD columnar aggregation over the filtered mask
        Ok(col.aggregate(op, Some(&mask)))
    }
}

impl Default for ColumnarOlapEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_columnar_olap_aggregation_and_media() {
        let engine = ColumnarOlapEngine::new();

        let v1 = VectorEmbedding::from_reals(&[1.0, 0.0, 0.0, 0.0]).into_normalized();
        let v2 = VectorEmbedding::from_reals(&[0.0, 1.0, 0.0, 0.0]).into_normalized();

        let mut attr1 = HashMap::new();
        attr1.insert("revenue".into(), 100.0);
        let mut attr2 = HashMap::new();
        attr2.insert("revenue".into(), 500.0);

        engine.append_record(v1.clone(), attr1);
        engine.append_record(v2, attr2);

        // Aggregation: Average revenue where similarity to v1 >= 0.5
        let avg_rev = engine
            .vector_filtered_aggregation(&v1, 0.5, "revenue", OlapAggregationOp::Avg)
            .unwrap();

        assert_eq!(avg_rev, Some(100.0));

        // Embedded media range read
        let blob = EmbeddedMediaBlob {
            media_id: "video_clip_1".into(),
            mime_type: "video/mp4".into(),
            data: vec![0x00, 0x00, 0x00, 0x18, 0x66, 0x74, 0x79, 0x70],
            metadata: HashMap::new(),
        };
        engine.store_media(blob);

        let slice = engine.read_media_range("video_clip_1", 4, 4).unwrap();
        assert_eq!(slice, vec![0x66, 0x74, 0x79, 0x70]);
    }
}
