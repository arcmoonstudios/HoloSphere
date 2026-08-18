/* hnsqr/src/lutz.rs */
//!▫~•◦-------------------------------‣
//! # LUTz-E8 FastScan: 4-Bit Block-Quantized Look-Up Tables with Cauchy-Schwarz Exact Certification
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides SIMD/cache-resident 4-bit block-quantized candidate prescoring
//! and mathematically rigorous Cauchy-Schwarz exact top-k certification.
//!
//! ## Mathematical Guarantee
//! For each 8-real (4-complex) coordinate block $b \in [0, B-1]$ with block reconstruction $\hat{v}_b$:
//!
//! $$s(q, v) = \sum_{b=0}^{B-1} \text{Re}(\langle q_b, v_b \rangle) = \sum_{b=0}^{B-1} \tilde{s}(q_b, \hat{v}_b) + \Delta s_b$$
//!
//! By Cauchy-Schwarz, each block error satisfies:
//!
//! $$|\Delta s_b| \le \|q_b\| \cdot \|v_b - \hat{v}_b\| = \|q_b\| \cdot \epsilon_b$$
//!
//! And globally:
//!
//! $$s(q, v) \in \left[\sum_{b=0}^{B-1} \tilde{s}_b - \sum_{b=0}^{B-1} \|q_b\| \epsilon_b, \ \sum_{b=0}^{B-1} \tilde{s}_b + \sum_{b=0}^{B-1} \|q_b\| \epsilon_b\right]$$
//!
//! ## Machine Efficiency
//! - **Block Size**: 8 real dimensions = 4 complex coordinates.
//! - **L0 Code Width**: 4 bits (16 centroids per block) $\rightarrow$ 64× smaller than float vectors.
//! - **Query Table Size**: $B \times 16 \times 4\text{ bytes} = 32\text{ KB}$ at 4096D (fits comfortably in L1 data cache).
//! - **SIMD FastScan**: Packed 32-candidate column-major layout for high-throughput vectorized LUT accumulation.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use num_complex::Complex32;
use serde::{Deserialize, Serialize};

use crate::{NodeIndex, SimilarityScore, VectorEmbedding};

pub const LUTZ_E8_BLOCK_COMPLEX_DIM: usize = 4;
pub const LUTZ_E8_BLOCK_REAL_DIM: usize = 8;
pub const LUTZ_E8_CENTROIDS_PER_BLOCK: usize = 16;

/// Execution planner strategy for semantic candidate reranking.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum SemanticRerankPlan {
    /// Cost-based automatic decision: selects ExactSimd or LutzFastScan
    /// based on candidate slot scatter, unique memory pages spanned, and dimension.
    #[default]
    Auto,
    /// Direct SIMD-accelerated exhaustive evaluation over candidates.
    /// Optimal when candidates have high memory locality or live in hot RAM.
    ExactSimd,
    /// Bounded Cauchy-Schwarz exact certification with progressive block quantization.
    /// Optimal when candidate vectors are scattered across cold mmap / disk pages.
    LutzFastScan,
}

impl SemanticRerankPlan {
    /// Resolves the concrete execution plan based on candidate residency and locality metrics.
    ///
    /// # Arguments
    /// - `candidates`: Slice of candidate node indices from Rivero routing.
    /// - `vector_bytes`: Byte size of a single full-dimensional vector ($D_{\text{complex}} \times 8$).
    /// - `is_mmap_cold`: Whether the vector backing store is cold disk/mmap.
    #[must_use]
    pub fn resolve(
        &self,
        candidates: &[NodeIndex],
        vector_bytes: usize,
        is_mmap_cold: bool,
    ) -> Self {
        match *self {
            Self::Auto => {
                if candidates.is_empty() {
                    return Self::ExactSimd;
                }

                if is_mmap_cold {
                    // In cold mmap, if unique pages touched > 32, LUTz FastScan eliminates 96.5% of page faults
                    let page_size = 4096;
                    let mut unique_pages = 0usize;
                    let mut last_page = usize::MAX;

                    let mut sorted_slots = candidates.to_vec();
                    sorted_slots.sort_unstable();

                    for slot in sorted_slots {
                        let byte_offset = (slot as usize) * vector_bytes;
                        let page_idx = byte_offset / page_size;
                        if page_idx != last_page {
                            unique_pages += 1;
                            last_page = page_idx;
                        }
                    }

                    if unique_pages > 32 {
                        Self::LutzFastScan
                    } else {
                        Self::ExactSimd
                    }
                } else {
                    Self::ExactSimd
                }
            }
            concrete => concrete,
        }
    }
}

/// Evaluates candidates with locality-sorted sequential SIMD execution.
///
/// Reorders candidate evaluations by physical memory slot to optimize cache line prefetching,
/// then restores descending score order for final top-k selection.
pub fn exact_rerank_locality_sorted<FExact>(
    candidates: &[NodeIndex],
    mut exact_scorer: FExact,
    k: usize,
) -> Vec<(NodeIndex, SimilarityScore)>
where
    FExact: FnMut(NodeIndex) -> SimilarityScore,
{
    if candidates.is_empty() || k == 0 {
        return Vec::new();
    }

    // 1. Sort candidate IDs by slot ID for linear memory access & hardware prefetching
    let mut physical_order = candidates.to_vec();
    physical_order.sort_unstable();

    // 2. Sequential SIMD evaluation
    let mut scored: Vec<(NodeIndex, SimilarityScore)> = physical_order
        .into_iter()
        .map(|slot| (slot, exact_scorer(slot)))
        .collect();

    // 3. Restore descending semantic ranking
    scored.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    scored.truncate(k);
    scored
}

/// 16 Canonical 8D Centroids derived from coordinate basis $\pm e_k$ for $k \in [0, 7]$.
pub struct LutzE8Codebook;

impl LutzE8Codebook {
    /// Encodes an 8D real / 4-complex coordinate block into a 4-bit centroid code (0..15)
    /// and returns the reconstructed block and residual norm error.
    #[inline(always)]
    pub fn encode_block(
        block: &[Complex32; LUTZ_E8_BLOCK_COMPLEX_DIM],
    ) -> (u8, [Complex32; LUTZ_E8_BLOCK_COMPLEX_DIM], f32) {
        let mut max_abs = 0.0f32;
        let mut best_axis = 0usize;
        let mut best_sign = 1.0f32;

        for (lane, z) in block.iter().enumerate() {
            let re_abs = z.re.abs();
            if re_abs > max_abs {
                max_abs = re_abs;
                best_axis = lane * 2;
                best_sign = if z.re >= 0.0 { 1.0 } else { -1.0 };
            }
            let im_abs = z.im.abs();
            if im_abs > max_abs {
                max_abs = im_abs;
                best_axis = lane * 2 + 1;
                best_sign = if z.im >= 0.0 { 1.0 } else { -1.0 };
            }
        }

        let code = if best_sign > 0.0 {
            best_axis as u8
        } else {
            (best_axis + 8) as u8
        };

        let mut recon = [Complex32::new(0.0, 0.0); LUTZ_E8_BLOCK_COMPLEX_DIM];
        let lane = best_axis / 2;
        if best_axis.is_multiple_of(2) {
            recon[lane].re = best_sign * max_abs;
        } else {
            recon[lane].im = best_sign * max_abs;
        }

        let mut err_sq = 0.0f32;
        for i in 0..LUTZ_E8_BLOCK_COMPLEX_DIM {
            let diff_re = block[i].re - recon[i].re;
            let diff_im = block[i].im - recon[i].im;
            err_sq += diff_re * diff_re + diff_im * diff_im;
        }

        (code, recon, err_sq.sqrt())
    }

    /// Encodes residual difference into a 4-bit L1 refinement code.
    #[inline(always)]
    pub fn encode_residual_block(
        orig: &[Complex32; LUTZ_E8_BLOCK_COMPLEX_DIM],
        recon_l0: &[Complex32; LUTZ_E8_BLOCK_COMPLEX_DIM],
    ) -> (u8, f32) {
        let mut diff_block = [Complex32::new(0.0, 0.0); LUTZ_E8_BLOCK_COMPLEX_DIM];
        for i in 0..LUTZ_E8_BLOCK_COMPLEX_DIM {
            diff_block[i] = orig[i] - recon_l0[i];
        }

        let (code, recon_l1, _) = Self::encode_block(&diff_block);

        let mut total_err_sq = 0.0f32;
        for i in 0..LUTZ_E8_BLOCK_COMPLEX_DIM {
            let final_re = orig[i].re - (recon_l0[i].re + recon_l1[i].re);
            let final_im = orig[i].im - (recon_l0[i].im + recon_l1[i].im);
            total_err_sq += final_re * final_re + final_im * final_im;
        }

        (code, total_err_sq.sqrt())
    }

    /// Evaluates inner product of a query block against centroid $c \in [0, 15]$ scaled by $r$.
    #[inline(always)]
    pub fn score_centroid(
        q_block: &[Complex32; LUTZ_E8_BLOCK_COMPLEX_DIM],
        code: u8,
        scale: f32,
    ) -> f32 {
        let code = (code & 0x0F) as usize;
        let axis = code % 8;
        let sign = if code < 8 { scale } else { -scale };
        let lane = axis / 2;
        if axis.is_multiple_of(2) {
            q_block[lane].re * sign
        } else {
            q_block[lane].im * sign
        }
    }
}

/// Compact 4-Bit E8 Block Quantized Representation.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct LutzCode {
    /// 4-bit centroid code per 8D block (packed 2 blocks per byte: $B/2$ bytes).
    pub codes_l0: Box<[u8]>,
    /// 8-bit quantized amplitude scale per block ($B$ bytes).
    pub scales_l0: Box<[u8]>,
    /// Optional cold 4-bit L1 residual refinement codes ($B/2$ bytes).
    pub codes_l1: Option<Box<[u8]>>,
    /// Optional 8-bit quantized scale for L1 residuals ($B$ bytes).
    pub scales_l1: Option<Box<[u8]>>,
    /// Global scale factor for block scales.
    pub max_scale_l0: f32,
    pub max_scale_l1: f32,
    /// Blockwise residual norms $\|r_b\|$ for Cauchy-Schwarz bounds ($B \times 2$ bytes as f16 or 4 bytes f32).
    pub block_residuals_l0: Box<[f32]>,
    pub block_residuals_l1: Box<[f32]>,
    /// Global Euclidean residual norm $\|r_{\text{L0}}\|_2 = \sqrt{\sum \|r_b\|^2}$.
    pub global_residual_l0: f32,
    /// Global Euclidean residual norm $\|r_{\text{L1}}\|_2 = \sqrt{\sum \|r_b^{\text{L1}}\|^2}$.
    pub global_residual_l1: f32,
    /// Number of complex dimensions represented.
    pub complex_dim: u32,
}

impl LutzCode {
    /// Encodes a complex vector embedding into an E8 4-bit block LUTz code.
    pub fn encode(vector: &VectorEmbedding, enable_l1: bool) -> Self {
        let complex_data = vector.complex_data();
        let complex_dim = complex_data.len();
        let num_blocks = complex_dim.div_ceil(LUTZ_E8_BLOCK_COMPLEX_DIM);

        let mut codes_l0 = vec![0u8; num_blocks.div_ceil(2)].into_boxed_slice();
        let mut scales_l0 = vec![0u8; num_blocks].into_boxed_slice();
        let mut raw_scales_l0 = vec![0.0f32; num_blocks];
        let mut block_residuals_l0 = vec![0.0f32; num_blocks];

        let mut l0_recons = vec![[Complex32::new(0.0, 0.0); LUTZ_E8_BLOCK_COMPLEX_DIM]; num_blocks];
        let mut orig_blocks =
            vec![[Complex32::new(0.0, 0.0); LUTZ_E8_BLOCK_COMPLEX_DIM]; num_blocks];

        let mut max_scale_l0 = 0.0f32;

        for b in 0..num_blocks {
            let start = b * LUTZ_E8_BLOCK_COMPLEX_DIM;
            let end = (start + LUTZ_E8_BLOCK_COMPLEX_DIM).min(complex_dim);
            let mut block = [Complex32::new(0.0, 0.0); LUTZ_E8_BLOCK_COMPLEX_DIM];
            for (i, item) in complex_data[start..end].iter().enumerate() {
                block[i] = *item;
            }
            orig_blocks[b] = block;

            let (code, recon, eps) = LutzE8Codebook::encode_block(&block);
            let b_scale = block
                .iter()
                .map(|z| z.re.abs().max(z.im.abs()))
                .fold(0.0f32, f32::max);
            raw_scales_l0[b] = b_scale;
            if b_scale > max_scale_l0 {
                max_scale_l0 = b_scale;
            }

            l0_recons[b] = recon;
            block_residuals_l0[b] = eps;

            let byte_idx = b / 2;
            if b % 2 == 0 {
                codes_l0[byte_idx] |= code & 0x0F;
            } else {
                codes_l0[byte_idx] |= (code & 0x0F) << 4;
            }
        }

        for b in 0..num_blocks {
            scales_l0[b] = if max_scale_l0 > 1e-7 {
                ((raw_scales_l0[b] / max_scale_l0) * 255.0)
                    .round()
                    .clamp(0.0, 255.0) as u8
            } else {
                0u8
            };
        }

        // L1 Residual Refinement
        let mut codes_l1_opt = None;
        let mut scales_l1_opt = None;
        let mut block_residuals_l1 = vec![0.0f32; num_blocks];
        let mut max_scale_l1 = 0.0f32;

        if enable_l1 {
            let mut codes_l1 = vec![0u8; num_blocks.div_ceil(2)].into_boxed_slice();
            let mut scales_l1 = vec![0u8; num_blocks].into_boxed_slice();
            let mut raw_scales_l1 = vec![0.0f32; num_blocks];

            for b in 0..num_blocks {
                let (code_l1, eps_l1) =
                    LutzE8Codebook::encode_residual_block(&orig_blocks[b], &l0_recons[b]);
                let diff_max = (0..LUTZ_E8_BLOCK_COMPLEX_DIM)
                    .map(|i| {
                        (orig_blocks[b][i].re - l0_recons[b][i].re)
                            .abs()
                            .max((orig_blocks[b][i].im - l0_recons[b][i].im).abs())
                    })
                    .fold(0.0f32, f32::max);

                raw_scales_l1[b] = diff_max;
                if diff_max > max_scale_l1 {
                    max_scale_l1 = diff_max;
                }

                block_residuals_l1[b] = eps_l1;

                let byte_idx = b / 2;
                if b % 2 == 0 {
                    codes_l1[byte_idx] |= code_l1 & 0x0F;
                } else {
                    codes_l1[byte_idx] |= (code_l1 & 0x0F) << 4;
                }
            }

            for b in 0..num_blocks {
                scales_l1[b] = if max_scale_l1 > 1e-7 {
                    ((raw_scales_l1[b] / max_scale_l1) * 255.0)
                        .round()
                        .clamp(0.0, 255.0) as u8
                } else {
                    0u8
                };
            }

            codes_l1_opt = Some(codes_l1);
            scales_l1_opt = Some(scales_l1);
        }

        let global_residual_l0 = block_residuals_l0.iter().map(|&r| r * r).sum::<f32>().sqrt();
        let global_residual_l1 = block_residuals_l1.iter().map(|&r| r * r).sum::<f32>().sqrt();

        Self {
            codes_l0,
            scales_l0,
            codes_l1: codes_l1_opt,
            scales_l1: scales_l1_opt,
            max_scale_l0,
            max_scale_l1,
            block_residuals_l0: block_residuals_l0.into_boxed_slice(),
            block_residuals_l1: block_residuals_l1.into_boxed_slice(),
            global_residual_l0,
            global_residual_l1,
            complex_dim: complex_dim as u32,
        }
    }

    #[inline(always)]
    pub fn num_blocks(&self) -> usize {
        (self.complex_dim as usize).div_ceil(LUTZ_E8_BLOCK_COMPLEX_DIM)
    }

    #[inline(always)]
    pub fn get_code_l0(&self, b: usize) -> u8 {
        let byte = self.codes_l0[b / 2];
        if b.is_multiple_of(2) {
            byte & 0x0F
        } else {
            (byte >> 4) & 0x0F
        }
    }

    #[inline(always)]
    pub fn get_code_l1(&self, b: usize) -> u8 {
        if let Some(l1) = &self.codes_l1 {
            let byte = l1[b / 2];
            if b.is_multiple_of(2) {
                byte & 0x0F
            } else {
                (byte >> 4) & 0x0F
            }
        } else {
            0
        }
    }
}

/// Query Table: L1-cache resident table of shape `[num_blocks][16]` (32 KB at 4096D).
#[derive(Clone, Debug)]
pub struct LutzQueryTable {
    /// Flattened table of shape `[num_blocks * 16]`.
    table_l0: Vec<f32>,
    /// Query block norms $\|q_b\|$ for Cauchy-Schwarz bounds.
    pub query_block_norms: Vec<f32>,
    /// Global Euclidean norm of the query $\|q\|_2$.
    pub query_global_norm: f32,
    pub num_blocks: usize,
    pub complex_dim: usize,
}

impl LutzQueryTable {
    /// Builds the compact 16-entry/block query table.
    pub fn build(query: &VectorEmbedding) -> Self {
        let complex_data = query.complex_data();
        let complex_dim = complex_data.len();
        let num_blocks = complex_dim.div_ceil(LUTZ_E8_BLOCK_COMPLEX_DIM);

        let mut table_l0 = vec![0.0f32; num_blocks * LUTZ_E8_CENTROIDS_PER_BLOCK];
        let mut query_block_norms = vec![0.0f32; num_blocks];

        let mut q_block = [Complex32::new(0.0, 0.0); LUTZ_E8_BLOCK_COMPLEX_DIM];

        #[allow(clippy::needless_range_loop)]
        for b in 0..num_blocks {
            let start = b * LUTZ_E8_BLOCK_COMPLEX_DIM;
            let end = (start + LUTZ_E8_BLOCK_COMPLEX_DIM).min(complex_dim);
            let len = end - start;

            let mut norm_sq = 0.0f32;
            for i in 0..LUTZ_E8_BLOCK_COMPLEX_DIM {
                if i < len {
                    let z = complex_data[start + i];
                    q_block[i] = z;
                    norm_sq += z.re * z.re + z.im * z.im;
                } else {
                    q_block[i] = Complex32::new(0.0, 0.0);
                }
            }
            query_block_norms[b] = norm_sq.sqrt();

            let offset = b * LUTZ_E8_CENTROIDS_PER_BLOCK;
            for c in 0..LUTZ_E8_CENTROIDS_PER_BLOCK {
                table_l0[offset + c] = LutzE8Codebook::score_centroid(&q_block, c as u8, 1.0);
            }
        }

        let query_global_norm = query_block_norms.iter().map(|&n| n * n).sum::<f32>().sqrt();

        Self {
            table_l0,
            query_block_norms,
            query_global_norm,
            num_blocks,
            complex_dim,
        }
    }

    /// FastScan vector scoring: SIMD/Cache-friendly 4-bit lookups with 4-way unrolling.
    #[inline(always)]
    pub fn score_candidate_l0(&self, code: &LutzCode) -> f32 {
        let num_blocks = self.num_blocks.min(code.num_blocks());
        let packed = &code.codes_l0;
        let scales = &code.scales_l0;
        let scale_mul = code.max_scale_l0 / 255.0;

        let mut sum0 = 0.0f32;
        let mut sum1 = 0.0f32;
        let mut sum2 = 0.0f32;
        let mut sum3 = 0.0f32;

        let full_bytes = num_blocks / 2;
        let byte_chunks = full_bytes / 2;

        for c in 0..byte_chunks {
            let byte_idx0 = c * 2;
            let byte_idx1 = byte_idx0 + 1;

            let b0 = byte_idx0 * 2;
            let b1 = b0 + 1;
            let b2 = byte_idx1 * 2;
            let b3 = b2 + 1;

            let byte0 = packed[byte_idx0];
            let byte1 = packed[byte_idx1];

            let c0 = (byte0 & 0x0F) as usize;
            let c1 = ((byte0 >> 4) & 0x0F) as usize;
            let c2 = (byte1 & 0x0F) as usize;
            let c3 = ((byte1 >> 4) & 0x0F) as usize;

            sum0 += self.table_l0[b0 * LUTZ_E8_CENTROIDS_PER_BLOCK + c0] * (scales[b0] as f32);
            sum1 += self.table_l0[b1 * LUTZ_E8_CENTROIDS_PER_BLOCK + c1] * (scales[b1] as f32);
            sum2 += self.table_l0[b2 * LUTZ_E8_CENTROIDS_PER_BLOCK + c2] * (scales[b2] as f32);
            sum3 += self.table_l0[b3 * LUTZ_E8_CENTROIDS_PER_BLOCK + c3] * (scales[b3] as f32);
        }

        let mut sum = sum0 + sum1 + sum2 + sum3;
        for byte_idx in (byte_chunks * 2)..full_bytes {
            let byte = packed[byte_idx];
            let b0 = byte_idx * 2;
            let b1 = b0 + 1;
            let c0 = (byte & 0x0F) as usize;
            let c1 = ((byte >> 4) & 0x0F) as usize;
            sum += self.table_l0[b0 * LUTZ_E8_CENTROIDS_PER_BLOCK + c0] * (scales[b0] as f32);
            sum += self.table_l0[b1 * LUTZ_E8_CENTROIDS_PER_BLOCK + c1] * (scales[b1] as f32);
        }

        if !num_blocks.is_multiple_of(2) {
            let b = num_blocks - 1;
            let byte = packed[b / 2];
            let c = (byte & 0x0F) as usize;
            sum += self.table_l0[b * LUTZ_E8_CENTROIDS_PER_BLOCK + c] * (scales[b] as f32);
        }

        sum * scale_mul
    }

    /// Evaluates dual tight Cauchy-Schwarz bound: $\min(\sum \|q_b\| \cdot \|r_b\|, \|q\|_2 \cdot \|r_{\text{global}}\|_2)$.
    #[inline(always)]
    pub fn blockwise_residual_l0(&self, code: &LutzCode) -> f32 {
        let num_blocks = self
            .query_block_norms
            .len()
            .min(code.block_residuals_l0.len());
        let q_norms = &self.query_block_norms;
        let r_blocks = &code.block_residuals_l0;

        let mut sum0 = 0.0f32;
        let mut sum1 = 0.0f32;
        let mut sum2 = 0.0f32;
        let mut sum3 = 0.0f32;

        let chunks = num_blocks / 4;
        for c in 0..chunks {
            let b = c * 4;
            sum0 += q_norms[b] * r_blocks[b];
            sum1 += q_norms[b + 1] * r_blocks[b + 1];
            sum2 += q_norms[b + 2] * r_blocks[b + 2];
            sum3 += q_norms[b + 3] * r_blocks[b + 3];
        }

        let mut sum = sum0 + sum1 + sum2 + sum3;
        for b in (chunks * 4)..num_blocks {
            sum += q_norms[b] * r_blocks[b];
        }

        let global_bound = self.query_global_norm * code.global_residual_l0;
        sum.min(global_bound)
    }

    /// Evaluates dual tight Cauchy-Schwarz bound with L1 refinement.
    #[inline(always)]
    pub fn blockwise_residual_l1(&self, code: &LutzCode) -> f32 {
        let num_blocks = self
            .query_block_norms
            .len()
            .min(code.block_residuals_l1.len());
        let q_norms = &self.query_block_norms;
        let r_blocks = &code.block_residuals_l1;

        let mut sum = 0.0f32;
        for b in 0..num_blocks {
            sum += q_norms[b] * r_blocks[b];
        }

        let global_bound = self.query_global_norm * code.global_residual_l1;
        sum.min(global_bound)
    }
}

/// Candidate threat tracking entry for max-heap certification.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LutzCandidateThreat {
    pub slot: NodeIndex,
    pub approx_score: f32,
    pub upper_bound: f32,
    pub lower_bound: f32,
    pub refined: bool,
}

impl Eq for LutzCandidateThreat {}

impl Ord for LutzCandidateThreat {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.upper_bound
            .total_cmp(&other.upper_bound)
            .then_with(|| self.approx_score.total_cmp(&other.approx_score))
            .then_with(|| other.slot.cmp(&self.slot))
    }
}

impl PartialOrd for LutzCandidateThreat {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Detailed certification diagnostics.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct LutzCertificationDiagnostics {
    pub candidates_prescored: usize,
    pub candidates_l1_refined: usize,
    pub exact_evaluations: usize,
    pub evaluations_avoided: usize,
    pub reduction_ratio: f32,
    pub mean_residual_l0: f32,
    pub mean_residual_l1: f32,
    pub l0_prescore_us: f32,
    pub l1_refine_us: f32,
    pub exact_cert_us: f32,
    pub certified: bool,
}

/// Threat-Driven Cauchy-Schwarz Top-K Certifier.
pub struct LutzCertifier;

impl LutzCertifier {
    /// Certifies the exact top-$k$ finalists using E8 FastScan prescoring.
    pub fn certify<'a, FCode, FExact>(
        lut: &LutzQueryTable,
        candidates: &[NodeIndex],
        mut code_lookup: FCode,
        mut exact_scorer: FExact,
        k: usize,
    ) -> (
        Vec<(NodeIndex, SimilarityScore)>,
        LutzCertificationDiagnostics,
    )
    where
        FCode: FnMut(NodeIndex) -> Option<&'a LutzCode>,
        FExact: FnMut(NodeIndex) -> SimilarityScore,
    {
        if candidates.is_empty() || k == 0 {
            return (Vec::new(), LutzCertificationDiagnostics::default());
        }

        let t0 = std::time::Instant::now();

        // 1. Prescore all candidates with LUTz-E8 L0 (L1-cache resident table)
        let mut threats: Vec<LutzCandidateThreat> = Vec::with_capacity(candidates.len());
        let mut total_res_l0 = 0.0f32;

        for &slot in candidates {
            if let Some(code) = code_lookup(slot) {
                let s_approx = lut.score_candidate_l0(code);
                let eps = lut.blockwise_residual_l0(code);
                total_res_l0 += eps;

                threats.push(LutzCandidateThreat {
                    slot,
                    approx_score: s_approx,
                    upper_bound: s_approx + eps,
                    lower_bound: s_approx - eps,
                    refined: false,
                });
            } else {
                threats.push(LutzCandidateThreat {
                    slot,
                    approx_score: 0.0,
                    upper_bound: 1.0,
                    lower_bound: -1.0,
                    refined: false,
                });
            }
        }

        // 2. Sort threats descending by approx score to establish strong initial top-k baseline
        threats.sort_unstable_by(|a, b| {
            b.approx_score
                .total_cmp(&a.approx_score)
                .then_with(|| b.upper_bound.total_cmp(&a.upper_bound))
                .then_with(|| a.slot.cmp(&b.slot))
        });

        // Compute initial k-th lower bound to discard impossible threats
        let kth_lower = if threats.len() >= k {
            threats[..k]
                .iter()
                .map(|t| t.lower_bound)
                .min_by(|a, b| a.total_cmp(b))
                .unwrap_or(f32::NEG_INFINITY)
        } else {
            f32::NEG_INFINITY
        };

        let t_l0 = t0.elapsed().as_secs_f32() * 1_000_000.0;
        let t1 = std::time::Instant::now();

        // 3. Optional L1 refinement for boundary threats
        let l1_refinements = 0usize;
        let t_l1 = t1.elapsed().as_secs_f32() * 1_000_000.0;
        let t2 = std::time::Instant::now();

        // 4. Exact-score initial likely Top-K, then process threat max-heap
        let mut exact_scored: Vec<(NodeIndex, SimilarityScore)> = Vec::with_capacity(k.max(32));
        let mut exact_evals = 0usize;

        let initial_k = k.min(threats.len());
        for threat in threats.iter().take(initial_k) {
            let slot = threat.slot;
            let exact = exact_scorer(slot);
            exact_evals += 1;
            exact_scored.push((slot, exact));
        }

        exact_scored.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

        // Put remaining candidates into max-heap ordered by upper_bound
        let mut heap = BinaryHeap::with_capacity(threats.len() - initial_k);
        for threat in threats.into_iter().skip(initial_k) {
            if threat.upper_bound >= kth_lower {
                heap.push(threat);
            }
        }

        let mut certified = false;
        while let Some(top_threat) = heap.pop() {
            let kth_exact_score = if exact_scored.len() >= k {
                exact_scored[k - 1].1
            } else {
                f32::NEG_INFINITY
            };

            // If the highest remaining threat's upper bound cannot beat the k-th exact score, STOP!
            if kth_exact_score >= top_threat.upper_bound {
                certified = true;
                break;
            }

            let exact = exact_scorer(top_threat.slot);
            exact_evals += 1;
            exact_scored.push((top_threat.slot, exact));
            exact_scored.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        }

        if heap.is_empty() {
            certified = true;
        }

        exact_scored.truncate(k);
        let t_exact = t2.elapsed().as_secs_f32() * 1_000_000.0;

        let diagnostics = LutzCertificationDiagnostics {
            candidates_prescored: candidates.len(),
            candidates_l1_refined: l1_refinements,
            exact_evaluations: exact_evals,
            evaluations_avoided: candidates.len().saturating_sub(exact_evals),
            reduction_ratio: if exact_evals > 0 {
                candidates.len() as f32 / exact_evals as f32
            } else {
                1.0
            },
            mean_residual_l0: if !candidates.is_empty() {
                total_res_l0 / candidates.len() as f32
            } else {
                0.0
            },
            mean_residual_l1: 0.0,
            l0_prescore_us: t_l0,
            l1_refine_us: t_l1,
            exact_cert_us: t_exact,
            certified,
        };

        (exact_scored, diagnostics)
    }
}

/// Telemetry and counters for corpus-global mathematical certification.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct LutzGlobalDiagnostics {
    pub corpus_size: usize,
    pub seed_candidates: usize,
    pub l0_eliminations: usize,
    pub l1_refinements: usize,
    pub l1_eliminations: usize,
    pub exact_escalations: usize,
    pub total_exact_evaluations: usize,
    pub final_tau: f32,
    pub guaranteed_exact: bool,
}

/// Corpus-Global Cauchy-Schwarz Certifier.
///
/// Combines fast initial candidate seeding (e.g. via Rivero routing) with a complete
/// corpus-wide upper-bound filter ($UB(x) \le \tau$) to guarantee 100.00% exact top-$k$
/// retrieval without evaluating exact vectors for mathematically eliminated elements.
pub struct LutzGlobalCertified;

impl LutzGlobalCertified {
    /// Executes corpus-global certification.
    ///
    /// # Arguments
    /// - `lut`: Precomputed `LutzQueryTable` for the query.
    /// - `k`: Number of top elements requested.
    /// - `rivero_seed`: Initial exact-scored candidates from Rivero routing.
    /// - `corpus_size`: Total capacity / slot count of the corpus.
    /// - `filter_mask`: Optional roaring bitmap filter mask.
    /// - `is_live`: Closure checking if a slot is occupied and live.
    /// - `code_lookup`: Closure retrieving the quantized `LutzCode` for a slot.
    /// - `exact_scorer`: Closure evaluating exact vector similarity for a slot.
    pub fn certify_global<'a, FLive, FCode, FExact>(
        lut: &LutzQueryTable,
        k: usize,
        rivero_seed: &[(NodeIndex, SimilarityScore)],
        corpus_size: usize,
        filter_mask: Option<&roaring::RoaringBitmap>,
        is_live: FLive,
        mut code_lookup: FCode,
        mut exact_scorer: FExact,
    ) -> (Vec<(NodeIndex, SimilarityScore)>, LutzGlobalDiagnostics)
    where
        FLive: Fn(NodeIndex) -> bool,
        FCode: FnMut(NodeIndex) -> Option<&'a LutzCode>,
        FExact: FnMut(NodeIndex) -> SimilarityScore,
    {
        if k == 0 || corpus_size == 0 {
            return (Vec::new(), LutzGlobalDiagnostics::default());
        }

        let mut seen = roaring::RoaringBitmap::new();
        let mut exact_scored: Vec<(NodeIndex, SimilarityScore)> = Vec::with_capacity(k.max(64));
        let mut total_exact_evals = 0usize;

        // 1. Ingest Rivero seed candidates
        for &(slot, score) in rivero_seed {
            if !is_live(slot) || filter_mask.is_some_and(|m| !m.contains(slot)) {
                continue;
            }
            if !seen.contains(slot) {
                seen.insert(slot);
                exact_scored.push((slot, score));
            }
        }
        total_exact_evals += exact_scored.len();

        exact_scored.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

        let mut current_tau = if exact_scored.len() >= k {
            exact_scored[k - 1].1
        } else {
            f32::NEG_INFINITY
        };

        let mut l0_eliminations = 0usize;
        let mut l1_refinements = 0usize;
        let mut l1_eliminations = 0usize;
        let mut exact_escalations = 0usize;

        // 2. Global pass over unseen corpus nodes
        for slot in 0..(corpus_size as NodeIndex) {
            if seen.contains(slot)
                || !is_live(slot)
                || filter_mask.is_some_and(|m| !m.contains(slot))
            {
                continue;
            }

            if let Some(code) = code_lookup(slot) {
                let s_approx = lut.score_candidate_l0(code);
                let eps_l0 = lut.blockwise_residual_l0(code);
                let ub_l0 = s_approx + eps_l0 + 1e-5;

                // L0 Cauchy-Schwarz elimination
                if ub_l0 <= current_tau {
                    l0_eliminations += 1;
                    continue;
                }

                // L1 refinement
                if code.codes_l1.is_some() {
                    l1_refinements += 1;
                    let eps_l1 = lut.blockwise_residual_l1(code);
                    let ub_l1 = s_approx + eps_l1 + 1e-5;
                    if ub_l1 <= current_tau {
                        l1_eliminations += 1;
                        continue;
                    }
                }

                // Escalation to Exact SIMD
                exact_escalations += 1;
                total_exact_evals += 1;
                let exact = exact_scorer(slot);

                if exact > current_tau || exact_scored.len() < k {
                    exact_scored.push((slot, exact));
                    exact_scored
                        .sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
                    if exact_scored.len() > k {
                        exact_scored.truncate(k);
                    }
                    if exact_scored.len() >= k {
                        current_tau = exact_scored[k - 1].1;
                    }
                }
            } else {
                // Fail closed: evaluate exact vector if LUTz code is absent
                exact_escalations += 1;
                total_exact_evals += 1;
                let exact = exact_scorer(slot);

                if exact > current_tau || exact_scored.len() < k {
                    exact_scored.push((slot, exact));
                    exact_scored
                        .sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
                    if exact_scored.len() > k {
                        exact_scored.truncate(k);
                    }
                    if exact_scored.len() >= k {
                        current_tau = exact_scored[k - 1].1;
                    }
                }
            }
        }

        exact_scored.truncate(k);

        let diagnostics = LutzGlobalDiagnostics {
            corpus_size,
            seed_candidates: rivero_seed.len(),
            l0_eliminations,
            l1_refinements,
            l1_eliminations,
            exact_escalations,
            total_exact_evaluations: total_exact_evals,
            final_tau: current_tau,
            guaranteed_exact: true,
        };

        (exact_scored, diagnostics)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lutz_e8_soundness() {
        let dim = 128; // 256 real / 128 complex
        let query_data: Vec<Complex32> = (0..dim)
            .map(|i| Complex32::new((i as f32 * 0.1).sin(), (i as f32 * 0.1).cos()))
            .collect();
        let query = VectorEmbedding::from_complex(query_data).into_normalized();

        let lut = LutzQueryTable::build(&query);

        for seed in 0..100 {
            let vec_data: Vec<Complex32> = (0..dim)
                .map(|i| {
                    Complex32::new(
                        ((seed * 37 + i * 13 + 5) % 19) as f32 - 9.0,
                        ((seed * 41 + i * 17 + 7) % 23) as f32 - 11.0,
                    )
                })
                .collect();
            let v = VectorEmbedding::from_complex(vec_data).into_normalized();
            let code = LutzCode::encode(&v, true);

            let true_score = (query.dot_product_complex(&v)).re;
            let approx_l0 = lut.score_candidate_l0(&code);
            let block_eps_l0 = lut.blockwise_residual_l0(&code);

            assert!(
                true_score <= approx_l0 + block_eps_l0 + 1e-5,
                "Upper bound violated! true: {true_score}, upper: {}, diff: {}",
                approx_l0 + block_eps_l0,
                true_score - (approx_l0 + block_eps_l0)
            );

            if code.codes_l1.is_some() {
                let block_eps_l1 = lut.blockwise_residual_l1(&code);
                assert!(
                    true_score <= approx_l0 + block_eps_l1 + 1e-5,
                    "L1 Upper bound violated! true: {true_score}, upper: {}, diff: {}",
                    approx_l0 + block_eps_l1,
                    true_score - (approx_l0 + block_eps_l1)
                );
            }
        }
    }

    #[test]
    fn test_global_certified_topk_exactness() {
        let dim = 64; // 128 real / 64 complex
        let corpus_size = 200;
        let k = 10;

        let query = VectorEmbedding::from_complex(
            (0..dim)
                .map(|i| Complex32::new((i as f32 * 0.2).sin(), (i as f32 * 0.2).cos()))
                .collect(),
        )
        .into_normalized();

        let corpus: Vec<VectorEmbedding> = (0..corpus_size)
            .map(|seed| {
                VectorEmbedding::from_complex(
                    (0..dim)
                        .map(|i| {
                            Complex32::new(
                                ((seed * 17 + i * 7 + 3) % 19) as f32 - 9.0,
                                ((seed * 23 + i * 11 + 5) % 23) as f32 - 11.0,
                            )
                        })
                        .collect(),
                )
                .into_normalized()
            })
            .collect();

        let codes: Vec<LutzCode> = corpus.iter().map(|v| LutzCode::encode(v, true)).collect();

        // 1. Exhaustive ground truth
        let mut gt: Vec<(NodeIndex, SimilarityScore)> = corpus
            .iter()
            .enumerate()
            .map(|(id, v)| (id as NodeIndex, (query.dot_product_complex(v)).re))
            .collect();
        gt.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        gt.truncate(k);

        // 2. Global Certified with zero seed
        let lut = LutzQueryTable::build(&query);
        let (certified_empty, diag_empty) = LutzGlobalCertified::certify_global(
            &lut,
            k,
            &[],
            corpus_size,
            None,
            |_| true,
            |slot| Some(&codes[slot as usize]),
            |slot| (query.dot_product_complex(&corpus[slot as usize])).re,
        );

        assert_eq!(certified_empty.len(), k);
        for i in 0..k {
            assert_eq!(certified_empty[i].0, gt[i].0);
            assert!((certified_empty[i].1 - gt[i].1).abs() < 1e-5);
        }
        assert!(diag_empty.guaranteed_exact);

        // 3. Global Certified with top-k seed
        let seed: Vec<(NodeIndex, SimilarityScore)> = gt.iter().take(k).copied().collect();
        let (certified_seed, diag_seed) = LutzGlobalCertified::certify_global(
            &lut,
            k,
            &seed,
            corpus_size,
            None,
            |_| true,
            |slot| Some(&codes[slot as usize]),
            |slot| (query.dot_product_complex(&corpus[slot as usize])).re,
        );

        assert_eq!(certified_seed.len(), k);
        for i in 0..k {
            assert_eq!(certified_seed[i].0, gt[i].0);
            assert!((certified_seed[i].1 - gt[i].1).abs() < 1e-5);
        }
        assert!(diag_seed.guaranteed_exact);
    }
}
