/* hnsqr/src/vector/hypercube.rs */
//!▫~•◦-------------------------------‣
//! # N-Dimensional Hypercube & Volumetric Tensor Slicing Engine (TileDB Rival)
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides $N$-dimensional coordinate tensor geometry ($N \ge 3$) for volumetric
//! arrays (3D MRI/CT scans, 4D spatio-temporal climate grids $T \times L \times X \times Y$,
//! genomic expression matrices), hypercube bounding box subvolume slicing, and voxel cell indexing.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::collections::BTreeMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::{HNSQRError, HNSQRResult};

/// N-Dimensional Coordinate Index (e.g. [x, y, z] or [t, level, lat, lon]).
pub type CoordinateND = Vec<usize>;

/// N-Dimensional Bounding Box representing a hyper-rectangle subvolume.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HypercubeBoundingBox {
    pub min_coords: CoordinateND,
    pub max_coords: CoordinateND,
}

impl HypercubeBoundingBox {
    pub fn new(min_coords: CoordinateND, max_coords: CoordinateND) -> Option<Self> {
        if min_coords.len() != max_coords.len() || min_coords.is_empty() {
            return None;
        }
        for (min, max) in min_coords.iter().zip(max_coords.iter()) {
            if min > max {
                return None;
            }
        }
        Some(Self { min_coords, max_coords })
    }

    /// Evaluates whether a coordinate point falls inside this hypercube subvolume.
    pub fn contains(&self, coords: &[usize]) -> bool {
        if coords.len() != self.min_coords.len() {
            return false;
        }
        for (i, &c) in coords.iter().enumerate() {
            if c < self.min_coords[i] || c > self.max_coords[i] {
                return false;
            }
        }
        true
    }

    /// Number of dimensions in this hypercube.
    pub fn dimensions(&self) -> usize {
        self.min_coords.len()
    }
}

/// N-Dimensional Tensor Space managing dense or sparse hypercube voxels.
#[allow(dead_code)]
pub struct HypercubeTensorSpace {
    shape: CoordinateND,
    strides: Vec<usize>,
    sparse_cells: RwLock<BTreeMap<CoordinateND, f32>>,
}

impl HypercubeTensorSpace {
    pub fn new(shape: CoordinateND) -> Self {
        let n = shape.len();
        let mut strides = vec![1; n];
        for i in (0..n.saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }

        Self {
            shape,
            strides,
            sparse_cells: RwLock::new(BTreeMap::new()),
        }
    }

    /// Returns the number of dimensions in this tensor space.
    pub fn dimensions(&self) -> usize {
        self.shape.len()
    }

    /// Sets a voxel value at the given $N$-dimensional coordinate.
    pub fn set_voxel(&self, coords: CoordinateND, value: f32) -> HNSQRResult<()> {
        if coords.len() != self.shape.len() {
            return Err(HNSQRError::InvalidRequest(format!(
                "Dimension mismatch: expected {}, got {}",
                self.shape.len(),
                coords.len()
            )));
        }

        for (i, &c) in coords.iter().enumerate() {
            if c >= self.shape[i] {
                return Err(HNSQRError::InvalidRequest(format!(
                    "Coordinate index out of bounds at dimension {i}: {c} >= {}",
                    self.shape[i]
                )));
            }
        }

        self.sparse_cells.write().insert(coords, value);
        Ok(())
    }

    /// Reads a voxel value at the given $N$-dimensional coordinate.
    pub fn get_voxel(&self, coords: &[usize]) -> Option<f32> {
        self.sparse_cells.read().get(coords).copied()
    }

    /// Slices a subvolume hypercube and returns all non-zero voxels inside the slice.
    pub fn slice_subvolume(
        &self,
        bbox: &HypercubeBoundingBox,
    ) -> HNSQRResult<Vec<(CoordinateND, f32)>> {
        if bbox.dimensions() != self.shape.len() {
            return Err(HNSQRError::InvalidRequest(format!(
                "Bounding box dimensions {} do not match tensor dimensions {}",
                bbox.dimensions(),
                self.shape.len()
            )));
        }

        let cells = self.sparse_cells.read();
        let mut results = Vec::new();

        for (coord, &val) in cells.iter() {
            if bbox.contains(coord) {
                results.push((coord.clone(), val));
            }
        }

        Ok(results)
    }

    /// Calculates total tensor hyper-volume (number of possible voxel positions).
    pub fn total_volume(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hypercube_4d_spatiotemporal_slicing() {
        // 4D Space: Time(10) x Level(5) x Latitude(180) x Longitude(360)
        let space = HypercubeTensorSpace::new(vec![10, 5, 180, 360]);
        assert_eq!(space.shape(), &[10, 5, 180, 360]);

        // Place climate reading at t=2, level=1, lat=40, lon=74
        space.set_voxel(vec![2, 1, 40, 74], 298.15).unwrap();
        // Place unrelated reading at t=8, level=4, lat=120, lon=200
        space.set_voxel(vec![8, 4, 120, 200], 215.30).unwrap();

        // Slice subvolume: t in [0..5], level in [0..2], lat in [30..50], lon in [70..80]
        let bbox = HypercubeBoundingBox::new(
            vec![0, 0, 30, 70],
            vec![5, 2, 50, 80],
        ).unwrap();

        let slice = space.slice_subvolume(&bbox).unwrap();
        assert_eq!(slice.len(), 1);
        assert_eq!(slice[0].0, vec![2, 1, 40, 74]);
        assert_eq!(slice[0].1, 298.15);
    }
}
