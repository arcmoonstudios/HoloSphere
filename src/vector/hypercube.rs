/* holosphere/src/vector/hypercube.rs */
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

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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
        Some(Self {
            min_coords,
            max_coords,
        })
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

/// 64-bit Morton Z-Order Space-Filling Curve Encoder for $N$-dimensional space.
#[derive(Clone, Debug)]
pub struct MortonEncoderND {
    dimensions: usize,
    bits_per_dim: usize,
}

impl MortonEncoderND {
    pub fn new(dimensions: usize) -> Self {
        let bits_per_dim = if dimensions > 0 {
            (64 / dimensions).min(21).max(1)
        } else {
            1
        };
        Self {
            dimensions,
            bits_per_dim,
        }
    }

    /// Encodes an N-dimensional coordinate into a 64-bit Morton index by bit interleaving.
    #[inline]
    pub fn encode(&self, coords: &[usize]) -> u64 {
        let mut morton = 0u64;
        let bits = self.bits_per_dim;
        for bit in 0..bits {
            for (dim_idx, &coord) in coords.iter().enumerate().take(self.dimensions) {
                let bit_val = ((coord >> bit) & 1) as u64;
                morton |= bit_val << (bit * self.dimensions + dim_idx);
            }
        }
        morton
    }

    /// Decodes a 64-bit Morton index back into an N-dimensional coordinate.
    #[inline]
    pub fn decode(&self, morton: u64, out: &mut [usize]) {
        for val in out.iter_mut().take(self.dimensions) {
            *val = 0;
        }
        let bits = self.bits_per_dim;
        for bit in 0..bits {
            for (dim_idx, out_val) in out.iter_mut().enumerate().take(self.dimensions) {
                let bit_pos = bit * self.dimensions + dim_idx;
                let bit_val = ((morton >> bit_pos) & 1) as usize;
                *out_val |= bit_val << bit;
            }
        }
    }
}

/// Fixed hyper-tile edge size for spatial chunking (16^N voxels per chunk max, dynamically packed).
pub const TILE_EDGE: usize = 16;

/// Contiguous Dense Hyper-Tile containing localized voxels in flat memory.
#[derive(Clone, Debug)]
pub struct HyperTile {
    pub tile_origin: CoordinateND,
    pub dense_buffer: Vec<f32>,
    pub occupancy: roaring::RoaringBitmap,
}

impl HyperTile {
    fn new(origin: CoordinateND, capacity: usize) -> Self {
        Self {
            tile_origin: origin,
            dense_buffer: vec![0.0; capacity],
            occupancy: roaring::RoaringBitmap::new(),
        }
    }

    #[inline]
    fn set(&mut self, local_morton: u32, value: f32) {
        let idx = local_morton as usize;
        if idx >= self.dense_buffer.len() {
            self.dense_buffer.resize(idx + 1, 0.0);
        }
        self.dense_buffer[idx] = value;
        self.occupancy.insert(local_morton);
    }

    #[inline]
    fn get(&self, local_morton: u32) -> Option<f32> {
        if self.occupancy.contains(local_morton) {
            self.dense_buffer.get(local_morton as usize).copied()
        } else {
            None
        }
    }
}

/// N-Dimensional Tensor Space managing dense chunked hypercube tiles with Morton Z-Order indexing.
pub struct HypercubeTensorSpace {
    shape: CoordinateND,
    strides: Vec<usize>,
    encoder: MortonEncoderND,
    tile_capacity: usize,
    tiles: RwLock<HashMap<u64, HyperTile>>,
}

impl HypercubeTensorSpace {
    pub fn new(shape: CoordinateND) -> Self {
        let n = shape.len();
        let mut strides = vec![1; n];
        for i in (0..n.saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }

        let encoder = MortonEncoderND::new(n);
        let tile_capacity = TILE_EDGE.pow(n.min(5) as u32);

        Self {
            shape,
            strides,
            encoder,
            tile_capacity,
            tiles: RwLock::new(HashMap::new()),
        }
    }

    /// Returns the number of dimensions in this tensor space.
    pub fn dimensions(&self) -> usize {
        self.shape.len()
    }

    /// Splits an N-dimensional coordinate into (Tile Origin, Local Tile Offset).
    #[inline]
    fn split_tile_coord(&self, coords: &[usize]) -> (CoordinateND, u32) {
        let mut origin = Vec::with_capacity(self.shape.len());
        let mut local = Vec::with_capacity(self.shape.len());
        for &c in coords {
            origin.push((c / TILE_EDGE) * TILE_EDGE);
            local.push(c % TILE_EDGE);
        }
        let local_morton = self.encoder.encode(&local) as u32;
        (origin, local_morton)
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

        let (tile_origin, local_offset) = self.split_tile_coord(&coords);
        let tile_morton = self.encoder.encode(&tile_origin);

        let mut tiles = self.tiles.write();
        let tile = tiles
            .entry(tile_morton)
            .or_insert_with(|| HyperTile::new(tile_origin, self.tile_capacity));
        tile.set(local_offset, value);
        Ok(())
    }

    /// Reads a voxel value at the given $N$-dimensional coordinate.
    pub fn get_voxel(&self, coords: &[usize]) -> Option<f32> {
        if coords.len() != self.shape.len() {
            return None;
        }
        for (i, &c) in coords.iter().enumerate() {
            if c >= self.shape[i] {
                return None;
            }
        }

        let (tile_origin, local_offset) = self.split_tile_coord(coords);
        let tile_morton = self.encoder.encode(&tile_origin);

        let tiles = self.tiles.read();
        let tile = tiles.get(&tile_morton)?;
        tile.get(local_offset)
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

        let tiles = self.tiles.read();
        let mut results = Vec::new();
        let mut local_coords = vec![0usize; self.shape.len()];

        for tile in tiles.values() {
            // Check if this tile overlaps with the bounding box
            let mut overlaps = true;
            for d in 0..self.shape.len() {
                let tile_min = tile.tile_origin[d];
                let tile_max = tile_min + TILE_EDGE - 1;
                if tile_max < bbox.min_coords[d] || tile_min > bbox.max_coords[d] {
                    overlaps = false;
                    break;
                }
            }

            if !overlaps {
                continue;
            }

            // Inspect occupied voxels inside the overlapping tile
            for local_morton in tile.occupancy.iter() {
                if let Some(val) = tile.get(local_morton) {
                    self.encoder.decode(local_morton as u64, &mut local_coords);
                    let mut global_coords = Vec::with_capacity(self.shape.len());
                    for d in 0..self.shape.len() {
                        global_coords.push(tile.tile_origin[d] + local_coords[d]);
                    }

                    if bbox.contains(&global_coords) {
                        results.push((global_coords, val));
                    }
                }
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

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Captures an immutable point-in-time snapshot of the hypercube tensor space.
    pub fn snapshot(&self) -> HypercubeSnapshot {
        let tiles = self.tiles.read().clone();
        HypercubeSnapshot {
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            tiles,
        }
    }
}

/// Immutable point-in-time snapshot of Hypercube Tensor Space at a specific LSN.
#[derive(Clone, Debug)]
pub struct HypercubeSnapshot {
    pub shape: CoordinateND,
    pub strides: Vec<usize>,
    pub tiles: HashMap<u64, HyperTile>,
}

impl HypercubeSnapshot {
    pub fn dimensions(&self) -> usize {
        self.shape.len()
    }

    pub fn get_voxel(&self, coords: &[usize]) -> Option<f32> {
        if coords.len() != self.shape.len() {
            return None;
        }
        for (i, &c) in coords.iter().enumerate() {
            if c >= self.shape[i] {
                return None;
            }
        }
        let mut origin = Vec::with_capacity(self.shape.len());
        let mut local = Vec::with_capacity(self.shape.len());
        for &c in coords {
            origin.push((c / TILE_EDGE) * TILE_EDGE);
            local.push(c % TILE_EDGE);
        }
        let encoder = MortonEncoderND::new(self.shape.len());
        let local_morton = encoder.encode(&local) as u32;
        let tile_morton = encoder.encode(&origin);
        let tile = self.tiles.get(&tile_morton)?;
        tile.get(local_morton)
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
        let bbox = HypercubeBoundingBox::new(vec![0, 0, 30, 70], vec![5, 2, 50, 80]).unwrap();

        let slice = space.slice_subvolume(&bbox).unwrap();
        assert_eq!(slice.len(), 1);
        assert_eq!(slice[0].0, vec![2, 1, 40, 74]);
        assert_eq!(slice[0].1, 298.15);
    }

    #[test]
    fn test_morton_encoding_roundtrip() {
        let encoder = MortonEncoderND::new(4);
        let coords = [5, 12, 30, 42];
        let morton = encoder.encode(&coords);
        let mut decoded = [0usize; 4];
        encoder.decode(morton, &mut decoded);
        assert_eq!(coords, decoded);
    }
}
