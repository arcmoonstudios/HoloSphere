/* holosphere/src/learning/inference/rune_evo/reasoning/blade.rs */
//!▫~•◦-------------------------------‣
//! # Cl(24) Multivector Blade Algebra & Sparse Geometric Product
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Direct reference-equivalent mathematical implementation of Cl(24,0,0) blade algebra,
//! parity-based geometric product sign computation, Top-K energy truncation,
//! grade-1 extraction, and canonical Leech-to-E8 projection.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use thiserror::Error;

use crate::entity::id::EntityId;
use crate::relation::instance::DurableRelationInstance;

#[derive(Error, Clone, Debug, PartialEq, Eq)]
pub enum Cl24BasisError {
    #[error("Cl(24) basis contains {0} entities; maximum is 24")]
    BasisTooWide(usize),
    #[error("Entity {0} appears more than once in the local Cl(24) basis")]
    DuplicateBasisEntity(EntityId),
    #[error("Entity {0} is not present in the local Cl(24) basis")]
    UnknownEntity(EntityId),
    #[error(
        "Relation {relation_id} binds entity {entity_id} more than once; a blade cannot encode role multiplicity"
    )]
    RepeatedRelationEntity {
        relation_id: u64,
        entity_id: EntityId,
    },
}

/// Explicit local mapping between durable entity identities and Cl(24) basis bits.
/// A blade grade denotes the number of distinct participating entities only under
/// this mapping; it never treats global `EntityId` values as bit positions.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Cl24EntityBasis {
    entities: Vec<EntityId>,
}

impl Cl24EntityBasis {
    pub fn new(entities: Vec<EntityId>) -> Result<Self, Cl24BasisError> {
        if entities.len() > 24 {
            return Err(Cl24BasisError::BasisTooWide(entities.len()));
        }
        let mut seen = std::collections::HashSet::new();
        for &entity in &entities {
            if !seen.insert(entity) {
                return Err(Cl24BasisError::DuplicateBasisEntity(entity));
            }
        }
        Ok(Self { entities })
    }

    pub fn entities(&self) -> &[EntityId] {
        &self.entities
    }

    pub fn blade_for_relation(
        &self,
        relation: &DurableRelationInstance,
        coeff: f32,
    ) -> Result<Cl24Blade, Cl24BasisError> {
        let mut seen = std::collections::HashSet::new();
        let mut bitmap = 0u32;
        for binding in &relation.bindings {
            if !seen.insert(binding.entity_id) {
                return Err(Cl24BasisError::RepeatedRelationEntity {
                    relation_id: relation.id,
                    entity_id: binding.entity_id,
                });
            }
            let bit = self
                .entities
                .iter()
                .position(|&entity| entity == binding.entity_id)
                .ok_or(Cl24BasisError::UnknownEntity(binding.entity_id))?;
            bitmap |= 1u32 << bit;
        }
        Ok(Cl24Blade::new(bitmap, coeff))
    }
}

/// Basis blade in Cl(24,0,0): bitmap indicates present basis vectors $e_0 \dots e_{23}$,
/// and coeff is the floating-point scalar weight.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Cl24Blade {
    pub bitmap: u32,
    pub coeff: f32,
}

impl Cl24Blade {
    #[inline]
    pub fn new(bitmap: u32, coeff: f32) -> Self {
        Self { bitmap, coeff }
    }

    #[inline]
    pub fn grade(&self) -> u32 {
        self.bitmap.count_ones()
    }
}

/// Compute the geometric product sign of two basis blades in Cl(24,0,0).
///
/// For each bit k set in bitmap_b, count how many bits in bitmap_a are at
/// positions strictly greater than k. The result sign is (-1)^(total count).
#[inline]
pub fn blade_product_sign(bitmap_a: u32, bitmap_b: u32) -> f32 {
    let mut parity = 0u32;
    let mut b = bitmap_b;
    while b != 0 {
        let k = b.trailing_zeros();
        parity ^= (bitmap_a >> (k + 1)).count_ones() & 1;
        b &= b - 1; // clear lowest set bit
    }
    if parity & 1 == 0 { 1.0 } else { -1.0 }
}

/// Sparse multivector in Cl(24,0,0).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct MultivectorCl24Sparse {
    pub blades: Vec<Cl24Blade>,
}

impl MultivectorCl24Sparse {
    pub fn identity() -> Self {
        Self {
            blades: vec![Cl24Blade::new(0, 1.0)],
        }
    }

    pub fn zero() -> Self {
        Self::default()
    }

    pub fn from_blades(blades: &[Cl24Blade]) -> Self {
        Self {
            blades: blades
                .iter()
                .filter(|b| b.coeff.is_finite() && b.coeff.abs() > f32::EPSILON)
                .copied()
                .collect(),
        }
    }

    pub fn from_grade1(coords: &[f32; 24]) -> Self {
        let mut blades = Vec::new();
        for (i, &c) in coords.iter().enumerate() {
            if c.is_finite() && c.abs() > f32::EPSILON {
                blades.push(Cl24Blade::new(1u32 << i, c));
            }
        }
        Self { blades }
    }

    /// Full Cl(24) geometric product `self ⊗ other`.
    ///
    /// Complexity is bounded by the sparse input blade counts; callers that need
    /// an operational cap apply `truncate_topk` after the product. No grade is
    /// discarded here because higher-grade blades encode N-ary relation structure.
    #[must_use]
    pub fn geometric_product(&self, other: &Self) -> Self {
        // Zero-allocation flat accumulator: linear scan for existing bitmap key,
        // insert-or-add in place. SmallVec<64> fits on the stack for the common
        // sparse case (truncate_topk caps inputs to ≤32 blades each in practice).
        let mut accum: SmallVec<[(u32, f32); 64]> = SmallVec::new();

        for ai in &self.blades {
            for bi in &other.blades {
                let result_bitmap = ai.bitmap ^ bi.bitmap;
                let sign = blade_product_sign(ai.bitmap, bi.bitmap);
                let contribution = sign * ai.coeff * bi.coeff;
                // Linear scan: fast for small blade counts, avoids heap allocation.
                if let Some(entry) = accum.iter_mut().find(|(bm, _)| *bm == result_bitmap) {
                    entry.1 += contribution;
                } else {
                    accum.push((result_bitmap, contribution));
                }
            }
        }

        let mut blades: Vec<Cl24Blade> = accum
            .into_iter()
            .filter(|&(_, c)| c.is_finite() && c.abs() > f32::EPSILON)
            .map(|(bitmap, coeff)| Cl24Blade::new(bitmap, coeff))
            .collect();

        // Deterministic sort by |coeff| descending, then bitmap ascending for stable ties
        blades.sort_by(|a, b| {
            b.coeff
                .abs()
                .total_cmp(&a.coeff.abs())
                .then_with(|| a.bitmap.cmp(&b.bitmap))
        });

        Self { blades }
    }

    /// Truncates multivector to top-K blades by |coeff| descending with deterministic tie-breaking.
    #[must_use]
    pub fn truncate_topk(&self, k: usize) -> Self {
        let mut blades = self.blades.clone();
        if blades.len() > k {
            blades.sort_by(|a, b| {
                b.coeff
                    .abs()
                    .total_cmp(&a.coeff.abs())
                    .then_with(|| a.bitmap.cmp(&b.bitmap))
            });
            blades.truncate(k);
        }
        Self { blades }
    }

    /// Total blade energy $E = \sum c_i^2$.
    pub fn energy(&self) -> f32 {
        self.blades.iter().map(|b| b.coeff * b.coeff).sum()
    }

    /// Extracts grade-1 coordinates as [f32; 24].
    pub fn to_grade1_coords(&self) -> [f32; 24] {
        let mut out = [0.0f32; 24];
        for b in &self.blades {
            if b.grade() == 1 {
                let idx = b.bitmap.trailing_zeros() as usize;
                if idx < 24 {
                    out[idx] += b.coeff;
                }
            }
        }
        out
    }
}

/// Project a 24D coordinate to 8D E8 under the canonical $E_8^3$ Leech embedding
/// (first 8 coordinates).
#[inline]
pub fn leech_to_e8_f32(coords24: &[f32; 24]) -> [f32; 8] {
    std::array::from_fn(|i| coords24[i])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entity::status::EpistemicStatus;
    use crate::relation::id::DurableRoleBinding;

    #[test]
    fn relation_members_map_to_explicit_basis_and_grade() {
        let basis = Cl24EntityBasis::new(vec![100, 200, 300, 400]).unwrap();
        let relation = DurableRelationInstance::new(
            9,
            77,
            1,
            vec![
                DurableRoleBinding {
                    entity_id: 300,
                    role_id: 3,
                },
                DurableRoleBinding {
                    entity_id: 100,
                    role_id: 1,
                },
                DurableRoleBinding {
                    entity_id: 200,
                    role_id: 2,
                },
            ],
            5,
            EpistemicStatus::Observed,
        );
        let blade = basis.blade_for_relation(&relation, 0.75).unwrap();
        assert_eq!(blade.bitmap, 0b0111);
        assert_eq!(blade.grade(), 3);
    }

    #[test]
    fn relation_role_multiplicity_is_not_silently_collapsed() {
        let basis = Cl24EntityBasis::new(vec![100]).unwrap();
        let relation = DurableRelationInstance::new(
            9,
            77,
            1,
            vec![
                DurableRoleBinding {
                    entity_id: 100,
                    role_id: 1,
                },
                DurableRoleBinding {
                    entity_id: 100,
                    role_id: 2,
                },
            ],
            5,
            EpistemicStatus::Observed,
        );
        assert_eq!(
            basis.blade_for_relation(&relation, 1.0),
            Err(Cl24BasisError::RepeatedRelationEntity {
                relation_id: 9,
                entity_id: 100,
            })
        );
    }

    #[test]
    fn geometric_product_retains_high_grade_hyperrelation_blades() {
        let grade_twelve =
            MultivectorCl24Sparse::from_blades(&[Cl24Blade::new((1u32 << 12) - 1, 1.0)]);
        let disjoint_grade_twelve =
            MultivectorCl24Sparse::from_blades(&[Cl24Blade::new(((1u32 << 12) - 1) << 12, 1.0)]);
        let product = grade_twelve.geometric_product(&disjoint_grade_twelve);
        assert_eq!(product.blades.len(), 1);
        assert_eq!(product.blades[0].grade(), 24);
        assert_eq!(product.blades[0].bitmap, (1u32 << 24) - 1);
    }
}
