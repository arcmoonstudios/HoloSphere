/* holosphere/tests/vector_primitives_invariants.rs */
//!▫~•◦-------------------------------‣
//! # Vector Primitives & Geometric Invariants Test Suite
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Validates the mathematical and systems invariants of the 4 generalized vector primitives:
//!   1. `VectorEmbedding::dot_product_real` and `cosine_similarity_real`
//!   2. `ComplexSliceCast` (Zero-copy bidirectional slice reinterpretation)
//!   3. `RotaryPhaseTransformer` (RoPE, Unitary Isometry & Relative Shift Equivariance)
//!   4. `CircularAngularMetric` (Polar Projection & Periodic $S^1$ Metric)
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::f32::consts::PI;
use num_complex::Complex32;

use hnsqr::vector::{CircularAngularMetric, ComplexSliceCast, RotaryPhaseTransformer};
use hnsqr::VectorEmbedding;

// ════════════════════════════════════════════════════════════════════════════════
// 1. REAL SIMD INNER PRODUCT & COSINE INVARIANTS
// ════════════════════════════════════════════════════════════════════════════════

#[test]
fn test_real_dot_product_exact_equivalence_and_symmetry() {
    let dim = 128;
    let u = VectorEmbedding::from_complex(
        (0..dim)
            .map(|i| Complex32::new((i * 7 + 3) as f32, (i * 11 + 5) as f32))
            .collect(),
    )
    .into_normalized();

    let v = VectorEmbedding::from_complex(
        (0..dim)
            .map(|i| Complex32::new((i * 13 + 1) as f32, (i * 17 + 9) as f32))
            .collect(),
    )
    .into_normalized();

    // Invariant 1: dot_product_real(u, v) == Re(<u, v>_C)
    let complex_ip = u.dot_product_complex(&v);
    let real_ip = u.dot_product_real(&v);
    assert!(
        (real_ip - complex_ip.re).abs() < 1e-5,
        "Real dot product must equal real part of complex dot product! got {real_ip}, expected {}",
        complex_ip.re
    );

    // Invariant 2: Symmetry <u, v>_R == <v, u>_R
    let real_ip_rev = v.dot_product_real(&u);
    assert!(
        (real_ip - real_ip_rev).abs() < 1e-5,
        "Real dot product must be symmetric! <u, v>={real_ip}, <v, u>={real_ip_rev}"
    );

    // Invariant 3: Self-cosine is exactly 1.0 for normalized vectors
    let self_cosine = u.cosine_similarity_real(&u);
    assert!(
        (self_cosine - 1.0).abs() < 1e-5,
        "Self cosine similarity must be 1.0, got {self_cosine}"
    );

    // Invariant 4: Cosine is bounded in [-1.0, 1.0]
    let cos_uv = u.cosine_similarity_real(&v);
    assert!((-1.0..=1.0).contains(&cos_uv));
}

#[test]
fn test_orthogonal_and_negated_vectors() {
    let dim = 16;
    let u = VectorEmbedding::from_complex(
        (0..dim)
            .map(|i| Complex32::new((i + 1) as f32, 0.0))
            .collect(),
    )
    .into_normalized();

    let neg_u = VectorEmbedding::from_complex(
        u.complex_data()
            .iter()
            .map(|z| Complex32::new(-z.re, -z.im))
            .collect(),
    );

    let cos_neg = u.cosine_similarity_real(&neg_u);
    assert!(
        (cos_neg - (-1.0)).abs() < 1e-5,
        "Negated vector cosine must be -1.0, got {cos_neg}"
    );
}

// ════════════════════════════════════════════════════════════════════════════════
// 2. ZERO-COPY SLICE REINTERPRETATION INVARIANTS (ComplexSliceCast)
// ════════════════════════════════════════════════════════════════════════════════

#[test]
fn test_complex_slice_cast_bidirectional_roundtrip() {
    let mut complex_vec = vec![
        Complex32::new(1.0, 2.0),
        Complex32::new(3.0, 4.0),
        Complex32::new(5.0, 6.0),
        Complex32::new(7.0, 8.0),
    ];

    // Invariant 1: Length is exactly doubled
    let real_view = ComplexSliceCast::as_real_slice(&complex_vec);
    assert_eq!(real_view.len(), 8);
    assert_eq!(real_view, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

    // Invariant 2: In-place mutation through real view reflects in complex representation
    let real_mut = ComplexSliceCast::as_real_slice_mut(&mut complex_vec);
    real_mut[0] = 42.0;
    real_mut[7] = 99.0;
    assert_eq!(complex_vec[0], Complex32::new(42.0, 2.0));
    assert_eq!(complex_vec[3], Complex32::new(7.0, 99.0));

    // Invariant 3: Casting even-length real slice to complex slice
    let real_data = [10.0, 20.0, 30.0, 40.0];
    let complex_view = ComplexSliceCast::try_as_complex_slice(&real_data).unwrap();
    assert_eq!(complex_view.len(), 2);
    assert_eq!(complex_view[0], Complex32::new(10.0, 20.0));
    assert_eq!(complex_view[1], Complex32::new(30.0, 40.0));

    // Invariant 4: Odd-length slice returns error
    let odd_data = [1.0, 2.0, 3.0];
    assert!(ComplexSliceCast::try_as_complex_slice(&odd_data).is_err());
}

#[test]
fn test_vector_embedding_slice_view_methods() {
    let mut v = VectorEmbedding::from_complex(vec![
        Complex32::new(1.5, -2.5),
        Complex32::new(3.5, -4.5),
    ]);

    assert_eq!(v.as_real_slice(), &[1.5, -2.5, 3.5, -4.5]);

    v.as_real_slice_mut()[1] = 100.0;
    assert_eq!(v.complex_data()[0], Complex32::new(1.5, 100.0));
}

// ════════════════════════════════════════════════════════════════════════════════
// 3. ROTARY PHASE TRANSFORMER INVARIANTS (RoPE)
// ════════════════════════════════════════════════════════════════════════════════

#[test]
fn test_rotary_transformer_unitary_norm_invariance() {
    let dim = 64;
    let rope = RotaryPhaseTransformer::default_for_dim(dim);

    let original = VectorEmbedding::from_complex(
        (0..dim)
            .map(|i| Complex32::new((i * 3 + 1) as f32, (i * 5 + 2) as f32))
            .collect(),
    )
    .into_normalized();

    let norm_sq_0 = original.norm_squared();
    assert!((norm_sq_0 - 1.0).abs() < 1e-5);

    // Invariant 1: Rotation by any position preserves Euclidean norm exactly
    for pos in [0, 1, 2, 7, 31, 100, 2048] {
        let rotated = rope.apply_rotation(&original, pos);
        let norm_sq_p = rotated.norm_squared();
        assert!(
            (norm_sq_p - norm_sq_0).abs() < 1e-5,
            "Norm must be strictly preserved at pos={pos}: initial={norm_sq_0}, rotated={norm_sq_p}"
        );
    }
}

#[test]
fn test_rotary_transformer_additivity_and_relative_shift() {
    let dim = 32;
    let rope = RotaryPhaseTransformer::default_for_dim(dim);

    let u = VectorEmbedding::from_complex(
        (0..dim)
            .map(|i| Complex32::new((i + 1) as f32, (i * 2 + 1) as f32))
            .collect(),
    )
    .into_normalized();

    let v = VectorEmbedding::from_complex(
        (0..dim)
            .map(|i| Complex32::new((i * 4 + 3) as f32, (i + 5) as f32))
            .collect(),
    )
    .into_normalized();

    // Invariant 2: Additive composition R_p1(R_p2(u)) == R_(p1+p2)(u)
    let p1 = 13;
    let p2 = 29;
    let u_p2 = rope.apply_rotation(&u, p2);
    let u_p1_p2 = rope.apply_rotation(&u_p2, p1);
    let u_combined = rope.apply_rotation(&u, p1 + p2);

    for (a, b) in u_p1_p2.complex_data().iter().zip(u_combined.complex_data()) {
        assert!((a.re - b.re).abs() < 1e-5);
        assert!((a.im - b.im).abs() < 1e-5);
    }

    // Invariant 3: Relative shift equivariance <R_(p+k)(u), R_p(v)> == <R_k(u), v>
    let p = 42;
    let k = 15;
    let u_pk = rope.apply_rotation(&u, p + k);
    let v_p = rope.apply_rotation(&v, p);
    let dot_shifted = u_pk.dot_product_real(&v_p);

    let u_k = rope.apply_rotation(&u, k);
    let dot_relative = u_k.dot_product_real(&v);

    assert!(
        (dot_shifted - dot_relative).abs() < 1e-5,
        "Relative shift property violated! shifted={dot_shifted}, relative={dot_relative}"
    );
}

// ════════════════════════════════════════════════════════════════════════════════
// 4. CIRCULAR ANGULAR METRIC INVARIANTS (S^1 Periodic Geometry)
// ════════════════════════════════════════════════════════════════════════════════

#[test]
fn test_circular_metric_axioms_and_triangle_inequality() {
    let angles = [-3.0, -1.5, 0.0, 0.7, 2.5, 3.14];

    for &a in &angles {
        // Identity: d(a, a) == 0
        assert_eq!(CircularAngularMetric::angular_distance(a, a), 0.0);

        for &b in &angles {
            let d_ab = CircularAngularMetric::angular_distance(a, b);
            let d_ba = CircularAngularMetric::angular_distance(b, a);

            // Non-negativity and range in [0, PI]
            assert!((0.0..=PI + 1e-5).contains(&d_ab));

            // Symmetry: d(a, b) == d(b, a)
            assert!((d_ab - d_ba).abs() < 1e-6);

            for &c in &angles {
                let d_bc = CircularAngularMetric::angular_distance(b, c);
                let d_ac = CircularAngularMetric::angular_distance(a, c);

                // Triangle inequality: d(a, c) <= d(a, b) + d(b, c)
                assert!(
                    d_ac <= d_ab + d_bc + 1e-5,
                    "Triangle inequality violated: d(a,c)={d_ac} > d(a,b)={d_ab} + d(b,c)={d_bc}"
                );
            }
        }
    }
}

#[test]
fn test_circular_metric_branch_cut_continuity() {
    // Distance across branch cut (+PI - eps) and (-PI + eps) must be 2*eps
    let eps = 0.0123f32;
    let theta1 = PI - eps;
    let theta2 = -PI + eps;
    let dist = CircularAngularMetric::angular_distance(theta1, theta2);
    assert!(
        (dist - (2.0 * eps)).abs() < 1e-5,
        "Wrap-around metric failed across branch cut: expected {}, got {dist}",
        2.0 * eps
    );
}
