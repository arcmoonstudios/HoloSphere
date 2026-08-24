/* holosphere/src/learning/inference/rune_evo/reasoning/mod.rs */
//!▫~•◦-------------------------------‣
//! # Cl(24) Operator Composition & Evidence-Bound Closure Reasoning
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Subsystem for composing verified multi-hop relation paths via sparse Cl(24)
//! geometric operator products with identity-grounded link continuity and explicit
//! derivation loss sidecars.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

pub mod blade;
pub mod closure;
pub mod composition;
pub mod operator;

pub use blade::{
    Cl24BasisError, Cl24Blade, Cl24EntityBasis, MultivectorCl24Sparse, blade_product_sign,
    leech_to_e8_f32,
};
pub use closure::{
    ClosureCandidate, ClosureKind, CompositionRule, CompositionRuleRegistry, CompositionSemantics,
    RUNE_CLOSURE_METHOD_ID, RUNE_CLOSURE_METHOD_VERSION, RuneClosureEvidenceV1, compile_closure,
};
pub use composition::{
    Cl24CompositionArtifact, DEFAULT_MAX_TRUNCATION_LOSS_RATIO, DEFAULT_TRUNCATION_TOPK,
    MAX_OPERATOR_CHAIN, RuneCl24CompositionConfig, execute_operator_chain,
};
pub use operator::{DurableEvidenceRef, ReasoningOperator, ReasoningOperatorId, RuneOperatorClass};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entity::status::EpistemicStatus;

    fn make_test_operator(
        id: u64,
        from_entity: u64,
        to_entity: u64,
        rel: u32,
        from_c: [f32; 8],
        to_c: [f32; 8],
        blade_bm: u32,
    ) -> ReasoningOperator {
        ReasoningOperator {
            operator_id: ReasoningOperatorId(id),
            from_entity,
            to_entity,
            relation_type: rel,
            from_coords: from_c,
            to_coords: to_c,
            transform: vec![Cl24Blade::new(blade_bm, 1.0)],
            evidence: Vec::new(),
            provenance_id: 1,
            reference_confidence: 0.95,
        }
    }

    #[test]
    fn test_cl24_blade_algebra_reference_identities_and_orthogonality() {
        // 1. Basis blade square e1 * e1 = 1
        let e1 = MultivectorCl24Sparse::from_grade1(&{
            let mut c = [0.0f32; 24];
            c[0] = 1.0;
            c
        });
        let prod_e1_e1 = e1.geometric_product(&e1);
        assert_eq!(prod_e1_e1.blades.len(), 1);
        assert_eq!(prod_e1_e1.blades[0].bitmap, 0); // scalar
        assert!((prod_e1_e1.blades[0].coeff - 1.0).abs() < 1e-6);

        // 2. Basis blade anticommutativity: e1 * e2 = - (e2 * e1)
        let e2 = MultivectorCl24Sparse::from_grade1(&{
            let mut c = [0.0f32; 24];
            c[1] = 1.0;
            c
        });
        let e12 = e1.geometric_product(&e2);
        let e21 = e2.geometric_product(&e1);
        assert_eq!(e12.blades[0].bitmap, 0b11);
        assert_eq!(e21.blades[0].bitmap, 0b11);
        assert!((e12.blades[0].coeff - 1.0).abs() < 1e-6);
        assert!((e21.blades[0].coeff - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_cl24_operator_chain_composition_and_truncation_energy() {
        let op1 = make_test_operator(
            1,
            100,
            200,
            10,
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            1,
        );
        let op2 = make_test_operator(
            2,
            200,
            300,
            10,
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            2,
        );

        let config = RuneCl24CompositionConfig::default();
        let artifact = execute_operator_chain(&[op1, op2], &config);

        assert_eq!(artifact.chain_depth, 2);
        assert_eq!(artifact.retained_blades.len(), 1);
        assert_eq!(artifact.retained_blades[0].bitmap, 3); // 1 ^ 2 = 3
        assert_eq!(artifact.max_truncation_loss_ratio, 0.0);
    }

    #[test]
    fn test_closure_compilation_identity_continuity_and_rule_resolution() {
        let op1 = make_test_operator(
            1,
            1001,
            1002,
            10,
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            1,
        );
        let op2 = make_test_operator(
            2,
            1002,
            1003,
            10,
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            2,
        );

        let mut registry = CompositionRuleRegistry::new();
        registry.register(CompositionRule {
            lhs: 10,    // PART_OF
            rhs: 10,    // PART_OF
            result: 10, // PART_OF
            semantics: CompositionSemantics::DeclaredExact,
        });

        let config = RuneCl24CompositionConfig::default();
        let closure =
            compile_closure(&[op1, op2], &registry, &config, 100).expect("compile closure");

        // Canonical semantic endpoint MUST be final operator's to_entity
        assert_eq!(closure.start_entity, 1001);
        assert_eq!(closure.semantic_endpoint, 1003);
        assert_eq!(closure.result_relation_type, Some(10));
        assert_eq!(closure.closure_kind, ClosureKind::ComposedReasoning);
        assert_eq!(closure.epistemic_status, EpistemicStatus::Provisional);
    }

    #[test]
    fn test_closure_compilation_rejects_identity_discontinuity() {
        // Discontinuous chain: op1 ends at 1002, but op2 starts at 9999
        let op1 = make_test_operator(
            1,
            1001,
            1002,
            10,
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            1,
        );
        let op2 = make_test_operator(
            2,
            9999,
            1003,
            10,
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            2,
        );

        let registry = CompositionRuleRegistry::new();
        let config = RuneCl24CompositionConfig::default();
        let res = compile_closure(&[op1, op2], &registry, &config, 100);

        assert!(res.is_err(), "Must reject identity discontinuity");
    }
}
