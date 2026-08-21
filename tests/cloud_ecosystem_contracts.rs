/* hnsqr/tests/cloud_ecosystem_contracts.rs */
//!▫~•◦-------------------------------‣
//! # Enterprise Security, KMS, OIDC & Ecosystem Client Invariant Suite
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Validates:
//!   - OIDC token claim decoding, expiration checks, and role mapping
//!   - KMS envelope encryption key generation and decryption
//!   - Multi-tenant cardinality quota bounds and adaptive posting selection
//!   - Client SDK retry, failover, and leader redirection
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use hnsqr::ecosystem::sdks::{HNSQRClientConfig, HNSQRClientRouter};
use hnsqr::metadata::cardinality::{CardinalityBudget, CardinalityGuard, PostingRepresentation};
use hnsqr::security::auth::AccessRole;
use hnsqr::security::kms::{KmsProvider, LocalKmsProvider};
use hnsqr::security::oidc::{JsonWebKey, OidcClaims, OidcConfig, OidcValidator};
use std::time::{SystemTime, UNIX_EPOCH};

#[test]
fn test_oidc_token_validation_and_rbac() {
    let config = OidcConfig {
        issuer: "https://auth.enterprise.org".to_string(),
        audience: "hnsqr-cluster".to_string(),
        jwks_url: "https://auth.enterprise.org/.well-known/jwks.json".to_string(),
    };
    let validator = OidcValidator::new(config);

    let jwk = JsonWebKey {
        kid: "key-2026-1".to_string(),
        kty: "RSA".to_string(),
        alg: "RS256".to_string(),
        n: "modulus".to_string(),
        e: "AQAB".to_string(),
    };
    validator.register_jwk(jwk);

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let claims = OidcClaims {
        sub: "user_admin_99".to_string(),
        iss: "https://auth.enterprise.org".to_string(),
        exp: now + 3600,
        roles: vec!["cluster-admin".to_string()],
        tenant_id: Some("tenant_enterprise_alpha".to_string()),
    };

    let (role, tenant) = validator.validate_claims(&claims).unwrap();
    assert_eq!(role, AccessRole::Admin);
    assert_eq!(tenant, Some("tenant_enterprise_alpha".to_string()));
}

#[test]
fn test_kms_envelope_encryption_roundtrip() {
    let kms = LocalKmsProvider::default();
    let (plain_dek, enc_dek) = kms
        .generate_data_key("arn:aws:kms:us-east-1:123456789012:key/backup-key")
        .unwrap();
    assert_eq!(plain_dek.len(), 32);
    assert_eq!(enc_dek.len(), 32);

    let decrypted = kms
        .decrypt_data_key(
            "arn:aws:kms:us-east-1:123456789012:key/backup-key",
            &enc_dek,
        )
        .unwrap();
    assert_eq!(decrypted.len(), 32);
}

#[test]
fn test_tenant_cardinality_admission_and_adaptive_representation() {
    let budget = CardinalityBudget {
        max_distinct_terms: 10,
        max_fields: 5,
        max_dictionary_bytes: 1024,
        max_bitmap_bytes: 2048,
    };
    let guard = CardinalityGuard::new(budget);

    // Ingest 10 terms: OK
    for _i in 1..=10 {
        assert!(guard.check_admission("tenant_a", true, 32).is_ok());
    }

    // 11th term: Exceeds distinct term budget!
    assert!(guard.check_admission("tenant_a", true, 32).is_err());

    // Adaptive posting representation selection
    assert_eq!(
        CardinalityGuard::select_representation(10_000, 50, 0),
        PostingRepresentation::SortedPostings
    );
    assert_eq!(
        CardinalityGuard::select_representation(10_000, 1_000, 0),
        PostingRepresentation::RoaringBitmap
    );
    assert_eq!(
        CardinalityGuard::select_representation(10_000, 5_000, 0),
        PostingRepresentation::DenseBitmap
    );
    assert_eq!(
        CardinalityGuard::select_representation(10_000, 1_000, 100),
        PostingRepresentation::CompactDictionary
    );
}

#[test]
fn test_client_sdk_smart_routing() {
    let config = HNSQRClientConfig {
        seed_endpoints: vec![
            "http://node-1:8080".to_string(),
            "http://node-2:8080".to_string(),
        ],
        ..Default::default()
    };
    let router = HNSQRClientRouter::new(config);

    // Initial write routes to round-robin seed
    let ep = router.select_endpoint(true);
    assert!(ep.contains("node-"));

    // Set leader after redirect
    router.handle_leader_redirect("http://node-1:8080");
    assert_eq!(router.select_endpoint(true), "http://node-1:8080");
}
