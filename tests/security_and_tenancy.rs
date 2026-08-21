/* hnsqr/tests/security_and_tenancy.rs */
//!▫~•◦-------------------------------‣
//! # Multi-Tenant Isolation & Authentication Test Suite
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Validates:
//!   - Tenant namespace qualification & parse isolation
//!   - Cross-tenant access and leakage prevention
//!   - API key registration, hashing, and RBAC authorization
//!   - Tenant resource quota enforcement
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use hnsqr::security::auth::{AccessRole, AuthRegistry};
use hnsqr::security::tenant::{TenantManager, TenantNamespace, TenantQuota};

#[test]
fn test_tenant_namespace_isolation() {
    let tenant_a = TenantNamespace::new("tenant_alpha", "collection_docs");
    let tenant_b = TenantNamespace::new("tenant_beta", "collection_docs");

    let qid_a = tenant_a.qualify_id("doc_100");
    let qid_b = tenant_b.qualify_id("doc_100");

    assert_eq!(qid_a, "tenant_alpha:collection_docs:doc_100");
    assert_eq!(qid_b, "tenant_beta:collection_docs:doc_100");

    // Tenant A cannot parse Tenant B's qualified ID
    assert_eq!(tenant_a.parse_qualified_id(&qid_a), Some("doc_100"));
    assert_eq!(
        tenant_a.parse_qualified_id(&qid_b),
        None,
        "Cross-tenant ID parsing must be rejected!"
    );
}

#[test]
fn test_auth_registry_rbac_enforcement() {
    let auth = AuthRegistry::new();

    // Register API keys with different roles
    let _key_read = auth.register_key(
        "secret_token_readonly",
        "tenant_alpha",
        AccessRole::ReadOnly,
        100,
    );
    let _key_write = auth.register_key(
        "secret_token_readwrite",
        "tenant_alpha",
        AccessRole::ReadWrite,
        500,
    );
    let _key_admin = auth.register_key(
        "secret_token_admin",
        "tenant_alpha",
        AccessRole::Admin,
        1000,
    );

    // Read-only token can perform ReadOnly action
    assert!(
        auth.authenticate("secret_token_readonly", AccessRole::ReadOnly)
            .is_ok()
    );

    // Read-only token is rejected for ReadWrite action
    let write_attempt = auth.authenticate("secret_token_readonly", AccessRole::ReadWrite);
    assert!(
        write_attempt.is_err(),
        "ReadOnly token must not perform mutations!"
    );

    // ReadWrite token can perform ReadWrite and ReadOnly
    assert!(
        auth.authenticate("secret_token_readwrite", AccessRole::ReadOnly)
            .is_ok()
    );
    assert!(
        auth.authenticate("secret_token_readwrite", AccessRole::ReadWrite)
            .is_ok()
    );
    assert!(
        auth.authenticate("secret_token_readwrite", AccessRole::Admin)
            .is_err()
    );

    // Admin token can perform Admin
    assert!(
        auth.authenticate("secret_token_admin", AccessRole::Admin)
            .is_ok()
    );

    // Unknown token is rejected
    assert!(
        auth.authenticate("invalid_token", AccessRole::ReadOnly)
            .is_err()
    );
}

#[test]
fn test_tenant_quota_admission_checks() {
    let mgr = TenantManager::new();
    mgr.register_tenant(
        "tenant_omega",
        TenantQuota {
            max_vectors: 100,
            max_memory_bytes: 1024 * 1024,
            max_qps: 100,
        },
    );

    // Insert 50 vectors (admitted)
    assert!(mgr.check_admission("tenant_omega", 50, 1000).is_ok());

    // Insert 50 more vectors (admitted, at 100)
    assert!(mgr.check_admission("tenant_omega", 50, 1000).is_ok());

    // Insert 1 more vector (quota exceeded)
    let overflow = mgr.check_admission("tenant_omega", 1, 100);
    assert!(overflow.is_err(), "Tenant quota must reject overflow!");
}
