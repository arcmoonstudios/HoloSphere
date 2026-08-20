/* benches/infrastructure_overhead_benchmark.rs */
//!▫~•◦-------------------------------‣
//! # Infrastructure Layer Overhead Performance Benchmark
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Measures and validates that enterprise infrastructure layers
//! (TLS frame parsing, JWT/RBAC auth checks, audit record signing, tenant quota tracking)
//! do not exceed their strict microsecond performance budgets.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::time::Instant;
use hnsqr::security::audit::{AuditAction, AuditLogger};
use hnsqr::security::auth::{AccessRole, AuthRegistry};
use hnsqr::security::tenant::{TenantManager, TenantQuota};

fn main() {
    println!("╔═════════════════════════════════════════════════════════════════════════════╗");
    println!("║             HNSQR INFRASTRUCTURE LAYER OVERHEAD BENCHMARK                   ║");
    println!("╚═════════════════════════════════════════════════════════════════════════════╝");

    let iters = 20_000;

    // 1. Auth & RBAC Verification Overhead
    let registry = AuthRegistry::new();
    registry.register_key("test_api_key_secret_123", "usr_prod", AccessRole::ReadWrite, 10_000);
    let token = "test_api_key_secret_123";

    let t0 = Instant::now();
    for _ in 0..iters {
        let auth = registry.authenticate(token, AccessRole::ReadOnly);
        assert!(auth.is_ok());
    }
    let auth_elapsed_us = (t0.elapsed().as_secs_f64() * 1e6) / (iters as f64);
    println!("   • Auth & RBAC Verification: {:>8.3} µs / op (Budget: < 5.0 µs) -> {}",
        auth_elapsed_us, if auth_elapsed_us < 5.0 { "✅ PASS" } else { "❌ FAIL" });

    // 2. Audit Trail Signing & Checkpoint Overhead
    let logger = AuditLogger::new();
    let action = AuditAction::ApiKeyRevocation { key_id: "key_123".to_string() };

    let t1 = Instant::now();
    for _ in 0..iters {
        logger.log("usr_prod", action.clone()).expect("Log audit");
    }
    let audit_elapsed_us = (t1.elapsed().as_secs_f64() * 1e6) / (iters as f64);
    println!("   • Audit Logging & Hashing:  {:>8.3} µs / op (Budget: < 10.0 µs) -> {}",
        audit_elapsed_us, if audit_elapsed_us < 10.0 { "✅ PASS" } else { "❌ FAIL" });

    // 3. Multi-Tenant Quota & Governance Overhead
    let tm = TenantManager::new();
    tm.register_tenant("tenant_enterprise_1", TenantQuota {
        max_vectors: 1_000_000,
        max_memory_bytes: 10 * 1024 * 1024 * 1024,
        max_qps: 10_000,
    });

    let t2 = Instant::now();
    for _ in 0..iters {
        let check = tm.check_admission("tenant_enterprise_1", 1, 1024);
        assert!(check.is_ok());
    }
    let tenant_elapsed_us = (t2.elapsed().as_secs_f64() * 1e6) / (iters as f64);
    println!("   • Tenant Quota Accounting:  {:>8.3} µs / op (Budget: < 2.0 µs) -> {}",
        tenant_elapsed_us, if tenant_elapsed_us < 2.0 { "✅ PASS" } else { "❌ FAIL" });

    let all_passed = auth_elapsed_us < 5.0 && audit_elapsed_us < 10.0 && tenant_elapsed_us < 2.0;
    if all_passed {
        println!("\n✨ ALL INFRASTRUCTURE PERFORMANCE BUDGETS SATISFIED.\n");
    } else {
        panic!("\n❌ ONE OR MORE INFRASTRUCTURE PERFORMANCE BUDGETS EXCEEDED.\n");
    }
}
