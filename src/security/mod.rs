/* holosphere/src/security/mod.rs */
//!▫~•◦-------------------------------‣
//! # HNSQR Security, Multi-Tenancy & Authorization Subsystem
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides strict multi-tenant namespace isolation, API Key / JWT authentication,
//! Role-Based Access Control (RBAC), and quota governance.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

pub mod audit;
pub mod auth;
pub mod cert_manager;
pub mod compliance;
pub mod fuzzing;
pub mod kms;
pub mod oidc;
pub mod siem;
pub mod tenant;
pub mod tls;

pub use audit::{AuditAction, AuditLogger, AuditRecord};
pub use auth::{AccessRole, AuthCredential, AuthRegistry, AuthenticatedSubject};
pub use cert_manager::{ActiveCertificate, CertificateManager};
pub use compliance::{ComplianceEvidenceGenerator, SecurityReportDocument};
pub use fuzzing::{ProtocolFuzzSummary, ProtocolFuzzer};
pub use kms::{KmsProvider, LocalKmsProvider};
pub use oidc::{JsonWebKey, OidcClaims, OidcConfig, OidcValidator};
pub use siem::{SiemExporter, SiemFormat};
pub use tenant::{TenantContext, TenantManager, TenantNamespace, TenantQuota};
pub use tls::{DEFAULT_MAX_FRAME_BYTES, TlsConfig};
