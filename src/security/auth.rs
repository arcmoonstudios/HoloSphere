/* hnsqr/src/security/auth.rs */
//!▫~•◦-------------------------------‣
//! # Authentication & Role-Based Access Control (RBAC)
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Enforces identity verification, API key lookups, and role-based permissions
//! (Admin, ReadWrite, ReadOnly) across all transport boundaries.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::collections::HashMap;

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use sha2::Digest;

use crate::{HNSQRError, HNSQRResult};

/// Access permission levels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AccessRole {
    /// Read-only search and health diagnostics.
    ReadOnly = 1,
    /// Read-only search + mutations (insert, delete, compact).
    ReadWrite = 2,
    /// Full administrative controls (tenant creation, quota modification, backup/restore).
    Admin = 3,
}

/// An authenticated caller identity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AuthenticatedSubject {
    pub tenant_id: String,
    pub role: AccessRole,
    pub key_id: String,
}

/// Stored credential profile.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AuthCredential {
    pub key_id: String,
    pub hashed_token: String,
    pub tenant_id: String,
    pub role: AccessRole,
    pub rate_limit_qps: u32,
    pub active: bool,
}

/// Thread-safe registry for API Keys and authentication tokens.
#[derive(Default)]
pub struct AuthRegistry {
    credentials: RwLock<HashMap<String, AuthCredential>>,
}

impl AuthRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a new API key credential.
    pub fn register_key(
        &self,
        raw_token: &str,
        tenant_id: &str,
        role: AccessRole,
        rate_limit_qps: u32,
    ) -> String {
        let key_id = format!("key_{:08x}", crc32fast::hash(raw_token.as_bytes()));
        let mut hasher = sha2::Sha256::new();
        sha2::Digest::update(&mut hasher, raw_token.as_bytes());
        let hashed_token = sha2::Digest::finalize(hasher)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>();

        let cred = AuthCredential {
            key_id: key_id.clone(),
            hashed_token,
            tenant_id: tenant_id.to_string(),
            role,
            rate_limit_qps,
            active: true,
        };

        self.credentials.write().insert(raw_token.to_string(), cred);
        key_id
    }

    /// Authenticates a raw token and checks required role.
    pub fn authenticate(
        &self,
        raw_token: &str,
        required_role: AccessRole,
    ) -> HNSQRResult<AuthenticatedSubject> {
        let cred_guard = self.credentials.read();
        let cred = cred_guard
            .get(raw_token)
            .ok_or_else(|| HNSQRError::Internal("Invalid or missing API key".to_string()))?;

        if !cred.active {
            return Err(HNSQRError::Internal("API key has been revoked".to_string()));
        }

        if cred.role < required_role {
            return Err(HNSQRError::Internal(format!(
                "Permission denied: {:?} required, but key has {:?}",
                required_role, cred.role
            )));
        }

        Ok(AuthenticatedSubject {
            tenant_id: cred.tenant_id.clone(),
            role: cred.role,
            key_id: cred.key_id.clone(),
        })
    }
}
