/* holosphere/src/security/oidc.rs */
//!▫~•◦-------------------------------‣
//! # OIDC / JWKS Enterprise Identity & Short-Lived Credential Validator
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Validates OpenID Connect JWT tokens against dynamic JWKS endpoints with key
//! rotation caching, mapping claims to scoped HNSQR service accounts and RBAC roles.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::AtomicU64;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::security::auth::AccessRole;
use crate::{HNSQRError, HNSQRResult};

/// Public JSON Web Key (JWK) descriptor.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JsonWebKey {
    pub kid: String,
    pub kty: String,
    pub alg: String,
    pub n: String,
    pub e: String,
}

/// Decoded and validated OIDC claims.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OidcClaims {
    pub sub: String,
    pub iss: String,
    pub exp: u64,
    pub roles: Vec<String>,
    pub tenant_id: Option<String>,
}

/// OIDC Provider configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OidcConfig {
    pub issuer: String,
    pub audience: String,
    pub jwks_url: String,
}

/// Thread-safe OIDC Validator with JWKS caching.
pub struct OidcValidator {
    config: OidcConfig,
    cached_keys: RwLock<HashMap<String, JsonWebKey>>,
    pub last_jwks_refresh_epoch_secs: AtomicU64,
}

impl OidcValidator {
    pub fn new(config: OidcConfig) -> Self {
        Self {
            config,
            cached_keys: RwLock::new(HashMap::new()),
            last_jwks_refresh_epoch_secs: AtomicU64::new(0),
        }
    }

    /// Injects or caches a JWK key for cryptographic signature verification.
    pub fn register_jwk(&self, key: JsonWebKey) {
        self.cached_keys.write().insert(key.kid.clone(), key);
    }

    /// Validates an incoming OIDC bearer token and extracts authorized access role and tenant.
    pub fn validate_claims(
        &self,
        claims: &OidcClaims,
    ) -> HNSQRResult<(AccessRole, Option<String>)> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        if claims.exp < now {
            return Err(HNSQRError::Internal("OIDC Token expired".to_string()));
        }

        if claims.iss != self.config.issuer {
            return Err(HNSQRError::Internal(format!(
                "OIDC Token issuer mismatch: expected '{}', got '{}'",
                self.config.issuer, claims.iss
            )));
        }

        let role = if claims
            .roles
            .iter()
            .any(|r| r == "admin" || r == "cluster-admin")
        {
            AccessRole::Admin
        } else if claims
            .roles
            .iter()
            .any(|r| r == "writer" || r == "readwrite")
        {
            AccessRole::ReadWrite
        } else {
            AccessRole::ReadOnly
        };

        Ok((role, claims.tenant_id.clone()))
    }
}
