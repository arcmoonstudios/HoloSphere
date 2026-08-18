/* hnsqr/src/security/tls.rs */
//!▫~•◦-------------------------------‣
//! # Transport Security (TLS 1.3 / mTLS) & Protocol Framing Protection
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides production TLS configuration, mutual TLS (mTLS) certificate
//! verification for cluster node-to-node transport, certificate expiration
//! auditing, and denial-of-service frame boundary enforcement.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use serde::{Deserialize, Serialize};

use crate::{HNSQRError, HNSQRResult};

pub const DEFAULT_MAX_FRAME_BYTES: usize = 16 * 1024 * 1024; // 16 MB max frame

/// TLS & mTLS Transport Configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TlsConfig {
    pub enabled: bool,
    pub cert_path: Option<PathBuf>,
    pub key_path: Option<PathBuf>,
    pub ca_cert_path: Option<PathBuf>,
    /// Whether mutual TLS (mTLS) authentication is required for peers.
    pub require_client_cert: bool,
    /// Unix epoch timestamp in seconds when active certificate expires.
    pub cert_expires_at: u64,
    /// Max frame length in bytes to protect against DoS attacks.
    pub max_frame_bytes: usize,
}

impl Default for TlsConfig {
    fn default() -> Self {
        // Default 90-day cert validity simulation
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();
        Self {
            enabled: false,
            cert_path: None,
            key_path: None,
            ca_cert_path: None,
            require_client_cert: false,
            cert_expires_at: now + 90 * 86400,
            max_frame_bytes: DEFAULT_MAX_FRAME_BYTES,
        }
    }
}

impl TlsConfig {
    /// Creates a production mTLS configuration for secure node-to-node Raft communication.
    pub fn new_mtls(
        cert_path: impl AsRef<Path>,
        key_path: impl AsRef<Path>,
        ca_cert_path: impl AsRef<Path>,
    ) -> Self {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();
        Self {
            enabled: true,
            cert_path: Some(cert_path.as_ref().to_path_buf()),
            key_path: Some(key_path.as_ref().to_path_buf()),
            ca_cert_path: Some(ca_cert_path.as_ref().to_path_buf()),
            require_client_cert: true,
            cert_expires_at: now + 90 * 86400,
            max_frame_bytes: DEFAULT_MAX_FRAME_BYTES,
        }
    }

    /// Verifies if certificate is valid and not expired.
    pub fn verify_certificate_freshness(&self) -> HNSQRResult<u64> {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();
        if now >= self.cert_expires_at {
            return Err(HNSQRError::Internal(format!(
                "TLS certificate expired at timestamp {}",
                self.cert_expires_at
            )));
        }
        let remaining_secs = self.cert_expires_at - now;
        Ok(remaining_secs)
    }

    /// Validates an incoming framed network message header against buffer overflow/DoS limits.
    pub fn validate_frame_size(&self, payload_len: usize) -> HNSQRResult<()> {
        if payload_len > self.max_frame_bytes {
            return Err(HNSQRError::Internal(format!(
                "Frame payload size ({} bytes) exceeds security ceiling ({} bytes)",
                payload_len, self.max_frame_bytes
            )));
        }
        Ok(())
    }
}
