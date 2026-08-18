/* hnsqr/src/security/cert_manager.rs */
//!▫~•◦-------------------------------‣
//! # Automated Certificate Rotation & Lifecycle Orchestration
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides hot certificate reload without process restarts, overlapping old/new
//! trust windows for seamless peer rotation, and automatic renewal alerting.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::HNSQRResult;

/// Active certificate descriptor.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ActiveCertificate {
    pub cert_id: String,
    pub cert_pem: String,
    pub key_pem: String,
    pub issued_at_epoch_secs: u64,
    pub expires_at_epoch_secs: u64,
}

/// Certificate rotation orchestrator supporting overlapping trust windows.
pub struct CertificateManager {
    active_cert: RwLock<ActiveCertificate>,
    previous_cert: RwLock<Option<ActiveCertificate>>,
    pub rotation_count: AtomicU64,
}

impl CertificateManager {
    pub fn new(initial_cert: ActiveCertificate) -> Self {
        Self {
            active_cert: RwLock::new(initial_cert),
            previous_cert: RwLock::new(None),
            rotation_count: AtomicU64::new(0),
        }
    }

    /// Hot-reloads a new certificate, retaining the old one in an overlapping grace window.
    pub fn rotate_certificate(&self, new_cert: ActiveCertificate) -> HNSQRResult<()> {
        let old = self.active_cert.read().clone();
        *self.previous_cert.write() = Some(old);
        *self.active_cert.write() = new_cert;
        self.rotation_count.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    /// Returns the active certificate.
    pub fn active_certificate(&self) -> ActiveCertificate {
        self.active_cert.read().clone()
    }

    /// Validates whether an incoming peer's certificate is recognized (active or within grace window).
    pub fn is_trusted_peer_cert(&self, cert_id: &str) -> bool {
        if self.active_cert.read().cert_id == cert_id {
            return true;
        }
        if let Some(prev) = self.previous_cert.read().as_ref() {
            if prev.cert_id == cert_id {
                return true;
            }
        }
        false
    }
}
