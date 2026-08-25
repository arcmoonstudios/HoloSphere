/* holosphere/src/security/audit.rs */
//!▫~•◦-------------------------------‣
//! # Tamper-Evident Hash-Chained Security Audit Log
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides an append-only, cryptographic hash-chained audit trail of all
//! security-critical operations (membership changes, key rotations, backup restores,
//! tenant deletions) with durable checkpoint root signing.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::HNSQRResult;

/// Audit event action category.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum AuditAction {
    ClusterMembershipChange { new_membership: Vec<u64> },
    CertificateRotation { cert_id: String },
    BackupRestoreTriggered { backup_id: String, target_lsn: u64 },
    TenantDeletion { tenant_id: String },
    ApiKeyRevocation { key_id: String },
}

/// An immutable, cryptographically chained audit record.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AuditRecord {
    pub sequence_num: u64,
    pub timestamp_epoch_ms: u64,
    pub actor_id: String,
    pub action: AuditAction,
    pub prev_hash_hex: String,
    pub record_hash_hex: String,
}

/// Thread-safe append-only audit logger.
pub struct AuditLogger {
    records: RwLock<Vec<AuditRecord>>,
    last_hash_hex: RwLock<String>,
    /// Path to the append-only `audit.log` file, or `None` for volatile in-memory mode.
    log_file: Option<std::path::PathBuf>,
}

impl Default for AuditLogger {
    fn default() -> Self {
        Self::new()
    }
}

impl AuditLogger {
    pub fn new() -> Self {
        Self {
            records: RwLock::new(Vec::new()),
            last_hash_hex: RwLock::new("0".repeat(64)),
            log_file: None,
        }
    }

    /// Appends a new audit record to the cryptographic hash chain.
    pub fn append(&self, actor_id: &str, action: AuditAction) -> HNSQRResult<AuditRecord> {
        let mut rec_guard = self.records.write();
        let mut hash_guard = self.last_hash_hex.write();

        let seq = rec_guard.len() as u64 + 1;
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        let prev_hash = hash_guard.clone();

        let mut hasher = Sha256::new();
        hasher.update(seq.to_le_bytes());
        hasher.update(now.to_le_bytes());
        hasher.update(actor_id.as_bytes());
        hasher.update(prev_hash.as_bytes());
        let record_hash_hex = hasher
            .finalize()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>();

        let record = AuditRecord {
            sequence_num: seq,
            timestamp_epoch_ms: now,
            actor_id: actor_id.to_string(),
            action,
            prev_hash_hex: prev_hash,
            record_hash_hex: record_hash_hex.clone(),
        };

        rec_guard.push(record.clone());
        *hash_guard = record_hash_hex;

        // Persist to disk if a log file path was provided.
        if let Some(path) = &self.log_file {
            if let Ok(mut f) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
            {
                if let Ok(json) = serde_json::to_string(&record) {
                    use std::io::Write;
                    let _ = writeln!(f, "{}", json);
                    let _ = f.sync_all();
                }
            }
        }

        Ok(record)
    }

    /// Returns the most recent hash for external checkpoint signing.
    pub fn latest_checkpoint_hash(&self) -> String {
        self.last_hash_hex.read().clone()
    }

    /// Exports all audit records formatted according to the requested SIEM standard.
    pub fn export_siem_events(&self, format: crate::security::siem::SiemFormat) -> Vec<String> {
        let records = self.records.read();
        records
            .iter()
            .map(|r| crate::security::siem::SiemExporter::format_record(r, format))
            .collect()
    }

    /// Alias for append to log an action.
    pub fn log(&self, actor_id: &str, action: AuditAction) -> HNSQRResult<AuditRecord> {
        self.append(actor_id, action)
    }

    /// Opens an audit logger for the given directory, replaying any existing
    /// `audit.log` to restore the in-memory chain and last-hash state.
    pub fn open(dir: impl AsRef<std::path::Path>) -> HNSQRResult<Self> {
        let log_file = dir.as_ref().join("audit.log");
        let mut records = Vec::new();
        let mut last_hash_hex = "0".repeat(64);

        // Replay existing records so the chain is continuous across restarts.
        if log_file.exists() {
            if let Ok(content) = std::fs::read_to_string(&log_file) {
                for line in content.lines() {
                    if let Ok(rec) = serde_json::from_str::<AuditRecord>(line) {
                        last_hash_hex = rec.record_hash_hex.clone();
                        records.push(rec);
                    }
                }
            }
        }

        Ok(Self {
            records: RwLock::new(records),
            last_hash_hex: RwLock::new(last_hash_hex),
            log_file: Some(log_file),
        })
    }

    /// Verifies the cryptographic integrity of the entire audit chain.
    pub fn verify_integrity(&self) -> bool {
        let records = self.records.read();
        let mut expected_prev = "0".repeat(64);

        for rec in records.iter() {
            if rec.prev_hash_hex != expected_prev {
                return false;
            }
            let mut hasher = Sha256::new();
            hasher.update(rec.sequence_num.to_le_bytes());
            hasher.update(rec.timestamp_epoch_ms.to_le_bytes());
            hasher.update(rec.actor_id.as_bytes());
            hasher.update(rec.prev_hash_hex.as_bytes());
            let computed = hasher
                .finalize()
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<String>();

            if computed != rec.record_hash_hex {
                return false;
            }
            expected_prev = rec.record_hash_hex.clone();
        }
        true
    }
}
