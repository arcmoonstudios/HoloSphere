/* hnsqr/src/security/siem.rs */
//!▫~•◦-------------------------------‣
//! # SIEM Integration & Open Security Event Export
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Streams tamper-evident audit records across OpenTelemetry (OTLP), RFC 5424 Syslog,
//! and structured JSON formats, with external root checkpoint verification (`hnsqr audit verify`).
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use serde::{Deserialize, Serialize};

use crate::HNSQRResult;
use crate::security::audit::{AuditLogger, AuditRecord};

/// Supported SIEM export formats.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SiemFormat {
    StructuredJson,
    Rfc5424Syslog,
    OtlpJson,
}

/// SIEM Event Streamer.
pub struct SiemExporter;

impl SiemExporter {
    /// Formats an audit record into the target SIEM standard.
    pub fn format_record(record: &AuditRecord, format: SiemFormat) -> String {
        match format {
            SiemFormat::StructuredJson => serde_json::to_string(record).unwrap_or_default(),
            SiemFormat::Rfc5424Syslog => {
                format!(
                    "<134>1 {} hnsqr-engine security-audit - - [audit seq=\"{}\" actor=\"{}\" hash=\"{}\"] {:?}",
                    record.timestamp_epoch_ms,
                    record.sequence_num,
                    record.actor_id,
                    record.record_hash_hex,
                    record.action
                )
            }
            SiemFormat::OtlpJson => {
                serde_json::json!({
                    "resourceLogs": [{
                        "resource": {
                            "attributes": [{
                                "key": "service.name",
                                "value": {"stringValue": "hnsqr"}
                            }]
                        },
                        "scopeLogs": [{
                            "logRecords": [{
                                "timeUnixNano": record.timestamp_epoch_ms * 1_000_000,
                                "severityNumber": 9,
                                "severityText": "INFO",
                                "body": { "stringValue": format!("{:?}", record.action) },
                                "attributes": [
                                    { "key": "audit.seq", "value": { "intValue": record.sequence_num } },
                                    { "key": "audit.actor", "value": { "stringValue": record.actor_id } },
                                    { "key": "audit.hash", "value": { "stringValue": record.record_hash_hex } }
                                ]
                            }]
                        }]
                    }]
                }).to_string()
            }
        }
    }

    /// Verifies the cryptographic integrity of the entire audit chain against an expected root hash.
    pub fn verify_audit_chain(logger: &AuditLogger, expected_last_hash: &str) -> HNSQRResult<bool> {
        let is_valid = logger.verify_integrity();
        let last_hash = logger.latest_checkpoint_hash();
        Ok(is_valid && (expected_last_hash.is_empty() || last_hash == expected_last_hash))
    }
}
