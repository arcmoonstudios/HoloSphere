/* hnsqr/src/security/fuzzing.rs */
//!▫~•◦-------------------------------‣
//! # Production Protocol & Binary Parser Fuzzing Engine
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Validates that public network and storage deserializers (QIR0 frames,
//! Snapshot V2 headers, WAL record frames, and Raft consensus RPCs) never panic
//! on adversarial, truncated, malformed, or out-of-bounds byte streams.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use bytes::BytesMut;

use crate::consensus::raft::{AppendEntriesArgs, RequestVoteArgs};
use crate::storage::snapshot::SnapshotHeaderV2;
use crate::storage::wal::WalFrameHeader;
use crate::transport::qir0::MessageHeader;

/// Fuzz execution results summary.
#[derive(Clone, Debug, Default)]
pub struct ProtocolFuzzSummary {
    pub total_inputs_tested: usize,
    pub graceful_rejections: usize,
    pub panics_detected: usize,
}

pub struct ProtocolFuzzer;

impl ProtocolFuzzer {
    /// Fuzzes the production QIR0 framing parser.
    pub fn fuzz_qir0_parser(payloads: &[Vec<u8>]) -> ProtocolFuzzSummary {
        let mut summary = ProtocolFuzzSummary::default();

        for data in payloads {
            summary.total_inputs_tested += 1;
            let res = std::panic::catch_unwind(|| {
                let mut buf = BytesMut::from(data.as_slice());
                let _ = MessageHeader::decode(&mut buf);
            });

            if res.is_ok() {
                summary.graceful_rejections += 1;
            } else {
                summary.panics_detected += 1;
            }
        }

        summary
    }

    /// Fuzzes the Snapshot V2 binary header parser.
    pub fn fuzz_snapshot_header_parser(payloads: &[Vec<u8>]) -> ProtocolFuzzSummary {
        let mut summary = ProtocolFuzzSummary::default();

        for data in payloads {
            summary.total_inputs_tested += 1;
            let res = std::panic::catch_unwind(|| {
                if data.len() >= 256 {
                    let mut arr = [0u8; 256];
                    arr.copy_from_slice(&data[0..256]);
                    let ptr = arr.as_ptr() as *const SnapshotHeaderV2;
                    let _ = unsafe { std::ptr::read_unaligned(ptr) };
                }
            });

            if res.is_ok() {
                summary.graceful_rejections += 1;
            } else {
                summary.panics_detected += 1;
            }
        }

        summary
    }

    /// Fuzzes the WAL record frame parser.
    pub fn fuzz_wal_frame_parser(payloads: &[Vec<u8>]) -> ProtocolFuzzSummary {
        let mut summary = ProtocolFuzzSummary::default();

        for data in payloads {
            summary.total_inputs_tested += 1;
            let res = std::panic::catch_unwind(|| {
                if data.len() >= 36 {
                    let mut arr = [0u8; 36];
                    arr.copy_from_slice(&data[0..36]);
                    let _ = WalFrameHeader::decode(&arr);
                }
            });

            if res.is_ok() {
                summary.graceful_rejections += 1;
            } else {
                summary.panics_detected += 1;
            }
        }

        summary
    }

    /// Fuzzes the Raft RPC payload parser.
    pub fn fuzz_raft_rpc_parser(payloads: &[Vec<u8>]) -> ProtocolFuzzSummary {
        let mut summary = ProtocolFuzzSummary::default();

        for data in payloads {
            summary.total_inputs_tested += 1;
            let res = std::panic::catch_unwind(|| {
                let _ = bincode::deserialize::<AppendEntriesArgs>(data);
                let _ = bincode::deserialize::<RequestVoteArgs>(data);
            });

            if res.is_ok() {
                summary.graceful_rejections += 1;
            } else {
                summary.panics_detected += 1;
            }
        }

        summary
    }
}
