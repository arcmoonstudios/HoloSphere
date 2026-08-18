/* hnsqr/src/telemetry/tracing.rs */
//!▫~•◦-------------------------------‣
//! # OpenTelemetry Distributed Tracing & Context Propagation
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides distributed trace context, W3C `traceparent` header propagation,
//! and structured span instrumentation across retrieval stages:
//!   `search_request -> rivero_route -> proof_tree_bound -> lutz_cascade -> simd_exact`
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::collections::HashMap;
use std::time::Instant;

use serde::{Deserialize, Serialize};

/// W3C Distributed Trace Context.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TraceContext {
    pub trace_id: [u8; 16],
    pub span_id: [u8; 8],
    pub sampled: bool,
}

impl TraceContext {
    /// Generates a new root trace context.
    pub fn new_root() -> Self {
        let mut trace_id = [0u8; 16];
        let mut span_id = [0u8; 8];
        for b in &mut trace_id {
            *b = rand::random();
        }
        for b in &mut span_id {
            *b = rand::random();
        }
        Self {
            trace_id,
            span_id,
            sampled: true,
        }
    }

    /// Creates a child span context under this trace.
    pub fn child_span(&self) -> Self {
        let mut span_id = [0u8; 8];
        for b in &mut span_id {
            *b = rand::random();
        }
        Self {
            trace_id: self.trace_id,
            span_id,
            sampled: self.sampled,
        }
    }

    /// Formats as standard W3C `traceparent` header (`00-<trace_id>-<span_id>-01`).
    pub fn to_w3c_header(&self) -> String {
        let trace_hex = hex_encode(&self.trace_id);
        let span_hex = hex_encode(&self.span_id);
        let flags = if self.sampled { "01" } else { "00" };
        format!("00-{trace_hex}-{span_hex}-{flags}")
    }
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Recorded trace span for production diagnostics.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SpanRecord {
    pub name: String,
    pub trace_id_hex: String,
    pub span_id_hex: String,
    pub parent_span_id_hex: Option<String>,
    pub duration_micros: u64,
    pub attributes: HashMap<String, String>,
}

/// Active scoped execution span.
pub struct ExecutionSpan {
    pub name: &'static str,
    pub context: TraceContext,
    pub parent_id: Option<[u8; 8]>,
    pub start_time: Instant,
    pub attributes: HashMap<String, String>,
}

impl ExecutionSpan {
    pub fn start(name: &'static str, context: &TraceContext) -> Self {
        Self {
            name,
            context: context.child_span(),
            parent_id: Some(context.span_id),
            start_time: Instant::now(),
            attributes: HashMap::new(),
        }
    }

    pub fn set_attribute(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.attributes.insert(key.into(), value.into());
    }

    pub fn finish(self) -> SpanRecord {
        let duration_micros = self.start_time.elapsed().as_micros() as u64;
        SpanRecord {
            name: self.name.to_string(),
            trace_id_hex: hex_encode(&self.context.trace_id),
            span_id_hex: hex_encode(&self.context.span_id),
            parent_span_id_hex: self.parent_id.map(|p| hex_encode(&p)),
            duration_micros,
            attributes: self.attributes,
        }
    }
}
