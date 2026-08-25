/* holosphere/src/transport/mod.rs */
//!▫~•◦-------------------------------‣
//! # Wire Transport & QIR0 Network Protocol Subsystem
//!▫~•◦-------------------------------------------------------------------‣
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

pub mod arrow_flight;
pub mod mcp;
pub mod model_gateway;
pub mod qir0;
pub mod resp;
pub mod swagger;
pub mod web_console;

pub use arrow_flight::{
    ArrowFieldDescriptor, ArrowFieldType, ArrowFlightService, ArrowRecordBatchPayload,
    ArrowSchemaDescriptor,
};
pub use mcp::{MCP_PROTOCOL_VERSION, create_mcp_router, process_mcp_payload};
pub use model_gateway::{
    EmbeddingDescriptor, EvidenceEnvelope, KnowledgeRecord, ModelGatewayAuth, ModelKnowledgeStore,
    ModelOutcomeRecord, ModelToolService, RecordOutcomeToolRequest, RememberToolRequest,
    ResolveToolRequest, SearchToolRequest, TraverseToolRequest, create_model_api_router,
};
pub use qir0::{HNSQRClient, HNSQRServer, MessageHeader, OpCode, PROTOCOL_MAGIC};
pub use resp::{PubSubBroker, RedisStreamEngine, RespFrame, RespServer, StreamEntry};
pub use swagger::{OpenApiSpecGenerator, SWAGGER_HTML, openapi_spec_handler, swagger_handler};
pub use web_console::{CONSOLE_HTML, console_handler};
