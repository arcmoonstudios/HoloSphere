/* holosphere/src/transport/mod.rs */
//!▫~•◦-------------------------------‣
//! # Wire Transport & QIR0 Network Protocol Subsystem
//!▫~•◦-------------------------------------------------------------------‣
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

pub mod arrow_flight;
pub mod embedding_provider;
pub mod mcp;
pub mod model_gateway;
pub mod qir0;
pub mod resp;
pub mod swagger;
pub mod web_console;
pub mod web_search;

pub use arrow_flight::{
    ArrowFieldDescriptor, ArrowFieldType, ArrowFlightService, ArrowRecordBatchPayload,
    ArrowSchemaDescriptor,
};
pub use embedding_provider::{
    ConfiguredProviders, EmbeddingBackend, EmbeddingProviderConfig, HoloSphereConfig,
    TextEmbeddingProvider, load_config, provider_from_config, provider_from_file_if_exists,
    providers_from_file_if_exists,
};
pub use mcp::{MCP_PROTOCOL_VERSION, create_mcp_router, process_mcp_payload};
pub use model_gateway::{
    ActionGate, CaseBudget, EmbeddingDescriptor, EvidenceEnvelope, KnowledgeRecord,
    ModelGatewayAuth, ModelKnowledgeStore, ModelOutcomeRecord, ModelToolService,
    RecordOutcomeToolRequest, RememberToolRequest, ResolveToolRequest, RunCaseResult,
    RunCaseToolRequest, RuntimeStatus, SearchToolRequest, TraverseToolRequest,
    create_model_api_router,
};
pub use qir0::{HNSQRClient, HNSQRServer, MessageHeader, OpCode, PROTOCOL_MAGIC};
pub use resp::{PubSubBroker, RedisStreamEngine, RespFrame, RespServer, StreamEntry};
pub use swagger::{OpenApiSpecGenerator, SWAGGER_HTML, openapi_spec_handler, swagger_handler};
pub use web_console::{CONSOLE_HTML, console_handler};
pub use web_search::{
    WebSearchBackend, WebSearchConfig, WebSearchProvider, WebSearchResponse, WebSearchResult,
    WebSearchToolRequest,
};
