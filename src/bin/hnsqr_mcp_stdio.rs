/* holosphere/src/bin/hnsqr_mcp_stdio.rs */
//! Local MCP STDIO server shared by Codex, Gemini CLI, Claude Code, and MCP SDK clients.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::path::PathBuf;
use std::sync::Arc;

use hnsqr::{
    AccessRole, AuthenticatedSubject, GatewayRouter, ModelGatewayAuth, ModelKnowledgeStore,
    ModelToolService, process_mcp_payload, providers_from_file_if_exists,
};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data_dir = data_directory();
    let tenant_id =
        std::env::var("HNSQR_MCP_TENANT").unwrap_or_else(|_| "local-agents".to_string());
    let role = access_role()?;
    let subject = AuthenticatedSubject {
        tenant_id: tenant_id.clone(),
        role,
        key_id: format!("stdio:{tenant_id}"),
    };
    let vectors = Arc::new(GatewayRouter::new(&data_dir.to_string_lossy(), false));
    let store = Arc::new(ModelKnowledgeStore::open(
        data_dir.join("model-knowledge.jsonl"),
    )?);
    let auth = Arc::new(ModelGatewayAuth::development_anonymous());
    let config_path = std::env::var("HNSQR_CONFIG").unwrap_or_else(|_| "Config.toml".to_string());
    let service = match providers_from_file_if_exists(&config_path)? {
        Some(providers) => ModelToolService::with_providers(
            vectors,
            store,
            auth,
            providers.embedding,
            providers.web_search,
        ),
        None => ModelToolService::new(vectors, store, auth),
    };

    let stdin = tokio::io::stdin();
    let mut lines = BufReader::new(stdin).lines();
    let mut stdout = tokio::io::stdout();
    while let Some(line) = lines.next_line().await? {
        if line.trim().is_empty() {
            continue;
        }
        let response = match serde_json::from_str::<serde_json::Value>(&line) {
            Ok(payload) => process_mcp_payload(&service, &subject, payload),
            Err(error) => Some(serde_json::json!({
                "jsonrpc": "2.0",
                "id": null,
                "error": {"code": -32700, "message": error.to_string()}
            })),
        };
        if let Some(response) = response {
            stdout
                .write_all(serde_json::to_string(&response)?.as_bytes())
                .await?;
            stdout.write_all(b"\n").await?;
            stdout.flush().await?;
        }
    }
    Ok(())
}

fn access_role() -> Result<AccessRole, String> {
    let value = std::env::var("HNSQR_MCP_ROLE").unwrap_or_else(|_| "readwrite".to_string());
    match value.trim().to_ascii_lowercase().as_str() {
        "readonly" | "read-only" | "read" => Ok(AccessRole::ReadOnly),
        "readwrite" | "read-write" | "write" => Ok(AccessRole::ReadWrite),
        "admin" => Ok(AccessRole::Admin),
        _ => Err(format!(
            "HNSQR_MCP_ROLE must be readonly, readwrite, or admin; received '{value}'"
        )),
    }
}

fn data_directory() -> PathBuf {
    if let Some(path) = std::env::var_os("HNSQR_DATA_DIR") {
        return PathBuf::from(path);
    }
    if let Some(path) = std::env::var_os("LOCALAPPDATA") {
        return PathBuf::from(path).join("HoloSphere").join("model-agent");
    }
    if let Some(path) = std::env::var_os("XDG_DATA_HOME") {
        return PathBuf::from(path).join("holosphere").join("model-agent");
    }
    if let Some(path) = std::env::var_os("HOME") {
        return PathBuf::from(path)
            .join(".local")
            .join("share")
            .join("holosphere")
            .join("model-agent");
    }
    PathBuf::from("hnsqr_data").join("model-agent")
}
