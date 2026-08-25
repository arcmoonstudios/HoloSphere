# OpenAI, Gemini, and Claude Integration

HoloSphere exposes one provider-neutral remote MCP server at `POST /mcp` using the
MCP `2025-06-18` Streamable HTTP protocol. The same operations are available as REST
endpoints for applications that prefer to drive each provider's function-calling loop.

## Start the gateway

Model endpoints are fail-closed unless at least one token is configured or anonymous
development access is explicitly enabled.

```powershell
$env:HNSQR_MODEL_READ_TOKEN = "replace-with-read-token"
$env:HNSQR_MODEL_WRITE_TOKEN = "replace-with-write-token"
$env:HNSQR_DATA_DIR = ".\hnsqr_data"
cargo run --release --bin hnsqr_daemon
```

For loopback-only development:

```powershell
$env:HNSQR_MODEL_ALLOW_ANONYMOUS = "true"
cargo run --bin hnsqr_daemon
```

The MCP URL is `http://127.0.0.1:8080/mcp`. Production model APIs require a stable
public HTTPS URL; put the daemon behind a TLS reverse proxy and keep anonymous access
disabled.

## Model tools

| MCP tool | REST endpoint | Role | Behavior |
| --- | --- | --- | --- |
| `holosphere.search` | `/v1/knowledge/search` | ReadOnly | Certified, tenant-isolated retrieval at one LSN |
| `holosphere.traverse` | `/v1/knowledge/traverse` | ReadOnly | Bounded N-ary relation traversal |
| `holosphere.resolve` | `/v1/knowledge/resolve` | ReadOnly | Evidence-backed hypotheses; never executes actions |
| `holosphere.remember` | `/v1/knowledge/remember` | ReadWrite | Idempotent, provenance-bearing durable knowledge write |
| `holosphere.record_outcome` | `/v1/knowledge/outcomes` | ReadWrite | Idempotent empirical feedback write |

All retrieved content is marked `content_is_untrusted: true`. A caller must treat it as
data rather than system instructions. Resolution results are marked
`hypothesis_requires_external_validation`.

## MCP verification

```powershell
$headers = @{
  Authorization = "Bearer $env:HNSQR_MODEL_READ_TOKEN"
  "Content-Type" = "application/json"
}

$initialize = @{
  jsonrpc = "2.0"
  id = 1
  method = "initialize"
  params = @{
    protocolVersion = "2025-06-18"
    capabilities = @{}
    clientInfo = @{ name = "smoke-test"; version = "1.0" }
  }
} | ConvertTo-Json -Depth 8

Invoke-RestMethod http://127.0.0.1:8080/mcp -Method Post -Headers $headers -Body $initialize
```

Call `tools/list` next, then `tools/call`. The server is stateless; snapshot consistency
is explicit through `snapshot_lsn` rather than hidden in an HTTP session.

## OpenAI Responses API

Configure a remote MCP tool with:

```json
{
  "type": "mcp",
  "server_label": "holosphere",
  "server_url": "https://holosphere.example.com/mcp",
  "headers": {"Authorization": "Bearer READ_TOKEN"},
  "allowed_tools": ["holosphere.search", "holosphere.traverse", "holosphere.resolve"],
  "require_approval": "never"
}
```

Give ordinary conversations only the three read tools. Add write tools only in an
application flow that applies its own approval and idempotency policy. See the official
[OpenAI MCP guide](https://developers.openai.com/api/docs/guides/tools-connectors-mcp).

## Gemini API

In the Gemini Interactions API, configure a tool with type `mcp_server`, the name
`holosphere`, URL `https://holosphere.example.com/mcp`, the bearer header, and an
allow-list of the three read tools. Gemini's remote MCP integration uses Streamable HTTP.
See the official [Gemini function-calling and MCP guide](https://ai.google.dev/gemini-api/docs/function-calling).

## Claude Messages API

Configure the Messages API `mcp_servers` entry with the HoloSphere HTTPS URL and
authorization token, and send the MCP connector beta header required by the current
Claude API. Restrict normal conversations to read tools. See the official
[Claude MCP connector guide](https://platform.claude.com/docs/en/agents-and-tools/mcp-connector).

## Embedding spaces

Text-only calls use HoloSphere's deterministic `holosphere/lexical-hash/1` embedding.
It is a dependency-free operational baseline, not a substitute for a semantic embedding
model. For semantic cross-domain retrieval, the application should compute an embedding
and send both the vector and its complete descriptor:

```json
{
  "provider": "openai",
  "model": "your-pinned-embedding-model",
  "version": "your-pinned-version",
  "dimensions": 1536,
  "normalization": "l2",
  "distance_metric": "cosine"
}
```

The first write pins a collection to that exact descriptor. HoloSphere rejects vectors
from a different provider, model, version, dimension, normalization, or metric. The chat
model may be OpenAI, Gemini, or Claude independently of the embedding model.

## Persistence and authorization

- Model knowledge and outcomes are fsynced to `HNSQR_DATA_DIR/model-knowledge.jsonl`.
- The journal is replayed at startup and reconstructs the vector projection.
- Tenant identity comes from the authenticated credential, never from tool arguments.
- ReadOnly tokens cannot call mutation tools.
- Write requests require an idempotency key.
- Historical reads reject future LSNs and exclude later records.
- Provider credentials stay in the calling application; HoloSphere stores only its own access tokens.

