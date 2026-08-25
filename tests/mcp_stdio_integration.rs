use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

struct McpProcess {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl McpProcess {
    fn start() -> Self {
        let data_dir =
            std::env::temp_dir().join(format!("holosphere-mcp-stdio-{:x}", rand::random::<u64>()));
        let mut child = Command::new(env!("CARGO_BIN_EXE_hnsqr_mcp_stdio"))
            .env("HNSQR_DATA_DIR", data_dir)
            .env("HNSQR_MCP_TENANT", "integration-agent")
            .env("HNSQR_MCP_ROLE", "readwrite")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let stdin = child.stdin.take().unwrap();
        let stdout = BufReader::new(child.stdout.take().unwrap());
        Self {
            child,
            stdin,
            stdout,
        }
    }

    fn request(&mut self, value: serde_json::Value) -> serde_json::Value {
        writeln!(self.stdin, "{value}").unwrap();
        self.stdin.flush().unwrap();
        let mut line = String::new();
        self.stdout.read_line(&mut line).unwrap();
        assert!(
            !line.is_empty(),
            "STDIO MCP server closed without a response"
        );
        serde_json::from_str(&line).unwrap()
    }
}

impl Drop for McpProcess {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[test]
fn stdio_server_initializes_persists_and_retrieves_shared_agent_knowledge() {
    let mut server = McpProcess::start();
    let initialized = server.request(serde_json::json!({
        "jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "integration-test", "version": "1.0"}
        }
    }));
    assert_eq!(initialized["result"]["protocolVersion"], "2025-06-18");
    assert!(
        initialized["result"]["instructions"]
            .as_str()
            .unwrap()
            .contains("Autonomously consult HoloSphere")
    );

    let tools = server.request(serde_json::json!({
        "jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}
    }));
    assert_eq!(tools["result"]["tools"].as_array().unwrap().len(), 5);

    let remembered = server.request(serde_json::json!({
        "jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {
            "name": "holosphere.remember",
            "arguments": {
                "idempotency_key": "stdio-integration-memory-v1",
                "id": "stdio-integration-memory",
                "kind": "verified_resolution",
                "content": "A native STDIO MCP transport lets local agents share HoloSphere knowledge without public hosting.",
                "provenance": [{"source_id": "mcp-stdio-test", "content_hash": "sha256:mcp-stdio-test"}]
            }
        }
    }));
    assert_eq!(
        remembered["result"]["structuredContent"]["results"]["id"],
        "stdio-integration-memory"
    );

    let searched = server.request(serde_json::json!({
        "jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {
            "name": "holosphere.search",
            "arguments": {"query": "local agents shared knowledge public hosting", "k": 5}
        }
    }));
    assert_eq!(
        searched["result"]["structuredContent"]["results"][0]["id"],
        "stdio-integration-memory"
    );
}
