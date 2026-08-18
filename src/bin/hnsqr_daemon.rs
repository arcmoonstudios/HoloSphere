/* hnsqr/src/bin/hnsqr_daemon.rs */
//!▫~•◦-------------------------------‣
//! # HNSQR Distributed Operational Database Daemon
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Exposes both:
//! 1. High-Performance Zero-Copy Async Binary TCP Engine (Port 9090)
//! 2. Classical-to-Quantum LLM HTTP REST Gateway (Port 8080)
//!
//! ## Key Capabilities
//! - **Concurrent Multi-Protocol Transport:** Serves native TCP binary streams and HTTP REST queries concurrently.
//! - **Zero-Copy Disk Synchronization:** Dispatches memory-mapped flushed segments across multi-collection namespaces.
//! - **Production Signal Handling:** Graceful shutdown via Tokio signal integration.
//!
//! ### Architectural Notes
//! Standalone server entrypoint executing on Tokio async runtime.
//!
//! #### Example
//! ```bash
//! cargo run --release --bin hnsqr_daemon
//! ```
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use hnsqr::transport::qir0::HNSQRServer;
use hnsqr::vector::folding::{GatewayRouter, create_http_router};
use hnsqr::{HNSQRConfig, HNSQRIndex};
use std::net::SocketAddr;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_target(false)
        .with_level(true)
        .init();

    println!(
        r#"
    ██╗  ██╗███╗   ██╗███████╗ ██████╗ ██████╗ 
    ██║  ██║████╗  ██║██╔════╝██╔═══██╗██╔══██╗
    ███████║██╔██╗ ██║███████╗██║   ██║██████╔╝
    ██╔══██║██║╚██╗██║╚════██║██║▄▄ ██║██╔══██╗
    ██║  ██║██║ ╚████║███████║╚██████╔╝██║  ██║
    ╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝ ╚══▀▀═╝ ╚═╝  ╚═╝
     Quantum Superposition Vector Engine Daemon
    "#
    );

    let tcp_str = std::env::var("HNSQR_TCP_ADDR").unwrap_or_else(|_| "127.0.0.1:9090".to_string());
    let tcp_addr: SocketAddr = tcp_str
        .parse()
        .unwrap_or_else(|_| "127.0.0.1:9090".parse().unwrap());

    let http_str =
        std::env::var("HNSQR_HTTP_ADDR").unwrap_or_else(|_| "127.0.0.1:8080".to_string());
    let http_addr: SocketAddr = http_str
        .parse()
        .unwrap_or_else(|_| "127.0.0.1:8080".parse().unwrap());

    let data_dir = std::env::var("HNSQR_DATA_DIR").unwrap_or_else(|_| "./hnsqr_data".to_string());
    let dim: usize = std::env::var("HNSQR_DIM")
        .unwrap_or_else(|_| "64".to_string())
        .parse()
        .unwrap_or(64);

    println!(
        "⚡ Initializing Lock-Free Arena & Inverted Metadata Index (Dim: {})...",
        dim
    );
    let config = HNSQRConfig::default();
    let index = Arc::new(HNSQRIndex::new(config, dim));

    let tcp_server = HNSQRServer::new(index.clone(), tcp_addr);
    let gateway_router = Arc::new(GatewayRouter::new(&data_dir, false));
    let app = create_http_router(gateway_router);

    println!("🚀 Starting Async Binary TCP Server on: {}", tcp_addr);
    let tcp_handle = tokio::spawn(async move {
        if let Err(e) = tcp_server.run().await {
            eprintln!("❌ TCP Server Error: {}", e);
        }
    });

    println!(
        "🌐 Starting HTTP REST & LLM Gateway on:  http://{}",
        http_addr
    );
    let http_handle = tokio::spawn(async move {
        let listener = tokio::net::TcpListener::bind(http_addr)
            .await
            .expect("Failed to bind HTTP listener");
        if let Err(e) = axum::serve(listener, app).await {
            eprintln!("❌ HTTP Gateway Error: {}", e);
        }
    });

    println!("✅ HNSQR Operational Database Daemon is online and ready for traffic.\n");

    tokio::select! {
        _ = tcp_handle => eprintln!("TCP Server exited"),
        _ = http_handle => eprintln!("HTTP Gateway Server exited"),
        _ = tokio::signal::ctrl_c() => println!("\n🛑 Graceful shutdown initiated..."),
    }

    Ok(())
}
