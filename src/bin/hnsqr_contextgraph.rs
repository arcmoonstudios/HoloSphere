/* holosphere/src/bin/hnsqr_contextgraph.rs */
//!▫~•◦-------------------------------‣
//! # HoloSphere ContextGraph CLI — Ingestion, Compilation & Graph Reasoning
//!▫~•◦-------------------------------------------------------------------‣
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use hnsqr::contextgraph::{
    ContextBudget, ContextCompiler, ContextGraphStore, ContextQueryEngine, HtmlVisualizerView,
    JsonExportView, MarkdownReportView, Namespace, adapters::fs::FilesystemSourceAdapter,
    schema::EntityId,
};
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        println!("HoloSphere ContextGraph Universal Reasoning Engine");
        println!("Usage: hnsqr_contextgraph <command> [options]");
        println!("Commands:");
        println!("  build <workspace_path>      Compile workspace into ContextGraph");
        println!("  search <query>              Search entities by query");
        println!("  explore <entity_id>         Explore entity neighborhood");
        println!("  path <from_id> <to_id>      Find shortest path between entities");
        println!("  report <workspace_path>     Generate CONTEXT_REPORT.md and HTML visualizer");
        return Ok(());
    }

    let command = &args[1];
    match command.as_str() {
        "build" | "ingest" => {
            let path = args
                .get(2)
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("."));
            println!(
                "Compiling workspace at `{}` into ContextGraph...",
                path.display()
            );
            let fs_adapter = FilesystemSourceAdapter::new();
            let sources = fs_adapter.crawl_directory(&path)?;
            println!("Discovered {} source items.", sources.len());

            let ns = Namespace::new("workspace:holosphere");
            let compiler = ContextCompiler::default();
            let output = compiler.compile(&ns, &sources)?;

            let store = ContextGraphStore::new();
            let lsn = store.commit_delta(output.into_delta());
            let state = store.snapshot();

            println!("Compilation complete in {} ms!", state.commit_lsn);
            println!("- Total Entities: {}", state.entities.len());
            println!("- Total Relations: {}", state.relations.len());
            println!(
                "- Canonical Fingerprint: {:x?}",
                state.canonical_fingerprint
            );
            println!("- Published Commit LSN: {}", lsn);
        }
        "report" => {
            let path = args
                .get(2)
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("."));
            let fs_adapter = FilesystemSourceAdapter::new();
            let sources = fs_adapter.crawl_directory(&path)?;
            let ns = Namespace::new("workspace:holosphere");
            let compiler = ContextCompiler::default();
            let output = compiler.compile(&ns, &sources)?;

            let store = ContextGraphStore::new();
            store.commit_delta(output.into_delta());
            let state = store.snapshot();

            let out_dir = PathBuf::from(".holosphere");
            std::fs::create_dir_all(&out_dir)?;

            MarkdownReportView::write_to_file(&state, out_dir.join("CONTEXT_REPORT.md"))?;
            HtmlVisualizerView::write_to_file(&state, out_dir.join("contextgraph.html"))?;
            JsonExportView::write_to_file(&state, out_dir.join("contextgraph.json"))?;

            println!("Generated reports in `.holosphere/`:");
            println!("- .holosphere/CONTEXT_REPORT.md");
            println!("- .holosphere/contextgraph.html");
            println!("- .holosphere/contextgraph.json");
        }
        "search" => {
            let query = args.get(2).cloned().unwrap_or_default();
            let fs_adapter = FilesystemSourceAdapter::new();
            let sources = fs_adapter.crawl_directory(".")?;
            let ns = Namespace::new("workspace:holosphere");
            let compiler = ContextCompiler::default();
            let output = compiler.compile(&ns, &sources)?;
            let store = ContextGraphStore::new();
            store.commit_delta(output.into_delta());
            let state = store.snapshot();

            let budget = ContextBudget::default();
            let res = ContextQueryEngine::search(&state, &query, None, &budget);
            println!("{}", res.summary);
        }
        _ => {
            println!("Unknown command: `{command}`");
        }
    }

    Ok(())
}
