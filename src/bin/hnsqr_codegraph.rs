/* holosphere/src/bin/hnsqr_codegraph.rs */
//!▫~•◦-------------------------------‣
//! # HoloSphere CodeGraph CLI compatibility wrapper over ContextGraph
//!▫~•◦-------------------------------------------------------------------‣
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use hnsqr::contextgraph::{
    ContextBudget, ContextCompiler, ContextGraphStore, ContextQueryEngine, HtmlVisualizerView,
    JsonExportView, MarkdownReportView, Namespace, adapters::fs::FilesystemSourceAdapter,
};
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        println!("HoloSphere CodeGraph (ContextGraph Code Profile)");
        println!("Usage: hnsqr_codegraph <command> [options]");
        println!("Commands:");
        println!("  build <workspace_path>      Compile workspace into CodeGraph");
        println!("  search <query>              Search code symbols");
        println!("  report <workspace_path>     Generate CODE_REPORT.md and HTML visualizer");
        return Ok(());
    }

    let command = &args[1];
    match command.as_str() {
        "build" => {
            let path = args
                .get(2)
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("."));
            println!("Compiling workspace at `{}`...", path.display());
            let fs_adapter = FilesystemSourceAdapter::new();
            let sources = fs_adapter.crawl_directory(&path)?;
            let ns = Namespace::new("workspace:holosphere");
            let compiler = ContextCompiler::default();
            let output = compiler.compile(&ns, &sources)?;

            let store = ContextGraphStore::new();
            let lsn = store.commit_delta(output.into_delta());
            let state = store.snapshot();

            println!("CodeGraph compilation complete!");
            println!("- Total Entities: {}", state.entities.len());
            println!("- Total Relations: {}", state.relations.len());
            println!(
                "- Canonical Fingerprint: {:x?}",
                state.canonical_fingerprint
            );
            println!("- Commit LSN: {}", lsn);
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

            MarkdownReportView::write_to_file(&state, out_dir.join("CODE_REPORT.md"))?;
            HtmlVisualizerView::write_to_file(&state, out_dir.join("codegraph.html"))?;
            JsonExportView::write_to_file(&state, out_dir.join("codegraph.json"))?;

            println!("Generated reports in `.holosphere/`:");
            println!("- .holosphere/CODE_REPORT.md");
            println!("- .holosphere/codegraph.html");
            println!("- .holosphere/codegraph.json");
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
