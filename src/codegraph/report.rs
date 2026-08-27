/* holosphere/src/codegraph/report.rs */
//!▫~•◦-------------------------------‣
//! # CodeGraph Markdown Architecture Report Generator (CODE_REPORT.md)
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Synthesizes a structured architectural overview detailing topological hubs,
//! community modules, circular dependencies, and engineering rationale comments.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::collections::BTreeMap;
use std::path::Path;

use super::analysis::CodeGraphAnalyzer;
use super::community::CommunityDetector;
use super::ingest::CodeGraphStoreState;
use super::schema::CodeNodeKind;
use crate::HNSQRResult;

pub struct CodeGraphReportGenerator;

impl CodeGraphReportGenerator {
    /// Generates markdown architecture report content.
    #[must_use]
    pub fn generate_markdown(state: &CodeGraphStoreState) -> String {
        let total_files = state.nodes_by_file.len();
        let total_symbols = state
            .nodes
            .values()
            .filter(|n| n.kind != CodeNodeKind::File && n.kind != CodeNodeKind::Rationale)
            .count();
        let total_relations = state.edges.len();
        let total_rationale = state
            .nodes
            .values()
            .filter(|n| n.kind == CodeNodeKind::Rationale)
            .count();

        let communities = CommunityDetector::detect_communities(state);
        let god_nodes = CodeGraphAnalyzer::find_god_nodes(state, 10);
        let cycles = CodeGraphAnalyzer::find_dependency_cycles(state);

        let mut doc = Vec::new();

        doc.push(format!(
            "# HoloSphere CodeGraph Architecture Report — `{}`",
            state.workspace_id
        ));
        doc.push(String::new());

        // Overview
        doc.push("## Workspace Metrics".to_string());
        doc.push(format!("- **Source Files:** {total_files}"));
        doc.push(format!("- **Indexed Symbols:** {total_symbols}"));
        doc.push(format!("- **Structural Relations:** {total_relations}"));
        doc.push(format!(
            "- **Architectural Communities:** {}",
            communities.len()
        ));
        doc.push(format!(
            "- **Extracted Rationale Notes:** {total_rationale}"
        ));
        doc.push(format!("- **Snapshot LSN:** {}", state.commit_lsn));
        doc.push(String::new());

        // Top Architectural Hubs
        doc.push("## Architectural Hubs (Highest Centrality)".to_string());
        doc.push(
            "| Symbol | Kind | In-Degree | Out-Degree | Total Connections | Source File |"
                .to_string(),
        );
        doc.push("| :--- | :--- | :--- | :--- | :--- | :--- |".to_string());
        for hub in &god_nodes {
            doc.push(format!(
                "| `{}` | `{}` | {} | {} | **{}** | `{}` |",
                hub.name,
                hub.kind,
                hub.in_degree,
                hub.out_degree,
                hub.total_degree,
                hub.source_file.display()
            ));
        }
        doc.push(String::new());

        // Communities
        doc.push("## Architectural Communities".to_string());
        for comm in &communities {
            doc.push(format!(
                "### Community {}: {}",
                comm.community_id, comm.label
            ));
            doc.push(format!("- **Symbols:** {}", comm.symbol_count));
            doc.push(format!(
                "- **Dominant Symbols:** {}",
                comm.top_symbols.join(", ")
            ));
            let files_str = comm
                .top_files
                .iter()
                .map(|f| f.display().to_string())
                .collect::<Vec<_>>()
                .join(", ");
            doc.push(format!("- **Key Files:** {files_str}"));
            doc.push(String::new());
        }

        // Circular Dependencies
        if !cycles.is_empty() {
            doc.push("## ⚠️ Dependency Cycles Detected".to_string());
            for (idx, cycle) in cycles.iter().enumerate() {
                doc.push(format!(
                    "**Cycle {}:** {}",
                    idx + 1,
                    cycle.symbol_names.join(" ↔ ")
                ));
            }
            doc.push(String::new());
        }

        // Rationale Nodes
        let mut rationale_by_tag: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for node in state.nodes.values() {
            if node.kind == CodeNodeKind::Rationale {
                let tag = node
                    .attributes
                    .get("rationale_tag")
                    .and_then(|v| v.as_str())
                    .unwrap_or("note")
                    .to_uppercase();
                rationale_by_tag.entry(tag).or_default().push(format!(
                    "`{}` in `{}`",
                    node.name,
                    node.source_file.display()
                ));
            }
        }

        if !rationale_by_tag.is_empty() {
            doc.push("## Extracted Engineering Rationale".to_string());
            for (tag, items) in rationale_by_tag {
                doc.push(format!("### {tag} ({})", items.len()));
                for item in items.into_iter().take(8) {
                    doc.push(format!("- {item}"));
                }
                doc.push(String::new());
            }
        }

        // Suggested Questions
        doc.push("## Suggested Architecture Queries".to_string());
        doc.push(
            "1. `code_explain(\"HNSQRIndex::search\")` — Trace primary retrieval execution path."
                .to_string(),
        );
        doc.push("2. `code_path(\"ModelToolService::search\", \"HNSQRIndex::search\")` — Structural flow from MCP gateway to index.".to_string());
        doc.push("3. `code_impact(\"CodeGraphDelta\", 3)` — Downstream dependency blast radius for storage mutations.".to_string());
        doc.push(String::new());

        doc.join("\n")
    }

    /// Writes report to disk.
    pub fn write_to_file(state: &CodeGraphStoreState, path: impl AsRef<Path>) -> HNSQRResult<()> {
        if let Some(parent) = path.as_ref().parent() {
            std::fs::create_dir_all(parent)?;
        }
        let md = Self::generate_markdown(state);
        std::fs::write(path, md)?;
        Ok(())
    }
}
