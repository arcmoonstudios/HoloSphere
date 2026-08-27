/* holosphere/src/contextgraph/views/markdown.rs */
//!▫~•◦-------------------------------‣
//! # ContextGraph Architecture Report View (CONTEXT_REPORT.md)
//!▫~•◦-------------------------------------------------------------------‣
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::path::Path;

use super::super::analytics::ContextAnalytics;
use super::super::community::ScopeClustering;
use super::super::store::ContextGraphStoreState;
use super::GraphView;
use crate::HNSQRResult;

pub struct MarkdownReportView;

impl Default for MarkdownReportView {
    fn default() -> Self {
        Self::new()
    }
}

impl MarkdownReportView {
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    #[must_use]
    pub fn generate(state: &ContextGraphStoreState) -> String {
        let total_entities = state.entities.len();
        let total_relations = state.relations.len();
        let scopes = ScopeClustering::detect_scopes(state);
        let hubs = ContextAnalytics::find_hubs(state, 10);

        let mut doc = Vec::new();
        doc.push(format!(
            "# HoloSphere ContextGraph Report — `{}`",
            state.namespace
        ));
        doc.push(String::new());

        doc.push("## Universal Metrics".to_string());
        doc.push(format!("- **Total Entities:** {total_entities}"));
        doc.push(format!("- **Total Relations:** {total_relations}"));
        doc.push(format!("- **Detected Scopes:** {}", scopes.len()));
        doc.push(format!(
            "- **Canonical Graph Fingerprint:** `{:x?}`",
            state.canonical_fingerprint
        ));
        doc.push(format!("- **Commit LSN:** {}", state.commit_lsn));
        doc.push(String::new());

        doc.push("## Top Centrality Hubs".to_string());
        doc.push("| Entity Label | Kind | Total Connections |".to_string());
        doc.push("| :--- | :--- | :--- |".to_string());
        for hub in hubs {
            doc.push(format!(
                "| `{}` | `{}` | **{}** |",
                hub.label, hub.kind, hub.total_degree
            ));
        }
        doc.push(String::new());

        doc.push("## Topological Scopes".to_string());
        for sc in scopes {
            doc.push(format!("### {}", sc.label));
            doc.push(format!("- **Entities:** {}", sc.entity_count));
            doc.push(format!(
                "- **Key Entities:** {}",
                sc.top_entities.join(", ")
            ));
            doc.push(String::new());
        }

        doc.join("\n")
    }

    pub fn write_to_file(
        state: &ContextGraphStoreState,
        path: impl AsRef<Path>,
    ) -> HNSQRResult<()> {
        if let Some(parent) = path.as_ref().parent() {
            std::fs::create_dir_all(parent)?;
        }
        let md = Self::generate(state);
        std::fs::write(path, md)?;
        Ok(())
    }
}

impl GraphView for MarkdownReportView {
    fn render(&self, state: &ContextGraphStoreState) -> HNSQRResult<Vec<u8>> {
        Ok(Self::generate(state).into_bytes())
    }
}
