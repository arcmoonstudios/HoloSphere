/* holosphere/src/contextgraph/adapters/code_jsx.rs */
//!▫~•◦-------------------------------‣
//! # JavaScript JSX Context Adapter (tree-sitter-javascript)
//!▫~•◦-------------------------------------------------------------------‣
//!
//! JSX is parsed by tree-sitter-javascript, but retains its own source dialect and registry
//! identity so callers can select it explicitly.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use super::super::adapter::{AdapterCapabilities, SourceAdapter, SourceInput};
use super::super::ir::ExtractionBatch;
use super::super::schema::Namespace;
use super::code_js::JsSourceAdapter;
use crate::HNSQRResult;

pub struct JsxSourceAdapter;

impl Default for JsxSourceAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl JsxSourceAdapter {
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl SourceAdapter for JsxSourceAdapter {
    fn name(&self) -> &'static str {
        "jsx_treesitter_adapter"
    }

    fn version(&self) -> &'static str {
        "1.0.0"
    }

    fn capabilities(&self) -> AdapterCapabilities {
        JsSourceAdapter::new().capabilities()
    }

    fn detect(&self, source: &SourceInput) -> bool {
        source.locator.ends_with(".jsx") || source.source_type == "jsx"
    }

    fn extract(&self, source: &SourceInput, namespace: &Namespace) -> HNSQRResult<ExtractionBatch> {
        let mut batch = JsSourceAdapter::new().extract(source, namespace)?;
        batch.source.source_type = "jsx".to_string();
        Ok(batch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contextgraph::schema::{EntityKind, Namespace};

    #[test]
    fn extracts_a_jsx_component() {
        let input = SourceInput::from_text(
            "function Banner() { return <section>hello</section>; }",
            "file:///Banner.jsx",
            "jsx",
        );
        let batch = JsxSourceAdapter::new()
            .extract(&input, &Namespace::new("test"))
            .unwrap();

        assert!(batch
            .entities
            .iter()
            .any(|entity| entity.kind == EntityKind::code_function() && entity.label == "Banner"));
        assert_eq!(batch.source.source_type, "jsx");
    }
}
