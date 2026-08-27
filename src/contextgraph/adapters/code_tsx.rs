/* holosphere/src/contextgraph/adapters/code_tsx.rs */
//!▫~•◦-------------------------------‣
//! # TypeScript JSX Context Adapter (tree-sitter-typescript)
//!▫~•◦-------------------------------------------------------------------‣
//!
//! TSX shares TypeScript's extraction contract while requiring the TSX grammar for JSX syntax.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use super::super::adapter::{AdapterCapabilities, SourceAdapter, SourceInput};
use super::super::ir::ExtractionBatch;
use super::super::schema::Namespace;
use super::code_ts::TsSourceAdapter;
use crate::HNSQRResult;

pub struct TsxSourceAdapter;

impl Default for TsxSourceAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl TsxSourceAdapter {
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl SourceAdapter for TsxSourceAdapter {
    fn name(&self) -> &'static str {
        "tsx_treesitter_adapter"
    }

    fn version(&self) -> &'static str {
        "1.0.0"
    }

    fn capabilities(&self) -> AdapterCapabilities {
        TsSourceAdapter::new().capabilities()
    }

    fn detect(&self, source: &SourceInput) -> bool {
        source.locator.ends_with(".tsx")
            || source.source_type == "tsx"
            || source.source_type == "typescriptreact"
    }

    fn extract(
        &self,
        source: &SourceInput,
        _namespace: &Namespace,
    ) -> HNSQRResult<ExtractionBatch> {
        TsSourceAdapter::extract_with_language(
            source,
            tree_sitter_typescript::LANGUAGE_TSX.into(),
            "tsx",
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contextgraph::schema::{EntityKind, Namespace};

    #[test]
    fn extracts_a_tsx_component_with_jsx() {
        let input = SourceInput::from_text(
            "export function Banner(): JSX.Element { return <section>hello</section>; }",
            "file:///Banner.tsx",
            "tsx",
        );

        let batch = TsxSourceAdapter::new()
            .extract(&input, &Namespace::new("test"))
            .unwrap();

        assert!(batch
            .entities
            .iter()
            .any(|entity| entity.kind == EntityKind::code_function() && entity.label == "Banner"));
    }
}
