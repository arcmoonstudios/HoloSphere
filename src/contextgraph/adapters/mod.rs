/* holosphere/src/contextgraph/adapters/mod.rs */
//!▫~•◦-------------------------------‣
//! # Source Adapters Registry & Concrete Implementations
//!▫~•◦-------------------------------------------------------------------‣
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

pub mod code_js;
pub mod code_jsx;
pub mod code_python;
pub mod code_rust;
pub mod code_ts;
pub mod code_tsx;
pub mod data_json;
pub mod data_toml;
pub mod fs;
pub mod markdown;

use std::sync::Arc;

use super::adapter::{SourceAdapter, SourceInput};

/// Registry of pluggable SourceAdapters.
pub struct AdapterRegistry {
    adapters: Vec<Arc<dyn SourceAdapter>>,
}

impl Default for AdapterRegistry {
    fn default() -> Self {
        let mut registry = Self {
            adapters: Vec::new(),
        };
        registry.register(Arc::new(code_rust::RustSourceAdapter::new()));
        registry.register(Arc::new(code_tsx::TsxSourceAdapter::new()));
        registry.register(Arc::new(code_ts::TsSourceAdapter::new()));
        registry.register(Arc::new(code_jsx::JsxSourceAdapter::new()));
        registry.register(Arc::new(code_js::JsSourceAdapter::new()));
        registry.register(Arc::new(code_python::PythonSourceAdapter::new()));
        registry.register(Arc::new(data_json::JsonSourceAdapter::new()));
        registry.register(Arc::new(data_toml::TomlSourceAdapter::new()));
        registry.register(Arc::new(markdown::MarkdownSourceAdapter::new()));
        registry.register(Arc::new(fs::FilesystemSourceAdapter::new()));
        registry
    }
}

impl AdapterRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, adapter: Arc<dyn SourceAdapter>) {
        self.adapters.push(adapter);
    }

    #[must_use]
    pub fn find_adapter(&self, source: &SourceInput) -> Option<Arc<dyn SourceAdapter>> {
        self.adapters.iter().find(|a| a.detect(source)).cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_selects_each_supported_programming_language() {
        let registry = AdapterRegistry::new();
        for (locator, source_type, expected) in [
            ("file:///module.rs", "rust", "rust_treesitter_adapter"),
            (
                "file:///module.ts",
                "typescript",
                "typescript_treesitter_adapter",
            ),
            ("file:///Component.tsx", "tsx", "tsx_treesitter_adapter"),
            ("file:///Component.jsx", "jsx", "jsx_treesitter_adapter"),
            (
                "file:///module.js",
                "javascript",
                "javascript_treesitter_adapter",
            ),
            ("file:///module.py", "python", "python_treesitter_adapter"),
            ("file:///package.json", "json", "json_structural_adapter"),
            ("file:///Cargo.toml", "toml", "toml_structural_adapter"),
        ] {
            let input = SourceInput::from_text("", locator, source_type);
            assert_eq!(registry.find_adapter(&input).unwrap().name(), expected);
        }
    }
}
