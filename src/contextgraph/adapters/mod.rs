/* holosphere/src/contextgraph/adapters/mod.rs */
//!▫~•◦-------------------------------‣
//! # Source Adapters Registry & Concrete Implementations
//!▫~•◦-------------------------------------------------------------------‣
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

pub mod code_rust;
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
