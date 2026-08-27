/* holosphere/src/codegraph/languages/mod.rs */
//!▫~•◦-------------------------------‣
//! # Language Extractor Registry & Front-End Implementations
//!▫~•◦-------------------------------------------------------------------‣
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

pub mod rust;

use std::collections::HashMap;
use std::sync::Arc;

use super::parser::LanguageExtractor;
use super::schema::Language;
use rust::RustExtractor;

/// Registry of available language extractors.
pub struct LanguageRegistry {
    extractors: HashMap<Language, Arc<dyn LanguageExtractor>>,
}

impl Default for LanguageRegistry {
    fn default() -> Self {
        let mut registry = Self {
            extractors: HashMap::new(),
        };
        registry.register(Arc::new(RustExtractor::new()));
        registry
    }
}

impl LanguageRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, extractor: Arc<dyn LanguageExtractor>) {
        self.extractors.insert(extractor.language(), extractor);
    }

    #[must_use]
    pub fn get(&self, language: Language) -> Option<Arc<dyn LanguageExtractor>> {
        self.extractors.get(&language).cloned()
    }
}
