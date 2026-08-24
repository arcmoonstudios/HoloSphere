/* holosphere/src/learning/inference/registry.rs */
//!▫~•◦-------------------------------‣
//! # Inference Method Registry
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides the thread-safe registry of all admitted inference generators.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

use crate::learning::inference::contract::{InferenceMethod, InferenceMethodId};

/// Thread-safe registry for inference hypothesis generators.
#[derive(Default)]
pub struct InferenceRegistry {
    methods: RwLock<HashMap<InferenceMethodId, Arc<dyn InferenceMethod>>>,
}

impl InferenceRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers an inference method into the registry.
    pub fn register(&self, method: Arc<dyn InferenceMethod>) {
        self.methods.write().insert(method.id(), method);
    }

    /// Retrieves an inference method by ID.
    pub fn get(&self, id: InferenceMethodId) -> Option<Arc<dyn InferenceMethod>> {
        self.methods.read().get(&id).cloned()
    }

    /// Returns all registered method IDs.
    pub fn list_methods(&self) -> Vec<InferenceMethodId> {
        self.methods.read().keys().copied().collect()
    }
}
