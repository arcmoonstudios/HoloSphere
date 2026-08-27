/* holosphere/src/contextgraph/views/json.rs */
//!▫~•◦-------------------------------‣
//! # Canonical JSON Export View
//!▫~•◦-------------------------------------------------------------------‣
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::path::Path;

use super::super::store::ContextGraphStoreState;
use super::GraphView;
use crate::HNSQRResult;

pub struct JsonExportView;

impl Default for JsonExportView {
    fn default() -> Self {
        Self::new()
    }
}

impl JsonExportView {
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    pub fn write_to_file(
        state: &ContextGraphStoreState,
        path: impl AsRef<Path>,
    ) -> HNSQRResult<()> {
        if let Some(parent) = path.as_ref().parent() {
            std::fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(state)
            .map_err(|e| crate::HNSQRError::SerializationError(e.to_string()))?;
        std::fs::write(path, json)?;
        Ok(())
    }
}

impl GraphView for JsonExportView {
    fn render(&self, state: &ContextGraphStoreState) -> HNSQRResult<Vec<u8>> {
        serde_json::to_vec_pretty(state)
            .map_err(|e| crate::HNSQRError::SerializationError(e.to_string()))
    }
}
