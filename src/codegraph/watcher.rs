/* holosphere/src/codegraph/watcher.rs */
//!▫~•◦-------------------------------‣
//! # Live Background Filesystem Watcher & Recompile Coordinator
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Watches repository workspace files and triggers debounced incremental compilation passes.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use super::incremental::IncrementalCompiler;
use super::ingest::CodeGraphStore;
use super::manifest::WorkspaceManifest;
use crate::HNSQRResult;

pub struct CodeGraphWatcher {
    workspace_id: String,
    workspace_root: PathBuf,
    compiler: IncrementalCompiler,
    store: Arc<CodeGraphStore>,
    running: Arc<AtomicBool>,
}

impl CodeGraphWatcher {
    #[must_use]
    pub fn new(
        workspace_id: impl Into<String>,
        workspace_root: impl AsRef<Path>,
        store: Arc<CodeGraphStore>,
    ) -> Self {
        Self {
            workspace_id: workspace_id.into(),
            workspace_root: workspace_root.as_ref().to_path_buf(),
            compiler: IncrementalCompiler::default(),
            store,
            running: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Performs one poll-and-recompile cycle against the workspace.
    pub fn poll_once(&self, manifest: &mut WorkspaceManifest) -> HNSQRResult<bool> {
        if let Some(delta) =
            self.compiler
                .compile_incremental(&self.workspace_id, &self.workspace_root, manifest)?
        {
            self.store.commit_delta(delta);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Stops any running watcher loop.
    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
    }
}
