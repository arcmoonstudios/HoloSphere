/* holosphere/src/contextgraph/watcher.rs */
//!▫~•◦-------------------------------‣
//! # Live Background Workspace Watcher & Invalidation Coordinator
//!▫~•◦-------------------------------------------------------------------‣
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use super::adapters::fs::FilesystemSourceAdapter;
use super::compiler::ContextCompiler;
use super::manifest::ContextGraphManifest;
use super::schema::Namespace;
use super::store::ContextGraphStore;
use crate::HNSQRResult;

pub struct ContextGraphWatcher {
    namespace: Namespace,
    root_path: PathBuf,
    compiler: ContextCompiler,
    store: Arc<ContextGraphStore>,
    running: Arc<AtomicBool>,
}

impl ContextGraphWatcher {
    #[must_use]
    pub fn new(
        namespace: Namespace,
        root_path: impl AsRef<Path>,
        store: Arc<ContextGraphStore>,
    ) -> Self {
        Self {
            namespace,
            root_path: root_path.as_ref().to_path_buf(),
            compiler: ContextCompiler::default(),
            store,
            running: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Performs one scan-and-compile update against the workspace.
    pub fn poll_once(&self, _manifest: &mut ContextGraphManifest) -> HNSQRResult<bool> {
        let fs_adapter = FilesystemSourceAdapter::new();
        let sources = fs_adapter.crawl_directory(&self.root_path)?;
        let output = self.compiler.compile(&self.namespace, &sources)?;
        self.store.commit_delta(output.into_delta());
        Ok(true)
    }

    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
    }
}
