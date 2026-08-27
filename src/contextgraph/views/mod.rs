/* holosphere/src/contextgraph/views/mod.rs */
//!▫~•◦-------------------------------‣
//! # Graph Views Subsystem (Markdown, JSON, Interactive HTML)
//!▫~•◦-------------------------------------------------------------------‣
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

pub mod html;
pub mod json;
pub mod markdown;

use super::store::ContextGraphStoreState;
use crate::HNSQRResult;

pub trait GraphView: Send + Sync {
    fn render(&self, state: &ContextGraphStoreState) -> HNSQRResult<Vec<u8>>;
}
