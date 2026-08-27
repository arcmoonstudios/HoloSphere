/* holosphere/src/contextgraph/views/html.rs */
//!▫~•◦-------------------------------‣
//! # Universal Interactive HTML Graph Visualizer View
//!▫~•◦-------------------------------------------------------------------‣
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::path::Path;

use super::super::store::ContextGraphStoreState;
use super::GraphView;
use crate::HNSQRResult;

pub struct HtmlVisualizerView;

impl Default for HtmlVisualizerView {
    fn default() -> Self {
        Self::new()
    }
}

impl HtmlVisualizerView {
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    #[must_use]
    pub fn generate_html(state: &ContextGraphStoreState) -> String {
        let json = serde_json::to_string(state).unwrap_or_else(|_| "{}".to_string());
        format!(
            r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>HoloSphere ContextGraph — {ns}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace; background: #0b0f19; color: #e2e8f0; display: flex; height: 100vh; overflow: hidden; }}
  #sidebar {{ width: 340px; background: #131b2e; border-right: 1px solid #1e293b; display: flex; flex-direction: column; padding: 16px; gap: 14px; z-index: 10; }}
  #main {{ flex: 1; position: relative; }}
  canvas {{ width: 100%; height: 100%; display: block; }}
  h1 {{ font-size: 1.1rem; color: #38bdf8; font-weight: 700; }}
  .stat {{ font-size: 0.82rem; color: #94a3b8; }}
  input {{ width: 100%; padding: 8px 12px; background: #0b0f19; border: 1px solid #334155; border-radius: 6px; color: #f8fafc; font-size: 0.85rem; outline: none; }}
  #details {{ flex: 1; overflow-y: auto; background: #0b0f19; padding: 12px; border-radius: 6px; border: 1px solid #1e293b; font-size: 0.8rem; line-height: 1.5; }}
</style>
</head>
<body>
<div id="sidebar">
  <div>
    <h1>HoloSphere ContextGraph</h1>
    <div class="stat">Namespace: <b>{ns}</b> | LSN: {lsn}</div>
    <div class="stat">Entities: {ent_count} | Relations: {rel_count}</div>
  </div>
  <input type="text" id="search" placeholder="Search entity...">
  <div id="details">Click any node to inspect details, relations, and provenance.</div>
</div>
<div id="main">
  <canvas id="graph-canvas"></canvas>
</div>
<script>
const GRAPH_STATE = {json};
const canvas = document.getElementById('graph-canvas');
const ctx = canvas.getContext('2d');
let width = canvas.width = canvas.parentElement.clientWidth;
let height = canvas.height = canvas.parentElement.clientHeight;

window.addEventListener('resize', () => {{
  width = canvas.width = canvas.parentElement.clientWidth;
  height = canvas.height = canvas.parentElement.clientHeight;
}});

const entities = Object.values(GRAPH_STATE.entities || {{}}).map(e => ({{
  ...e,
  x: width/2 + (Math.random() - 0.5) * width * 0.7,
  y: height/2 + (Math.random() - 0.5) * height * 0.7,
  vx: 0, vy: 0,
  radius: 6
}}));

function render() {{
  ctx.fillStyle = '#0b0f19';
  ctx.fillRect(0, 0, width, height);

  for(const e of entities) {{
    ctx.beginPath();
    ctx.arc(e.x, e.y, e.radius, 0, Math.PI * 2);
    ctx.fillStyle = e.kind.startsWith('code') ? '#38bdf8' : (e.kind.startsWith('doc') ? '#34d399' : '#f59e0b');
    ctx.fill();
  }}
  requestAnimationFrame(render);
}}
render();
</script>
</body>
</html>"#,
            ns = state.namespace,
            lsn = state.commit_lsn,
            ent_count = state.entities.len(),
            rel_count = state.relations.len(),
            json = json
        )
    }

    pub fn write_to_file(
        state: &ContextGraphStoreState,
        path: impl AsRef<Path>,
    ) -> HNSQRResult<()> {
        if let Some(parent) = path.as_ref().parent() {
            std::fs::create_dir_all(parent)?;
        }
        let html = Self::generate_html(state);
        std::fs::write(path, html)?;
        Ok(())
    }
}

impl GraphView for HtmlVisualizerView {
    fn render(&self, state: &ContextGraphStoreState) -> HNSQRResult<Vec<u8>> {
        Ok(Self::generate_html(state).into_bytes())
    }
}
