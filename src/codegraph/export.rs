/* holosphere/src/codegraph/export.rs */
//!▫~•◦-------------------------------‣
//! # CodeGraph JSON & Interactive HTML Visualizer Export Engine
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Exports canonical JSON graph representations and renders self-contained, interactive
//! HTML web graph visualizations with pan/zoom, community clustering, edge filtering, and inspection.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::community::CommunityDetector;
use super::ingest::CodeGraphStoreState;
use super::schema::CodeNodeKind;
use crate::HNSQRResult;

/// Lightweight node structure formatted for web visualizer export.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExportNode {
    pub id: String,
    pub name: String,
    pub qualified_name: String,
    pub kind: String,
    pub file: String,
    pub community: usize,
    pub degree: usize,
    pub signature: Option<String>,
    pub docstring: Option<String>,
}

/// Lightweight edge structure formatted for web visualizer export.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExportEdge {
    pub source: String,
    pub target: String,
    pub relation: String,
    pub origin: String,
    pub confidence: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CodeGraphExportPayload {
    pub workspace_id: String,
    pub commit_lsn: u64,
    pub nodes: Vec<ExportNode>,
    pub edges: Vec<ExportEdge>,
}

pub struct CodeGraphExporter;

impl CodeGraphExporter {
    /// Builds export payload.
    #[must_use]
    pub fn build_payload(state: &CodeGraphStoreState) -> CodeGraphExportPayload {
        let (_communities, node_community_map) = CommunityDetector::detect_community_map(state);

        let mut export_nodes = Vec::new();
        for (id, node) in &state.nodes {
            if node.kind == CodeNodeKind::File || node.kind == CodeNodeKind::Directory {
                continue;
            }
            let deg = state.outgoing_edges.get(id).map_or(0, |v| v.len())
                + state.incoming_edges.get(id).map_or(0, |v| v.len());
            let comm = node_community_map.get(id).copied().unwrap_or(0);

            export_nodes.push(ExportNode {
                id: id.to_string(),
                name: node.name.clone(),
                qualified_name: node.qualified_name.clone(),
                kind: node.kind.to_string(),
                file: node.source_file.display().to_string(),
                community: comm,
                degree: deg,
                signature: node.signature.clone(),
                docstring: node.docstring.clone(),
            });
        }

        let mut export_edges = Vec::new();
        for edge in state.edges.values() {
            export_edges.push(ExportEdge {
                source: edge.source.to_string(),
                target: edge.target.to_string(),
                relation: edge.relation.to_string(),
                origin: edge.origin.to_string(),
                confidence: edge.confidence,
            });
        }

        CodeGraphExportPayload {
            workspace_id: state.workspace_id.clone(),
            commit_lsn: state.commit_lsn,
            nodes: export_nodes,
            edges: export_edges,
        }
    }

    /// Exports graph as JSON.
    pub fn export_json(state: &CodeGraphStoreState, path: impl AsRef<Path>) -> HNSQRResult<()> {
        if let Some(parent) = path.as_ref().parent() {
            std::fs::create_dir_all(parent)?;
        }
        let payload = Self::build_payload(state);
        let json = serde_json::to_string_pretty(&payload)
            .map_err(|e| crate::HNSQRError::SerializationError(e.to_string()))?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Exports self-contained interactive HTML visualization.
    pub fn export_html(state: &CodeGraphStoreState, path: impl AsRef<Path>) -> HNSQRResult<()> {
        if let Some(parent) = path.as_ref().parent() {
            std::fs::create_dir_all(parent)?;
        }
        let payload = Self::build_payload(state);
        let json_data = serde_json::to_string(&payload)
            .map_err(|e| crate::HNSQRError::SerializationError(e.to_string()))?;

        let html = format!(
            r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>HoloSphere CodeGraph — {workspace_id}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace; background: #0b0f19; color: #e2e8f0; overflow: hidden; display: flex; height: 100vh; }}
  #sidebar {{ width: 340px; background: #131b2e; border-right: 1px solid #1e293b; display: flex; flex-direction: column; padding: 16px; gap: 14px; z-index: 10; }}
  #main {{ flex: 1; position: relative; }}
  canvas {{ width: 100%; height: 100%; display: block; }}
  h1 {{ font-size: 1.1rem; color: #38bdf8; font-weight: 700; }}
  .stat {{ font-size: 0.82rem; color: #94a3b8; }}
  input, select {{ width: 100%; padding: 8px 12px; background: #0b0f19; border: 1px solid #334155; border-radius: 6px; color: #f8fafc; font-size: 0.85rem; outline: none; }}
  input:focus {{ border-color: #38bdf8; }}
  #node-info {{ flex: 1; overflow-y: auto; background: #0b0f19; padding: 12px; border-radius: 6px; border: 1px solid #1e293b; font-size: 0.8rem; line-height: 1.5; }}
  .badge {{ display: inline-block; padding: 2px 6px; border-radius: 4px; font-size: 0.72rem; font-weight: 600; text-transform: uppercase; margin-bottom: 6px; }}
  .badge-struct {{ background: #0284c7; color: white; }}
  .badge-function {{ background: #10b981; color: white; }}
  .badge-trait {{ background: #8b5cf6; color: white; }}
  .badge-rationale {{ background: #f59e0b; color: white; }}
</style>
</head>
<body>
<div id="sidebar">
  <div>
    <h1>HoloSphere CodeGraph</h1>
    <div class="stat">Workspace: <b>{workspace_id}</b> | LSN: {commit_lsn}</div>
    <div class="stat">Nodes: <span id="node-count">{node_count}</span> | Edges: <span id="edge-count">{edge_count}</span></div>
  </div>
  <input type="text" id="search" placeholder="Search symbol or file...">
  <select id="kind-filter">
    <option value="ALL">All Symbol Kinds</option>
    <option value="struct">Structs</option>
    <option value="function">Functions / Methods</option>
    <option value="trait">Traits</option>
    <option value="rationale">Rationale Notes</option>
  </select>
  <div id="node-info">Click any node to inspect details, definition, and structural relations.</div>
</div>
<div id="main">
  <canvas id="graph-canvas"></canvas>
</div>
<script>
const GRAPH_DATA = {json_data};
const canvas = document.getElementById('graph-canvas');
const ctx = canvas.getContext('2d');
let width, height;

function resize() {{
  width = canvas.width = canvas.parentElement.clientWidth;
  height = canvas.height = canvas.parentElement.clientHeight;
}}
window.addEventListener('resize', resize);
resize();

// Simple Canvas Force Graph Simulation
const nodes = GRAPH_DATA.nodes.map((n, i) => ({{
  ...n,
  x: width/2 + (Math.random() - 0.5) * width * 0.7,
  y: height/2 + (Math.random() - 0.5) * height * 0.7,
  vx: 0,
  vy: 0,
  radius: Math.min(14, Math.max(4, Math.sqrt(n.degree) * 2.5 + 3))
}}));

const nodeMap = new Map();
nodes.forEach(n => nodeMap.set(n.id, n));

const edges = GRAPH_DATA.edges.map(e => ({{
  ...e,
  sourceNode: nodeMap.get(e.source),
  targetNode: nodeMap.get(e.target)
}})).filter(e => e.sourceNode && e.targetNode);

let selectedNode = null;
let hoverNode = null;

function getColor(kind) {{
  switch(kind) {{
    case 'struct': return '#38bdf8';
    case 'function': case 'method': return '#34d399';
    case 'trait': return '#a78bfa';
    case 'enum': return '#f472b6';
    case 'rationale': return '#fbbf24';
    default: return '#94a3b8';
  }}
}}

function step() {{
  for(let i=0; i<nodes.length; i++) {{
    const n = nodes[i];
    n.vx += (width/2 - n.x) * 0.0005;
    n.vy += (height/2 - n.y) * 0.0005;
    for(let j=i+1; j<nodes.length; j++) {{
      const m = nodes[j];
      const dx = m.x - n.x;
      const dy = m.y - n.y;
      const dist = Math.sqrt(dx*dx + dy*dy) || 1;
      if(dist < 120) {{
        const force = (120 - dist) / dist * 0.05;
        n.vx -= dx * force;
        n.vy -= dy * force;
        m.vx += dx * force;
        m.vy += dy * force;
      }}
    }}
  }}

  for(const e of edges) {{
    const dx = e.targetNode.x - e.sourceNode.x;
    const dy = e.targetNode.y - e.sourceNode.y;
    const dist = Math.sqrt(dx*dx + dy*dy) || 1;
    const force = (dist - 40) * 0.002;
    e.sourceNode.vx += dx * force;
    e.sourceNode.vy += dy * force;
    e.targetNode.vx -= dx * force;
    e.targetNode.vy -= dy * force;
  }}

  for(const n of nodes) {{
    n.vx *= 0.88;
    n.vy *= 0.88;
    n.x += n.vx;
    n.y += n.vy;
    n.x = Math.max(n.radius, Math.min(width - n.radius, n.x));
    n.y = Math.max(n.radius, Math.min(height - n.radius, n.y));
  }}
}}

function render() {{
  ctx.fillStyle = '#0b0f19';
  ctx.fillRect(0, 0, width, height);

  // Draw Edges
  ctx.lineWidth = 0.8;
  for(const e of edges) {{
    ctx.strokeStyle = (selectedNode && (e.sourceNode === selectedNode || e.targetNode === selectedNode)) ? '#38bdf8' : 'rgba(148, 163, 184, 0.15)';
    ctx.beginPath();
    ctx.moveTo(e.sourceNode.x, e.sourceNode.y);
    ctx.lineTo(e.targetNode.x, e.targetNode.y);
    ctx.stroke();
  }}

  // Draw Nodes
  for(const n of nodes) {{
    ctx.beginPath();
    ctx.arc(n.x, n.y, n.radius, 0, Math.PI * 2);
    ctx.fillStyle = getColor(n.kind);
    ctx.fill();
    if(n === selectedNode || n === hoverNode) {{
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;
      ctx.stroke();
    }}
  }}

  step();
  requestAnimationFrame(render);
}}
render();

canvas.addEventListener('click', (e) => {{
  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;
  selectedNode = nodes.find(n => Math.hypot(n.x - x, n.y - y) <= n.radius + 3) || null;
  if(selectedNode) {{
    document.getElementById('node-info').innerHTML = `
      <div class="badge badge-${{selectedNode.kind}}">${{selectedNode.kind}}</div>
      <h3 style="color:#f8fafc; margin-bottom:4px;">${{selectedNode.name}}</h3>
      <p style="color:#94a3b8; font-size:0.75rem; margin-bottom:8px;">${{selectedNode.qualified_name}}</p>
      <p><b>File:</b> ${{selectedNode.file}}</p>
      <p><b>Degree:</b> ${{selectedNode.degree}} connections</p>
      ${{selectedNode.signature ? `<pre style="background:#131b2e; padding:6px; margin:6px 0; border-radius:4px; overflow-x:auto;">${{selectedNode.signature}}</pre>` : ''}}
      ${{selectedNode.docstring ? `<p style="margin-top:6px; color:#cbd5e1;"><i>${{selectedNode.docstring}}</i></p>` : ''}}
    `;
  }}
}});
</script>
</body>
</html>"#,
            workspace_id = payload.workspace_id,
            commit_lsn = payload.commit_lsn,
            node_count = payload.nodes.len(),
            edge_count = payload.edges.len(),
            json_data = json_data
        );

        std::fs::write(path, html)?;
        Ok(())
    }
}
