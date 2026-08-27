# HoloSphere ContextGraph Report — `workspace:holosphere`

## Universal Metrics
- **Total Entities:** 261
- **Total Relations:** 301
- **Detected Scopes:** 6
- **Canonical Graph Fingerprint:** `[3b, 3b, 3c, 55, c4, fc, 1d, b0, 6a, cf, f4, 3f, b2, 1d, eb, bd, 1, 79, cf, d5, 19, 6c, cc, 34, 3c, 84, bc, d2, 7d, 93, 0, 24]`
- **Commit LSN:** 1

## Top Centrality Hubs
| Entity Label | Kind | Total Connections |
| :--- | :--- | :--- |
| `file:///src/contextgraph/schema.rs` | `code:file` | **68** |
| `file:///src/contextgraph/adapters/code_rust.rs` | `code:file` | **24** |
| `file:///src/contextgraph/mod.rs` | `code:file` | **16** |
| `file:///src/contextgraph/adapters/fs.rs` | `code:file` | **10** |
| `file:///src/contextgraph/query.rs` | `code:file` | **10** |
| `file:///src/contextgraph/store.rs` | `code:file` | **10** |
| `RustSourceAdapter::text_of` | `code:function` | **9** |
| `file:///src/contextgraph/ir.rs` | `code:file` | **9** |
| `file:///src/contextgraph/manifest.rs` | `code:file` | **9** |
| `MarkdownSourceAdapter::extract` | `code:function` | **8** |

## Topological Scopes
### Scope: ContextGraphWatcher::new
- **Entities:** 184
- **Key Entities:** ContextGraphWatcher::new, FilesystemSourceAdapter::capabilities, ContextGraphWatcher::poll_once, UniversalReferenceResolver, file:///src/contextgraph/ir.rs

### Scope: HtmlVisualizerView
- **Entities:** 35
- **Key Entities:** HtmlVisualizerView, GraphView, MarkdownReportView, file:///src/contextgraph/community.rs, JsonExportView::render

### Scope: adapters
- **Entities:** 17
- **Key Entities:** adapters, watcher, query, schema, planner

### Scope: ContextQueryEngine
- **Entities:** 11
- **Key Entities:** ContextQueryEngine, ContextQueryEngine::search, ContextQueryEngine::explore, ContextQueryEngine::impact, ContextQueryEngine::path

### Scope: QueryPlanner
- **Entities:** 7
- **Key Entities:** QueryPlanner, ContextBudget, ContextQueryRequest, file:///src/contextgraph/planner.rs, ContextBudget::default

### Scope: file:///src/contextgraph/invalidation.rs
- **Entities:** 7
- **Key Entities:** file:///src/contextgraph/invalidation.rs, InvalidationGraph::compute_affected_scope, InvalidationGraph::invalidate_locators, InvalidationGraph::new, InvalidationGraph::register_dependency
