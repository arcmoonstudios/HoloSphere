/* hnsqr/src/graph/query/morsel.rs */
//!▫~•◦-------------------------------‣
//! # Morsel — Columnar Intermediate Result Representation
//!▫~•◦-------------------------------------------------------------------‣
//!
//! A `Morsel` holds a batch of query result rows in columnar form.
//! One column exists per binding in the current plan scope.
//! A `SelectionVector` marks which rows are logically active so filters can
//! operate without physically removing rows until a compaction step.
//!
//! Example for pattern `(p)-[:WORKS_AT]->(c)<-[:INVESTED_IN]-(v)`:
//! ```text
//! p column:  [17, 17, 22, 31, ...]
//! c column:  [42, 48, 59, 60, ...]
//! v column:  [88, 91, 97, 99, ...]
//! selection: [true, true, false, true, ...]
//! ```
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣
 
use smallvec::SmallVec;

use crate::NodeIndex;

/// One column binding in a morsel.
#[derive(Clone, Debug)]
pub enum BindingColumn {
    /// Column of node indices.
    Node(Vec<NodeIndex>),
    /// Column of relationship IDs.
    Relationship(Vec<u32>),
    /// Column of f32 scalar values (e.g. edge weights, scores).
    Scalar(Vec<f32>),
}

impl BindingColumn {
    pub fn len(&self) -> usize {
        match self {
            Self::Node(v) => v.len(),
            Self::Relationship(v) => v.len(),
            Self::Scalar(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the node ID at position `i`, panicking if not a Node column.
    #[inline]
    pub fn node_at(&self, i: usize) -> NodeIndex {
        match self {
            Self::Node(v) => v[i],
            _ => panic!("BindingColumn::node_at called on non-Node column"),
        }
    }
}

/// A batch of query result rows in columnar form.
#[derive(Clone, Debug, Default)]
pub struct Morsel {
    /// Number of logical rows (including filtered-out rows).
    pub rows: usize,
    /// Columns in binding-declaration order.
    pub columns: SmallVec<[BindingColumn; 8]>,
    /// `true` at position `i` means row `i` is still active.
    pub selection: Vec<bool>,
}

impl Morsel {
    pub fn new_empty() -> Self {
        Self::default()
    }

    /// Creates a morsel from a single Node column (e.g. after a NodeScan).
    pub fn from_node_column(nodes: Vec<NodeIndex>) -> Self {
        let rows = nodes.len();
        let selection = vec![true; rows];
        let mut columns = SmallVec::new();
        columns.push(BindingColumn::Node(nodes));
        Self { rows, columns, selection }
    }

    /// Returns the number of logically active (non-filtered) rows.
    pub fn active_row_count(&self) -> usize {
        self.selection.iter().filter(|&&b| b).count()
    }

    /// Compacts the morsel by removing filtered-out rows.  Returns a new `Morsel`.
    pub fn compact(&self) -> Morsel {
        let active_indices: Vec<usize> = self
            .selection
            .iter()
            .enumerate()
            .filter_map(|(i, &b)| if b { Some(i) } else { None })
            .collect();

        let rows = active_indices.len();
        let selection = vec![true; rows];

        let columns = self
            .columns
            .iter()
            .map(|col| match col {
                BindingColumn::Node(v) => {
                    BindingColumn::Node(active_indices.iter().map(|&i| v[i]).collect())
                }
                BindingColumn::Relationship(v) => {
                    BindingColumn::Relationship(active_indices.iter().map(|&i| v[i]).collect())
                }
                BindingColumn::Scalar(v) => {
                    BindingColumn::Scalar(active_indices.iter().map(|&i| v[i]).collect())
                }
            })
            .collect();

        Morsel { rows, columns, selection }
    }

    /// Appends a new Node column (e.g. after an Expand step).
    pub fn push_node_column(&mut self, nodes: Vec<NodeIndex>) {
        assert_eq!(nodes.len(), self.rows, "Column length mismatch");
        self.columns.push(BindingColumn::Node(nodes));
    }

    /// Appends a new Scalar column.
    pub fn push_scalar_column(&mut self, values: Vec<f32>) {
        assert_eq!(values.len(), self.rows, "Column length mismatch");
        self.columns.push(BindingColumn::Scalar(values));
    }
}
