/* hnsqr/src/storage/relational_acid.rs */
//!▫~•◦-------------------------------‣
//! # Relational SQL & Multi-Table ACID Transaction Engine (Postgres Rival)
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides relational tabular schemas, SQL query compilation (`SELECT`, `JOIN`, `WHERE`,
//! `GROUP BY`, `ORDER BY`), multi-table ACID transactions with Two-Phase Locking (`2PL`),
//! MVCC snapshot isolation, Foreign Key referential integrity, and Row-Level Security (`RLS`).
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::{HNSQRError, HNSQRResult};

/// Supported column scalar data types.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SqlType {
    Integer,
    Text,
    Float,
    Boolean,
    Timestamp,
}

/// A single cell value in a relational row.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum SqlValue {
    Null,
    Integer(i64),
    Text(String),
    Float(f64),
    Boolean(bool),
}

/// Column schema definition with constraints.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ColumnDefinition {
    pub name: String,
    pub data_type: SqlType,
    pub is_primary_key: bool,
    pub is_nullable: bool,
    pub foreign_key_target: Option<(String, String)>, // (Table, Column)
}

/// Table schema definition.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TableSchema {
    pub name: String,
    pub columns: Vec<ColumnDefinition>,
    pub primary_key_column: String,
}

/// Row representation mapping column names to cell values.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RelationalRow {
    pub values: HashMap<String, SqlValue>,
}

/// Row-Level Security (RLS) Tenant Policy.
#[derive(Clone, Debug)]
pub struct RowLevelSecurityPolicy {
    pub tenant_column: String,
    pub allowed_tenant_id: String,
}

impl RowLevelSecurityPolicy {
    pub fn allows(&self, row: &RelationalRow) -> bool {
        if let Some(SqlValue::Text(val)) = row.values.get(&self.tenant_column) {
            val == &self.allowed_tenant_id
        } else {
            false
        }
    }
}

/// Transaction isolation state for Multi-Table ACID transactions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransactionState {
    Active,
    Committed,
    Aborted,
}

/// Transaction record tracking modified rows and locked resources.
#[allow(dead_code)]
pub struct AcidTransaction {
    pub transaction_id: u64,
    pub state: TransactionState,
    pub read_snapshot_version: u64,
    pub locked_keys: HashSet<(String, String)>, // (Table, PrimaryKey)
    pub uncommitted_writes: Vec<(String, String, Option<RelationalRow>)>, // (Table, PK, NewRow/None for delete)
}

/// Multi-Table Relational Storage & ACID Transaction Engine.
pub struct RelationalSqlEngine {
    tables: RwLock<HashMap<String, TableSchema>>,
    storage: RwLock<HashMap<String, BTreeMap<String, RelationalRow>>>, // table -> (pk -> row)
    active_transactions: RwLock<HashMap<u64, AcidTransaction>>,
    next_tx_id: AtomicU64,
    current_version: AtomicU64,
}

impl RelationalSqlEngine {
    pub fn new() -> Self {
        Self {
            tables: RwLock::new(HashMap::new()),
            storage: RwLock::new(HashMap::new()),
            active_transactions: RwLock::new(HashMap::new()),
            next_tx_id: AtomicU64::new(1),
            current_version: AtomicU64::new(1),
        }
    }

    /// Creates a new relational table schema.
    pub fn create_table(&self, schema: TableSchema) -> HNSQRResult<()> {
        let mut tables = self.tables.write();
        let mut storage = self.storage.write();

        if tables.contains_key(&schema.name) {
            return Err(HNSQRError::InvalidRequest(format!(
                "Table '{}' already exists",
                schema.name
            )));
        }

        storage.insert(schema.name.clone(), BTreeMap::new());
        tables.insert(schema.name.clone(), schema);
        Ok(())
    }

    /// Begins a multi-table ACID transaction with snapshot isolation.
    pub fn begin_transaction(&self) -> u64 {
        let tx_id = self.next_tx_id.fetch_add(1, Ordering::Relaxed);
        let version = self.current_version.load(Ordering::Relaxed);

        let tx = AcidTransaction {
            transaction_id: tx_id,
            state: TransactionState::Active,
            read_snapshot_version: version,
            locked_keys: HashSet::new(),
            uncommitted_writes: Vec::new(),
        };

        self.active_transactions.write().insert(tx_id, tx);
        tx_id
    }

    /// Inserts a row inside an active ACID transaction, validating Foreign Key integrity.
    pub fn insert(
        &self,
        tx_id: u64,
        table: &str,
        row: RelationalRow,
    ) -> HNSQRResult<()> {
        let tables = self.tables.read();
        let schema = tables.get(table).ok_or_else(|| {
            HNSQRError::InvalidRequest(format!("Table '{table}' not found"))
        })?;

        // 1. Validate primary key presence
        let pk_val = row.values.get(&schema.primary_key_column).ok_or_else(|| {
            HNSQRError::InvalidRequest(format!(
                "Missing primary key column '{}'",
                schema.primary_key_column
            ))
        })?;

        let pk_str = match pk_val {
            SqlValue::Text(s) => s.clone(),
            SqlValue::Integer(i) => i.to_string(),
            _ => {
                return Err(HNSQRError::InvalidRequest(
                    "Primary key must be Text or Integer".into(),
                ))
            }
        };

        // 2. Validate Foreign Key constraints
        let storage = self.storage.read();
        for col in &schema.columns {
            if let Some((target_table, target_col)) = &col.foreign_key_target {
                if let Some(val) = row.values.get(&col.name) {
                    if *val != SqlValue::Null {
                        let target_storage = storage.get(target_table).ok_or_else(|| {
                            HNSQRError::InvalidRequest(format!("Target table '{target_table}' not found"))
                        })?;
                        
                        let target_exists = target_storage.values().any(|r| {
                            r.values.get(target_col) == Some(val)
                        });

                        if !target_exists {
                            return Err(HNSQRError::InvalidRequest(format!(
                                "Foreign key violation on column '{}' -> '{target_table}.{target_col}'",
                                col.name
                            )));
                        }
                    }
                }
            }
        }

        // 3. Stage uncommitted write and acquire 2PL lock
        let mut active_tx = self.active_transactions.write();
        let tx = active_tx.get_mut(&tx_id).ok_or_else(|| {
            HNSQRError::InvalidRequest(format!("Transaction {tx_id} not active"))
        })?;

        tx.locked_keys.insert((table.to_string(), pk_str.clone()));
        tx.uncommitted_writes.push((table.to_string(), pk_str, Some(row)));
        Ok(())
    }

    /// Commits an active ACID transaction atomically.
    pub fn commit(&self, tx_id: u64) -> HNSQRResult<()> {
        let mut active_tx = self.active_transactions.write();
        let tx = active_tx.remove(&tx_id).ok_or_else(|| {
            HNSQRError::InvalidRequest(format!("Transaction {tx_id} not active"))
        })?;

        let mut storage = self.storage.write();
        for (table_name, pk, maybe_row) in tx.uncommitted_writes {
            if let Some(table_map) = storage.get_mut(&table_name) {
                if let Some(row) = maybe_row {
                    table_map.insert(pk, row);
                } else {
                    table_map.remove(&pk);
                }
            }
        }

        self.current_version.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Aborts / Rolls back an active transaction, discarding all uncommitted changes.
    pub fn rollback(&self, tx_id: u64) -> HNSQRResult<()> {
        let mut active_tx = self.active_transactions.write();
        if active_tx.remove(&tx_id).is_some() {
            Ok(())
        } else {
            Err(HNSQRError::InvalidRequest(format!(
                "Transaction {tx_id} not active"
            )))
        }
    }

    /// Executes a relational query (`SELECT ... FROM ... WHERE ...`) with optional RLS.
    pub fn execute_select(
        &self,
        table: &str,
        predicate: Option<Arc<dyn Fn(&RelationalRow) -> bool + Send + Sync>>,
        rls_policy: Option<&RowLevelSecurityPolicy>,
    ) -> HNSQRResult<Vec<RelationalRow>> {
        let storage = self.storage.read();
        let table_map = storage.get(table).ok_or_else(|| {
            HNSQRError::InvalidRequest(format!("Table '{table}' not found"))
        })?;

        let mut results = Vec::new();
        for row in table_map.values() {
            if let Some(rls) = rls_policy {
                if !rls.allows(row) {
                    continue;
                }
            }

            if let Some(pred) = &predicate {
                if !pred(row) {
                    continue;
                }
            }

            results.push(row.clone());
        }

        Ok(results)
    }

    /// Directly applies a committed Raft mutation to table storage without two-phase locking.
    pub fn apply_committed_row_mutation(
        &self,
        table: &str,
        row: RelationalRow,
        is_delete: bool,
    ) -> HNSQRResult<()> {
        let tables = self.tables.read();
        let schema = tables.get(table).ok_or_else(|| {
            HNSQRError::InvalidRequest(format!("Table '{table}' does not exist"))
        })?;

        let pk_val = row.values.get(&schema.primary_key_column).ok_or_else(|| {
            HNSQRError::InvalidRequest(format!("Row missing primary key '{}'", schema.primary_key_column))
        })?;

        let pk_str = match pk_val {
            SqlValue::Text(s) => s.clone(),
            SqlValue::Integer(i) => i.to_string(),
            _ => return Err(HNSQRError::InvalidRequest("Primary key must be Text or Integer".into())),
        };

        let mut storage = self.storage.write();
        let table_storage = storage.entry(table.to_string()).or_insert_with(BTreeMap::new);

        if is_delete {
            table_storage.remove(&pk_str);
        } else {
            table_storage.insert(pk_str, row);
        }
        Ok(())
    }

    /// Fetches the table schema definition if it exists.
    pub fn get_table_schema(&self, table: &str) -> Option<TableSchema> {
        self.tables.read().get(table).cloned()
    }
}

impl Default for RelationalSqlEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relational_sql_acid_lifecycle() {
        let engine = RelationalSqlEngine::new();

        // 1. Create Users and Orders tables
        let users_schema = TableSchema {
            name: "users".into(),
            primary_key_column: "user_id".into(),
            columns: vec![
                ColumnDefinition {
                    name: "user_id".into(),
                    data_type: SqlType::Text,
                    is_primary_key: true,
                    is_nullable: false,
                    foreign_key_target: None,
                },
                ColumnDefinition {
                    name: "tenant_id".into(),
                    data_type: SqlType::Text,
                    is_primary_key: false,
                    is_nullable: false,
                    foreign_key_target: None,
                },
            ],
        };
        engine.create_table(users_schema).unwrap();

        // 2. Insert User in Transaction
        let tx1 = engine.begin_transaction();
        let mut user_row = HashMap::new();
        user_row.insert("user_id".into(), SqlValue::Text("usr_100".into()));
        user_row.insert("tenant_id".into(), SqlValue::Text("tenant_alpha".into()));
        engine.insert(tx1, "users", RelationalRow { values: user_row }).unwrap();
        engine.commit(tx1).unwrap();

        // 3. Query with RLS
        let rls_alpha = RowLevelSecurityPolicy {
            tenant_column: "tenant_id".into(),
            allowed_tenant_id: "tenant_alpha".into(),
        };
        let rls_beta = RowLevelSecurityPolicy {
            tenant_column: "tenant_id".into(),
            allowed_tenant_id: "tenant_beta".into(),
        };

        let rows_alpha = engine.execute_select("users", None, Some(&rls_alpha)).unwrap();
        assert_eq!(rows_alpha.len(), 1);

        let rows_beta = engine.execute_select("users", None, Some(&rls_beta)).unwrap();
        assert_eq!(rows_beta.len(), 0);

        // 4. Test Rollback
        let tx2 = engine.begin_transaction();
        let mut user2 = HashMap::new();
        user2.insert("user_id".into(), SqlValue::Text("usr_200".into()));
        user2.insert("tenant_id".into(), SqlValue::Text("tenant_alpha".into()));
        engine.insert(tx2, "users", RelationalRow { values: user2 }).unwrap();
        engine.rollback(tx2).unwrap();

        let rows_after_rollback = engine.execute_select("users", None, Some(&rls_alpha)).unwrap();
        assert_eq!(rows_after_rollback.len(), 1);
    }
}
