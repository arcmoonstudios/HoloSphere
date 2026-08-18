# HNSQR Enterprise Operational Runbooks & Incident Response SOPs

## 1. Raft Quorum Loss / Leader Election Failure

### Symptoms
- Writes failing with `HNSQRError::Internal("Node X is not leader")`.
- `hnsqr doctor` reports `⚠️ Quorum Election Pending`.

### Diagnosis & Triaging
1. Execute `hnsqr doctor` to inspect cluster membership and active terms.
2. Verify node connectivity across peer IP addresses.
3. If $N/2$ or more nodes are unreachable, restore network connectivity or downscale cluster membership via joint consensus:
   ```rust
   leader.propose(RaftCommand::MembershipChange { new_peers: healthy_nodes });
   ```

---

## 2. Low Disk Headroom & ENOSPC Fail-Safe Mode

### Symptoms
- Ingestion requests rejected with `Engine is in read-only fail-safe mode due to storage resource pressure`.

### Resolution Steps
1. Verify storage volume capacity with `df -h` or `hnsqr doctor`.
2. Trigger online snapshot checkpoint to prune older WAL segments:
   ```rust
   wal_manager.truncate_before(last_snapshot_lsn);
   ```
3. Run engine compaction to reclaim dead tombstones and metadata postings:
   ```rust
   engine.compact()?;
   ```
4. Reset read-only protection once headroom exceeds threshold:
   ```rust
   backpressure.set_read_only(false);
   ```

---

## 3. Disaster Recovery: Point-in-Time Recovery (PITR) Drill

### Procedure
1. Identify target recovery point (e.g., $LSN_{\text{target}} = 450000$ or UTC timestamp).
2. Locate the newest Full Backup prior to target LSN (`backup_full_gen_X`).
3. Locate subsequent Incremental Backup chain (`backup_inc_Y`).
4. Execute PITR restore into a clean directory:
   ```rust
   BackupManager::restore_pitr(
       backup_dir,
       target_restore_dir,
       "backup_full_gen_X",
       Some("backup_inc_Y"),
       target_lsn,
       |lsn, mutation| engine.apply(mutation),
   )?;
   ```
5. Run `hnsqr doctor` against `target_restore_dir` to confirm zero CRC failures.
