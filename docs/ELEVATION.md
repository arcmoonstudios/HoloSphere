HNSQR Phase 4 Directive — Cloud-Scale Throughput, Operability & Ecosystem

Implement every item below. Do not merely add APIs or configuration knobs; each change must include production wiring, failure handling, telemetry, tests, and benchmarks.

Break the Raft leader write bottleneck without weakening linearizability. The current leader must not serialize vector ingestion around one mutation, one network round-trip, or one fsync at a time. Implement leader-side append batching, pipelined AppendEntries, multiple in-flight replication windows, group WAL durability, contiguous LSN batch acknowledgement, follower disk batching, and adaptive batch sizing driven by queue depth and latency. Separate mutation admission from commit completion with bounded queues and backpressure. Preserve the invariant that a quorum-durable acknowledgement is returned only after the configured durability condition is actually satisfied. Benchmark writes/sec and p50/p95/p99 across 3-, 5-, and 7-node groups under 1, 8, 32, 128, and 512 concurrent writers. Add a controller that chooses batch size and flush cadence to minimize tail latency while maintaining throughput. Target throughput close to storage/network hardware limits rather than a fixed hand-tuned batch size.
Decouple Raft liveness from slow storage. A saturated WAL must never starve heartbeats and cause avoidable election churn. Put consensus timers, heartbeat transmission, replication scheduling, disk completion, snapshot transfer, and application callbacks on separated execution lanes or runtimes with explicit priorities. Heartbeats must remain schedulable when WAL queues are full. Track event-loop delay, heartbeat send delay, WAL queue depth, follower replication lag, fsync latency, election count, and leader tenure. Add fault tests where fsync latency is injected at 5 ms, 25 ms, 100 ms, and multi-second stalls; healthy peers must not trigger elections merely because the leader’s storage path is congested. If the leader cannot maintain safe timing, apply admission backpressure before Raft stability is endangered.
Introduce adaptive durability batching rather than requiring operators to tune NVMe physics manually. Build a DurabilityController using measured fsync latency, outstanding WAL bytes, mutation arrival rate, replication RTT, and requested SLA. Support an explicit maximum acknowledgement delay so group commit cannot optimize throughput by silently accumulating unacceptable latency. Let OperatorIntent expose semantic controls such as DurableLowLatency, DurableThroughput, and explicit maximum commit latency; derive WAL/group-commit parameters internally. Expose the resulting plan through EXPLAIN and telemetry.
Isolate snapshot/WAL compaction I/O from the query path. Dr. Cross identifies snapshot generation as a potential P99 killer. Implement an I/O budget manager covering WAL flush, compaction reads/writes, snapshot creation, backup upload, shard migration, and cold-segment fetches. Use token-bucket or feedback-controlled bandwidth limits per maintenance class, with query and Raft durability I/O receiving higher priority. Where hardware topology allows it, support separate WAL/snapshot/vector devices; where it does not, enforce queue-depth limits. Stream snapshots rather than constructing large intermediate files. Benchmark certified query p99 during worst-case compaction and require a bounded degradation budget.
Make maintenance self-throttling. Snapshotting, compaction, backup, migration and index rebuilding should automatically slow when foreground p95/p99, disk queue depth, replication lag, CPU pressure, or memory pressure cross thresholds. Resume automatically when pressure clears. The controller must include hysteresis so it does not oscillate between full-speed and paused states. Emit explicit telemetry explaining every throttle decision.
Eliminate dangerous eager-prefault behavior under cgroups. PrefaultMode::Eager must become guarded rather than blindly touching an entire mapping. Detect cgroup v2 memory limits, current RSS, working-set estimate, page-cache pressure where available, snapshot size, and configured reserve. Add PrefaultMode::Adaptive as the production default. It should prefault only the hot structures first—manifest, ProofTree, LUTz, Rivero directories—and progressively warm dense-vector pages according to available headroom. Add bounded parallel prefetch, cancellation, and page-touch rate limiting. Refuse full eager prefault when projected memory use breaches configured safety headroom rather than allowing Kubernetes to OOMKill the process.
Exploit HNSQR’s proof architecture for smarter warming. Do not warm immutable segments uniformly. Persist or derive block-access heatmaps and proof-region query frequency. On restart, warm the ProofTree and LUTz hierarchy first, then the highest-probability exact-vector pages. This should reduce time-to-first-low-latency-query compared with sequential full-segment prefaulting. Benchmark cold-start p50/p99 after 0%, 10%, 25%, and 100% warmup.
Automate certificate lifecycle management. verify_certificate_freshness is not enough. Add certificate rotation orchestration with overlapping old/new trust windows, hot reload without process restart, node identity validation, expiration telemetry, rotation audit records, and configurable renewal thresholds. Cluster membership must not collapse because every node reaches the same certificate expiration date. Support external issuers through a clean provider interface—Kubernetes cert-manager, SPIFFE/SPIRE, Vault or another future issuer—without coupling consensus logic to one PKI product. Test certificate rotation during active replication and shard migration.
Strengthen authentication beyond static tokens. Preserve API-key/Bearer support, but add pluggable OIDC validation, JWKS refresh, short-lived credentials, key rotation, scoped service accounts, and authorization policy caching with safe invalidation. Administrative operations such as membership changes, backup restore, key rotation and tenant deletion should require separate privileges from ordinary write access. Audit all security-sensitive operations.
Prevent metadata-cardinality attacks per tenant. Existing global quota protection is not enough for high-frequency SaaS. Track dictionary bytes, bitmap bytes, term count, field count, filter complexity, posting density, and compaction debt per tenant and collection. Add hard and soft budgets plus weighted admission costs so a tenant cannot create millions of one-off tag values while remaining under a simplistic byte quota. Detect pathological cardinality trends before exhaustion and surface them through hnsqr doctor. At segment seal, choose representation adaptively—Roaring, sorted postings, dense bitmap, compact dictionary or other representation—based on cardinality and density.
Remove manual RRF tuning. RRF_DEFAULT_K should not remain a workload-specific magic constant. Add a fusion calibration layer that learns or selects dense/sparse fusion parameters from offline relevance judgments when available, and falls back to robust distribution-aware heuristics otherwise. Support RRF as one policy, but benchmark weighted normalized fusion and calibrated alternatives. Maintain deterministic behavior and a static configuration option for regulated deployments. AutoForge should recommend the policy and explain why it selected it.
Build first-class read replicas using Raft learners. Implement non-voting learner replicas that consume committed log entries and immutable segment/snapshot updates without increasing quorum size. Add read-consistency contracts such as Linearizable, Committed, and BoundedStaleness { max_lag }. A learner must only serve a requested consistency level when its applied index satisfies that contract. Route high-volume exact reads toward healthy learners based on CPU pressure, cache residency, replication lag and local segment availability. Promote learners safely through joint consensus when required. Benchmark read scaling from 1 to N learners and demonstrate that adding read capacity does not increase write quorum latency.
Exploit proof-aware replica routing. Go beyond generic least-loaded routing. Track per-replica hot proof-tree/LUTz pages and segment residency. Prefer replicas already warm for the target shard/tenant where consistency permits. Feed cache-hit statistics back into routing. This turns replicas into useful locality domains rather than interchangeable machines.
Build S3/blob disaggregation correctly—do not turn MmapArena into fake remote mmap. Immutable segment artifacts should become versioned, checksummed, independently range-addressable objects. Keep manifests, Rivero routing structures, ProofTree nodes and LUTz codes in local hot storage. Dense exact-vector blocks may reside remotely. Build a bounded content-addressed local NVMe cache supporting async range reads, request coalescing, read-ahead, checksum validation, admission control and eviction. A Certified query that reaches an unresolved vector must fetch the necessary exact block or return an explicit availability error; it must never silently downgrade exactness.
Make remote layout proof-aware. Store dense vector blocks in the same or compatible locality order as proof-tree leaves so opening one semantic leaf requires as few range requests as possible. Align object chunks to natural storage/network sizes and benchmark 64 KiB through multi-MiB blocks. Let LUTz eliminate threats before remote dense payload fetch. The key metric is not merely cache hit rate; measure remote bytes per certified query and remote requests per exact vector.
Add cache-thrash protection. Multi-tenant or adversarial random queries must not flush the useful working set. Implement TinyLFU/segmented admission or comparable frequency-aware cache admission instead of unconditional LRU insertion. Separate metadata/proof cache budgets from dense-vector cache budgets. Add per-tenant cache fairness and detect scan pollution. Benchmark Zipfian, uniform random, tenant-hotspot and adversarial scan workloads.
Build a real Kubernetes Operator. Define CRDs for HNSQR clusters, shard groups, durability policy, storage classes, backup policy and upgrade policy. The operator—not humans—must orchestrate safe scaling using the existing migration state machine. Scaling out should create learners, wait for snapshot/WAL catch-up, verify health, commit membership/ownership through Raft, and only then remove old ownership. Scaling down must reverse the operation safely. Add reconciliation idempotence so operator restarts do not corrupt migration state.
Make Kubernetes upgrades quorum-safe. Implement controlled rolling upgrades with PodDisruptionBudget, readiness gates, topology spread constraints, anti-affinity, graceful leadership transfer, learner-first replacement where appropriate, and explicit maximum unavailable replicas. The operator must refuse an action that would destroy quorum. Add upgrade simulation tests for 3-, 5-, and 7-member groups.
Teach the operator storage topology. Prefer WAL volumes appropriate for sync-heavy workloads, vector cache volumes appropriate for capacity/throughput, and spread replicas across failure domains. Validate storage classes and warn or reject configurations that obviously violate declared latency/durability intentions. Add capacity planning output instead of leaving users to infer this from Raft behavior.
Provide autoscaling based on HNSQR-native signals, not CPU alone. Scale read learners using query queue depth, exact-evaluation work, proof-tree hit rates, cache pressure and p99. Scale shard groups or recommend resharding based on storage, WAL bandwidth, metadata pressure and write queueing. Use stabilization windows and cooldown periods so autoscaling does not create topology churn.
Build Python, TypeScript and Go SDKs against a stable protocol abstraction. Do not hand-code three independent QIR0 implementations that will drift. Define a protocol/schema source of truth and generate or share framing definitions where practical. SDKs must support typed search contracts, filters, batch upsert, streaming, tenant/auth configuration, timeout/cancellation, retry semantics, idempotency keys, execution proof retrieval and structured errors. Retries must understand which operations are safe to repeat.
Give SDKs intelligent cluster behavior. Support endpoint discovery, leader redirection, bounded retry, connection pooling, circuit breaking and optional learner/read-replica selection according to consistency requirements. Prevent retry storms during elections using randomized exponential backoff and server-provided hints.
Add compatibility testing for every client. Run the current server against the previous supported SDK release and the current SDK against the previous supported server. Create wire-level golden tests and protocol-version negotiation. QIR0, HTTP/gRPC and SDK evolution must become independently deployable rather than requiring lockstep upgrades.
Build native LangChain, LlamaIndex and Haystack adapters on top of the official SDKs. Do not duplicate protocol code in framework integrations. Expose standard approximate/high-recall behavior where framework interfaces require it, but also expose an HNSQR-specific Certified mode and execution-proof metadata where extension points allow. Add metadata filtering, namespaces, hybrid dense/sparse retrieval and async batching.
Add an ecosystem integration conformance suite. Framework adapters must be tested against a real ephemeral HNSQR instance for insert, delete, metadata filtering, persistence, restart, authentication, tenant isolation and exact retrieval. Prevent integrations from becoming marketing-only repositories that silently diverge from server behavior.
Create a DBaaS control-plane boundary without contaminating the data plane. The data plane remains the optimized Rust HNSQR server. Build a separate control plane responsible for cluster lifecycle, tenancy, billing/metering hooks, certificate provisioning, backups, upgrades, capacity management and placement. Do not put cloud-provider API calls in the retrieval process.
Separate desired state from observed state. DBaaS control-plane resources should be reconciled, not executed as one-shot scripts. Persist cluster desired state and continually converge infrastructure toward it. Every operation—create, resize, upgrade, restore, rotate certs—must be resumable and idempotent.
Design serverless cautiously. Do not promise “scale-to-zero exact vector database” until cold-start physics is measured. Build segment attachment and adaptive proof/LUTz warming benchmarks first. If control-plane scale-to-zero is implemented, preserve manifests/proof structures in a warm metadata tier and use lazy dense-block attachment. Define an explicit cold-start SLA rather than hiding seconds of hydration behind the first customer query.
Implement federated cross-cluster Certified search with a global proof contract. A naive scatter/gather Top-K is insufficient if you want to preserve HNSQR’s exactness story. Each participating cluster must return local Top-K plus enough proof information to establish a local maximum-unseen upper bound. The coordinator maintains global τ and continues requesting work from clusters whose unresolved bound can still beat it. Terminate only when every participating cluster satisfies:
UB
cluster
	​

<τ
global
	​


subject to tie semantics.

That yields a genuine federated globally exact Top-K, rather than merely merging local Top-K lists.

Make federation region- and policy-aware. Respect data residency. EU data should not be copied to US regions merely to answer a query if policy forbids it. Send query representations to data where allowed and merge proof/result metadata centrally. Add tenant policies for permitted regions and failure behavior.
Support degraded federated semantics explicitly. If a region is unreachable, a Certified global query cannot claim global exactness. Return a structured IncompleteGlobalProof describing unreachable shards/regions and their last known state. Allow callers to opt into best-effort partial results, but never label them Certified.
Build multi-region disaster recovery before attempting active-active mutation. Establish asynchronous cross-region snapshot/WAL replication, measurable RPO/RTO, integrity verification and automated failover exercises. Active-active writes are a substantially harder consistency problem and should not be introduced merely because the market report says “enterprise-grade.” Prove DR first.
If active-active mutation is eventually required, define conflict semantics before code. HNSQR IDs, metadata mutations and deletes need explicit ownership/conflict rules. Do not casually layer multi-leader replication over Raft shards. Prefer single-writer ownership per shard across regions unless a concrete workload proves otherwise.
Integrate KMS-backed key management. Move TLS private keys, token-signing material and encrypted backup keys behind a provider abstraction capable of AWS KMS, GCP KMS, Azure Key Vault or HSM-backed deployments. Add envelope encryption for backups and optionally encrypted immutable segment objects. Rotate data-encryption keys without rewriting everything synchronously by tracking key versions per object.
Prepare for regulated cryptographic modes without falsely claiming compliance. Add a crypto-provider abstraction and test a FIPS-capable provider configuration such as appropriate aws-lc-rs deployments where supported. Do not print “FIPS compliant,” “HIPAA compliant,” “SOC2 compliant,” etc. merely because a crypto crate is enabled. Produce a machine-readable security configuration report that auditors can use as evidence.
Add SSO/SAML/OIDC enterprise identity integration at the control plane. Human administrative access should not rely on static database tokens. Map external groups/claims into HNSQR administrative roles, maintain audit history and require stronger controls for destructive operations.
Create audit-log integrity guarantees. Security and administrative audit events should be append-only, timestamped, identity-bound and tamper-evident. Chain event hashes or periodically checkpoint audit roots into durable storage. Provide export to SIEM systems using an adapter architecture rather than embedding vendor-specific code in the engine.
Do not implement patents in the codebase. Dr. Cross’s recommendation to patent the five-stage migration or protect AutoForge is a legal/business action, not an engineering requirement. Instead, maintain invention-disclosure evidence: algorithm definitions, Git chronology, benchmark provenance, design alternatives, public-disclosure dates and contributor/inventorship records. Counsel can then decide what is protectable.
Do not hard-code open-core licensing or pricing decisions. Preserve package boundaries so core retrieval, clustering, cloud control plane and enterprise integrations could be licensed separately, but do not change licenses or create artificial feature restrictions without an explicit product/legal decision.
Turn every competitive claim into a reproducible benchmark. Claims such as “70% cheaper than Elasticsearch,” “4× more vectors per GB,” “<5 ms P99,” “vastly superior to Qdrant,” “100M–1B vectors per shard group,” and “5B–10B ceiling” are hypotheses until measured. Build a public benchmark suite with identical hardware, identical embeddings, identical metrics, equal durability settings, equal replication factor and equivalent filtering workloads. Publish configuration manifests and raw results. Never compare HNSQR Certified exact search against an approximate competitor configuration without labeling the semantic difference.
Separate folding from compression in all technical claims. Pairwise real→complex isometric folding preserves byte size when 2 × f32 becomes Complex32; CPQ-8/LUTz or other quantized representations provide compression. Make documentation and SDK marketing metadata reflect that distinction.
Benchmark distributed exactness, not merely single-node exactness. Add exhaustive ground-truth verification across multiple shards, learners, migrations and federated clusters. Certified must remain exact while topology changes occur. Test that a query racing an ownership transition operates against one well-defined snapshot/epoch.
Define query consistency during migration. A five-stage write migration is not enough unless reads have explicit semantics. Decide exactly which owner serves each epoch and how in-flight Certified queries maintain a stable corpus view. Use generation/epoch pinning so one query cannot see half of the corpus before ownership transfer and half afterward.
Add MVCC/generation pinning where needed for linearizable Certified queries. Certified proof traversal depends on a stable eligible universe. If concurrent mutations may invalidate the proof tree or τ, pin the query to an immutable generation plus a defined mutable-log frontier, or establish equivalent snapshot semantics. Do not let “linearizable writes” coexist with fuzzy read snapshots without documenting the contract.
Protect Gate B3 performance permanently. Add CI performance canaries around Rivero proposal quality, ProofTree region pruning, LUTz L0/L1 pruning, exact SIMD percentage, full-vector bytes touched, p50/p95/p99, allocations/query and syscalls/query. Infrastructure features must not regress the query core unnoticed. Keep a fast smoke benchmark in CI and a statistically rigorous hardware benchmark in scheduled/release infrastructure.
Protect Raft/WAL performance with the same rigor. Track mutations/sec, commit p99, fsync latency distribution, bytes/fsync, AppendEntries batch size, follower lag, election rate, snapshot-install bandwidth and recovery throughput. Establish regression budgets.
Add end-to-end workload benchmarks. Benchmark realistic mixed workloads, not isolated microbenchmarks only: 90/10 read/write, 70/30, metadata-heavy SaaS, hybrid sparse+dense commerce, Certified legal/RAG, migration-under-load, backup-under-load, certificate rotation-under-load, and learner scaling. Report service-level throughput at a fixed p99 target rather than maximum QPS alone.
Turn hnsqr doctor into a real operational expert system. Extend it to inspect Raft quorum health, term/index agreement, follower lag, disk latency, WAL backlog, snapshot age, backup verification age, certificate expiration, cgroup headroom, prefault safety, tenant cardinality, cache health, migration progress, proof-tree integrity and SIMD availability. Every failure should include a precise remediation command or operator action rather than merely printing red text.
Add capacity planning. Provide hnsqr plan or equivalent. Given N,D,QPS, write rate, durability level, replication factor, tenant count and latency target, estimate vector storage, LUTz/proof bytes, metadata headroom, WAL rate, recommended NVMe throughput, RAM, cache size, CPU cores, learner count and shard count. Initially model-based; continuously calibrate from measured telemetry.
Make operational defaults conservative and boring. A default deployment should not require an E8 expert, Raft expert or storage expert. Default to adaptive prefault, automatic durability batching, safe snapshot I/O throttling, AutoForge query planning, managed certificate warnings/rotation hooks, bounded metadata, bounded queues, and strict Certified semantics when requested. Advanced tuning remains available but should not be necessary for a healthy installation.
Phase 4 acceptance gate

Do not declare this phase complete because all modules compile. Require the following end state:

DISTRIBUTED WRITE PATH
  Batched/pipelined Raft replication              PASS
  Group commit without heartbeat starvation       PASS
  Election churn under disk saturation            0 avoidable elections
  Backpressure under overload                      bounded
  Learner/read-replica scaling                     PASS


STORAGE / CLOUD
  Maintenance I/O isolation                        PASS
  Adaptive prefault under cgroups                  PASS
  Remote immutable segment cache                   PASS
  Certified query over cold S3-backed data         exact or explicit failure
  Cache-thrash adversarial workload                bounded


KUBERNETES
  Operator reconciliation                          idempotent
  Scale-out migration                              PASS
  Scale-in migration                               PASS
  Quorum-safe rolling upgrade                      PASS
  Operator restart mid-migration                   PASS
  Failure-domain placement validation              PASS


ECOSYSTEM
  Python SDK                                       PASS
  TypeScript SDK                                   PASS
  Go SDK                                           PASS
  Version compatibility matrix                     PASS
  LangChain adapter                                PASS
  LlamaIndex adapter                               PASS
  Haystack adapter                                 PASS


SECURITY / ENTERPRISE
  Automated certificate rotation                   PASS
  OIDC/JWKS                                        PASS
  KMS envelope encryption                          PASS
  Audit-log integrity                              PASS
  Security configuration evidence                  PASS


FEDERATION / DR
  Cross-cluster Top-K merge                        PASS
  Global Certified proof                           PASS
  Unreachable-region downgrade semantics           PASS
  Cross-region DR restore                          PASS
  Measured RPO/RTO                                 documented


PERFORMANCE
  Gate B3 exactness                                100.0000%
  Gate B3 regression                              within threshold
  Raft write throughput                            hardware-bound, measured
  Raft commit p99                                  measured
  Query p99 during compaction                      within SLA
  Query p99 during migration                       within SLA
  Query p99 during backup                          within SLA
  Remote bytes/query                               measured
  S3 requests/query                                measured


OPERABILITY
  hnsqr doctor                                     comprehensive
  capacity planner                                 PASS
  AutoForge                                        default path
  no unbounded resource queues                     PASS

The strategic instruction to the coding agent is:

Do not touch the mathematical search core unless a benchmark exposes a regression or a demonstrable improvement. Gate B3 is now a protected subsystem. Move the optimization frontier outward: consensus throughput, storage hierarchy, cloud orchestration, ecosystem integration, and operational automation.

The outside review’s most useful conclusion is that the failure mode has changed. Earlier HNSQR could lose data. The current system’s next risks are availability, operational friction, and cloud-scale economics. Those are much better problems to have—and they are now the ones the coding agent should eliminate.