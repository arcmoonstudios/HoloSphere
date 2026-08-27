/**
 * HNSQR Official TypeScript SDK
 * Copyright (c) 2026 ArcMoon Studios. MIT / Apache-2.0 License.
 */

export enum ReadConsistency {
  Linearizable = "Linearizable",
  Committed = "Committed",
  BoundedStaleness = "BoundedStaleness",
}

export class HNSQRError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "HNSQRError";
  }
}

export class HNSQRConnectionError extends HNSQRError {
  constructor(message: string) {
    super(message);
    this.name = "HNSQRConnectionError";
  }
}

export class HNSQRCircuitOpenError extends HNSQRError {
  constructor(endpoint: string) {
    super(`Fast-fail circuit breaker open for endpoint ${endpoint}`);
    this.name = "HNSQRCircuitOpenError";
  }
}

export interface SearchResult {
  id: string;
  score: number;
  isCertified: boolean;
  proofUpperBound?: number;
  metadata?: Record<string, unknown>;
}

export interface MutationReceipt {
  id: string;
  lsn: number;
  appliedGeneration: number;
  isQuorumReplicated: boolean;
}

export interface GraphQueryResult {
  columns: string[];
  rows: unknown[][];
  executionTimeMicros: number;
}

export interface SqlExecutionResult {
  columns: string[];
  rows: Record<string, unknown>[];
  affectedRows: number;
}

export interface HypercubeSliceResult {
  coordinates: number[][];
  values: number[];
  totalVoxels: number;
}

export interface HNSQRClientOptions {
  endpoints?: string[];
  apiKey?: string;
  tenantId?: string;
  timeoutMs?: number;
  maxRetries?: number;
  readConsistency?: ReadConsistency;
}

class CircuitBreaker {
  private failures = 0;
  private lastFailure = 0;
  private isOpen = false;

  constructor(
    private readonly threshold = 5,
    private readonly recoveryTimeMs = 10000
  ) {}

  public canExecute(): boolean {
    if (!this.isOpen) return true;
    if (Date.now() - this.lastFailure > this.recoveryTimeMs) {
      return true;
    }
    return false;
  }

  public recordSuccess(): void {
    this.failures = 0;
    this.isOpen = false;
  }

  public recordFailure(): void {
    this.failures++;
    this.lastFailure = Date.now();
    if (this.failures >= this.threshold) {
      this.isOpen = true;
    }
  }
}

export class HNSQRClient {
  private readonly endpoints: string[];
  private readonly apiKey?: string;
  private readonly tenantId?: string;
  private readonly timeoutMs: number;
  private readonly maxRetries: number;
  private readonly readConsistency: ReadConsistency;
  private readonly circuitBreakers: Map<string, CircuitBreaker>;
  private activeLeader?: string;
  private roundRobinIdx = 0;

  constructor(options: HNSQRClientOptions = {}) {
    this.endpoints = options.endpoints && options.endpoints.length > 0
      ? options.endpoints
      : ["http://127.0.0.1:8080"];
    this.apiKey = options.apiKey;
    this.tenantId = options.tenantId;
    this.timeoutMs = options.timeoutMs ?? 5000;
    this.maxRetries = options.maxRetries ?? 3;
    this.readConsistency = options.readConsistency ?? ReadConsistency.Committed;

    this.circuitBreakers = new Map();
    for (const ep of this.endpoints) {
      this.circuitBreakers.set(ep, new CircuitBreaker());
    }
  }

  private selectEndpoint(isWrite = false): string {
    if (isWrite && this.activeLeader) {
      const cb = this.circuitBreakers.get(this.activeLeader);
      if (cb && cb.canExecute()) {
        return this.activeLeader;
      }
    }

    const healthy = this.endpoints.filter((ep) =>
      this.circuitBreakers.get(ep)?.canExecute()
    );

    if (healthy.length === 0) {
      throw new HNSQRCircuitOpenError("All cluster endpoints circuit breakers are OPEN");
    }

    const chosen = healthy[this.roundRobinIdx % healthy.length];
    this.roundRobinIdx++;
    return chosen;
  }

  private headers(idempotencyKey?: string): HeadersInit {
    const h: Record<string, string> = {
      "Content-Type": "application/json",
    };
    if (this.apiKey) {
      h["Authorization"] = `Bearer ${this.apiKey}`;
    }
    if (this.tenantId) {
      h["X-HNSQR-Tenant-ID"] = this.tenantId;
    }
    if (idempotencyKey) {
      h["X-Idempotency-Key"] = idempotencyKey;
    }
    return h;
  }

export interface SearchOptions {
  k?: number;
  filter?: Record<string, unknown>;
  retrievalContract?: "exact" | "certified" | "high_recall" | "auto" | "rivero" | "hnsw" | string;
  certifiedExact?: boolean;
}

  public async search(
    collection: string,
    vector: number[],
    options: SearchOptions = {}
  ): Promise<SearchResult[]> {
    const k = options.k ?? 10;
    const retrievalContract = options.retrievalContract ?? (options.certifiedExact ? "certified" : "exact");
    let lastError: Error | undefined;

    for (let attempt = 0; attempt < this.maxRetries; attempt++) {
      const endpoint = this.selectEndpoint(false);
      const cb = this.circuitBreakers.get(endpoint);
      const url = `${endpoint}/v1/collections/${collection}/search`;

      try {
        const bodyPayload: Record<string, unknown> = {
          vector,
          k,
          filter: options.filter,
          retrieval_contract: retrievalContract,
          consistency: this.readConsistency,
        };
        if (options.certifiedExact !== undefined) {
          bodyPayload.certified_exact = options.certifiedExact;
        }

        const res = await fetch(url, {
          method: "POST",
          headers: this.headers(),
          body: JSON.stringify(bodyPayload),
          signal: AbortSignal.timeout(this.timeoutMs),
        });

        if (res.status === 200) {
          cb?.recordSuccess();
          const data = (await res.json()) as { results: SearchResult[] };
          return data.results || [];
        }

        if (res.status === 307 || res.status === 308) {
          const leader = res.headers.get("Location");
          if (leader) {
            this.activeLeader = leader;
            continue;
          }
        }

        cb?.recordFailure();
        lastError = new HNSQRError(`Search failed on ${endpoint}: ${res.status} ${res.statusText}`);
      } catch (err) {
        cb?.recordFailure();
        lastError = err instanceof Error ? err : new HNSQRConnectionError(String(err));
      }

      await new Promise((r) => setTimeout(r, 50 * Math.pow(2, attempt) + Math.random() * 30));
    }

    throw lastError || new HNSQRError("Search retries exhausted");
  }

  public async embedAndSearch(
    collection: string,
    queryText: string,
    k = 10,
    retrievalContract = "exact",
    certifiedExact?: boolean
  ): Promise<SearchResult[]> {
    const endpoint = this.selectEndpoint(false);
    const url = `${endpoint}/v1/collections/${collection}/search`;
    const bodyPayload: Record<string, unknown> = {
      query_text: queryText,
      k,
      retrieval_contract: certifiedExact ? "certified" : retrievalContract,
    };
    if (certifiedExact !== undefined) {
      bodyPayload.certified_exact = certifiedExact;
    }

    const res = await fetch(url, {
      method: "POST",
      headers: this.headers(),
      body: JSON.stringify(bodyPayload),
      signal: AbortSignal.timeout(this.timeoutMs),
    });
    if (!res.ok) {
      throw new HNSQRError(`Embed & Search failed: ${res.status}`);
    }
    const data = (await res.json()) as { results: SearchResult[] };
    return data.results || [];
  }

  public async queryGraph(cypherQuery: string): Promise<GraphQueryResult> {
    const endpoint = this.selectEndpoint(false);
    const url = `${endpoint}/v1/graph/query`;
    const res = await fetch(url, {
      method: "POST",
      headers: this.headers(),
      body: JSON.stringify({ query: cypherQuery }),
      signal: AbortSignal.timeout(this.timeoutMs),
    });
    if (!res.ok) {
      throw new HNSQRError(`Graph query failed: ${res.status}`);
    }
    return (await res.json()) as GraphQueryResult;
  }

  public async executeSql(sqlQuery: string): Promise<SqlExecutionResult> {
    const endpoint = this.selectEndpoint(false);
    const url = `${endpoint}/v1/sql/execute`;
    const res = await fetch(url, {
      method: "POST",
      headers: this.headers(),
      body: JSON.stringify({ sql: sqlQuery }),
      signal: AbortSignal.timeout(this.timeoutMs),
    });
    if (!res.ok) {
      throw new HNSQRError(`SQL execute failed: ${res.status}`);
    }
    return (await res.json()) as SqlExecutionResult;
  }

  public async sliceHypercube(
    spaceId: string,
    minCoords: number[],
    maxCoords: number[]
  ): Promise<HypercubeSliceResult> {
    const endpoint = this.selectEndpoint(false);
    const url = `${endpoint}/v1/hypercube/slice`;
    const res = await fetch(url, {
      method: "POST",
      headers: this.headers(),
      body: JSON.stringify({ space_id: spaceId, min_coords: minCoords, max_coords: maxCoords }),
      signal: AbortSignal.timeout(this.timeoutMs),
    });
    if (!res.ok) {
      throw new HNSQRError(`Hypercube slice failed: ${res.status}`);
    }
    return (await res.json()) as HypercubeSliceResult;
  }

  public async getBillingReport(tenantId: string): Promise<Record<string, unknown>> {
    const endpoint = this.selectEndpoint(false);
    const url = `${endpoint}/v1/dbaas/tenants/${tenantId}/usage`;
    const res = await fetch(url, {
      method: "GET",
      headers: this.headers(),
      signal: AbortSignal.timeout(this.timeoutMs),
    });
    if (!res.ok) {
      throw new HNSQRError(`Get billing report failed: ${res.status}`);
    }
    return (await res.json()) as Record<string, unknown>;
  }

  public async upsert(
    collection: string,
    id: string,
    vector: number[],
    metadata?: Record<string, unknown>,
    idempotencyKey?: string
  ): Promise<MutationReceipt> {
    let lastError: Error | undefined;

    for (let attempt = 0; attempt < this.maxRetries; attempt++) {
      const endpoint = this.selectEndpoint(true);
      const cb = this.circuitBreakers.get(endpoint);
      const url = `${endpoint}/v1/collections/${collection}/insert`;

      try {
        const res = await fetch(url, {
          method: "POST",
          headers: this.headers(idempotencyKey),
          body: JSON.stringify({ id, vector, metadata }),
          signal: AbortSignal.timeout(this.timeoutMs),
        });

        if (res.status === 200) {
          cb?.recordSuccess();
          const data = (await res.json()) as Partial<MutationReceipt>;
          return {
            id: data.id ?? id,
            lsn: data.lsn ?? 1,
            appliedGeneration: data.appliedGeneration ?? 1,
            isQuorumReplicated: data.isQuorumReplicated ?? true,
          };
        }

        if (res.status === 307 || res.status === 308) {
          const leader = res.headers.get("Location");
          if (leader) {
            this.activeLeader = leader;
            continue;
          }
        }

        cb?.recordFailure();
        lastError = new HNSQRError(`Upsert failed on ${endpoint}: ${res.status} ${res.statusText}`);
      } catch (err) {
        cb?.recordFailure();
        lastError = err instanceof Error ? err : new HNSQRConnectionError(String(err));
      }

      await new Promise((r) => setTimeout(r, 50 * Math.pow(2, attempt) + Math.random() * 30));
    }

    throw lastError || new HNSQRError("Upsert retries exhausted");
  }

  public get mcpUrl(): string {
    return `${this.endpoints[0].replace(/\/$/, "")}/mcp`;
  }

  public async callModelTool<T = Record<string, unknown>>(
    operation: string,
    payload: Record<string, unknown>
  ): Promise<EvidenceEnvelope<T>> {
    const paths: Record<string, string> = {
      search: "/v1/knowledge/search",
      web_search: "/v1/knowledge/web/search",
      websearch: "/v1/knowledge/web/search",
      traverse: "/v1/knowledge/traverse",
      resolve: "/v1/knowledge/resolve",
      remember: "/v1/knowledge/remember",
      record_outcome: "/v1/knowledge/outcomes",
      recordoutcome: "/v1/knowledge/outcomes",
      task_begin: "/v1/knowledge/tasks/begin",
      taskbegin: "/v1/knowledge/tasks/begin",
      task_context: "/v1/knowledge/tasks/context",
      taskcontext: "/v1/knowledge/tasks/context",
      task_complete: "/v1/knowledge/tasks/complete",
      taskcomplete: "/v1/knowledge/tasks/complete",
      explore: "/v1/knowledge/explore",
      status: "/v1/knowledge/status",
      run_case: "/v1/knowledge/cases/run",
      runcase: "/v1/knowledge/cases/run",
    };
    const normalized = operation.toLowerCase().replace(/[-.]/g, "_");
    const path = paths[normalized];
    if (!path) throw new HNSQRError(`Unknown model tool operation: ${operation}`);
    const isWrite = [
      "remember",
      "record_outcome",
      "recordoutcome",
      "task_begin",
      "taskbegin",
      "task_complete",
      "taskcomplete",
    ].includes(normalized);
    const endpoint = this.selectEndpoint(isWrite);
    const res = await fetch(`${endpoint.replace(/\/$/, "")}${path}`, {
      method: "POST",
      headers: this.headers(),
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(this.timeoutMs),
    });
    if (!res.ok) {
      const errorText = await res.text().catch(() => "");
      throw new HNSQRError(`Model tool '${operation}' failed (${res.status}): ${errorText}`);
    }
    return (await res.json()) as EvidenceEnvelope<T>;
  }

  public async taskBegin(params: TaskBeginParams): Promise<EvidenceEnvelope<TaskBeginResult>> {
    return this.callModelTool<TaskBeginResult>("task_begin", params as unknown as Record<string, unknown>);
  }

  public async taskContext(params: TaskContextParams): Promise<EvidenceEnvelope<TaskContextResult>> {
    return this.callModelTool<TaskContextResult>("task_context", params as unknown as Record<string, unknown>);
  }

  public async webSearch(params: WebSearchParams): Promise<EvidenceEnvelope<WebSearchResultSet>> {
    return this.callModelTool<WebSearchResultSet>("web_search", params as unknown as Record<string, unknown>);
  }

  public async remember(params: RememberParams): Promise<EvidenceEnvelope<KnowledgeRecord>> {
    return this.callModelTool<KnowledgeRecord>("remember", params as unknown as Record<string, unknown>);
  }

  public async searchKnowledge(params: SearchKnowledgeParams): Promise<EvidenceEnvelope<KnowledgeRecord[]>> {
    return this.callModelTool<KnowledgeRecord[]>("search", params as unknown as Record<string, unknown>);
  }

  public async traverse(params: TraverseParams): Promise<EvidenceEnvelope<TraverseRecord[]>> {
    return this.callModelTool<TraverseRecord[]>("traverse", params as unknown as Record<string, unknown>);
  }

  public async resolve(params: ResolveParams): Promise<EvidenceEnvelope<CandidateResolution[]>> {
    return this.callModelTool<CandidateResolution[]>("resolve", params as unknown as Record<string, unknown>);
  }

  public async recordOutcome(params: RecordOutcomeParams): Promise<EvidenceEnvelope<ModelOutcomeRecord>> {
    return this.callModelTool<ModelOutcomeRecord>("record_outcome", params as unknown as Record<string, unknown>);
  }

  public async taskComplete(params: TaskCompleteParams): Promise<EvidenceEnvelope<TaskCompleteResult>> {
    return this.callModelTool<TaskCompleteResult>("task_complete", params as unknown as Record<string, unknown>);
  }

  public async explore(params: ExploreParams): Promise<EvidenceEnvelope<ExploreResult>> {
    return this.callModelTool<ExploreResult>("explore", params as unknown as Record<string, unknown>);
  }

  public async status(): Promise<EvidenceEnvelope<RuntimeStatus>> {
    return this.callModelTool<RuntimeStatus>("status", {});
  }

  public async runCase(params: RunCaseParams): Promise<EvidenceEnvelope<RunCaseResult>> {
    return this.callModelTool<RunCaseResult>("run_case", params as unknown as Record<string, unknown>);
  }
}

// ---------------------------------------------------------------------------
// HoloSphere Epistemic Data Models & Types
// ---------------------------------------------------------------------------

export interface ProvenanceReference {
  source_id: string;
  content_hash: string;
  uri?: string;
  observed_at_lsn?: number;
}

export interface EvidenceEnvelope<T> {
  tenant_id: string;
  snapshot_lsn: number;
  retrieval_contract: string;
  certified: boolean;
  proof_upper_bound?: number;
  content_is_untrusted: boolean;
  results: T;
  contradictions?: unknown[];
}

export interface KnowledgeRecord {
  id: string;
  collection: string;
  kind: string;
  content: string;
  commit_lsn: number;
  tenant_id: string;
  members: string[];
  roles: Record<string, string>;
  metadata: Record<string, unknown>;
  provenance: ProvenanceReference[];
}

export interface CandidateResolution {
  hypothesis: string;
  confidence: number;
  ranking_components: Record<string, number>;
  evidence_ids: string[];
  successful_outcomes: number;
  failed_outcomes: number;
  status: string;
}

export interface TaskBeginParams {
  problem: string;
  case_id?: string;
  idempotency_key?: string;
  collection?: string;
  max_hypotheses?: number;
  provenance?: ProvenanceReference[];
}

export interface TaskBeginResult {
  case: KnowledgeRecord;
  related_cases: Array<{ id: string; record: KnowledgeRecord; score: number }>;
  candidate_resolutions: CandidateResolution[];
}

export interface TaskContextParams {
  case_id: string;
  snapshot_lsn?: number;
}

export interface TaskContextResult {
  case: KnowledgeRecord;
  related_cases: Array<{ id: string; record: KnowledgeRecord; score: number }>;
  relations: Array<{ depth: number; record: KnowledgeRecord }>;
  candidate_resolutions: CandidateResolution[];
}

export interface WebSearchParams {
  query?: string;
  query_text?: string;
  k?: number;
  /** Compatibility alias for k, accepted by the MCP server. */
  max_results?: number;
  time_range?: "day" | "month" | "year";
  language?: string;
}

export interface WebSearchResultItem {
  evidence_id: string;
  title: string;
  url: string;
  snippet: string;
  content_hash: string;
  engines?: string[];
}

export interface WebSearchResultSet {
  provider: string;
  results: WebSearchResultItem[];
  retrieved_at_unix_secs: number;
  content_is_untrusted: boolean;
}

export interface RememberParams {
  content: string;
  id?: string;
  idempotency_key?: string;
  collection?: string;
  kind?: string;
  evidence_class?: string;
  members?: string[];
  roles?: Record<string, string>;
  metadata?: Record<string, unknown>;
  provenance?: ProvenanceReference[];
  vector?: number[];
}

export interface SearchKnowledgeParams {
  query_text?: string;
  query?: string;
  query_vector?: number[];
  collection?: string;
  k?: number;
  kinds?: string[];
  retrieval_contract?: "exact" | "certified" | "high_recall" | "auto" | "rivero" | "hnsw" | string;
  snapshot_lsn?: number;
}

export interface TraverseParams {
  seed_ids: string[];
  max_depth?: number;
  max_results?: number;
  relation_kinds?: string[];
  snapshot_lsn?: number;
}

export interface TraverseRecord {
  depth: number;
  record: KnowledgeRecord;
}

export interface ResolveParams {
  problem: string;
  collection?: string;
  max_hypotheses?: number;
  snapshot_lsn?: number;
}

export interface RecordOutcomeParams {
  summary: string;
  successful: boolean;
  attempt_id?: string;
  idempotency_key?: string;
  evidence_ids?: string[];
  metrics?: Record<string, number>;
  provenance?: ProvenanceReference[];
  evidence_class?: string;
}

export interface ModelOutcomeRecord {
  attempt_id: string;
  summary: string;
  successful: boolean;
  commit_lsn: number;
  tenant_id: string;
  evidence_ids: string[];
  metrics: Record<string, number>;
  provenance: ProvenanceReference[];
}

export interface TaskCompleteParams {
  case_id: string;
  summary: string;
  successful: boolean;
  idempotency_key?: string;
  resolution_status?: "hypothesis" | "speculative_synthesis" | "empirically_verified" | "formally_verified";
  evidence_ids?: string[];
  metrics?: Record<string, number>;
  provenance?: ProvenanceReference[];
}

export interface TaskCompleteResult {
  outcome: ModelOutcomeRecord;
  resolution?: KnowledgeRecord;
  resolution_status?: string;
  verification_level?: string;
}

export interface ExploreParams {
  target: "stats" | "recent_cases" | "recent_memories" | "neighborhood";
  limit?: number;
  seed_id?: string;
  snapshot_lsn?: number;
}

export interface ExploreResult {
  target: string;
  stats?: {
    total_entities: number;
    total_outcomes: number;
    current_lsn: number;
    collections: string[];
    collection_embeddings: Record<string, EmbeddingDescriptor>;
    kinds: Record<string, number>;
  };
  recent_cases?: KnowledgeRecord[];
  recent_memories?: KnowledgeRecord[];
  neighborhood?: TraverseRecord[];
}

export interface EmbeddingDescriptor {
  provider: string;
  model: string;
  version: string;
  dimensions: number;
  normalization: "l2";
  distance_metric: "cosine";
}

export interface RuntimeStatus {
  ready: boolean;
  read_write_authorized: boolean;
  web_search_available: boolean;
  embedding_provider: EmbeddingDescriptor;
  collection_embeddings: Record<string, EmbeddingDescriptor>;
  limits: { max_search_results: number; max_web_results: number; max_hypotheses: number; max_traversal_depth: number };
  degradations: string[];
}

export interface CaseBudget {
  tool_calls?: number;
  retrieval_results?: number;
}

export interface RunCaseParams {
  objective: string;
  recipe?: "research_and_synthesize" | "diagnose_and_fix" | "implement_and_test" | "compare_options" | "incident_response" | "analyze_dataset" | "evaluate_strategy";
  collection?: string;
  web_query?: string;
  evidence_policy?: "none" | "knowledge_only" | "web_if_needed" | "web_required";
  execution_policy?: "propose_only" | "tests_only" | "authorized_executor";
  success_criteria?: string[];
  budgets?: CaseBudget;
  case_id?: string;
  idempotency_key?: string;
}

export interface RunCaseResult {
  status: RuntimeStatus;
  case?: KnowledgeRecord;
  evidence_ids: string[];
  candidate_resolutions: CandidateResolution[];
  plan: string[];
  tool_calls_used: number;
  tool_calls_remaining: number;
  action_gate: {
    execution_policy: string;
    external_execution_performed: boolean;
    approval_required: boolean;
    next_action: string;
  };
}

// ---------------------------------------------------------------------------
// Universal Epistemic Workflow Orchestrator
// ---------------------------------------------------------------------------

export interface EpistemicPipelineConfig<TMetrics extends Record<string, number> = Record<string, number>> {
  domain: string;
  problemStatement: string;
  searchQuery?: string;
  synthesizeEvidence?: (
    webResults: WebSearchResultItem[]
  ) => Array<{ id: string; content: string; kind?: string; roles?: Record<string, string> }>;
  evaluateCandidate: (
    hypotheses: CandidateResolution[],
    context: TaskContextResult
  ) => Promise<{ metrics: TMetrics; isSuccess: boolean; summary: string }>;
}

export async function executeUniversalEpistemicPipeline<TMetrics extends Record<string, number>>(
  client: HNSQRClient,
  config: EpistemicPipelineConfig<TMetrics>
): Promise<EvidenceEnvelope<TaskCompleteResult>> {
  const caseId = `case:${config.domain.toLowerCase().replace(/[^a-z0-9_-]/g, "_")}:${Date.now()}`;

  // Phase 1: Epistemic Inception & Context Hydration
  await client.taskBegin({
    case_id: caseId,
    problem: config.problemStatement,
    idempotency_key: `idemp:${caseId}`,
  });

  const contextEnvelope = await client.taskContext({ case_id: caseId });
  const evidenceIds: string[] = [];

  // Phase 2: External Entropy Gathering (if query supplied)
  if (config.searchQuery) {
    const webEnvelope = await client.webSearch({ query: config.searchQuery, k: 8 });
    if (config.synthesizeEvidence && webEnvelope.results?.results) {
      const synthesized = config.synthesizeEvidence(webEnvelope.results.results);

      // Phase 3: High-Dimensional Graph Topology Synthesis
      for (const item of synthesized) {
        evidenceIds.push(item.id);
        await client.remember({
          id: item.id,
          content: item.content,
          kind: item.kind || "domain_evidence",
          members: [caseId, item.id],
          roles: item.roles || { [caseId]: "target_scope" },
        });
      }
    }
  }

  // Phase 4: Deduction, Traversal & Constraint Resolution
  await client.traverse({ seed_ids: [caseId], max_depth: 3 });
  const resolveEnvelope = await client.resolve({ problem: config.problemStatement, max_hypotheses: 5 });

  // Phase 5: Empirical Verification & Durable Resolution Promotion
  const evalResult = await config.evaluateCandidate(resolveEnvelope.results, contextEnvelope.results);

  await client.recordOutcome({
    attempt_id: `attempt:${caseId}`,
    evidence_ids: evidenceIds,
    metrics: evalResult.metrics,
    successful: evalResult.isSuccess,
    summary: evalResult.summary,
  });

  return await client.taskComplete({
    case_id: caseId,
    evidence_ids: evidenceIds,
    metrics: evalResult.metrics,
    successful: evalResult.isSuccess,
    summary: evalResult.summary,
    // A callback result is not empirical proof unless it is backed by a MeasurementSpec.
    resolution_status: evalResult.isSuccess ? "speculative_synthesis" : "hypothesis",
  });
}
