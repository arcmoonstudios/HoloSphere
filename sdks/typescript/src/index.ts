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

  public async search(
    collection: string,
    vector: number[],
    options: {
      k?: number;
      filter?: Record<string, unknown>;
      certifiedExact?: boolean;
    } = {}
  ): Promise<SearchResult[]> {
    const k = options.k ?? 10;
    const certifiedExact = options.certifiedExact ?? true;
    let lastError: Error | undefined;

    for (let attempt = 0; attempt < this.maxRetries; attempt++) {
      const endpoint = this.selectEndpoint(false);
      const cb = this.circuitBreakers.get(endpoint);
      const url = `${endpoint}/v1/collections/${collection}/search`;

      try {
        const res = await fetch(url, {
          method: "POST",
          headers: this.headers(),
          body: JSON.stringify({
            vector,
            k,
            filter: options.filter,
            certified_exact: certifiedExact,
            consistency: this.readConsistency,
          }),
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
    certifiedExact = true
  ): Promise<SearchResult[]> {
    const endpoint = this.selectEndpoint(false);
    const url = `${endpoint}/v1/collections/${collection}/search`;
    const res = await fetch(url, {
      method: "POST",
      headers: this.headers(),
      body: JSON.stringify({
        query_text: queryText,
        k,
        certified_exact: certifiedExact,
      }),
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
}
