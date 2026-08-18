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
    this.endpoints = options.endpoints || ["http://127.0.0.1:8080"];
    this.apiKey = options.apiKey;
    this.tenantId = options.tenantId;
    this.timeoutMs = options.timeoutMs || 5000;
    this.maxRetries = options.maxRetries || 3;
    this.readConsistency = options.readConsistency || ReadConsistency.Committed;
    this.circuitBreakers = new Map();
    for (const ep of this.endpoints) {
      this.circuitBreakers.set(ep, new CircuitBreaker());
    }
  }

  private selectEndpoint(isWrite = false): string {
    if (isWrite && this.activeLeader) {
      return this.activeLeader;
    }
    for (let i = 0; i < this.endpoints.length; i++) {
      const ep = this.endpoints[this.roundRobinIdx % this.endpoints.length];
      this.roundRobinIdx++;
      const cb = this.circuitBreakers.get(ep);
      if (!cb || cb.canExecute()) {
        return ep;
      }
    }
    return this.endpoints[0];
  }

  private headers(idempotencyKey?: string): Record<string, string> {
    const h: Record<string, string> = {
      "Content-Type": "application/json",
    };
    if (this.apiKey) {
      h["Authorization"] = `Bearer ${this.apiKey}`;
    }
    if (this.tenantId) {
      h["X-Tenant-ID"] = this.tenantId;
    }
    if (idempotencyKey) {
      h["X-Idempotency-Key"] = idempotencyKey;
    }
    return h;
  }

  public async search(
    query: number[],
    k = 10,
    filter?: Record<string, unknown>,
    certifiedExact = true
  ): Promise<SearchResult[]> {
    let lastError: Error | undefined;

    for (let attempt = 0; attempt < this.maxRetries; attempt++) {
      const endpoint = this.selectEndpoint(false);
      const cb = this.circuitBreakers.get(endpoint);
      const url = `${endpoint}/search`;

      try {
        const res = await fetch(url, {
          method: "POST",
          headers: this.headers(),
          body: JSON.stringify({
            query,
            k,
            filter,
            certified_exact: certifiedExact,
            consistency: this.readConsistency,
          }),
          signal: AbortSignal.timeout(this.timeoutMs),
        });

        if (res.status === 200) {
          cb?.recordSuccess();
          const data = (await res.json()) as {
            results?: Array<{
              id: string;
              score: number;
              is_certified?: boolean;
              proof_upper_bound?: number;
              metadata?: Record<string, unknown>;
            }>;
          };
          return (data.results || []).map((item) => ({
            id: item.id,
            score: item.score,
            isCertified: item.is_certified ?? certifiedExact,
            proofUpperBound: item.proof_upper_bound,
            metadata: item.metadata,
          }));
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

  public async upsert(
    id: string,
    vector: number[],
    metadata?: Record<string, unknown>,
    idempotencyKey?: string
  ): Promise<MutationReceipt> {
    let lastError: Error | undefined;

    for (let attempt = 0; attempt < this.maxRetries; attempt++) {
      const endpoint = this.selectEndpoint(true);
      const cb = this.circuitBreakers.get(endpoint);
      const url = `${endpoint}/upsert`;

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

