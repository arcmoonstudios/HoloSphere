"""
HNSQR Official Python SDK Client
Provides high-throughput async and sync vector search with Certified exactness proofs,
native Graph-RAG queries, relational SQL ACID transactions, N-D Hypercube slicing, and in-memory KV caching.
"""
from dataclasses import dataclass
from enum import Enum
import random
import time
from typing import List, Dict, Any, Optional
import httpx

class ReadConsistency(str, Enum):
    LINEARIZABLE = "Linearizable"
    COMMITTED = "Committed"
    BOUNDED_STALENESS = "BoundedStaleness"

class HNSQRError(Exception):
    """Base error for HNSQR SDK operations."""
    pass

class HNSQRConnectionError(HNSQRError):
    """Network connection failure."""
    pass

class HNSQRCircuitOpenError(HNSQRError):
    """Fast-fail circuit breaker triggered due to consecutive endpoint failures."""
    pass

class HNSQRLeaderRedirect(HNSQRError):
    """Leader redirection notification."""
    def __init__(self, leader_endpoint: str):
        super().__init__(f"Redirect to active leader: {leader_endpoint}")
        self.leader_endpoint = leader_endpoint

@dataclass
class SearchResult:
    id: str
    score: float
    is_certified: bool
    proof_upper_bound: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class MutationReceipt:
    id: str
    lsn: int
    applied_generation: int
    is_quorum_replicated: bool

@dataclass
class GraphQueryResult:
    columns: List[str]
    rows: List[List[Any]]
    execution_time_micros: int

@dataclass
class SqlExecutionResult:
    columns: List[str]
    rows: List[Dict[str, Any]]
    affected_rows: int

@dataclass
class HypercubeSliceResult:
    coordinates: List[List[int]]
    values: List[float]
    total_voxels: int

@dataclass
class ProvenanceReference:
    source_id: str
    content_hash: str
    uri: Optional[str] = None
    observed_at_lsn: Optional[int] = None

@dataclass
class EvidenceEnvelope:
    tenant_id: str
    snapshot_lsn: int
    retrieval_contract: str
    certified: bool
    content_is_untrusted: bool
    results: Any
    proof_upper_bound: Optional[float] = None
    contradictions: Optional[List[Any]] = None

@dataclass
class KnowledgeRecord:
    id: str
    collection: str
    kind: str
    content: str
    commit_lsn: int
    tenant_id: str
    members: List[str]
    roles: Dict[str, str]
    metadata: Dict[str, Any]
    provenance: List[Dict[str, Any]]

@dataclass
class CandidateResolution:
    hypothesis: str
    confidence: float
    evidence_ids: List[str]
    successful_outcomes: int
    failed_outcomes: int
    status: str

@dataclass
class ModelOutcomeRecord:
    attempt_id: str
    summary: str
    successful: bool
    commit_lsn: int
    tenant_id: str
    evidence_ids: List[str]
    metrics: Dict[str, float]
    provenance: List[Dict[str, Any]]

@dataclass
class TaskCompleteResult:
    outcome: ModelOutcomeRecord
    resolution: Optional[KnowledgeRecord] = None
    resolution_status: Optional[str] = None
    verification_level: Optional[str] = None

class _CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout_s: float = 10.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout_s = recovery_timeout_s
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.is_open = False

    def record_success(self):
        self.failure_count = 0
        self.is_open = False

    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.is_open = True

    def can_execute(self) -> bool:
        if not self.is_open:
            return True
        if time.time() - self.last_failure_time > self.recovery_timeout_s:
            return True
        return False

class AsyncHNSQRClient:
    """Production asynchronous client utilizing httpx connection pooling and HTTP/2 multiplexing."""
    def __init__(
        self,
        endpoints: Optional[List[str]] = None,
        api_key: Optional[str] = None,
        tenant_id: Optional[str] = None,
        timeout: float = 5.0,
        max_retries: int = 3,
        read_consistency: ReadConsistency = ReadConsistency.COMMITTED,
        http2: bool = True,
    ):
        self.endpoints = endpoints or ["http://127.0.0.1:8080"]
        self.api_key = api_key
        self.tenant_id = tenant_id
        self.timeout = timeout
        self.max_retries = max_retries
        self.read_consistency = read_consistency
        self.active_leader: Optional[str] = None
        self._round_robin_idx = 0
        self._circuit_breakers: Dict[str, _CircuitBreaker] = {
            ep: _CircuitBreaker() for ep in self.endpoints
        }
        
        limits = httpx.Limits(max_keepalive_connections=50, max_connections=200)
        self._client = httpx.AsyncClient(
            timeout=self.timeout,
            limits=limits,
            http2=http2,
        )

    def _select_endpoint(self, is_write: bool = False) -> str:
        if is_write and self.active_leader:
            cb = self._circuit_breakers.get(self.active_leader)
            if cb and cb.can_execute():
                return self.active_leader

        healthy_eps = [
            ep for ep in self.endpoints
            if self._circuit_breakers[ep].can_execute()
        ]
        if not healthy_eps:
            raise HNSQRCircuitOpenError("All endpoints in cluster are failing or open")
        
        ep = healthy_eps[self._round_robin_idx % len(healthy_eps)]
        self._round_robin_idx += 1
        return ep

    def _headers(self, idempotency_key: Optional[str] = None) -> Dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        if self.tenant_id:
            h["X-HNSQR-Tenant-ID"] = self.tenant_id
        if idempotency_key:
            h["X-Idempotency-Key"] = idempotency_key
        return h

    async def search(
        self,
        collection: str,
        vector: List[float],
        k: int = 10,
        filter_expr: Optional[Dict[str, Any]] = None,
        retrieval_contract: str = "exact",
        certified_exact: Optional[bool] = None,
    ) -> List[SearchResult]:
        payload: Dict[str, Any] = {
            "vector": vector,
            "k": k,
            "filter": filter_expr,
            "retrieval_contract": retrieval_contract,
            "consistency": self.read_consistency.value if isinstance(self.read_consistency, ReadConsistency) else self.read_consistency,
        }
        if certified_exact is not None:
            payload["certified_exact"] = certified_exact
        
        for attempt in range(self.max_retries):
            endpoint = self._select_endpoint(is_write=False)
            cb = self._circuit_breakers[endpoint]
            url = f"{endpoint}/v1/collections/{collection}/search"
            try:
                resp = await self._client.post(url, json=payload, headers=self._headers())
                if resp.status_code == 200:
                    cb.record_success()
                    resp_json = resp.json()
                    return [
                        SearchResult(
                            id=item["id"],
                            score=item["score"],
                            is_certified=item.get("is_certified", False),
                            proof_upper_bound=item.get("proof_upper_bound"),
                            metadata=item.get("metadata"),
                        )
                        for item in resp_json.get("results", [])
                    ]
                elif resp.status_code in (307, 308):
                    leader = resp.headers.get("Location") or resp.json().get("leader_endpoint")
                    if leader:
                        self.active_leader = leader
                        continue
                resp.raise_for_status()
            except (httpx.RequestError, httpx.HTTPStatusError) as e:
                cb.record_failure()
                if attempt == self.max_retries - 1:
                    raise HNSQRConnectionError(f"HNSQR search failed on {endpoint}: {e}") from e
                import asyncio
                await asyncio.sleep((0.05 * (2 ** attempt)) + random.uniform(0.01, 0.05))
        return []

    async def embed_and_search(
        self,
        collection: str,
        query_text: str,
        k: int = 10,
        retrieval_contract: str = "exact",
        certified_exact: Optional[bool] = None,
    ) -> List[SearchResult]:
        """Direct text search utilizing in-database neural inference."""
        payload: Dict[str, Any] = {
            "query_text": query_text,
            "k": k,
            "retrieval_contract": retrieval_contract,
        }
        if certified_exact is not None:
            payload["certified_exact"] = certified_exact
        endpoint = self._select_endpoint(is_write=False)
        url = f"{endpoint}/v1/collections/{collection}/search"
        resp = await self._client.post(url, json=payload, headers=self._headers())
        resp.raise_for_status()
        return [
            SearchResult(
                id=item["id"],
                score=item["score"],
                is_certified=item.get("is_certified", False),
                metadata=item.get("metadata"),
            )
            for item in resp.json().get("results", [])
        ]

    async def query_graph(
        self,
        cypher_query: str,
    ) -> GraphQueryResult:
        """Executes a Cypher/GQL query with VECTOR MATCH against the graph engine."""
        payload = {"query": cypher_query}
        endpoint = self._select_endpoint(is_write=False)
        url = f"{endpoint}/v1/graph/query"
        resp = await self._client.post(url, json=payload, headers=self._headers())
        resp.raise_for_status()
        data = resp.json()
        return GraphQueryResult(
            columns=data.get("columns", []),
            rows=data.get("rows", []),
            execution_time_micros=data.get("execution_time_micros", 0),
        )

    async def execute_sql(
        self,
        sql_query: str,
    ) -> SqlExecutionResult:
        """Executes a relational SQL query with ACID transaction support."""
        payload = {"sql": sql_query}
        endpoint = self._select_endpoint(is_write=False)
        url = f"{endpoint}/v1/sql/execute"
        resp = await self._client.post(url, json=payload, headers=self._headers())
        resp.raise_for_status()
        data = resp.json()
        return SqlExecutionResult(
            columns=data.get("columns", []),
            rows=data.get("rows", []),
            affected_rows=data.get("affected_rows", 0),
        )

    async def slice_hypercube(
        self,
        space_id: str,
        min_coords: List[int],
        max_coords: List[int],
    ) -> HypercubeSliceResult:
        """Slices an N-dimensional volumetric tensor space."""
        payload = {
            "space_id": space_id,
            "min_coords": min_coords,
            "max_coords": max_coords,
        }
        endpoint = self._select_endpoint(is_write=False)
        url = f"{endpoint}/v1/hypercube/slice"
        resp = await self._client.post(url, json=payload, headers=self._headers())
        resp.raise_for_status()
        data = resp.json()
        return HypercubeSliceResult(
            coordinates=data.get("coordinates", []),
            values=data.get("values", []),
            total_voxels=data.get("total_voxels", 0),
        )

    async def get_billing_report(self, tenant_id: str) -> Dict[str, Any]:
        """Fetches usage-based metering and billing summary for a tenant."""
        endpoint = self._select_endpoint(is_write=False)
        url = f"{endpoint}/v1/dbaas/tenants/{tenant_id}/usage"
        resp = await self._client.get(url, headers=self._headers())
        resp.raise_for_status()
        return resp.json()

    async def upsert(
        self,
        collection: str,
        doc_id: str,
        vector: List[float],
        metadata: Optional[Dict[str, Any]] = None,
        idempotency_key: Optional[str] = None,
    ) -> MutationReceipt:
        payload = {
            "id": doc_id,
            "vector": vector,
            "metadata": metadata,
        }
        for attempt in range(self.max_retries):
            endpoint = self._select_endpoint(is_write=True)
            cb = self._circuit_breakers.setdefault(endpoint, _CircuitBreaker())
            url = f"{endpoint}/v1/collections/{collection}/insert"
            try:
                resp = await self._client.post(url, json=payload, headers=self._headers(idempotency_key))
                if resp.status_code == 200:
                    cb.record_success()
                    data = resp.json()
                    return MutationReceipt(
                        id=data.get("id", doc_id),
                        lsn=data.get("lsn", 0),
                        applied_generation=data.get("applied_generation", 1),
                        is_quorum_replicated=data.get("is_quorum_replicated", True),
                    )
                elif resp.status_code in (307, 308):
                    leader = resp.headers.get("Location") or resp.json().get("leader_endpoint")
                    if leader:
                        self.active_leader = leader
                        continue
                resp.raise_for_status()
            except (httpx.RequestError, httpx.HTTPStatusError) as e:
                cb.record_failure()
                if attempt == self.max_retries - 1:
                    raise HNSQRConnectionError(f"HNSQR upsert failed on {endpoint}: {e}") from e
                import asyncio
                await asyncio.sleep((0.05 * (2 ** attempt)) + random.uniform(0.01, 0.05))
        raise HNSQRError("Upsert retry budget exhausted")

    async def close(self):
        await self._client.aclose()

    @property
    def mcp_url(self) -> str:
        """Remote MCP Streamable HTTP URL for OpenAI, Gemini, or Claude."""
        return f"{self.endpoints[0].rstrip('/')}/mcp"

    async def call_model_tool(self, operation: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Call a provider-neutral HoloSphere knowledge operation."""
        paths = {
            "search": "/v1/knowledge/search",
            "web_search": "/v1/knowledge/web/search",
            "websearch": "/v1/knowledge/web/search",
            "traverse": "/v1/knowledge/traverse",
            "resolve": "/v1/knowledge/resolve",
            "remember": "/v1/knowledge/remember",
            "record_outcome": "/v1/knowledge/outcomes",
            "recordoutcome": "/v1/knowledge/outcomes",
            "task_begin": "/v1/knowledge/tasks/begin",
            "taskbegin": "/v1/knowledge/tasks/begin",
            "task_context": "/v1/knowledge/tasks/context",
            "taskcontext": "/v1/knowledge/tasks/context",
            "task_complete": "/v1/knowledge/tasks/complete",
            "taskcomplete": "/v1/knowledge/tasks/complete",
            "explore": "/v1/knowledge/explore",
            "status": "/v1/knowledge/status",
            "run_case": "/v1/knowledge/cases/run",
            "runcase": "/v1/knowledge/cases/run",
        }
        normalized = operation.lower().replace("-", "_").replace(".", "_")
        if normalized not in paths:
            raise ValueError(f"Unknown model tool operation: {operation}")
        is_write = normalized in {
            "remember",
            "record_outcome",
            "recordoutcome",
            "task_begin",
            "taskbegin",
            "task_complete",
            "taskcomplete",
        }
        endpoint = self._select_endpoint(is_write=is_write)
        response = await self._client.post(
            f"{endpoint.rstrip('/')}{paths[normalized]}",
            json=payload,
            headers=self._headers(),
        )
        response.raise_for_status()
        return response.json()

    async def task_begin(
        self,
        problem: str,
        case_id: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        collection: str = "knowledge",
        max_hypotheses: int = 5,
        provenance: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Starts a durable agent problem case with prior evidence linking."""
        payload: Dict[str, Any] = {
            "problem": problem,
            "collection": collection,
            "max_hypotheses": max_hypotheses,
        }
        if case_id:
            payload["case_id"] = case_id
        if idempotency_key:
            payload["idempotency_key"] = idempotency_key
        if provenance:
            payload["provenance"] = provenance
        return await self.call_model_tool("task_begin", payload)

    async def task_context(self, case_id: str, snapshot_lsn: Optional[int] = None) -> Dict[str, Any]:
        """Rehydrates a case's related evidence and graph context at a pinned snapshot."""
        payload: Dict[str, Any] = {"case_id": case_id}
        if snapshot_lsn is not None:
            payload["snapshot_lsn"] = snapshot_lsn
        return await self.call_model_tool("task_context", payload)

    async def web_search(
        self,
        query: str,
        k: int = 8,
        time_range: Optional[str] = None,
        language: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Queries live public-web results returning untrusted evidence records."""
        payload: Dict[str, Any] = {"query": query, "k": k}
        if time_range:
            payload["time_range"] = time_range
        if language:
            payload["language"] = language
        return await self.call_model_tool("web_search", payload)

    async def remember(
        self,
        content: str,
        doc_id: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        collection: str = "knowledge",
        kind: str = "knowledge",
        evidence_class: Optional[str] = None,
        members: Optional[List[str]] = None,
        roles: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        provenance: Optional[List[Dict[str, Any]]] = None,
        vector: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Durably remembers tenant-scoped knowledge, entities, or relations."""
        payload: Dict[str, Any] = {
            "content": content,
            "collection": collection,
            "kind": kind,
        }
        if doc_id:
            payload["id"] = doc_id
        if idempotency_key:
            payload["idempotency_key"] = idempotency_key
        if evidence_class:
            payload["evidence_class"] = evidence_class
        if members:
            payload["members"] = members
        if roles:
            payload["roles"] = roles
        if metadata:
            payload["metadata"] = metadata
        if provenance:
            payload["provenance"] = provenance
        if vector:
            payload["vector"] = vector
        return await self.call_model_tool("remember", payload)

    async def search_knowledge(
        self,
        query_text: Optional[str] = None,
        query_vector: Optional[List[float]] = None,
        collection: str = "knowledge",
        k: int = 10,
        kinds: Optional[List[str]] = None,
        retrieval_contract: str = "exact",
        snapshot_lsn: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Searches tenant-isolated HoloSphere knowledge at one snapshot."""
        payload: Dict[str, Any] = {
            "collection": collection,
            "k": k,
            "retrieval_contract": retrieval_contract,
        }
        if query_text:
            payload["query_text"] = query_text
        if query_vector:
            payload["query_vector"] = query_vector
        if kinds:
            payload["kinds"] = kinds
        if snapshot_lsn is not None:
            payload["snapshot_lsn"] = snapshot_lsn
        return await self.call_model_tool("search", payload)

    async def traverse(
        self,
        seed_ids: List[str],
        max_depth: int = 3,
        max_results: int = 100,
        relation_kinds: Optional[List[str]] = None,
        snapshot_lsn: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Traverses provenance-bearing N-ary knowledge relations from seed IDs."""
        payload: Dict[str, Any] = {
            "seed_ids": seed_ids,
            "max_depth": max_depth,
            "max_results": max_results,
        }
        if relation_kinds:
            payload["relation_kinds"] = relation_kinds
        if snapshot_lsn is not None:
            payload["snapshot_lsn"] = snapshot_lsn
        return await self.call_model_tool("traverse", payload)

    async def resolve(
        self,
        problem: str,
        collection: str = "knowledge",
        max_hypotheses: int = 5,
        snapshot_lsn: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Returns evidence-backed candidate resolutions and hypotheses."""
        payload: Dict[str, Any] = {
            "problem": problem,
            "collection": collection,
            "max_hypotheses": max_hypotheses,
        }
        if snapshot_lsn is not None:
            payload["snapshot_lsn"] = snapshot_lsn
        return await self.call_model_tool("resolve", payload)

    async def record_outcome(
        self,
        summary: str,
        successful: bool,
        attempt_id: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        evidence_ids: Optional[List[str]] = None,
        metrics: Optional[Dict[str, float]] = None,
        provenance: Optional[List[Dict[str, Any]]] = None,
        evidence_class: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Durably attaches measured outcomes and provenance to an attempted resolution."""
        payload: Dict[str, Any] = {
            "summary": summary,
            "successful": successful,
        }
        if attempt_id:
            payload["attempt_id"] = attempt_id
        if idempotency_key:
            payload["idempotency_key"] = idempotency_key
        if evidence_ids:
            payload["evidence_ids"] = evidence_ids
        if metrics:
            payload["metrics"] = metrics
        if provenance:
            payload["provenance"] = provenance
        if evidence_class:
            payload["evidence_class"] = evidence_class
        return await self.call_model_tool("record_outcome", payload)

    async def task_complete(
        self,
        case_id: str,
        summary: str,
        successful: bool,
        idempotency_key: Optional[str] = None,
        resolution_status: Optional[str] = None,
        evidence_ids: Optional[List[str]] = None,
        metrics: Optional[Dict[str, float]] = None,
        provenance: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Promotes a resolved case to a durable resolution linked by fixed_by."""
        payload: Dict[str, Any] = {
            "case_id": case_id,
            "summary": summary,
            "successful": successful,
        }
        if idempotency_key:
            payload["idempotency_key"] = idempotency_key
        if resolution_status:
            payload["resolution_status"] = resolution_status
        if evidence_ids:
            payload["evidence_ids"] = evidence_ids
        if metrics:
            payload["metrics"] = metrics
        if provenance:
            payload["provenance"] = provenance
        return await self.call_model_tool("task_complete", payload)

    async def explore(
        self,
        target: str = "stats",
        limit: int = 10,
        seed_id: Optional[str] = None,
        snapshot_lsn: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Explores memory topology stats, recent cases, or entity neighborhoods."""
        payload: Dict[str, Any] = {"target": target, "limit": limit}
        if seed_id:
            payload["seed_id"] = seed_id
        if snapshot_lsn is not None:
            payload["snapshot_lsn"] = snapshot_lsn
        return await self.call_model_tool("explore", payload)

    async def status(self) -> Dict[str, Any]:
        """Returns the runtime preflight contract before selecting a workflow."""
        return await self.call_model_tool("status", {})

    async def run_case(self, objective: str, **options: Any) -> Dict[str, Any]:
        """Prepares a bounded, evidence-first case without executing external actions."""
        return await self.call_model_tool("run_case", {"objective": objective, **options})


class HNSQRClient:
    """Production synchronous client with persistent connection pooling and robust failover."""
    def __init__(
        self,
        endpoints: Optional[List[str]] = None,
        api_key: Optional[str] = None,
        tenant_id: Optional[str] = None,
        timeout: float = 5.0,
        max_retries: int = 3,
        read_consistency: ReadConsistency = ReadConsistency.COMMITTED,
    ):
        self.endpoints = endpoints or ["http://127.0.0.1:8080"]
        self.api_key = api_key
        self.tenant_id = tenant_id
        self.timeout = timeout
        self.max_retries = max_retries
        self.read_consistency = read_consistency
        self.active_leader: Optional[str] = None
        self._round_robin_idx = 0
        self._circuit_breakers: Dict[str, _CircuitBreaker] = {
            ep: _CircuitBreaker() for ep in self.endpoints
        }
        
        limits = httpx.Limits(max_keepalive_connections=50, max_connections=200)
        self._client = httpx.Client(
            timeout=self.timeout,
            limits=limits,
        )

    def _select_endpoint(self, is_write: bool = False) -> str:
        if is_write and self.active_leader:
            cb = self._circuit_breakers.get(self.active_leader)
            if cb and cb.can_execute():
                return self.active_leader

        healthy_eps = [
            ep for ep in self.endpoints
            if self._circuit_breakers[ep].can_execute()
        ]
        if not healthy_eps:
            raise HNSQRCircuitOpenError("All endpoints in cluster are failing or open")
        
        ep = healthy_eps[self._round_robin_idx % len(healthy_eps)]
        self._round_robin_idx += 1
        return ep

    def _headers(self, idempotency_key: Optional[str] = None) -> Dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        if self.tenant_id:
            h["X-HNSQR-Tenant-ID"] = self.tenant_id
        if idempotency_key:
            h["X-Idempotency-Key"] = idempotency_key
        return h

    def search(
        self,
        collection: str,
        vector: List[float],
        k: int = 10,
        filter_expr: Optional[Dict[str, Any]] = None,
        retrieval_contract: str = "exact",
        certified_exact: Optional[bool] = None,
    ) -> List[SearchResult]:
        payload: Dict[str, Any] = {
            "vector": vector,
            "k": k,
            "filter": filter_expr,
            "retrieval_contract": retrieval_contract,
            "consistency": self.read_consistency.value if isinstance(self.read_consistency, ReadConsistency) else self.read_consistency,
        }
        if certified_exact is not None:
            payload["certified_exact"] = certified_exact
        
        for attempt in range(self.max_retries):
            endpoint = self._select_endpoint(is_write=False)
            cb = self._circuit_breakers[endpoint]
            url = f"{endpoint}/v1/collections/{collection}/search"
            try:
                resp = self._client.post(url, json=payload, headers=self._headers())
                if resp.status_code == 200:
                    cb.record_success()
                    resp_json = resp.json()
                    return [
                        SearchResult(
                            id=item["id"],
                            score=item["score"],
                            is_certified=item.get("is_certified", False),
                            proof_upper_bound=item.get("proof_upper_bound"),
                            metadata=item.get("metadata"),
                        )
                        for item in resp_json.get("results", [])
                    ]
                elif resp.status_code in (307, 308):
                    leader = resp.headers.get("Location") or resp.json().get("leader_endpoint")
                    if leader:
                        self.active_leader = leader
                        continue
                resp.raise_for_status()
            except (httpx.RequestError, httpx.HTTPStatusError) as e:
                cb.record_failure()
                if attempt == self.max_retries - 1:
                    raise HNSQRConnectionError(f"HNSQR search failed on {endpoint}: {e}") from e
                time.sleep((0.05 * (2 ** attempt)) + random.uniform(0.01, 0.05))
        return []

    def embed_and_search(
        self,
        collection: str,
        query_text: str,
        k: int = 10,
        retrieval_contract: str = "exact",
        certified_exact: Optional[bool] = None,
    ) -> List[SearchResult]:
        payload: Dict[str, Any] = {
            "query_text": query_text,
            "k": k,
            "retrieval_contract": retrieval_contract,
        }
        if certified_exact is not None:
            payload["certified_exact"] = certified_exact
        endpoint = self._select_endpoint(is_write=False)
        url = f"{endpoint}/v1/collections/{collection}/search"
        resp = self._client.post(url, json=payload, headers=self._headers())
        resp.raise_for_status()
        return [
            SearchResult(
                id=item["id"],
                score=item["score"],
                is_certified=item.get("is_certified", False),
                metadata=item.get("metadata"),
            )
            for item in resp.json().get("results", [])
        ]

    def query_graph(self, cypher_query: str) -> GraphQueryResult:
        payload = {"query": cypher_query}
        endpoint = self._select_endpoint(is_write=False)
        url = f"{endpoint}/v1/graph/query"
        resp = self._client.post(url, json=payload, headers=self._headers())
        resp.raise_for_status()
        data = resp.json()
        return GraphQueryResult(
            columns=data.get("columns", []),
            rows=data.get("rows", []),
            execution_time_micros=data.get("execution_time_micros", 0),
        )

    def execute_sql(self, sql_query: str) -> SqlExecutionResult:
        payload = {"sql": sql_query}
        endpoint = self._select_endpoint(is_write=False)
        url = f"{endpoint}/v1/sql/execute"
        resp = self._client.post(url, json=payload, headers=self._headers())
        resp.raise_for_status()
        data = resp.json()
        return SqlExecutionResult(
            columns=data.get("columns", []),
            rows=data.get("rows", []),
            affected_rows=data.get("affected_rows", 0),
        )

    def slice_hypercube(self, space_id: str, min_coords: List[int], max_coords: List[int]) -> HypercubeSliceResult:
        payload = {
            "space_id": space_id,
            "min_coords": min_coords,
            "max_coords": max_coords,
        }
        endpoint = self._select_endpoint(is_write=False)
        url = f"{endpoint}/v1/hypercube/slice"
        resp = self._client.post(url, json=payload, headers=self._headers())
        resp.raise_for_status()
        data = resp.json()
        return HypercubeSliceResult(
            coordinates=data.get("coordinates", []),
            values=data.get("values", []),
            total_voxels=data.get("total_voxels", 0),
        )

    def get_billing_report(self, tenant_id: str) -> Dict[str, Any]:
        """Fetches usage-based metering and billing summary for a tenant."""
        endpoint = self._select_endpoint(is_write=False)
        url = f"{endpoint}/v1/dbaas/tenants/{tenant_id}/usage"
        resp = self._client.get(url, headers=self._headers())
        resp.raise_for_status()
        return resp.json()

    def upsert(
        self,
        collection: str,
        doc_id: str,
        vector: List[float],
        metadata: Optional[Dict[str, Any]] = None,
        idempotency_key: Optional[str] = None,
    ) -> MutationReceipt:
        payload = {
            "id": doc_id,
            "vector": vector,
            "metadata": metadata,
        }
        for attempt in range(self.max_retries):
            endpoint = self._select_endpoint(is_write=True)
            cb = self._circuit_breakers.setdefault(endpoint, _CircuitBreaker())
            url = f"{endpoint}/v1/collections/{collection}/insert"
            try:
                resp = self._client.post(url, json=payload, headers=self._headers(idempotency_key))
                if resp.status_code == 200:
                    cb.record_success()
                    data = resp.json()
                    return MutationReceipt(
                        id=data.get("id", doc_id),
                        lsn=data.get("lsn", 0),
                        applied_generation=data.get("applied_generation", 1),
                        is_quorum_replicated=data.get("is_quorum_replicated", True),
                    )
                elif resp.status_code in (307, 308):
                    leader = resp.headers.get("Location") or resp.json().get("leader_endpoint")
                    if leader:
                        self.active_leader = leader
                        continue
                resp.raise_for_status()
            except (httpx.RequestError, httpx.HTTPStatusError) as e:
                cb.record_failure()
                if attempt == self.max_retries - 1:
                    raise HNSQRConnectionError(f"HNSQR upsert failed on {endpoint}: {e}") from e
                time.sleep((0.05 * (2 ** attempt)) + random.uniform(0.01, 0.05))
        raise HNSQRError("Upsert retry budget exhausted")

    def close(self):
        self._client.close()

    @property
    def mcp_url(self) -> str:
        """Remote MCP Streamable HTTP URL for OpenAI, Gemini, or Claude."""
        return f"{self.endpoints[0].rstrip('/')}/mcp"

    def call_model_tool(self, operation: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Call a provider-neutral HoloSphere knowledge operation."""
        paths = {
            "search": "/v1/knowledge/search",
            "web_search": "/v1/knowledge/web/search",
            "websearch": "/v1/knowledge/web/search",
            "traverse": "/v1/knowledge/traverse",
            "resolve": "/v1/knowledge/resolve",
            "remember": "/v1/knowledge/remember",
            "record_outcome": "/v1/knowledge/outcomes",
            "recordoutcome": "/v1/knowledge/outcomes",
            "task_begin": "/v1/knowledge/tasks/begin",
            "taskbegin": "/v1/knowledge/tasks/begin",
            "task_context": "/v1/knowledge/tasks/context",
            "taskcontext": "/v1/knowledge/tasks/context",
            "task_complete": "/v1/knowledge/tasks/complete",
            "taskcomplete": "/v1/knowledge/tasks/complete",
            "explore": "/v1/knowledge/explore",
            "status": "/v1/knowledge/status",
            "run_case": "/v1/knowledge/cases/run",
            "runcase": "/v1/knowledge/cases/run",
        }
        normalized = operation.lower().replace("-", "_").replace(".", "_")
        if normalized not in paths:
            raise ValueError(f"Unknown model tool operation: {operation}")
        is_write = normalized in {
            "remember",
            "record_outcome",
            "recordoutcome",
            "task_begin",
            "taskbegin",
            "task_complete",
            "taskcomplete",
        }
        endpoint = self._select_endpoint(is_write=is_write)
        response = self._client.post(
            f"{endpoint.rstrip('/')}{paths[normalized]}",
            json=payload,
            headers=self._headers(),
        )
        response.raise_for_status()
        return response.json()

    def task_begin(
        self,
        problem: str,
        case_id: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        collection: str = "knowledge",
        max_hypotheses: int = 5,
        provenance: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Starts a durable agent problem case with prior evidence linking."""
        payload: Dict[str, Any] = {
            "problem": problem,
            "collection": collection,
            "max_hypotheses": max_hypotheses,
        }
        if case_id:
            payload["case_id"] = case_id
        if idempotency_key:
            payload["idempotency_key"] = idempotency_key
        if provenance:
            payload["provenance"] = provenance
        return self.call_model_tool("task_begin", payload)

    def task_context(self, case_id: str, snapshot_lsn: Optional[int] = None) -> Dict[str, Any]:
        """Rehydrates a case's related evidence and graph context at a pinned snapshot."""
        payload: Dict[str, Any] = {"case_id": case_id}
        if snapshot_lsn is not None:
            payload["snapshot_lsn"] = snapshot_lsn
        return self.call_model_tool("task_context", payload)

    def web_search(
        self,
        query: str,
        k: int = 8,
        time_range: Optional[str] = None,
        language: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Queries live public-web results returning untrusted evidence records."""
        payload: Dict[str, Any] = {"query": query, "k": k}
        if time_range:
            payload["time_range"] = time_range
        if language:
            payload["language"] = language
        return self.call_model_tool("web_search", payload)

    def remember(
        self,
        content: str,
        doc_id: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        collection: str = "knowledge",
        kind: str = "knowledge",
        evidence_class: Optional[str] = None,
        members: Optional[List[str]] = None,
        roles: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        provenance: Optional[List[Dict[str, Any]]] = None,
        vector: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Durably remembers tenant-scoped knowledge, entities, or relations."""
        payload: Dict[str, Any] = {
            "content": content,
            "collection": collection,
            "kind": kind,
        }
        if doc_id:
            payload["id"] = doc_id
        if idempotency_key:
            payload["idempotency_key"] = idempotency_key
        if evidence_class:
            payload["evidence_class"] = evidence_class
        if members:
            payload["members"] = members
        if roles:
            payload["roles"] = roles
        if metadata:
            payload["metadata"] = metadata
        if provenance:
            payload["provenance"] = provenance
        if vector:
            payload["vector"] = vector
        return self.call_model_tool("remember", payload)

    def search_knowledge(
        self,
        query_text: Optional[str] = None,
        query_vector: Optional[List[float]] = None,
        collection: str = "knowledge",
        k: int = 10,
        kinds: Optional[List[str]] = None,
        retrieval_contract: str = "exact",
        snapshot_lsn: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Searches tenant-isolated HoloSphere knowledge at one snapshot."""
        payload: Dict[str, Any] = {
            "collection": collection,
            "k": k,
            "retrieval_contract": retrieval_contract,
        }
        if query_text:
            payload["query_text"] = query_text
        if query_vector:
            payload["query_vector"] = query_vector
        if kinds:
            payload["kinds"] = kinds
        if snapshot_lsn is not None:
            payload["snapshot_lsn"] = snapshot_lsn
        return self.call_model_tool("search", payload)

    def traverse(
        self,
        seed_ids: List[str],
        max_depth: int = 3,
        max_results: int = 100,
        relation_kinds: Optional[List[str]] = None,
        snapshot_lsn: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Traverses provenance-bearing N-ary knowledge relations from seed IDs."""
        payload: Dict[str, Any] = {
            "seed_ids": seed_ids,
            "max_depth": max_depth,
            "max_results": max_results,
        }
        if relation_kinds:
            payload["relation_kinds"] = relation_kinds
        if snapshot_lsn is not None:
            payload["snapshot_lsn"] = snapshot_lsn
        return self.call_model_tool("traverse", payload)

    def resolve(
        self,
        problem: str,
        collection: str = "knowledge",
        max_hypotheses: int = 5,
        snapshot_lsn: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Returns evidence-backed candidate resolutions and hypotheses."""
        payload: Dict[str, Any] = {
            "problem": problem,
            "collection": collection,
            "max_hypotheses": max_hypotheses,
        }
        if snapshot_lsn is not None:
            payload["snapshot_lsn"] = snapshot_lsn
        return self.call_model_tool("resolve", payload)

    def record_outcome(
        self,
        summary: str,
        successful: bool,
        attempt_id: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        evidence_ids: Optional[List[str]] = None,
        metrics: Optional[Dict[str, float]] = None,
        provenance: Optional[List[Dict[str, Any]]] = None,
        evidence_class: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Durably attaches measured outcomes and provenance to an attempted resolution."""
        payload: Dict[str, Any] = {
            "summary": summary,
            "successful": successful,
        }
        if attempt_id:
            payload["attempt_id"] = attempt_id
        if idempotency_key:
            payload["idempotency_key"] = idempotency_key
        if evidence_ids:
            payload["evidence_ids"] = evidence_ids
        if metrics:
            payload["metrics"] = metrics
        if provenance:
            payload["provenance"] = provenance
        if evidence_class:
            payload["evidence_class"] = evidence_class
        return self.call_model_tool("record_outcome", payload)

    def task_complete(
        self,
        case_id: str,
        summary: str,
        successful: bool,
        idempotency_key: Optional[str] = None,
        resolution_status: Optional[str] = None,
        evidence_ids: Optional[List[str]] = None,
        metrics: Optional[Dict[str, float]] = None,
        provenance: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Promotes a resolved case to a durable resolution linked by fixed_by."""
        payload: Dict[str, Any] = {
            "case_id": case_id,
            "summary": summary,
            "successful": successful,
        }
        if idempotency_key:
            payload["idempotency_key"] = idempotency_key
        if resolution_status:
            payload["resolution_status"] = resolution_status
        if evidence_ids:
            payload["evidence_ids"] = evidence_ids
        if metrics:
            payload["metrics"] = metrics
        if provenance:
            payload["provenance"] = provenance
        return self.call_model_tool("task_complete", payload)

    def explore(
        self,
        target: str = "stats",
        limit: int = 10,
        seed_id: Optional[str] = None,
        snapshot_lsn: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Explores memory topology stats, recent cases, or entity neighborhoods."""
        payload: Dict[str, Any] = {"target": target, "limit": limit}
        if seed_id:
            payload["seed_id"] = seed_id
        if snapshot_lsn is not None:
            payload["snapshot_lsn"] = snapshot_lsn
        return self.call_model_tool("explore", payload)

    def status(self) -> Dict[str, Any]:
        """Returns the runtime preflight contract before selecting a workflow."""
        return self.call_model_tool("status", {})

    def run_case(self, objective: str, **options: Any) -> Dict[str, Any]:
        """Prepares a bounded, evidence-first case without executing external actions."""
        return self.call_model_tool("run_case", {"objective": objective, **options})


import hashlib

def _sha256_hex(content: str) -> str:
    return f"sha256:{hashlib.sha256(content.encode('utf-8')).hexdigest()}"


def execute_universal_epistemic_pipeline(
    client: HNSQRClient,
    domain: str,
    problem_statement: str,
    search_query: Optional[str] = None,
    synthesize_evidence: Optional[Any] = None,
    evaluate_candidate: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Executes the 5-phase HoloSphere Universal Epistemic Pipeline:
    1. Epistemic Inception & Context Rehydration
    2. External Ground Truth & Empirical Entropy Injection
    3. High-Dimensional Hypergraph Topology Synthesis
    4. Deduction, Traversal & Constraint Resolution
    5. Empirical Verification & Durable Resolution Promotion
    """
    clean_domain = domain.lower().replace("-", "_").replace(".", "_")
    case_id = f"case:{clean_domain}:{int(time.time() * 1000)}"
    init_hash = _sha256_hex(problem_statement)

    # Phase 1: Inception & Context Hydration
    client.task_begin(
        case_id=case_id,
        problem=problem_statement,
        idempotency_key=f"idemp:{case_id}",
        provenance=[{"source_id": "agent_orchestrator", "content_hash": init_hash}],
    )
    context_data = client.task_context(case_id=case_id)
    evidence_ids: List[str] = []

    # Phase 2: External Entropy Injection
    if search_query:
        web_data = client.web_search(query=search_query, k=8)
        raw_items = web_data.get("results", {}).get("results", [])
        if synthesize_evidence and raw_items:
            synthesized = synthesize_evidence(raw_items)
            # Phase 3: Graph Synthesis
            for item in synthesized:
                eid = item["id"]
                content = item["content"]
                evidence_ids.append(eid)
                client.remember(
                    doc_id=eid,
                    content=content,
                    kind=item.get("kind", "domain_evidence"),
                    members=[case_id, eid],
                    roles=item.get("roles", {case_id: "target_scope"}),
                    provenance=[{"source_id": "epistemic_synthesizer", "content_hash": _sha256_hex(content)}],
                )

    # Phase 4: Deduction, Traversal & Resolution
    client.traverse(seed_ids=[case_id], max_depth=3)
    resolutions = client.resolve(problem=problem_statement, max_hypotheses=5)

    # Phase 5: Empirical Verification & Durable Promotion
    if evaluate_candidate is None:
        raise ValueError(
            "evaluate_candidate is required: the pipeline never fabricates a successful or verified outcome"
        )
    eval_result = evaluate_candidate(resolutions.get("results", []), context_data.get("results", {}))

    summary = eval_result["summary"]
    is_success = eval_result["is_success"]
    metrics = eval_result.get("metrics", {})
    eval_hash = _sha256_hex(summary)

    client.record_outcome(
        attempt_id=f"attempt:{case_id}",
        evidence_ids=evidence_ids,
        metrics=metrics,
        successful=is_success,
        summary=summary,
        provenance=[{"source_id": "evaluation_runner", "content_hash": eval_hash}],
    )

    return client.task_complete(
        case_id=case_id,
        evidence_ids=evidence_ids,
        metrics=metrics,
        successful=is_success,
        summary=summary,
        # Metrics without an admissible measurement are a reported claim, not empirical verification.
        resolution_status="speculative_synthesis" if is_success else "hypothesis",
        provenance=[{"source_id": "evaluation_runner", "content_hash": eval_hash}],
    )
