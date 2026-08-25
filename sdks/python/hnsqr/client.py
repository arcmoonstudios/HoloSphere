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
        certified_exact: bool = True,
    ) -> List[SearchResult]:
        payload = {
            "vector": vector,
            "k": k,
            "filter": filter_expr,
            "certified_exact": certified_exact,
            "consistency": self.read_consistency.value if isinstance(self.read_consistency, ReadConsistency) else self.read_consistency,
        }
        
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
                            is_certified=item.get("is_certified", certified_exact),
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
        certified_exact: bool = True,
    ) -> List[SearchResult]:
        """Direct text search utilizing in-database neural inference."""
        payload = {
            "query_text": query_text,
            "k": k,
            "certified_exact": certified_exact,
        }
        endpoint = self._select_endpoint(is_write=False)
        url = f"{endpoint}/v1/collections/{collection}/search"
        resp = await self._client.post(url, json=payload, headers=self._headers())
        resp.raise_for_status()
        return [
            SearchResult(
                id=item["id"],
                score=item["score"],
                is_certified=item.get("is_certified", certified_exact),
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
            "traverse": "/v1/knowledge/traverse",
            "resolve": "/v1/knowledge/resolve",
            "remember": "/v1/knowledge/remember",
            "record_outcome": "/v1/knowledge/outcomes",
        }
        if operation not in paths:
            raise ValueError(f"Unknown model tool operation: {operation}")
        endpoint = self._select_endpoint(is_write=operation in {"remember", "record_outcome"})
        response = await self._client.post(
            f"{endpoint.rstrip('/')}{paths[operation]}",
            json=payload,
            headers=self._headers(),
        )
        response.raise_for_status()
        return response.json()


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
        certified_exact: bool = True,
    ) -> List[SearchResult]:
        payload = {
            "vector": vector,
            "k": k,
            "filter": filter_expr,
            "certified_exact": certified_exact,
            "consistency": self.read_consistency.value if isinstance(self.read_consistency, ReadConsistency) else self.read_consistency,
        }
        
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
                            is_certified=item.get("is_certified", certified_exact),
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
        certified_exact: bool = True,
    ) -> List[SearchResult]:
        payload = {
            "query_text": query_text,
            "k": k,
            "certified_exact": certified_exact,
        }
        endpoint = self._select_endpoint(is_write=False)
        url = f"{endpoint}/v1/collections/{collection}/search"
        resp = self._client.post(url, json=payload, headers=self._headers())
        resp.raise_for_status()
        return [
            SearchResult(
                id=item["id"],
                score=item["score"],
                is_certified=item.get("is_certified", certified_exact),
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
            "traverse": "/v1/knowledge/traverse",
            "resolve": "/v1/knowledge/resolve",
            "remember": "/v1/knowledge/remember",
            "record_outcome": "/v1/knowledge/outcomes",
        }
        if operation not in paths:
            raise ValueError(f"Unknown model tool operation: {operation}")
        endpoint = self._select_endpoint(is_write=operation in {"remember", "record_outcome"})
        response = self._client.post(
            f"{endpoint.rstrip('/')}{paths[operation]}",
            json=payload,
            headers=self._headers(),
        )
        response.raise_for_status()
        return response.json()
