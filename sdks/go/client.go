// Package hnsqr provides the official Go client for HNSQR.
// Copyright (c) 2026 ArcMoon Studios. MIT / Apache-2.0 License.
package hnsqr

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math/rand"
	"net"
	"net/http"
	"sync"
	"sync/atomic"
	"time"
)

type ReadConsistency string

const (
	Linearizable     ReadConsistency = "Linearizable"
	Committed        ReadConsistency = "Committed"
	BoundedStaleness ReadConsistency = "BoundedStaleness"
)

type SearchResult struct {
	ID               string                 `json:"id"`
	Score            float32                `json:"score"`
	IsCertified      bool                   `json:"is_certified"`
	ProofUpperBound *float32               `json:"proof_upper_bound,omitempty"`
	Metadata         map[string]interface{} `json:"metadata,omitempty"`
}

type MutationReceipt struct {
	ID                 string `json:"id"`
	LSN                uint64 `json:"lsn"`
	AppliedGeneration  uint64 `json:"applied_generation"`
	IsQuorumReplicated bool   `json:"is_quorum_replicated"`
}

type GraphQueryResult struct {
	Columns              []string        `json:"columns"`
	Rows                 [][]interface{} `json:"rows"`
	ExecutionTimeMicros  uint64          `json:"execution_time_micros"`
}

type SqlExecutionResult struct {
	Columns      []string                 `json:"columns"`
	Rows         []map[string]interface{} `json:"rows"`
	AffectedRows int                      `json:"affected_rows"`
}

type HypercubeSliceResult struct {
	Coordinates [][]int   `json:"coordinates"`
	Values      []float32 `json:"values"`
	TotalVoxels int       `json:"total_voxels"`
}

type ClientConfig struct {
	Endpoints       []string
	APIKey          string
	TenantID        string
	Timeout         time.Duration
	MaxRetries      int
	ReadConsistency ReadConsistency
}

type circuitBreaker struct {
	mu           sync.Mutex
	failures     int
	lastFailure  time.Time
	isOpen       bool
	threshold    int
	recoveryTime time.Duration
}

func newCircuitBreaker() *circuitBreaker {
	return &circuitBreaker{
		threshold:    5,
		recoveryTime: 10 * time.Second,
	}
}

func (cb *circuitBreaker) canExecute() bool {
	cb.mu.Lock()
	defer cb.mu.Unlock()
	if !cb.isOpen {
		return true
	}
	if time.Since(cb.lastFailure) > cb.recoveryTime {
		return true
	}
	return false
}

func (cb *circuitBreaker) recordSuccess() {
	cb.mu.Lock()
	defer cb.mu.Unlock()
	cb.failures = 0
	cb.isOpen = false
}

func (cb *circuitBreaker) recordFailure() {
	cb.mu.Lock()
	defer cb.mu.Unlock()
	cb.failures++
	cb.lastFailure = time.Now()
	if cb.failures >= cb.threshold {
		cb.isOpen = true
	}
}

type Client struct {
	config       ClientConfig
	httpClient   *http.Client
	mu           sync.RWMutex
	activeLeader string
	counter      uint64
	breakers     map[string]*circuitBreaker
}

func NewClient(config ClientConfig) *Client {
	if len(config.Endpoints) == 0 {
		config.Endpoints = []string{"http://127.0.0.1:8080"}
	}
	if config.Timeout == 0 {
		config.Timeout = 5 * time.Second
	}
	if config.MaxRetries == 0 {
		config.MaxRetries = 3
	}
	if config.ReadConsistency == "" {
		config.ReadConsistency = Committed
	}

	transport := &http.Transport{
		MaxIdleConns:        100,
		MaxIdleConnsPerHost: 50,
		IdleConnTimeout:     90 * time.Second,
		DialContext: (&net.Dialer{
			Timeout:   2 * time.Second,
			KeepAlive: 30 * time.Second,
		}).DialContext,
	}

	breakers := make(map[string]*circuitBreaker, len(config.Endpoints))
	for _, ep := range config.Endpoints {
		breakers[ep] = newCircuitBreaker()
	}

	return &Client{
		config: config,
		httpClient: &http.Client{
			Transport: transport,
			Timeout:   config.Timeout,
		},
		breakers: breakers,
	}
}

func (c *Client) selectEndpoint(isWrite bool) (string, error) {
	c.mu.RLock()
	leader := c.activeLeader
	c.mu.RUnlock()

	if isWrite && leader != "" {
		if cb, ok := c.breakers[leader]; ok && cb.canExecute() {
			return leader, nil
		}
	}

	var healthy []string
	for _, ep := range c.config.Endpoints {
		if cb, ok := c.breakers[ep]; ok && cb.canExecute() {
			healthy = append(healthy, ep)
		}
	}

	if len(healthy) == 0 {
		return "", errors.New("circuit breaker open: all endpoints failing")
	}

	idx := atomic.AddUint64(&c.counter, 1) - 1
	return healthy[idx%uint64(len(healthy))], nil
}

func (c *Client) headers(req *http.Request, idempotencyKey string) {
	req.Header.Set("Content-Type", "application/json")
	if c.config.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.config.APIKey)
	}
	if c.config.TenantID != "" {
		req.Header.Set("X-HNSQR-Tenant-ID", c.config.TenantID)
	}
	if idempotencyKey != "" {
		req.Header.Set("X-Idempotency-Key", idempotencyKey)
	}
}

func (c *Client) Search(ctx context.Context, collection string, vector []float32, k int, certifiedExact bool) ([]SearchResult, error) {
	payload := map[string]interface{}{
		"vector":          vector,
		"k":               k,
		"certified_exact": certifiedExact,
		"consistency":     c.config.ReadConsistency,
	}
	data, _ := json.Marshal(payload)

	var lastErr error
	for attempt := 0; attempt < c.config.MaxRetries; attempt++ {
		endpoint, err := c.selectEndpoint(false)
		if err != nil {
			return nil, err
		}

		cb := c.breakers[endpoint]
		url := fmt.Sprintf("%s/v1/collections/%s/search", endpoint, collection)
		req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(data))
		if err != nil {
			return nil, err
		}
		c.headers(req, "")

		resp, err := c.httpClient.Do(req)
		if err == nil && resp.StatusCode == http.StatusOK {
			cb.recordSuccess()
			var wrapper struct {
				Results []SearchResult `json:"results"`
			}
			err = json.NewDecoder(resp.Body).Decode(&wrapper)
			resp.Body.Close()
			return wrapper.Results, err
		}

		if resp != nil {
			resp.Body.Close()
			if resp.StatusCode == http.StatusTemporaryRedirect || resp.StatusCode == http.StatusPermanentRedirect {
				loc := resp.Header.Get("Location")
				if loc != "" {
					c.mu.Lock()
					c.activeLeader = loc
					c.mu.Unlock()
					continue
				}
			}
		}

		cb.recordFailure()
		lastErr = err
		time.Sleep(time.Duration(50*int(1<<attempt)+rand.Intn(30)) * time.Millisecond)
	}

	if lastErr != nil {
		return nil, lastErr
	}
	return nil, errors.New("search retries exhausted")
}

func (c *Client) EmbedAndSearch(ctx context.Context, collection string, queryText string, k int, certifiedExact bool) ([]SearchResult, error) {
	endpoint, err := c.selectEndpoint(false)
	if err != nil {
		return nil, err
	}
	payload := map[string]interface{}{
		"query_text":      queryText,
		"k":               k,
		"certified_exact": certifiedExact,
	}
	data, _ := json.Marshal(payload)
	url := fmt.Sprintf("%s/v1/collections/%s/search", endpoint, collection)
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	c.headers(req, "")
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	var wrapper struct {
		Results []SearchResult `json:"results"`
	}
	err = json.NewDecoder(resp.Body).Decode(&wrapper)
	return wrapper.Results, err
}

func (c *Client) QueryGraph(ctx context.Context, cypherQuery string) (*GraphQueryResult, error) {
	endpoint, err := c.selectEndpoint(false)
	if err != nil {
		return nil, err
	}
	payload := map[string]interface{}{"query": cypherQuery}
	data, _ := json.Marshal(payload)
	url := fmt.Sprintf("%s/v1/graph/query", endpoint)
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	c.headers(req, "")
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	var result GraphQueryResult
	err = json.NewDecoder(resp.Body).Decode(&result)
	return &result, err
}

func (c *Client) ExecuteSql(ctx context.Context, sqlQuery string) (*SqlExecutionResult, error) {
	endpoint, err := c.selectEndpoint(false)
	if err != nil {
		return nil, err
	}
	payload := map[string]interface{}{"sql": sqlQuery}
	data, _ := json.Marshal(payload)
	url := fmt.Sprintf("%s/v1/sql/execute", endpoint)
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	c.headers(req, "")
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	var result SqlExecutionResult
	err = json.NewDecoder(resp.Body).Decode(&result)
	return &result, err
}

func (c *Client) SliceHypercube(ctx context.Context, spaceID string, minCoords []int, maxCoords []int) (*HypercubeSliceResult, error) {
	endpoint, err := c.selectEndpoint(false)
	if err != nil {
		return nil, err
	}
	payload := map[string]interface{}{
		"space_id":   spaceID,
		"min_coords": minCoords,
		"max_coords": maxCoords,
	}
	data, _ := json.Marshal(payload)
	url := fmt.Sprintf("%s/v1/hypercube/slice", endpoint)
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	c.headers(req, "")
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	var result HypercubeSliceResult
	err = json.NewDecoder(resp.Body).Decode(&result)
	return &result, err
}

func (c *Client) GetBillingReport(ctx context.Context, tenantID string) (map[string]interface{}, error) {
	endpoint, err := c.selectEndpoint(false)
	if err != nil {
		return nil, err
	}
	url := fmt.Sprintf("%s/v1/dbaas/tenants/%s/usage", endpoint, tenantID)
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, err
	}
	c.headers(req, "")
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	var report map[string]interface{}
	err = json.NewDecoder(resp.Body).Decode(&report)
	return report, err
}

func (c *Client) Upsert(ctx context.Context, collection string, id string, vector []float32, metadata map[string]interface{}, idempotencyKey string) (*MutationReceipt, error) {
	payload := map[string]interface{}{
		"id":       id,
		"vector":   vector,
		"metadata": metadata,
	}
	data, _ := json.Marshal(payload)

	var lastErr error
	for attempt := 0; attempt < c.config.MaxRetries; attempt++ {
		endpoint, err := c.selectEndpoint(true)
		if err != nil {
			return nil, err
		}

		cb := c.breakers[endpoint]
		url := fmt.Sprintf("%s/v1/collections/%s/insert", endpoint, collection)
		req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(data))
		if err != nil {
			return nil, err
		}
		c.headers(req, idempotencyKey)

		resp, err := c.httpClient.Do(req)
		if err == nil && resp.StatusCode == http.StatusOK {
			cb.recordSuccess()
			var receipt MutationReceipt
			err = json.NewDecoder(resp.Body).Decode(&receipt)
			resp.Body.Close()
			return &receipt, err
		}

		if resp != nil {
			body, _ := io.ReadAll(resp.Body)
			resp.Body.Close()
			if resp.StatusCode == http.StatusTemporaryRedirect || resp.StatusCode == http.StatusPermanentRedirect {
				loc := resp.Header.Get("Location")
				if loc != "" {
					c.mu.Lock()
					c.activeLeader = loc
					c.mu.Unlock()
					continue
				}
			}
			lastErr = fmt.Errorf("upsert failed with code %d: %s", resp.StatusCode, string(body))
		} else {
			lastErr = err
		}

		cb.recordFailure()
		time.Sleep(time.Duration(50*int(1<<attempt)+rand.Intn(30)) * time.Millisecond)
	}

	if lastErr != nil {
		return nil, lastErr
	}
	return nil, errors.New("upsert retries exhausted")
}
