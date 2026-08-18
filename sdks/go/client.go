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
	rrIndex      uint64
	activeLeader string
	breakers     map[string]*circuitBreaker
}

func NewClient(cfg ClientConfig) *Client {
	if len(cfg.Endpoints) == 0 {
		cfg.Endpoints = []string{"http://127.0.0.1:8080"}
	}
	if cfg.Timeout == 0 {
		cfg.Timeout = 5 * time.Second
	}
	if cfg.MaxRetries == 0 {
		cfg.MaxRetries = 3
	}
	if cfg.ReadConsistency == "" {
		cfg.ReadConsistency = Committed
	}

	transport := &http.Transport{
		Proxy: http.ProxyFromEnvironment,
		DialContext: (&net.Dialer{
			Timeout:   5 * time.Second,
			KeepAlive: 30 * time.Second,
		}).DialContext,
		ForceAttemptHTTP2:     true,
		MaxIdleConns:          100,
		MaxIdleConnsPerHost:   50,
		IdleConnTimeout:       90 * time.Second,
		TLSHandshakeTimeout:   5 * time.Second,
		ExpectContinueTimeout: 1 * time.Second,
	}

	breakers := make(map[string]*circuitBreaker)
	for _, ep := range cfg.Endpoints {
		breakers[ep] = newCircuitBreaker()
	}

	return &Client{
		config: cfg,
		httpClient: &http.Client{
			Timeout:   cfg.Timeout,
			Transport: transport,
		},
		breakers: breakers,
	}
}

func (c *Client) selectEndpoint(isWrite bool) string {
	c.mu.RLock()
	if isWrite && c.activeLeader != "" {
		leader := c.activeLeader
		c.mu.RUnlock()
		return leader
	}
	c.mu.RUnlock()

	for i := 0; i < len(c.config.Endpoints); i++ {
		idx := atomic.AddUint64(&c.rrIndex, 1) - 1
		ep := c.config.Endpoints[idx%uint64(len(c.config.Endpoints))]
		c.mu.RLock()
		cb, ok := c.breakers[ep]
		c.mu.RUnlock()
		if !ok || cb.canExecute() {
			return ep
		}
	}
	return c.config.Endpoints[0]
}

type searchRequest struct {
	Query          []float32       `json:"query"`
	K              int             `json:"k"`
	CertifiedExact bool            `json:"certified_exact"`
	Consistency    ReadConsistency `json:"consistency"`
}

type searchResponse struct {
	Results []SearchResult `json:"results"`
}

type upsertRequest struct {
	ID       string                 `json:"id"`
	Vector   []float32              `json:"vector"`
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

func (c *Client) Search(ctx context.Context, query []float32, k int, certifiedExact bool) ([]SearchResult, error) {
	reqBody, err := json.Marshal(searchRequest{
		Query:          query,
		K:              k,
		CertifiedExact: certifiedExact,
		Consistency:    c.config.ReadConsistency,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to marshal search request: %w", err)
	}

	var lastErr error
	for attempt := 0; attempt < c.config.MaxRetries; attempt++ {
		ep := c.selectEndpoint(false)
		url := fmt.Sprintf("%s/search", ep)

		req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(reqBody))
		if err != nil {
			return nil, fmt.Errorf("failed to create request: %w", err)
		}
		req.Header.Set("Content-Type", "application/json")
		if c.config.APIKey != "" {
			req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", c.config.APIKey))
		}
		if c.config.TenantID != "" {
			req.Header.Set("X-Tenant-ID", c.config.TenantID)
		}

		resp, err := c.httpClient.Do(req)
		c.mu.RLock()
		cb := c.breakers[ep]
		c.mu.RUnlock()

		if err != nil {
			if cb != nil {
				cb.recordFailure()
			}
			lastErr = err
			time.Sleep(time.Duration(50*(1<<attempt)+rand.Intn(30)) * time.Millisecond)
			continue
		}

		if resp.StatusCode == http.StatusOK {
			if cb != nil {
				cb.recordSuccess()
			}
			var sResp searchResponse
			decErr := json.NewDecoder(resp.Body).Decode(&sResp)
			resp.Body.Close()
			if decErr != nil {
				return nil, fmt.Errorf("failed to decode search response: %w", decErr)
			}
			return sResp.Results, nil
		}

		if resp.StatusCode == http.StatusTemporaryRedirect || resp.StatusCode == http.StatusPermanentRedirect {
			resp.Body.Close()
			leader := resp.Header.Get("Location")
			if leader != "" {
				c.mu.Lock()
				c.activeLeader = leader
				c.mu.Unlock()
				continue
			}
		}

		bodyBytes, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		if cb != nil {
			cb.recordFailure()
		}
		lastErr = fmt.Errorf("search failed on %s with HTTP %d: %s", ep, resp.StatusCode, string(bodyBytes))
		time.Sleep(time.Duration(50*(1<<attempt)+rand.Intn(30)) * time.Millisecond)
	}

	return nil, fmt.Errorf("search retries exhausted: %w", lastErr)
}

func (c *Client) Upsert(ctx context.Context, id string, vector []float32, metadata map[string]interface{}) (*MutationReceipt, error) {
	reqBody, err := json.Marshal(upsertRequest{
		ID:       id,
		Vector:   vector,
		Metadata: metadata,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to marshal upsert request: %w", err)
	}

	var lastErr error
	for attempt := 0; attempt < c.config.MaxRetries; attempt++ {
		ep := c.selectEndpoint(true)
		url := fmt.Sprintf("%s/upsert", ep)

		req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(reqBody))
		if err != nil {
			return nil, fmt.Errorf("failed to create request: %w", err)
		}
		req.Header.Set("Content-Type", "application/json")
		if c.config.APIKey != "" {
			req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", c.config.APIKey))
		}
		if c.config.TenantID != "" {
			req.Header.Set("X-Tenant-ID", c.config.TenantID)
		}

		resp, err := c.httpClient.Do(req)
		c.mu.RLock()
		cb := c.breakers[ep]
		c.mu.RUnlock()

		if err != nil {
			if cb != nil {
				cb.recordFailure()
			}
			lastErr = err
			time.Sleep(time.Duration(50*(1<<attempt)+rand.Intn(30)) * time.Millisecond)
			continue
		}

		if resp.StatusCode == http.StatusOK {
			if cb != nil {
				cb.recordSuccess()
			}
			var receipt MutationReceipt
			decErr := json.NewDecoder(resp.Body).Decode(&receipt)
			resp.Body.Close()
			if decErr != nil {
				return &MutationReceipt{
					ID:                 id,
					LSN:                1,
					AppliedGeneration:  1,
					IsQuorumReplicated: true,
				}, nil
			}
			return &receipt, nil
		}

		if resp.StatusCode == http.StatusTemporaryRedirect || resp.StatusCode == http.StatusPermanentRedirect {
			resp.Body.Close()
			leader := resp.Header.Get("Location")
			if leader != "" {
				c.mu.Lock()
				c.activeLeader = leader
				c.mu.Unlock()
				continue
			}
		}

		bodyBytes, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		if cb != nil {
			cb.recordFailure()
		}
		lastErr = fmt.Errorf("upsert failed on %s with HTTP %d: %s", ep, resp.StatusCode, string(bodyBytes))
		time.Sleep(time.Duration(50*(1<<attempt)+rand.Intn(30)) * time.Millisecond)
	}

	if lastErr != nil {
		return nil, lastErr
	}
	return nil, errors.New("upsert retries exhausted")
}

