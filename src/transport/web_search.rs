//! Safe, provider-configurable public web search for MCP clients.
//!
//! The initial provider is self-hosted SearXNG. This module intentionally exposes search
//! results only; it does not fetch arbitrary URLs and therefore cannot become an SSRF proxy.

use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{HNSQRError, HNSQRResult};

const MAX_QUERY_BYTES: usize = 4_096;
const MAX_RESULTS: usize = 20;

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WebSearchConfig {
    pub backend: WebSearchBackend,
    pub endpoint: String,
    #[serde(default = "default_timeout_ms")]
    pub timeout_ms: u64,
    #[serde(default = "default_max_results")]
    pub max_results: usize,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum WebSearchBackend {
    Searxng,
}

fn default_timeout_ms() -> u64 {
    15_000
}

fn default_max_results() -> usize {
    8
}

impl WebSearchConfig {
    pub fn validate(&self) -> HNSQRResult<()> {
        if !(self.endpoint.starts_with("http://") || self.endpoint.starts_with("https://")) {
            return Err(HNSQRError::InvalidConfig(
                "web_search.endpoint must be an http(s) URL".to_string(),
            ));
        }
        if self.timeout_ms == 0 || self.timeout_ms > 120_000 {
            return Err(HNSQRError::InvalidConfig(
                "web_search.timeout_ms must be between 1 and 120000".to_string(),
            ));
        }
        if self.max_results == 0 || self.max_results > MAX_RESULTS {
            return Err(HNSQRError::InvalidConfig(format!(
                "web_search.max_results must be between 1 and {MAX_RESULTS}"
            )));
        }
        Ok(())
    }
}

pub trait WebSearchProvider: Send + Sync {
    fn name(&self) -> &'static str;
    fn max_results(&self) -> usize;
    fn search(&self, request: &WebSearchToolRequest) -> HNSQRResult<WebSearchResponse>;
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WebSearchToolRequest {
    #[serde(alias = "query_text")]
    pub query: String,
    /// `max_results` is the conventional name used by many MCP clients.  Keep it as
    /// a wire-level alias for `k` so clients do not need provider-specific retries.
    #[serde(default = "default_request_k", alias = "max_results")]
    pub k: usize,
    #[serde(default)]
    pub language: Option<String>,
    #[serde(default)]
    pub time_range: Option<WebTimeRange>,
}

fn default_request_k() -> usize {
    8
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum WebTimeRange {
    Day,
    Month,
    Year,
}

impl WebSearchToolRequest {
    pub fn validate(&self, max_results: usize) -> HNSQRResult<()> {
        if self.query.trim().is_empty() || self.query.len() > MAX_QUERY_BYTES {
            return Err(HNSQRError::InvalidRequest(format!(
                "web search query must contain 1 to {MAX_QUERY_BYTES} bytes"
            )));
        }
        if self.k == 0 || self.k > max_results {
            return Err(HNSQRError::InvalidRequest(format!(
                "k must be between 1 and {max_results} for the configured web provider"
            )));
        }
        if let Some(language) = &self.language
            && (language.len() > 16
                || !language
                    .bytes()
                    .all(|byte| byte.is_ascii_alphabetic() || byte == b'-'))
        {
            return Err(HNSQRError::InvalidRequest(
                "language must be a short BCP-47-style language tag".to_string(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct WebSearchResponse {
    pub provider: String,
    pub retrieved_at_unix_secs: u64,
    pub content_is_untrusted: bool,
    pub results: Vec<WebSearchResult>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct WebSearchResult {
    /// Canonical durable evidence ID. It is stable for the retrieved content and
    /// can be supplied directly to case, traversal, and outcome workflows.
    pub evidence_id: String,
    pub title: String,
    pub url: String,
    pub snippet: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub published_at: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub engines: Vec<String>,
    pub content_hash: String,
}

pub fn provider_from_config(config: &WebSearchConfig) -> HNSQRResult<Arc<dyn WebSearchProvider>> {
    config.validate()?;
    match config.backend {
        WebSearchBackend::Searxng => Ok(Arc::new(SearxngProvider {
            endpoint: config.endpoint.trim_end_matches('/').to_string(),
            timeout: Duration::from_millis(config.timeout_ms),
            max_results: config.max_results,
        })),
    }
}

pub fn config_from_file(path: impl AsRef<Path>) -> HNSQRResult<WebSearchConfig> {
    let source = fs::read_to_string(path.as_ref()).map_err(|error| {
        HNSQRError::InvalidConfig(format!("cannot read web search config: {error}"))
    })?;
    let config: WebSearchConfig = toml::from_str(&source).map_err(|error| {
        HNSQRError::InvalidConfig(format!("cannot parse web search config: {error}"))
    })?;
    config.validate()?;
    Ok(config)
}

struct SearxngProvider {
    endpoint: String,
    timeout: Duration,
    max_results: usize,
}

impl WebSearchProvider for SearxngProvider {
    fn name(&self) -> &'static str {
        "searxng"
    }

    fn max_results(&self) -> usize {
        self.max_results
    }

    fn search(&self, request: &WebSearchToolRequest) -> HNSQRResult<WebSearchResponse> {
        request.validate(self.max_results)?;
        let endpoint = self.endpoint.clone();
        let timeout = self.timeout;
        let request = request.clone();
        std::thread::spawn(move || search_searxng(endpoint, timeout, request))
            .join()
            .map_err(|_| {
                HNSQRError::InvalidRequest("web search provider worker panicked".to_string())
            })?
    }
}

fn search_searxng(
    endpoint: String,
    timeout: Duration,
    request: WebSearchToolRequest,
) -> HNSQRResult<WebSearchResponse> {
    let client = reqwest::blocking::Client::builder()
        .timeout(timeout)
        .build()
        .map_err(|error| {
            HNSQRError::InvalidConfig(format!("invalid web search HTTP client: {error}"))
        })?;
    let mut parameters = vec![
        ("q", request.query),
        ("format", "json".to_string()),
        ("safesearch", "1".to_string()),
    ];
    if let Some(language) = request.language {
        parameters.push(("language", language));
    }
    if let Some(time_range) = request.time_range {
        parameters.push((
            "time_range",
            serde_json::to_string(&time_range)
                .unwrap()
                .trim_matches('"')
                .to_string(),
        ));
    }
    let response = client
        .get(endpoint)
        .query(&parameters)
        .send()
        .map_err(|error| HNSQRError::InvalidRequest(format!("SearXNG request failed: {error}")))?;
    let status = response.status();
    if !status.is_success() {
        return Err(HNSQRError::InvalidRequest(format!(
            "SearXNG returned HTTP {status}"
        )));
    }
    let body: SearxngResponse = response.json().map_err(|error| {
        HNSQRError::InvalidRequest(format!("SearXNG returned invalid JSON: {error}"))
    })?;
    let results = body
        .results
        .into_iter()
        .take(request.k)
        .filter(|result| result.url.starts_with("http://") || result.url.starts_with("https://"))
        .map(|result| {
            let content_hash = evidence_hash(&result.title, &result.url, &result.content);
            WebSearchResult {
                evidence_id: format!("source:web:{}", &content_hash[..16]),
                content_hash,
                title: result.title,
                url: result.url,
                snippet: result.content,
                published_at: result.published_date,
                engines: result.engines,
            }
        })
        .collect();
    Ok(WebSearchResponse {
        provider: "searxng".to_string(),
        retrieved_at_unix_secs: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        content_is_untrusted: true,
        results,
    })
}

fn evidence_hash(title: &str, url: &str, snippet: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"HOLOSPHERE_WEB_EVIDENCE_V1");
    hasher.update(title.as_bytes());
    hasher.update([0]);
    hasher.update(url.as_bytes());
    hasher.update([0]);
    hasher.update(snippet.as_bytes());
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[derive(Deserialize)]
struct SearxngResponse {
    #[serde(default)]
    results: Vec<SearxngResult>,
}

#[derive(Deserialize)]
struct SearxngResult {
    #[serde(default)]
    title: String,
    url: String,
    #[serde(default)]
    content: String,
    #[serde(default, rename = "publishedDate")]
    published_date: Option<String>,
    #[serde(default)]
    engines: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_bounded_safe_requests() {
        let request = WebSearchToolRequest {
            query: "HoloSphere repository indexing".to_string(),
            k: 8,
            language: Some("en-US".to_string()),
            time_range: Some(WebTimeRange::Month),
        };
        request.validate(8).unwrap();
        assert!(
            WebSearchToolRequest { k: 9, ..request }
                .validate(8)
                .is_err()
        );
    }

    #[test]
    fn accepts_max_results_as_a_compatibility_alias() {
        let request: WebSearchToolRequest = serde_json::from_value(serde_json::json!({
            "query": "MCP interoperability",
            "max_results": 4
        }))
        .unwrap();
        assert_eq!(request.k, 4);
    }

    #[test]
    fn evidence_hash_changes_with_source_content() {
        assert_ne!(
            evidence_hash("a", "https://a.example", "one"),
            evidence_hash("a", "https://a.example", "two")
        );
    }
}
