//! Configurable text-embedding providers for the model-facing knowledge service.
//!
//! HoloSphere indexes vectors, not model weights. Providers run outside the database
//! process and expose one version-pinned boundary. An OpenAI-compatible endpoint covers
//! LM Studio, llama.cpp servers, and hosted embedding services.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use serde::Deserialize;

use super::model_gateway::{EmbeddingDescriptor, local_text_embedding};
use super::web_search::{
    WebSearchConfig, WebSearchProvider, provider_from_config as web_provider_from_config,
};
use crate::{HNSQRError, HNSQRResult};

/// Produces normalized real-valued text embeddings in one declared embedding space.
pub trait TextEmbeddingProvider: Send + Sync {
    fn descriptor(&self) -> &EmbeddingDescriptor;
    fn embed(&self, text: &str) -> HNSQRResult<Vec<f32>>;
}

/// Runtime configuration loaded from `Config.toml`.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HoloSphereConfig {
    pub embedding: EmbeddingProviderConfig,
    #[serde(default)]
    pub web_search: Option<WebSearchConfig>,
}

/// `model_path` records the local artifact served by an external runtime. HoloSphere never
/// loads arbitrary GGUF/ONNX/PyTorch code in-process.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmbeddingProviderConfig {
    pub backend: EmbeddingBackend,
    pub provider: String,
    pub model: String,
    pub version: String,
    pub dimensions: usize,
    #[serde(default = "default_l2")]
    pub normalization: String,
    #[serde(default = "default_cosine")]
    pub distance_metric: String,
    #[serde(default)]
    pub endpoint: Option<String>,
    #[serde(default)]
    pub model_path: Option<PathBuf>,
    #[serde(default)]
    pub api_key_env: Option<String>,
    #[serde(default = "default_timeout_ms")]
    pub timeout_ms: u64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EmbeddingBackend {
    LexicalHash,
    OpenaiCompatible,
}

fn default_l2() -> String {
    "l2".to_string()
}
fn default_cosine() -> String {
    "cosine".to_string()
}
fn default_timeout_ms() -> u64 {
    30_000
}

impl EmbeddingProviderConfig {
    pub fn descriptor(&self) -> EmbeddingDescriptor {
        EmbeddingDescriptor {
            provider: self.provider.clone(),
            model: self.model.clone(),
            version: self.version.clone(),
            dimensions: self.dimensions,
            normalization: self.normalization.clone(),
            distance_metric: self.distance_metric.clone(),
        }
    }

    pub fn validate(&self) -> HNSQRResult<()> {
        self.descriptor().validate()?;
        if self.timeout_ms == 0 || self.timeout_ms > 300_000 {
            return Err(HNSQRError::InvalidConfig(
                "embedding.timeout_ms must be between 1 and 300000".to_string(),
            ));
        }
        match self.backend {
            EmbeddingBackend::LexicalHash => Ok(()),
            EmbeddingBackend::OpenaiCompatible => {
                let endpoint = self.endpoint.as_deref().unwrap_or("").trim();
                if endpoint.starts_with("http://") || endpoint.starts_with("https://") {
                    Ok(())
                } else {
                    Err(HNSQRError::InvalidConfig(
                        "embedding.endpoint must be an http(s) URL for openai_compatible"
                            .to_string(),
                    ))
                }
            }
        }
    }
}

/// Reads a complete configuration document and validates its provider declaration.
pub fn load_config(path: impl AsRef<Path>) -> HNSQRResult<HoloSphereConfig> {
    let path = path.as_ref();
    let source = fs::read_to_string(path).map_err(|error| {
        HNSQRError::InvalidConfig(format!("cannot read {}: {error}", path.display()))
    })?;
    let config: HoloSphereConfig = toml::from_str(&source).map_err(|error| {
        HNSQRError::InvalidConfig(format!("cannot parse {}: {error}", path.display()))
    })?;
    config.embedding.validate()?;
    if let Some(web_search) = &config.web_search {
        web_search.validate()?;
    }
    Ok(config)
}

/// The configured external capabilities available to one model tool service.
pub struct ConfiguredProviders {
    pub embedding: Arc<dyn TextEmbeddingProvider>,
    pub web_search: Option<Arc<dyn WebSearchProvider>>,
}

/// Loads all configured model-facing providers from one configuration document.
pub fn providers_from_file_if_exists(
    path: impl AsRef<Path>,
) -> HNSQRResult<Option<ConfiguredProviders>> {
    let path = path.as_ref();
    if !path.exists() {
        return Ok(None);
    }
    let config = load_config(path)?;
    Ok(Some(ConfiguredProviders {
        embedding: provider_from_config(&config.embedding)?,
        web_search: config
            .web_search
            .as_ref()
            .map(web_provider_from_config)
            .transpose()?,
    }))
}

pub fn provider_from_config(
    config: &EmbeddingProviderConfig,
) -> HNSQRResult<Arc<dyn TextEmbeddingProvider>> {
    config.validate()?;
    match config.backend {
        EmbeddingBackend::LexicalHash => Ok(Arc::new(LexicalHashProvider {
            descriptor: config.descriptor(),
        })),
        EmbeddingBackend::OpenaiCompatible => Ok(Arc::new(OpenAiCompatibleProvider::new(config)?)),
    }
}

/// Loads a configured provider when the file exists; callers retain lexical fallback if absent.
pub fn provider_from_file_if_exists(
    path: impl AsRef<Path>,
) -> HNSQRResult<Option<Arc<dyn TextEmbeddingProvider>>> {
    let path = path.as_ref();
    if !path.exists() {
        return Ok(None);
    }
    let config = load_config(path)?;
    provider_from_config(&config.embedding).map(Some)
}

/// Dependency-free deterministic fallback for tests and explicitly offline deployments.
pub struct LexicalHashProvider {
    pub(crate) descriptor: EmbeddingDescriptor,
}

impl TextEmbeddingProvider for LexicalHashProvider {
    fn descriptor(&self) -> &EmbeddingDescriptor {
        &self.descriptor
    }

    fn embed(&self, text: &str) -> HNSQRResult<Vec<f32>> {
        Ok(local_text_embedding(text, self.descriptor.dimensions))
    }
}

/// OpenAI `/embeddings` protocol provider, compatible with local and hosted servers.
pub struct OpenAiCompatibleProvider {
    descriptor: EmbeddingDescriptor,
    endpoint: String,
    api_key_env: Option<String>,
    timeout: Duration,
}

impl OpenAiCompatibleProvider {
    fn new(config: &EmbeddingProviderConfig) -> HNSQRResult<Self> {
        let endpoint = config
            .endpoint
            .as_ref()
            .expect("validated endpoint")
            .trim_end_matches('/');
        let endpoint = if endpoint.ends_with("/embeddings") {
            endpoint.to_string()
        } else {
            format!("{endpoint}/embeddings")
        };
        Ok(Self {
            descriptor: config.descriptor(),
            endpoint,
            api_key_env: config.api_key_env.clone(),
            timeout: Duration::from_millis(config.timeout_ms),
        })
    }
}

impl TextEmbeddingProvider for OpenAiCompatibleProvider {
    fn descriptor(&self) -> &EmbeddingDescriptor {
        &self.descriptor
    }

    fn embed(&self, text: &str) -> HNSQRResult<Vec<f32>> {
        let descriptor = self.descriptor.clone();
        let endpoint = self.endpoint.clone();
        let api_key_env = self.api_key_env.clone();
        let text = text.to_string();
        let timeout = self.timeout;
        std::thread::spawn(move || {
            embed_openai_compatible(descriptor, endpoint, api_key_env, timeout, text)
        })
        .join()
        .map_err(|_| HNSQRError::InvalidRequest("embedding provider worker panicked".to_string()))?
    }
}

fn embed_openai_compatible(
    descriptor: EmbeddingDescriptor,
    endpoint: String,
    api_key_env: Option<String>,
    timeout: Duration,
    text: String,
) -> HNSQRResult<Vec<f32>> {
    let client = reqwest::blocking::Client::builder()
        .timeout(timeout)
        .build()
        .map_err(|error| {
            HNSQRError::InvalidConfig(format!("invalid embedding HTTP client: {error}"))
        })?;
    let mut request = client.post(endpoint).json(&serde_json::json!({
        "model": descriptor.model,
        "input": text,
    }));
    if let Some(name) = api_key_env {
        let value = env::var(&name).map_err(|_| {
            HNSQRError::Unauthorized(format!(
                "embedding API key environment variable {name} is not set"
            ))
        })?;
        request = request.bearer_auth(value);
    }
    let response = request.send().map_err(|error| {
        HNSQRError::InvalidRequest(format!(
            "embedding provider {} request failed: {error}",
            descriptor.provider
        ))
    })?;
    let status = response.status();
    let body: OpenAiEmbeddingResponse = response.json().map_err(|error| {
        HNSQRError::InvalidRequest(format!(
            "embedding provider {} returned invalid JSON: {error}",
            descriptor.provider
        ))
    })?;
    if !status.is_success() {
        return Err(HNSQRError::InvalidRequest(format!(
            "embedding provider {} returned HTTP {status}",
            descriptor.provider
        )));
    }
    let vector = body
        .data
        .into_iter()
        .next()
        .map(|item| item.embedding)
        .ok_or_else(|| {
            HNSQRError::InvalidRequest("embedding provider returned no embeddings".to_string())
        })?;
    if vector.len() != descriptor.dimensions {
        return Err(HNSQRError::DimensionMismatch {
            expected: descriptor.dimensions,
            actual: vector.len(),
        });
    }
    if vector.iter().any(|value| !value.is_finite()) {
        return Err(HNSQRError::InvalidRequest(
            "embedding provider returned a non-finite value".to_string(),
        ));
    }
    Ok(vector)
}

#[derive(Deserialize)]
struct OpenAiEmbeddingResponse {
    data: Vec<OpenAiEmbeddingDatum>,
}

#[derive(Deserialize)]
struct OpenAiEmbeddingDatum {
    embedding: Vec<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::thread;

    #[test]
    fn bge_m3_configuration_preserves_embedding_identity() {
        let config = load_config(concat!(env!("CARGO_MANIFEST_DIR"), "/Config.toml")).unwrap();
        assert_eq!(config.embedding.backend, EmbeddingBackend::OpenaiCompatible);
        assert_eq!(config.embedding.descriptor().model, "text-embedding-bge-m3");
        assert_eq!(config.embedding.dimensions, 1024);
        assert!(
            config
                .embedding
                .model_path
                .as_ref()
                .unwrap()
                .ends_with("bge-m3-FP16.gguf")
        );
    }

    #[test]
    fn lexical_provider_is_available_without_an_external_runtime() {
        let config = EmbeddingProviderConfig {
            backend: EmbeddingBackend::LexicalHash,
            provider: "holosphere".to_string(),
            model: "lexical-hash".to_string(),
            version: "1".to_string(),
            dimensions: 16,
            normalization: "l2".to_string(),
            distance_metric: "cosine".to_string(),
            endpoint: None,
            model_path: None,
            api_key_env: None,
            timeout_ms: 1000,
        };
        let vector = provider_from_config(&config)
            .unwrap()
            .embed("retrieval evidence")
            .unwrap();
        assert_eq!(vector.len(), 16);
        assert!((vector.iter().map(|value| value * value).sum::<f32>() - 1.0).abs() < 1e-5);
    }

    #[tokio::test]
    async fn openai_compatible_provider_uses_the_standard_embeddings_contract() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0u8; 4096];
            let count = stream.read(&mut request).unwrap();
            let request = String::from_utf8_lossy(&request[..count]);
            assert!(request.starts_with("POST /v1/embeddings HTTP/1.1"));
            assert!(request.contains("\"model\":\"bge-m3\""));
            let body = r#"{"data":[{"embedding":[0.0,0.6,0.8]}]}"#;
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            )
            .unwrap();
        });
        let config = EmbeddingProviderConfig {
            backend: EmbeddingBackend::OpenaiCompatible,
            provider: "bge".to_string(),
            model: "bge-m3".to_string(),
            version: "test".to_string(),
            dimensions: 3,
            normalization: "l2".to_string(),
            distance_metric: "cosine".to_string(),
            endpoint: Some(format!("http://{address}/v1")),
            model_path: None,
            api_key_env: None,
            timeout_ms: 1_000,
        };
        let vector = provider_from_config(&config)
            .unwrap()
            .embed("repository indexing")
            .unwrap();
        server.join().unwrap();
        assert_eq!(vector, vec![0.0, 0.6, 0.8]);
    }
}
