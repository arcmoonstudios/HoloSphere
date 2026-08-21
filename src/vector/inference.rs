/* hnsqr/src/vector/inference.rs */
//!▫~•◦-------------------------------‣
//! # In-Process Neural Model Inference & Text Tokenization
//! # Front 4: Qdrant/Weaviate Rival
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides direct in-database text embedding generation without Python
//! sidecars, JSON serialization, or external microservice boundaries.
//!
//! ## Architecture honesty contract
//!
//! HNSQR exposes three materially different embedding paths:
//!
//! - [`ModelArchitecture::HashProjection`]
//!   - dependency-free
//!   - deterministic
//!   - content-sensitive
//!   - no learned model parameters
//!   - lexical/hash projection only
//!
//! - [`ModelArchitecture::CustomProjector`]
//!   - real externally supplied static token/subword vectors
//!   - greedy WordPiece-style tokenization
//!   - weighted static pooling
//!   - no transformer contextualization
//!
//! - [`ModelArchitecture::BertMiniLM`] / [`ModelArchitecture::BgeDense`]
//!   - require an injected [`TransformerInferenceBackend`]
//!   - backend owns the real model runtime and model-correct tokenizer
//!   - backend must perform the actual transformer forward pass
//!   - this module never silently substitutes hash/static projection
//!
//! MiniLM/BGE variants therefore cannot be constructed from a token embedding
//! table alone. A backend must explicitly implement real contextual inference.
//!
//! All successful paths converge on:
//!
//! ```text
//! raw text
//!    ↓
//! backend-specific inference
//!    ↓
//! real-valued embedding
//!    ↓
//! optional L2 normalization
//!    ↓
//! ComplexWeaver::fold_token_embeddings_in_place
//! ```
//!
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::collections::HashMap;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};

use crate::vector::ComplexWeaver;
use crate::{HNSQRResult, VectorEmbedding};

/// Number of deterministic lexical buckets used by the built-in hash projector.
///
/// [BENCH REQUIRED]: retrieval quality, memory residency, initialization cost,
/// and throughput should be evaluated if this value changes.
const HASH_BUCKET_COUNT: usize = 1024;

/// Default maximum source-word length accepted by the static greedy WordPiece
/// tokenizer before falling back to `[UNK]`.
const DEFAULT_MAX_WORDPIECE_CHARS: usize = 100;

/// FNV-1a 64-bit offset basis.
const FNV64_OFFSET: u64 = 0xcbf_29ce_4842_2325;

/// FNV-1a 64-bit prime.
const FNV64_PRIME: u64 = 0x0000_0100_0000_01b3;

/// SplitMix64 golden-ratio increment.
const SPLITMIX_INCREMENT: u64 = 0x9e37_79b9_7f4a_7c15;

/// Supported in-process embedding architecture families.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ModelArchitecture {
    /// Deterministic dependency-free lexical hash projection.
    #[default]
    HashProjection,

    /// Real MiniLM/BERT-style transformer inference.
    ///
    /// Requires a [`TransformerInferenceBackend`].
    BertMiniLM,

    /// Real BGE dense-retrieval transformer inference.
    ///
    /// Requires a [`TransformerInferenceBackend`].
    BgeDense,

    /// Learned static token/subword projection.
    ///
    /// This is explicitly not a transformer architecture.
    CustomProjector,
}

impl Display for ModelArchitecture {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::HashProjection => f.write_str("HashProjection"),
            Self::BertMiniLM => f.write_str("BertMiniLM"),
            Self::BgeDense => f.write_str("BgeDense"),
            Self::CustomProjector => f.write_str("CustomProjector"),
        }
    }
}

/// Configuration for in-process embedding generation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InferenceModelConfig {
    /// Logical model/embedding-space identifier.
    pub model_name: String,

    /// Backend architecture.
    pub architecture: ModelArchitecture,

    /// Number of real-valued components before complex folding.
    ///
    /// HNSQR folds pairs of real values into complex components, so this must
    /// be positive and even.
    pub output_dimension: usize,

    /// Maximum number of backend-visible tokens/subwords.
    pub max_sequence_length: usize,

    /// Apply L2 normalization before complex folding.
    pub normalize_embeddings: bool,
}

impl Default for InferenceModelConfig {
    fn default() -> Self {
        Self {
            model_name: "hash-projection-v2".to_string(),
            architecture: ModelArchitecture::HashProjection,
            output_dimension: 384,
            max_sequence_length: 512,
            normalize_embeddings: true,
        }
    }
}

impl InferenceModelConfig {
    /// Validates invariants shared by all inference backends.
    pub fn validate(&self) -> Result<(), InferenceConfigError> {
        if self.model_name.trim().is_empty() {
            return Err(InferenceConfigError::EmptyModelName);
        }

        if self.output_dimension < 2 {
            return Err(InferenceConfigError::OutputDimensionTooSmall {
                dimension: self.output_dimension,
            });
        }

        if self.output_dimension % 2 != 0 {
            return Err(InferenceConfigError::OutputDimensionMustBeEven {
                dimension: self.output_dimension,
            });
        }

        if self.max_sequence_length == 0 {
            return Err(InferenceConfigError::SequenceLengthZero);
        }

        Ok(())
    }
}

/// Static tokenizer options used only by [`ModelArchitecture::CustomProjector`].
///
/// Real MiniLM/BGE tokenization belongs to the injected transformer backend.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct StaticTokenizerConfig {
    /// Lowercase basic lexical units before WordPiece lookup.
    pub lowercase: bool,

    /// Words above this many Unicode scalar values become `[UNK]`.
    pub max_input_chars_per_word: usize,
}

impl Default for StaticTokenizerConfig {
    fn default() -> Self {
        Self {
            lowercase: true,
            max_input_chars_per_word: DEFAULT_MAX_WORDPIECE_CHARS,
        }
    }
}

/// Construction-time failures for the inference subsystem.
///
/// This error is intentionally local instead of assuming a particular
/// `HNSQRError` variant exists elsewhere in the crate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InferenceConfigError {
    EmptyModelName,

    OutputDimensionTooSmall {
        dimension: usize,
    },

    OutputDimensionMustBeEven {
        dimension: usize,
    },

    SequenceLengthZero,

    TokenizerMaxWordCharsZero,

    ArchitectureMismatch {
        required: ModelArchitecture,
        supplied: ModelArchitecture,
    },

    ExplicitBackendRequired {
        architecture: ModelArchitecture,
    },

    VocabRead {
        path: PathBuf,
        message: String,
    },

    VocabIndexOverflow {
        index: usize,
    },

    MissingUnknownToken {
        path: PathBuf,
    },

    ProjectionWeightsEmpty,

    ProjectionShapeMismatch {
        weights: usize,
        output_dimension: usize,
    },

    ProjectionTableTooSmall {
        rows: usize,
        required_rows: usize,
    },

    ProjectionAllocationFailed {
        elements: usize,
    },

    ProjectionSizeOverflow {
        rows: usize,
        output_dimension: usize,
    },

    TransformerArchitectureMismatch {
        configured: ModelArchitecture,
        backend: ModelArchitecture,
    },

    TransformerModelNameMismatch {
        configured: String,
        backend: String,
    },

    TransformerDimensionMismatch {
        configured: usize,
        backend: usize,
    },

    TransformerSequenceLengthExceeded {
        configured: usize,
        backend_maximum: usize,
    },
}

impl Display for InferenceConfigError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyModelName => f.write_str("model_name must not be empty"),

            Self::OutputDimensionTooSmall { dimension } => {
                write!(
                    f,
                    "output_dimension must be at least 2, received {dimension}"
                )
            }

            Self::OutputDimensionMustBeEven { dimension } => {
                write!(
                    f,
                    "output_dimension must be even for complex folding, received {dimension}"
                )
            }

            Self::SequenceLengthZero => {
                f.write_str("max_sequence_length must be greater than zero")
            }

            Self::TokenizerMaxWordCharsZero => {
                f.write_str("static tokenizer max_input_chars_per_word must be greater than zero")
            }

            Self::ArchitectureMismatch { required, supplied } => {
                write!(
                    f,
                    "constructor requires architecture {required}, received {supplied}"
                )
            }

            Self::ExplicitBackendRequired { architecture } => {
                write!(
                    f,
                    "architecture {architecture} requires an explicit backend constructor"
                )
            }

            Self::VocabRead { path, message } => {
                write!(f, "failed to read vocabulary {}: {message}", path.display())
            }

            Self::VocabIndexOverflow { index } => {
                write!(f, "vocabulary index {index} cannot be represented as u32")
            }

            Self::MissingUnknownToken { path } => {
                write!(
                    f,
                    "vocabulary {} is missing required [UNK] token",
                    path.display()
                )
            }

            Self::ProjectionWeightsEmpty => {
                f.write_str("static token projection weights must not be empty")
            }

            Self::ProjectionShapeMismatch {
                weights,
                output_dimension,
            } => {
                write!(
                    f,
                    "projection weight length {weights} is not a multiple of \
                     output_dimension {output_dimension}"
                )
            }

            Self::ProjectionTableTooSmall {
                rows,
                required_rows,
            } => {
                write!(
                    f,
                    "projection table exposes {rows} rows but vocabulary requires \
                     at least {required_rows}"
                )
            }

            Self::ProjectionAllocationFailed { elements } => {
                write!(
                    f,
                    "failed to reserve deterministic hash projection matrix \
                     containing {elements} f32 elements"
                )
            }

            Self::ProjectionSizeOverflow {
                rows,
                output_dimension,
            } => {
                write!(
                    f,
                    "projection size overflow for {rows} rows × \
                     {output_dimension} dimensions"
                )
            }

            Self::TransformerArchitectureMismatch {
                configured,
                backend,
            } => {
                write!(
                    f,
                    "transformer backend architecture {backend} does not match \
                     configured architecture {configured}"
                )
            }

            Self::TransformerModelNameMismatch {
                configured,
                backend,
            } => {
                write!(
                    f,
                    "transformer backend model {backend:?} does not match \
                     configured model {configured:?}"
                )
            }

            Self::TransformerDimensionMismatch {
                configured,
                backend,
            } => {
                write!(
                    f,
                    "transformer backend output dimension {backend} does not match \
                     configured dimension {configured}"
                )
            }

            Self::TransformerSequenceLengthExceeded {
                configured,
                backend_maximum,
            } => {
                write!(
                    f,
                    "configured max_sequence_length {configured} exceeds \
                     transformer backend maximum {backend_maximum}"
                )
            }
        }
    }
}

impl Error for InferenceConfigError {}

/// Contract implemented by a real in-process transformer runtime.
///
/// Candle, ONNX Runtime, tract, Burn, or an HNSQR-native tensor engine can
/// implement this trait.
///
/// The backend owns:
///
/// - model-correct tokenization,
/// - attention-mask construction,
/// - positional/token-type handling where applicable,
/// - transformer layers,
/// - model-correct pooling.
///
/// The backend writes a real-valued sentence embedding into `output`.
///
/// The caller guarantees:
///
/// ```text
/// output.len() == self.output_dimension()
/// max_sequence_length <= self.max_sequence_length()
/// ```
///
/// Any runtime/model failure propagates through the crate's existing
/// [`HNSQRResult`] boundary.
pub trait TransformerInferenceBackend: Send + Sync {
    /// Architecture implemented by this backend.
    fn architecture(&self) -> ModelArchitecture;

    /// Exact logical model identifier.
    fn model_name(&self) -> &str;

    /// Real-valued output dimensionality.
    fn output_dimension(&self) -> usize;

    /// Maximum sequence length physically supported by the loaded model.
    fn max_sequence_length(&self) -> usize;

    /// Execute the complete contextual model inference path.
    fn embed_text_into(
        &self,
        text: &str,
        max_sequence_length: usize,
        output: &mut [f32],
    ) -> HNSQRResult<()>;
}

/// Dependency-free lexical hashing tokenizer used exclusively by
/// [`ModelArchitecture::HashProjection`].
///
/// It extracts alphanumeric/underscore lexical units, lowercases them, then maps
/// each unit to a stable bucket using FNV-1a.
///
/// No `[CLS]` or `[SEP]` tokens are injected because the hash projector is not a
/// transformer and gains nothing from a constant control-token bias.
struct HashingTokenizer {
    bucket_count: usize,
}

impl HashingTokenizer {
    fn new(bucket_count: usize) -> Self {
        debug_assert!(bucket_count > 0);

        Self { bucket_count }
    }

    fn tokenize(&self, text: &str, max_len: usize) -> Vec<u32> {
        if max_len == 0 {
            return Vec::new();
        }

        let mut tokens = Vec::with_capacity(max_len.min(64));
        let mut current = String::new();

        for character in text.chars() {
            if character.is_alphanumeric() || character == '_' {
                for lowered in character.to_lowercase() {
                    current.push(lowered);
                }

                continue;
            }

            self.flush_token(&mut current, &mut tokens, max_len);

            if tokens.len() >= max_len {
                break;
            }
        }

        if tokens.len() < max_len {
            self.flush_token(&mut current, &mut tokens, max_len);
        }

        tokens
    }

    fn flush_token(&self, current: &mut String, tokens: &mut Vec<u32>, max_len: usize) {
        if current.is_empty() || tokens.len() >= max_len {
            current.clear();
            return;
        }

        let hash = stable_hash64(current.as_bytes());
        let bucket = (hash % self.bucket_count as u64) as u32;

        tokens.push(bucket);
        current.clear();
    }
}

/// Greedy WordPiece-style tokenizer used by
/// [`ModelArchitecture::CustomProjector`].
///
/// This is deliberately not advertised as a complete Hugging Face/BERT tokenizer
/// implementation. Exact MiniLM/BGE preprocessing belongs inside a real
/// [`TransformerInferenceBackend`].
struct GreedyWordPieceTokenizer {
    vocab: HashMap<String, u32>,
    unk_token_id: u32,
    max_vocab_id: u32,
    config: StaticTokenizerConfig,
}

impl GreedyWordPieceTokenizer {
    fn from_vocab_file(
        path: &Path,
        config: StaticTokenizerConfig,
    ) -> Result<Self, InferenceConfigError> {
        if config.max_input_chars_per_word == 0 {
            return Err(InferenceConfigError::TokenizerMaxWordCharsZero);
        }

        let contents =
            std::fs::read_to_string(path).map_err(|error| InferenceConfigError::VocabRead {
                path: path.to_path_buf(),
                message: error.to_string(),
            })?;

        let mut vocab = HashMap::new();
        let mut max_vocab_id = 0_u32;

        for (index, line) in contents.lines().enumerate() {
            let token = line.trim_end_matches('\r');

            if token.is_empty() {
                continue;
            }

            let token_id = u32::try_from(index)
                .map_err(|_| InferenceConfigError::VocabIndexOverflow { index })?;

            max_vocab_id = max_vocab_id.max(token_id);
            vocab.insert(token.to_string(), token_id);
        }

        let unk_token_id = vocab.get("[UNK]").copied().ok_or_else(|| {
            InferenceConfigError::MissingUnknownToken {
                path: path.to_path_buf(),
            }
        })?;

        Ok(Self {
            vocab,
            unk_token_id,
            max_vocab_id,
            config,
        })
    }

    fn required_projection_rows(&self) -> usize {
        self.max_vocab_id as usize + 1
    }

    fn tokenize(&self, text: &str, max_len: usize) -> Vec<u32> {
        if max_len == 0 {
            return Vec::new();
        }

        let basic_tokens = self.basic_tokenize(text);
        let mut output = Vec::with_capacity(max_len.min(64));

        for token in basic_tokens {
            if output.len() >= max_len {
                break;
            }

            let pieces = self.tokenize_word(&token);

            for token_id in pieces {
                if output.len() >= max_len {
                    return output;
                }

                output.push(token_id);
            }
        }

        output
    }

    fn basic_tokenize(&self, text: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        let mut current = String::new();

        for character in text.chars() {
            if character.is_whitespace() || character.is_control() {
                push_nonempty_token(&mut current, &mut tokens);

                continue;
            }

            if character.is_alphanumeric() || character == '_' {
                if self.config.lowercase {
                    for lowered in character.to_lowercase() {
                        current.push(lowered);
                    }
                } else {
                    current.push(character);
                }

                continue;
            }

            push_nonempty_token(&mut current, &mut tokens);

            let punctuation = if self.config.lowercase {
                character.to_lowercase().collect::<String>()
            } else {
                character.to_string()
            };

            tokens.push(punctuation);
        }

        push_nonempty_token(&mut current, &mut tokens);

        tokens
    }

    fn tokenize_word(&self, word: &str) -> Vec<u32> {
        if word.chars().count() > self.config.max_input_chars_per_word {
            return vec![self.unk_token_id];
        }

        if let Some(token_id) = self.vocab.get(word).copied() {
            return vec![token_id];
        }

        let characters = word.chars().collect::<Vec<_>>();
        let mut result = Vec::new();
        let mut start = 0;

        while start < characters.len() {
            let mut end = characters.len();
            let mut matched = None;

            while start < end {
                let body = characters[start..end].iter().collect::<String>();

                let candidate = if start == 0 {
                    body
                } else {
                    let mut continuation = String::with_capacity(body.len() + 2);

                    continuation.push_str("##");
                    continuation.push_str(&body);
                    continuation
                };

                if let Some(token_id) = self.vocab.get(&candidate).copied() {
                    matched = Some((token_id, end));
                    break;
                }

                end -= 1;
            }

            match matched {
                Some((token_id, matched_end)) => {
                    result.push(token_id);
                    start = matched_end;
                }

                None => {
                    return vec![self.unk_token_id];
                }
            }
        }

        result
    }
}

/// Built-in deterministic lexical projection backend.
struct HashProjectionBackend {
    tokenizer: HashingTokenizer,

    /// Row-major matrix:
    ///
    /// ```text
    /// [hash bucket][real output dimension]
    /// ```
    weights: Vec<f32>,

    rows: usize,
}

impl HashProjectionBackend {
    fn new(config: &InferenceModelConfig) -> Result<Self, InferenceConfigError> {
        let weights = build_hash_projection_weights(
            config.model_name.as_str(),
            config.output_dimension,
            HASH_BUCKET_COUNT,
        )?;

        Ok(Self {
            tokenizer: HashingTokenizer::new(HASH_BUCKET_COUNT),
            weights,
            rows: HASH_BUCKET_COUNT,
        })
    }

    fn embed_into(
        &self,
        text: &str,
        max_sequence_length: usize,
        output_dimension: usize,
        output: &mut [f32],
    ) {
        output.fill(0.0);

        let token_ids = self.tokenizer.tokenize(text, max_sequence_length);

        pool_token_rows(
            &token_ids,
            &self.weights,
            self.rows,
            output_dimension,
            output,
        );
    }
}

/// Learned static token/subword vector backend.
///
/// This backend performs weighted pooling over actual externally supplied token
/// vectors. It does not perform transformer contextualization.
struct StaticProjectionBackend {
    tokenizer: GreedyWordPieceTokenizer,

    /// Row-major token embedding table:
    ///
    /// ```text
    /// [vocabulary token ID][real output dimension]
    /// ```
    token_embeddings: Vec<f32>,

    rows: usize,
}

impl StaticProjectionBackend {
    fn new(
        tokenizer: GreedyWordPieceTokenizer,
        token_embeddings: Vec<f32>,
        output_dimension: usize,
    ) -> Result<Self, InferenceConfigError> {
        if token_embeddings.is_empty() {
            return Err(InferenceConfigError::ProjectionWeightsEmpty);
        }

        if token_embeddings.len() % output_dimension != 0 {
            return Err(InferenceConfigError::ProjectionShapeMismatch {
                weights: token_embeddings.len(),
                output_dimension,
            });
        }

        let rows = token_embeddings.len() / output_dimension;

        let required_rows = tokenizer.required_projection_rows();

        if rows < required_rows {
            return Err(InferenceConfigError::ProjectionTableTooSmall {
                rows,
                required_rows,
            });
        }

        Ok(Self {
            tokenizer,
            token_embeddings,
            rows,
        })
    }

    fn embed_into(
        &self,
        text: &str,
        max_sequence_length: usize,
        output_dimension: usize,
        output: &mut [f32],
    ) {
        output.fill(0.0);

        let token_ids = self.tokenizer.tokenize(text, max_sequence_length);

        pool_token_rows(
            &token_ids,
            &self.token_embeddings,
            self.rows,
            output_dimension,
            output,
        );
    }
}

enum InferenceBackend {
    HashProjection(HashProjectionBackend),
    StaticProjection(StaticProjectionBackend),
    Transformer(Arc<dyn TransformerInferenceBackend>),
}

/// In-process text embedder.
///
/// Backend behavior is selected at construction and cannot silently downgrade to
/// another architecture.
pub struct InProcessModelEmbedder {
    config: InferenceModelConfig,
    backend: InferenceBackend,

    /// Counts successful embedding operations.
    total_inferences: AtomicU64,
}

impl InProcessModelEmbedder {
    /// Backward-compatible constructor for the built-in hash architecture.
    ///
    /// For fallible configuration handling, prefer [`Self::try_new`].
    ///
    /// # Panics
    ///
    /// Panics if the configuration is invalid or specifies an architecture other
    /// than [`ModelArchitecture::HashProjection`].
    pub fn new(config: InferenceModelConfig) -> Self {
        match Self::try_new(config) {
            Ok(embedder) => embedder,

            Err(error) => {
                panic!("invalid in-process hash projection configuration: {error}")
            }
        }
    }

    /// Creates the built-in dependency-free hash projector.
    ///
    /// This constructor deliberately refuses MiniLM/BGE/CustomProjector
    /// configurations rather than mutating or downgrading them.
    pub fn try_new(config: InferenceModelConfig) -> Result<Self, InferenceConfigError> {
        Self::new_hash_projection(config)
    }

    /// Explicit constructor for [`ModelArchitecture::HashProjection`].
    pub fn new_hash_projection(config: InferenceModelConfig) -> Result<Self, InferenceConfigError> {
        config.validate()?;

        require_architecture(config.architecture, ModelArchitecture::HashProjection)?;

        let backend = HashProjectionBackend::new(&config)?;

        Ok(Self {
            config,
            backend: InferenceBackend::HashProjection(backend),
            total_inferences: AtomicU64::new(0),
        })
    }

    /// Creates a learned static token/subword projector.
    ///
    /// `token_embeddings` must be a row-major table indexed directly by
    /// vocabulary token ID:
    ///
    /// ```text
    /// token_embeddings[token_id][output_dimension]
    /// ```
    ///
    /// No modulo remapping is performed. Missing rows are rejected during
    /// construction.
    ///
    /// This constructor only accepts [`ModelArchitecture::CustomProjector`].
    pub fn with_static_projection(
        config: InferenceModelConfig,
        vocab_path: &Path,
        token_embeddings: Vec<f32>,
    ) -> Result<Self, InferenceConfigError> {
        Self::with_static_projection_options(
            config,
            vocab_path,
            token_embeddings,
            StaticTokenizerConfig::default(),
        )
    }

    /// Same as [`Self::with_static_projection`] with explicit tokenizer
    /// configuration.
    pub fn with_static_projection_options(
        config: InferenceModelConfig,
        vocab_path: &Path,
        token_embeddings: Vec<f32>,
        tokenizer_config: StaticTokenizerConfig,
    ) -> Result<Self, InferenceConfigError> {
        config.validate()?;

        require_architecture(config.architecture, ModelArchitecture::CustomProjector)?;

        let tokenizer = GreedyWordPieceTokenizer::from_vocab_file(vocab_path, tokenizer_config)?;

        let backend =
            StaticProjectionBackend::new(tokenizer, token_embeddings, config.output_dimension)?;

        Ok(Self {
            config,
            backend: InferenceBackend::StaticProjection(backend),
            total_inferences: AtomicU64::new(0),
        })
    }

    /// Creates a real transformer-backed MiniLM/BGE embedder.
    ///
    /// The supplied backend must exactly match:
    ///
    /// - architecture,
    /// - model name,
    /// - output dimension,
    /// - requested sequence-length capability.
    ///
    /// No fallback occurs if any contract fails.
    pub fn with_transformer_backend<B>(
        config: InferenceModelConfig,
        backend: B,
    ) -> Result<Self, InferenceConfigError>
    where
        B: TransformerInferenceBackend + 'static,
    {
        Self::with_shared_transformer_backend(config, Arc::new(backend))
    }

    /// Shared-backend variant of
    /// [`Self::with_transformer_backend`].
    pub fn with_shared_transformer_backend(
        config: InferenceModelConfig,
        backend: Arc<dyn TransformerInferenceBackend>,
    ) -> Result<Self, InferenceConfigError> {
        config.validate()?;

        match config.architecture {
            ModelArchitecture::BertMiniLM | ModelArchitecture::BgeDense => {}

            architecture => {
                return Err(InferenceConfigError::ExplicitBackendRequired { architecture });
            }
        }

        if backend.architecture() != config.architecture {
            return Err(InferenceConfigError::TransformerArchitectureMismatch {
                configured: config.architecture,
                backend: backend.architecture(),
            });
        }

        if backend.model_name() != config.model_name {
            return Err(InferenceConfigError::TransformerModelNameMismatch {
                configured: config.model_name.clone(),
                backend: backend.model_name().to_string(),
            });
        }

        if backend.output_dimension() != config.output_dimension {
            return Err(InferenceConfigError::TransformerDimensionMismatch {
                configured: config.output_dimension,
                backend: backend.output_dimension(),
            });
        }

        if config.max_sequence_length > backend.max_sequence_length() {
            return Err(InferenceConfigError::TransformerSequenceLengthExceeded {
                configured: config.max_sequence_length,
                backend_maximum: backend.max_sequence_length(),
            });
        }

        Ok(Self {
            config,
            backend: InferenceBackend::Transformer(backend),
            total_inferences: AtomicU64::new(0),
        })
    }

    /// Embeds raw UTF-8 text directly in-process.
    ///
    /// Hash/static paths execute locally in this module.
    ///
    /// MiniLM/BGE paths execute through the injected real transformer backend.
    ///
    /// All paths converge on optional normalization followed by direct
    /// `ComplexWeaver` folding.
    pub fn embed_text(&self, text: &str) -> HNSQRResult<VectorEmbedding> {
        let dimension = self.config.output_dimension;

        let mut real_embedding = vec![0.0_f32; dimension];

        match &self.backend {
            InferenceBackend::HashProjection(backend) => {
                backend.embed_into(
                    text,
                    self.config.max_sequence_length,
                    dimension,
                    &mut real_embedding,
                );
            }

            InferenceBackend::StaticProjection(backend) => {
                backend.embed_into(
                    text,
                    self.config.max_sequence_length,
                    dimension,
                    &mut real_embedding,
                );
            }

            InferenceBackend::Transformer(backend) => {
                backend.embed_text_into(
                    text,
                    self.config.max_sequence_length,
                    &mut real_embedding,
                )?;
            }
        }

        if self.config.normalize_embeddings {
            l2_normalize_in_place(&mut real_embedding);
        }

        let embedding = ComplexWeaver::fold_token_embeddings_in_place(&real_embedding, dimension);

        self.total_inferences.fetch_add(1, Ordering::Relaxed);

        Ok(embedding)
    }

    /// Number of successful embedding operations.
    pub fn total_inferences(&self) -> u64 {
        self.total_inferences.load(Ordering::Relaxed)
    }

    /// Real-valued dimensionality before complex folding.
    pub fn output_dimension(&self) -> usize {
        self.config.output_dimension
    }

    /// Complex dimensionality after pairwise folding.
    pub fn complex_output_dimension(&self) -> usize {
        self.config.output_dimension / 2
    }

    /// Configured architecture.
    pub fn architecture(&self) -> ModelArchitecture {
        self.config.architecture
    }

    /// Logical model identifier.
    pub fn model_name(&self) -> &str {
        self.config.model_name.as_str()
    }

    /// Configured maximum sequence length.
    pub fn max_sequence_length(&self) -> usize {
        self.config.max_sequence_length
    }

    /// Whether final real-valued embeddings are L2-normalized.
    pub fn normalize_embeddings(&self) -> bool {
        self.config.normalize_embeddings
    }

    /// Full immutable configuration.
    pub fn config(&self) -> &InferenceModelConfig {
        &self.config
    }
}

fn require_architecture(
    supplied: ModelArchitecture,
    required: ModelArchitecture,
) -> Result<(), InferenceConfigError> {
    if supplied == required {
        return Ok(());
    }

    Err(InferenceConfigError::ArchitectureMismatch { required, supplied })
}

/// Pools token-table rows using deterministic position weighting.
///
/// Tokens earlier in the sequence receive slightly larger weight:
///
/// ```text
/// w(position) = 1 / sqrt(position + 1)
/// ```
///
/// The accumulated vector is divided by total weight, preventing vector
/// magnitude from scaling directly with sequence length.
///
/// [BENCH REQUIRED]: semantic/retrieval impact of this weighting function should
/// be evaluated against uniform mean pooling for the intended corpus.
fn pool_token_rows(
    token_ids: &[u32],
    table: &[f32],
    rows: usize,
    dimension: usize,
    output: &mut [f32],
) {
    debug_assert_eq!(output.len(), dimension);

    output.fill(0.0);

    if token_ids.is_empty() {
        return;
    }

    let mut total_weight = 0.0_f32;

    for (position, token_id) in token_ids.iter().copied().enumerate() {
        let row = token_id as usize;

        debug_assert!(row < rows, "token row must be validated before pooling");

        let offset = row * dimension;

        let projection_row = &table[offset..offset + dimension];

        let weight = 1.0 / ((position + 1) as f32).sqrt();

        total_weight += weight;

        for (destination, source) in output.iter_mut().zip(projection_row.iter().copied()) {
            *destination += source * weight;
        }
    }

    if total_weight > f32::EPSILON {
        let inverse = total_weight.recip();

        for value in output {
            *value *= inverse;
        }
    }
}

/// Builds the deterministic hash-projection matrix.
///
/// SplitMix64 is used instead of transcendental `sin`/`cos` generation.
///
/// [BENCH REQUIRED]: constructor latency must be measured before claiming this
/// is faster than another deterministic initializer.
fn build_hash_projection_weights(
    model_name: &str,
    dimension: usize,
    rows: usize,
) -> Result<Vec<f32>, InferenceConfigError> {
    let elements =
        rows.checked_mul(dimension)
            .ok_or(InferenceConfigError::ProjectionSizeOverflow {
                rows,
                output_dimension: dimension,
            })?;

    let mut weights = Vec::new();

    weights
        .try_reserve_exact(elements)
        .map_err(|_| InferenceConfigError::ProjectionAllocationFailed { elements })?;

    let model_seed = stable_hash64(model_name.as_bytes()) ^ 0x484e_5351_525f_4850;

    let dimension_scale = (dimension as f32).sqrt().recip();

    for row in 0..rows {
        let row_seed = splitmix64(model_seed ^ (row as u64).wrapping_mul(0xa076_1d64_78bd_642f));

        for column in 0..dimension {
            let mixed = splitmix64(row_seed ^ (column as u64).wrapping_mul(0xe703_7ed1_a0b4_28db));

            let sample = ((mixed >> 40) & 0x00ff_ffff) as u32;

            let unit = sample as f32 / 16_777_215.0_f32;

            let signed = unit.mul_add(2.0, -1.0);

            weights.push(signed * dimension_scale);
        }
    }

    Ok(weights)
}

/// L2-normalizes a real-valued embedding in-place.
///
/// Accumulation uses `f64` to reduce avoidable norm error while retaining
/// `f32` storage/output.
fn l2_normalize_in_place(values: &mut [f32]) {
    let norm_squared = values
        .iter()
        .map(|value| {
            let value = f64::from(*value);
            value * value
        })
        .sum::<f64>();

    if norm_squared <= f64::EPSILON {
        return;
    }

    let inverse_norm = norm_squared.sqrt().recip() as f32;

    for value in values {
        *value *= inverse_norm;
    }
}

fn push_nonempty_token(current: &mut String, output: &mut Vec<String>) {
    if current.is_empty() {
        return;
    }

    output.push(std::mem::take(current));
}

/// Stable FNV-1a hash.
///
/// This is used for reproducible lexical hashing and deterministic namespace
/// seeding, not cryptographic integrity.
fn stable_hash64(bytes: &[u8]) -> u64 {
    let mut hash = FNV64_OFFSET;

    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV64_PRIME);
    }

    hash
}

/// SplitMix64 mixing step.
fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(SPLITMIX_INCREMENT);

    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);

    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);

    value ^ (value >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hash_config() -> InferenceModelConfig {
        InferenceModelConfig {
            model_name: "test-hash-projection".to_string(),
            architecture: ModelArchitecture::HashProjection,
            output_dimension: 384,
            max_sequence_length: 128,
            normalize_embeddings: true,
        }
    }

    fn temporary_vocab(contents: &str) -> PathBuf {
        static NEXT_ID: AtomicU64 = AtomicU64::new(0);

        let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);

        let path = std::env::temp_dir().join(format!(
            "hnsqr_inference_vocab_{}_{}.txt",
            std::process::id(),
            id,
        ));

        std::fs::write(&path, contents).unwrap();

        path
    }

    #[test]
    fn config_accepts_valid_hash_configuration() {
        assert_eq!(hash_config().validate(), Ok(()));
    }

    #[test]
    fn config_rejects_zero_dimension() {
        let mut config = hash_config();
        config.output_dimension = 0;

        assert_eq!(
            config.validate(),
            Err(InferenceConfigError::OutputDimensionTooSmall { dimension: 0 })
        );
    }

    #[test]
    fn config_rejects_odd_dimension() {
        let mut config = hash_config();
        config.output_dimension = 383;

        assert_eq!(
            config.validate(),
            Err(InferenceConfigError::OutputDimensionMustBeEven { dimension: 383 })
        );
    }

    #[test]
    fn config_rejects_zero_sequence_length() {
        let mut config = hash_config();
        config.max_sequence_length = 0;

        assert_eq!(
            config.validate(),
            Err(InferenceConfigError::SequenceLengthZero)
        );
    }

    #[test]
    fn hash_projection_is_deterministic() {
        let embedder = InProcessModelEmbedder::new_hash_projection(hash_config()).unwrap();

        let first = embedder
            .embed_text("Search for legal compliance documents")
            .unwrap();

        let second = embedder
            .embed_text("Search for legal compliance documents")
            .unwrap();

        assert_eq!(first.dimension(), 192);

        assert_eq!(first.complex_data(), second.complex_data());
    }

    #[test]
    fn hash_projection_is_content_sensitive_at_equal_token_count() {
        let embedder = InProcessModelEmbedder::new_hash_projection(hash_config()).unwrap();

        let first = embedder
            .embed_text("quarterly revenue exceeded forecast")
            .unwrap();

        let second = embedder
            .embed_text("purple elephants dislike cold soup")
            .unwrap();

        assert_eq!(first.dimension(), second.dimension());

        assert_ne!(first.complex_data(), second.complex_data());
    }

    #[test]
    fn hash_projection_model_name_namespaces_embedding_space() {
        let mut first_config = hash_config();

        first_config.model_name = "hash-space-a".to_string();

        let mut second_config = hash_config();

        second_config.model_name = "hash-space-b".to_string();

        let first = InProcessModelEmbedder::new_hash_projection(first_config).unwrap();

        let second = InProcessModelEmbedder::new_hash_projection(second_config).unwrap();

        let first_embedding = first.embed_text("identical lexical input").unwrap();

        let second_embedding = second.embed_text("identical lexical input").unwrap();

        assert_ne!(
            first_embedding.complex_data(),
            second_embedding.complex_data()
        );
    }

    #[test]
    fn hash_constructor_refuses_non_hash_architecture() {
        let mut config = hash_config();

        config.architecture = ModelArchitecture::BertMiniLM;

        let result = InProcessModelEmbedder::new_hash_projection(config);

        assert!(matches!(
            result,
            Err(InferenceConfigError::ArchitectureMismatch {
                required: ModelArchitecture::HashProjection,
                supplied: ModelArchitecture::BertMiniLM,
            })
        ));
    }

    #[test]
    fn static_wordpiece_projection_uses_exact_vocab_rows() {
        let vocab = temporary_vocab("[UNK]\nplay\n##ing\nthe\n");

        let config = InferenceModelConfig {
            model_name: "static-wordpiece-test".to_string(),
            architecture: ModelArchitecture::CustomProjector,
            output_dimension: 8,
            max_sequence_length: 32,
            normalize_embeddings: true,
        };

        let token_embeddings = (0..(4 * 8))
            .map(|index| index as f32 / 32.0)
            .collect::<Vec<_>>();

        let embedder =
            InProcessModelEmbedder::with_static_projection(config, &vocab, token_embeddings)
                .unwrap();

        let embedding = embedder.embed_text("the playing").unwrap();

        assert_eq!(embedding.dimension(), 4);

        let _ = std::fs::remove_file(vocab);
    }

    #[test]
    fn static_projection_rejects_missing_vocab_rows() {
        let vocab = temporary_vocab("[UNK]\nplay\n##ing\nthe\n");

        let config = InferenceModelConfig {
            model_name: "static-wordpiece-test".to_string(),
            architecture: ModelArchitecture::CustomProjector,
            output_dimension: 8,
            max_sequence_length: 32,
            normalize_embeddings: true,
        };

        // Vocabulary requires rows 0 through 3,
        // but this table only supplies rows 0 through 2.
        let token_embeddings = vec![0.1_f32; 3 * 8];

        let result =
            InProcessModelEmbedder::with_static_projection(config, &vocab, token_embeddings);

        assert!(matches!(
            result,
            Err(InferenceConfigError::ProjectionTableTooSmall {
                rows: 3,
                required_rows: 4,
            })
        ));

        let _ = std::fs::remove_file(vocab);
    }

    struct MockMiniLmBackend;

    impl TransformerInferenceBackend for MockMiniLmBackend {
        fn architecture(&self) -> ModelArchitecture {
            ModelArchitecture::BertMiniLM
        }

        fn model_name(&self) -> &str {
            "mock-minilm"
        }

        fn output_dimension(&self) -> usize {
            8
        }

        fn max_sequence_length(&self) -> usize {
            128
        }

        fn embed_text_into(
            &self,
            text: &str,
            max_sequence_length: usize,
            output: &mut [f32],
        ) -> HNSQRResult<()> {
            let lexical_signal = text.len() as f32 + max_sequence_length as f32;

            for (index, value) in output.iter_mut().enumerate() {
                *value = lexical_signal + index as f32;
            }

            Ok(())
        }
    }

    #[test]
    fn transformer_backend_requires_exact_contract_match() {
        let config = InferenceModelConfig {
            model_name: "mock-minilm".to_string(),
            architecture: ModelArchitecture::BertMiniLM,
            output_dimension: 8,
            max_sequence_length: 64,
            normalize_embeddings: true,
        };

        let embedder =
            InProcessModelEmbedder::with_transformer_backend(config, MockMiniLmBackend).unwrap();

        let embedding = embedder.embed_text("contextual transformer test").unwrap();

        assert_eq!(embedding.dimension(), 4);

        assert_eq!(embedder.total_inferences(), 1);
    }

    #[test]
    fn transformer_backend_architecture_mismatch_is_rejected() {
        let config = InferenceModelConfig {
            model_name: "mock-minilm".to_string(),
            architecture: ModelArchitecture::BgeDense,
            output_dimension: 8,
            max_sequence_length: 64,
            normalize_embeddings: true,
        };

        let result = InProcessModelEmbedder::with_transformer_backend(config, MockMiniLmBackend);

        assert!(matches!(
            result,
            Err(InferenceConfigError::TransformerArchitectureMismatch {
                configured: ModelArchitecture::BgeDense,
                backend: ModelArchitecture::BertMiniLM,
            })
        ));
    }

    #[test]
    fn successful_inference_counter_only_advances_after_success() {
        let embedder = InProcessModelEmbedder::new_hash_projection(hash_config()).unwrap();

        assert_eq!(embedder.total_inferences(), 0);

        embedder.embed_text("first").unwrap();

        embedder.embed_text("second").unwrap();

        assert_eq!(embedder.total_inferences(), 2);
    }

    #[test]
    fn complex_dimension_matches_real_dimension_contract() {
        let embedder = InProcessModelEmbedder::new_hash_projection(hash_config()).unwrap();

        assert_eq!(embedder.output_dimension(), 384);

        assert_eq!(embedder.complex_output_dimension(), 192);
    }
}
