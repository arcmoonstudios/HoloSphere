# HoloSphere Vector & Multi-Modal Evaluation Datasets

This directory contains real-world and standardized vector datasets, structured metadata attributes, and exact ground-truth nearest neighbor indices used for benchmarking, performance profiling, and formal retrieval contract audits across HoloSphere (`HNSQR`).

---

## Supported Binary & Attribute File Formats

* **`.fvecs` (Float Vector Array)**:
  - Binary format: Each vector is stored consecutively as a 4-byte little-endian unsigned integer representing dimension $D$, followed immediately by $D \times 32$-bit IEEE 754 floating-point values ($4 \times D$ bytes).
  - Total byte size per vector: $4 + 4 \times D$ bytes.
* **`.ivecs` (Integer Vector Array / Ground Truth)**:
  - Binary format: Each record starts with a 4-byte little-endian unsigned integer indicating count $K$, followed by $K \times 32$-bit little-endian integer indices representing exact ground-truth nearest neighbor IDs.
* **`.jsonl` (JSON Lines Metadata)**:
  - Text format: One newline-delimited JSON document per line containing structured attributes (strings, numeric ranges, categories, tag lists, booleans) aligned 1:1 with corresponding vector indices.
* **`.hdf5` (Hierarchical Data Format)**:
  - Standard ANN-benchmarks container bundling `train`, `test`, `distances`, and `neighbors` datasets in a single self-describing HDF5 file.

---

## Dataset Directory Catalog & Contents

### 1. `arxiv_4096/` — Ultra-High Dimensional Academic Paper Embeddings & Filtered Search
* **Real Dimension**: 4096 ($\mathbb{R}^{4096}$) | **Complex Dimension**: 2048 ($\mathbb{C}^{2048}$)
* **Source & Model**: Modern large language model representations (e.g. 4096-dim LLaMA / Mistral / Qwen embedding space) over ArXiv scientific publications.
* **Contents**:
  - `database_vectors.fvecs` (~44.8 GB): Primary high-dimensional base vector corpus.
  - `database_attributes.jsonl` (~1.1 GB): Structured metadata attributes (e.g. paper categories, authors, publication years, licenses) for hybrid filtering.
  - `query_vectors.fvecs` (~163.9 MB): 4096-dimensional evaluation queries.
  - `em_query_attributes.jsonl`: Queries with Exact Match (EM) metadata filter constraints.
  - `emis_query_attributes.jsonl`: Queries with Equality / In / Substring (EMIS) metadata filters.
  - `r_query_attributes.jsonl`: Queries with Numeric Range (R) metadata filter constraints.
  - `ground_truth_em.ivecs`: Precomputed exact top-$k$ ground truth indices under Exact Match filters.
  - `ground_truth_emis.ivecs`: Precomputed exact top-$k$ ground truth under EMIS filters.
  - `ground_truth_r.ivecs`: Precomputed exact top-$k$ ground truth under Numeric Range filters.
  - `.cache/huggingface/download/`: Hugging Face dataset download cache and `.metadata` checksum files ensuring dataset provenance and reproducibility.
* **Purpose**: Stress-testing Gate A/B capacity bounds at $D=4096$, complex isometric folding (`ComplexWeaver`), roaring bitmap precompiled filter execution, and exact candidate reranking.

---

### 2. `clip_512/` — Multimodal Visual-Semantic Embeddings
* **Real Dimension**: 512 ($\mathbb{R}^{512}$) | **Complex Dimension**: 256 ($\mathbb{C}^{256}$)
* **Source & Model**: OpenAI CLIP (Contrastive Language-Image Pre-Training) ViT-B/32 multimodal embeddings.
* **Contents**:
  - `clip_base.fvecs`: Base multimodal corpus (1,000 vectors @ 512D).
  - `clip_query.fvecs`: Evaluation query vectors (100 vectors @ 512D).
* **Purpose**: Cross-modal projection validation, spherical harmonic phase encoding, and mid-dimensional multimodal retrieval benchmarking.

---

### 3. `cohere_768/` & `cohere_768_large/` — Dense Enterprise Text Embeddings
* **Real Dimension**: 768 ($\mathbb{R}^{768}$) | **Complex Dimension**: 384 ($\mathbb{C}^{384}$)
* **Source & Model**: Cohere / BERT-large dense text embeddings.
* **Contents**:
  - `cohere_768/cohere_base.fvecs`: Standard baseline corpus (1,000 vectors @ 768D).
  - `cohere_768/cohere_query.fvecs`: Evaluation queries (100 vectors @ 768D).
  - `cohere_768_large/cohere_100k_base.fvecs`: Scale corpus containing 100,000 vectors @ 768D (~153.8 MB).
* **Purpose**: Standard enterprise NLP retrieval validation, empirical exact-scan crossover sweeps ($N_{\text{cross}}$ characterization), and multi-lane Rivero routing scaling up to $N=100\text{k}$.

---

### 4. `glove_25/`, `glove_50/`, `glove_100/` — Stanford GloVe Word Vector Corpora
* **Real Dimensions**: 25D, 50D, 100D | **Complex Dimensions**: 13D, 25D, 50D
* **Source & Model**: Stanford NLP GloVe (Global Vectors for Word Representation) 2 Billion Tweet / Wikipedia dataset.
* **Contents**:
  - `glove_25/glove25_base.fvecs` & `glove25_query.fvecs` (1,183,514 base vectors, 10,000 queries).
  - `glove_25/glove25.hdf5`: ANN-benchmarks standard HDF5 container.
  - `glove_50/glove50_base.fvecs` & `glove50_query.fvecs` (1,183,514 base vectors, 10,000 queries).
  - `glove_50/glove50.hdf5`: ANN-benchmarks standard HDF5 container.
  - `glove_100/glove100_base.fvecs` & `glove100_query.fvecs` (1,183,514 base vectors, 10,000 queries).
  - `glove_100/glove100.hdf5`: ANN-benchmarks standard HDF5 container.
* **Purpose**: Million-scale low-dimensional dense geometry testing, comparative ANN-benchmarks parity testing, and baseline Euclidean / Cosine verification.

---

### 5. `openai_1536/` & `openai_1536_large/` — Production LLM Text Embeddings
* **Real Dimension**: 1536 ($\mathbb{R}^{1536}$) | **Complex Dimension**: 768 ($\mathbb{C}^{768}$)
* **Source & Model**: OpenAI `text-embedding-ada-002` / `text-embedding-3-small` dense vectors.
* **Contents**:
  - `openai_1536/openai_base.fvecs`: Base vector corpus (1,000 vectors @ 1536D).
  - `openai_1536/openai_query.fvecs`: Query vectors (100 vectors @ 1536D).
  - `openai_1536_large/openai_1m_base.fvecs`: Million-scale corpus segment (~236.5 MB).
* **Purpose**: Modern LLM embedding benchmarks, lossless coordinate transforming via [`ComplexWeaver`](../src/vector/folding.rs), 8-bit Polar PQ-C memory bandwidth reduction, and Lutz $E_8$ hierarchical proof tree verification.

---

### 6. `sift_1m/` — Standard Computer Vision Feature Descriptors (Million-Scale)
* **Real Dimension**: 128 ($\mathbb{R}^{128}$) | **Complex Dimension**: 64 ($\mathbb{C}^{64}$)
* **Source & Model**: INRIA / Texmex 128-dimensional SIFT (Scale-Invariant Feature Transform) descriptors.
* **Contents**:
  - `sift_1m/sift1m_base.fvecs`: 1,000,000 base vectors @ 128D (~516 MB).
  - `sift_1m/sift1m_query.fvecs`: 10,000 evaluation queries @ 128D (~5.16 MB).
  - `sift_1m/sift1m.hdf5`: Complete ANN-benchmarks package with ground truth.
* **Purpose**: Classical million-scale Euclidean distance retrieval benchmarking, index ingestion rate ($v/s$) profiling, and Rayon parallel batch query throughput validation.

---

### 7. `siftsmall/` — Development & CI Fast Smoke-Test Corpus
* **Real Dimension**: 128 ($\mathbb{R}^{128}$) | **Complex Dimension**: 64 ($\mathbb{C}^{64}$)
* **Source & Model**: Texmex SIFT subset.
* **Contents**:
  - `siftsmall_base.fvecs`: 10,000 base vectors.
  - `siftsmall_query.fvecs`: 100 query vectors.
  - `siftsmall_learn.fvecs`: 25,000 training vectors for quantization / clustering.
  - `siftsmall_groundtruth.ivecs`: Exact top-$k$ ground truth nearest neighbors.
* **Purpose**: Sub-second deterministic CI integration tests, local developer iteration, and rapid regression testing.

---

## Coordinate Folding & Metric Compatibility Matrix

| Dataset | Real Dim ($D_{\mathbb{R}}$) | Folded Complex Dim ($D_{\mathbb{C}}$) | Memory Size / Vec | Primary Evaluated Metrics |
| :--- | :---: | :---: | :---: | :--- |
| `glove_25` | 25 | 13 (zero-padded) | 104 B | Cosine, Euclidean |
| `glove_50` | 50 | 25 | 200 B | Cosine, Euclidean |
| `glove_100` | 100 | 50 | 400 B | Cosine, Euclidean |
| `siftsmall` / `sift_1m` | 128 | 64 | 512 B | Euclidean, Cosine, Projective Overlap |
| `clip_512` | 512 | 256 | 2,048 B | Cosine, Projective Overlap |
| `cohere_768` | 768 | 384 | 3,072 B | Cosine, Folded Hermitian |
| `openai_1536` | 1536 | 768 | 6,144 B | Cosine, Folded Hermitian, Projective Overlap |
| `arxiv_4096` | 4096 | 2048 | 16,384 B | Filtered Cosine, Precompiled Roaring Mask, LUTz $E_8$ |
