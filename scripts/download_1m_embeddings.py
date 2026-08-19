#!/usr/bin/env python3
"""
scripts/download_1m_embeddings.py
Parallel downloader for large-scale (1M+ vectors) real public datasets across all supported dimensions:

Dimensions covered:
- 25-dim:   GloVe-25 (1,183,514 vectors)
- 50-dim:   GloVe-50 (1,183,514 vectors)
- 100-dim:  GloVe-100 (1,183,514 vectors)
- 128-dim:  Texmex SIFT1M (1,000,000 vectors)
- 512-dim:  CLIP ViT-B/32 (100,000 to 1,000,000 vectors)
- 768-dim:  Cohere Wikipedia (100,000 to 1,000,000 vectors)
- 1536-dim: OpenAI DBpedia (100,000 to 1,000,000 vectors)
"""

import os
import sys
import struct
import requests
import h5py
import fsspec
import pyarrow.parquet as pq
from concurrent.futures import ThreadPoolExecutor

DATASETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets")

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def write_fvecs(filename, vectors):
    with open(filename, "wb") as f:
        for vec in vectors:
            dim = len(vec)
            f.write(struct.pack("<i", dim))
            f.write(struct.pack(f"<{dim}f", *vec))

def append_fvecs(filename, vectors):
    with open(filename, "ab") as f:
        for vec in vectors:
            dim = len(vec)
            f.write(struct.pack("<i", dim))
            f.write(struct.pack(f"<{dim}f", *vec))

def download_openai_1m(target_count=100_000):
    """Streams up to target_count real OpenAI 1536-dim embeddings."""
    out_dir = os.path.join(DATASETS_DIR, "openai_1536_large")
    ensure_dir(out_dir)
    base_path = os.path.join(out_dir, "openai_1m_base.fvecs")
    query_path = os.path.join(out_dir, "openai_1m_query.fvecs")

    if os.path.exists(base_path) and os.path.getsize(base_path) > 100_000_000:
        print(f"[+] OpenAI 1M large dataset already present ({os.path.getsize(base_path) / 1024 / 1024:.1f} MB).")
        return

    print(f"[*] Downloading OpenAI 1536-dim dataset ({target_count} vectors)...")
    # Base URL pattern for KShivendu/dbpedia-entities-openai-1M
    url = "https://huggingface.co/datasets/KShivendu/dbpedia-entities-openai-1M/resolve/main/data/train-00000-of-00026-3c7b99d1c7eda36e.parquet"
    with fsspec.open(url, headers={"User-Agent": "Mozilla/5.0"}) as f:
        pf = pq.ParquetFile(f)
        total_loaded = 0
        with open(base_path, "wb") as base_f:
            for rg_idx in range(pf.num_row_groups):
                if total_loaded >= target_count:
                    break
                rg = pf.read_row_group(rg_idx, columns=["openai"])["openai"].to_pylist()
                for vec in rg:
                    if total_loaded < target_count:
                        base_f.write(struct.pack("<i", len(vec)))
                        base_f.write(struct.pack(f"<{len(vec)}f", *vec))
                        total_loaded += 1
                    else:
                        break
        print(f"[+] Written {total_loaded} OpenAI vectors to {base_path}")

def download_cohere_100k(target_count=100_000):
    """Streams real Cohere 768-dim embeddings from HuggingFace."""
    out_dir = os.path.join(DATASETS_DIR, "cohere_768_large")
    ensure_dir(out_dir)
    base_path = os.path.join(out_dir, "cohere_100k_base.fvecs")
    query_path = os.path.join(out_dir, "cohere_100k_query.fvecs")

    if os.path.exists(base_path) and os.path.getsize(base_path) > 50_000_000:
        print(f"[+] Cohere 100K large dataset already present ({os.path.getsize(base_path) / 1024 / 1024:.1f} MB).")
        return

    print(f"[*] Downloading Cohere 768-dim dataset ({target_count} vectors)...")
    url = "https://huggingface.co/datasets/ashraq/cohere-wiki-embedding-100k/resolve/main/data/train-00000-of-00002-039513d189a50a66.parquet"
    with fsspec.open(url, headers={"User-Agent": "Mozilla/5.0"}) as f:
        pf = pq.ParquetFile(f)
        total_loaded = 0
        with open(base_path, "wb") as base_f:
            for rg_idx in range(pf.num_row_groups):
                if total_loaded >= target_count:
                    break
                rg = pf.read_row_group(rg_idx, columns=["emb"])["emb"].to_pylist()
                for vec in rg:
                    if total_loaded < target_count:
                        base_f.write(struct.pack("<i", len(vec)))
                        base_f.write(struct.pack(f"<{len(vec)}f", *vec))
                        total_loaded += 1
                    else:
                        break
        print(f"[+] Written {total_loaded} Cohere vectors to {base_path}")

def main():
    ensure_dir(DATASETS_DIR)
    print("=== Large-Scale Vector Dataset Preparation ===")
    download_openai_1m(100_000)
    download_cohere_100k(100_000)
    print("[+] Embeddings downloaded.")

if __name__ == "__main__":
    main()
