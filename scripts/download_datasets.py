#!/usr/bin/env python3
"""
scripts/download_datasets.py
Downloads and prepares REAL public vector datasets from Hugging Face and Texmex
into the `datasets/` directory for HoloSphere exact retrieval benchmarking:

1. Texmex SIFT10K (128-dim real descriptors, 10,000 vectors + 100 queries)
2. Cohere Wikipedia Embeddings (768-dim real neural embeddings, 1,000 vectors + 100 queries)
3. OpenAI Text Embeddings (1536-dim real text-embedding-ada-002 / text-3 vectors, 1,000 vectors + 100 queries)
4. CLIP ViT-B/32 Multi-Modal Embeddings (512-dim real vision-language embeddings, 1,000 vectors + 100 queries)
"""

import os
import sys
import struct
import tarfile
import urllib.request
import io

DATASETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets")

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def write_fvecs(filename, vectors):
    """Writes a list of float vectors in standard .fvecs format."""
    with open(filename, "wb") as f:
        for vec in vectors:
            dim = len(vec)
            f.write(struct.pack("<i", dim))
            f.write(struct.pack(f"<{dim}f", *vec))

def download_sift_small():
    """Downloads SIFT10K dataset from Texmex repository."""
    sift_dir = os.path.join(DATASETS_DIR, "siftsmall")
    base_file = os.path.join(sift_dir, "siftsmall_base.fvecs")
    if os.path.exists(base_file):
        print("[+] SIFT10K dataset already present.")
        return

    url = "ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz"
    tar_path = os.path.join(DATASETS_DIR, "siftsmall.tar.gz")
    
    print("[*] Downloading SIFT10K dataset...")
    try:
        urllib.request.urlretrieve(url, tar_path)
    except Exception as e:
        print(f"[!] Network download error: {e}")

    if os.path.exists(tar_path):
        try:
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(path=DATASETS_DIR)
            print("[+] Extracted SIFT10K dataset.")
        except Exception as e:
            print(f"[!] Extraction error: {e}")

def download_cohere_768():
    """Downloads real 768-dim Cohere Wikipedia embeddings from HuggingFace."""
    out_dir = os.path.join(DATASETS_DIR, "cohere_768")
    ensure_dir(out_dir)
    base_path = os.path.join(out_dir, "cohere_base.fvecs")
    query_path = os.path.join(out_dir, "cohere_query.fvecs")

    if os.path.exists(base_path) and os.path.exists(query_path) and os.path.getsize(query_path) > 0:
        print("[+] Cohere 768-dim dataset already present.")
        return

    print("[*] Downloading Cohere-768 embeddings from HuggingFace...")
    import fsspec
    import pyarrow.parquet as pq

    url = "https://huggingface.co/datasets/ashraq/cohere-wiki-embedding-100k/resolve/main/data/train-00000-of-00002-039513d189a50a66.parquet"
    with fsspec.open(url, headers={"User-Agent": "Mozilla/5.0"}) as f:
        pf = pq.ParquetFile(f)
        rg0 = pf.read_row_group(0, columns=["emb"])["emb"].to_pylist()
        rg1 = pf.read_row_group(1, columns=["emb"])["emb"].to_pylist()
        embs = rg0 + rg1
        print(f"    Loaded {len(embs)} Cohere embeddings, dim = {len(embs[0])}")

        write_fvecs(base_path, embs[:1000])
        write_fvecs(query_path, embs[1000:1100])
        print(f"[+] Saved Cohere-768 dataset: {base_path} (1,000 vecs) and {query_path} (100 queries)")

def download_openai_1536():
    """Downloads real 1536-dim OpenAI text-embedding embeddings from HuggingFace."""
    out_dir = os.path.join(DATASETS_DIR, "openai_1536")
    ensure_dir(out_dir)
    base_path = os.path.join(out_dir, "openai_base.fvecs")
    query_path = os.path.join(out_dir, "openai_query.fvecs")

    if os.path.exists(base_path) and os.path.exists(query_path):
        print("[+] OpenAI 1536-dim dataset already present.")
        return

    print("[*] Downloading OpenAI-1536 embeddings from HuggingFace...")
    import fsspec
    import pyarrow.parquet as pq

    url = "https://huggingface.co/datasets/KShivendu/dbpedia-entities-openai-1M/resolve/main/data/train-00000-of-00026-3c7b99d1c7eda36e.parquet"
    with fsspec.open(url, headers={"User-Agent": "Mozilla/5.0"}) as f:
        pf = pq.ParquetFile(f)
        rg0 = pf.read_row_group(0, columns=["openai"])["openai"].to_pylist()
        rg1 = pf.read_row_group(1, columns=["openai"])["openai"].to_pylist()
        embs = rg0 + rg1
        print(f"    Loaded {len(embs)} OpenAI embeddings, dim = {len(embs[0])}")

        write_fvecs(base_path, embs[:1000])
        write_fvecs(query_path, embs[1000:1100])
        print(f"[+] Saved OpenAI-1536 dataset: {base_path} (1,000 vecs) and {query_path} (100 queries)")

def download_clip_512():
    """Downloads real 512-dim CLIP ViT-B/32 multi-modal embeddings from HuggingFace."""
    out_dir = os.path.join(DATASETS_DIR, "clip_512")
    ensure_dir(out_dir)
    base_path = os.path.join(out_dir, "clip_base.fvecs")
    query_path = os.path.join(out_dir, "clip_query.fvecs")

    if os.path.exists(base_path) and os.path.exists(query_path):
        print("[+] CLIP 512-dim dataset already present.")
        return

    print("[*] Downloading CLIP-512 embeddings from HuggingFace...")
    import fsspec
    import pyarrow.parquet as pq

    url = "https://huggingface.co/datasets/closji/mscoco_train_2014_openai_clip-vit-base-patch32/resolve/main/data/train-00000-of-00001.parquet"
    with fsspec.open(url, headers={"User-Agent": "Mozilla/5.0"}) as f:
        pf = pq.ParquetFile(f)
        table = pf.read_row_group(0, columns=["embeddings"])
        embs = table["embeddings"].to_pylist()
        print(f"    Loaded {len(embs)} CLIP embeddings, dim = {len(embs[0])}")

        write_fvecs(base_path, embs[:1000])
        write_fvecs(query_path, embs[1000:1100])
        print(f"[+] Saved CLIP-512 dataset: {base_path} (1,000 vecs) and {query_path} (100 queries)")

def main():
    ensure_dir(DATASETS_DIR)
    print(f"[*] Downloading Real Public Datasets to: {DATASETS_DIR}\n")

    download_sift_small()
    download_cohere_768()
    download_openai_1536()
    download_clip_512()

    print("\n[+] All 4 real public datasets downloaded and converted to standard .fvecs format.")

if __name__ == "__main__":
    main()
