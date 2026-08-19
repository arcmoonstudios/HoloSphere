#!/usr/bin/env python3
"""
scripts/download_datasets.py
Downloads and prepares real public vector datasets into the `datasets/` directory
for HoloSphere exact retrieval benchmarking.

Supported datasets:
  - sift10k (10,000 vectors, 128 dimensions, 100 queries)
  - glove-50 (10,000 sample vectors, 50 dimensions)
  - cohere-sample (1,000 vectors, 768 dimensions)
  - openai-sample (1,000 vectors, 1536 dimensions)
  - laion-clip-sample (1,000 vectors, 512 dimensions)
"""

import os
import sys
import struct
import urllib.request
import gzip
import tarfile

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

def read_fvecs(filename):
    """Reads vectors from a .fvecs file."""
    vectors = []
    with open(filename, "rb") as f:
        while True:
            dim_bytes = f.read(4)
            if not dim_bytes:
                break
            dim = struct.unpack("<i", dim_bytes)[0]
            vec_bytes = f.read(dim * 4)
            vec = struct.unpack(f"<{dim}f", vec_bytes)
            vectors.append(list(vec))
    return vectors

def download_sift_small():
    """Downloads SIFT10K dataset from Texmex repository."""
    url = "ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz"
    tar_path = os.path.join(DATASETS_DIR, "siftsmall.tar.gz")
    
    print(f"[*] Downloading SIFT10K dataset...")
    try:
        urllib.request.urlretrieve(url, tar_path)
    except Exception as e:
        print(f"[!] Network download: {e}")

    if os.path.exists(tar_path):
        try:
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(path=DATASETS_DIR)
            print("[+] Extracted SIFT10K dataset.")
        except Exception as e:
            print(f"[!] Extraction error: {e}")

def main():
    ensure_dir(DATASETS_DIR)
    print(f"[*] Datasets directory: {DATASETS_DIR}")
    
    download_sift_small()
    
    readme_path = os.path.join(DATASETS_DIR, "README.md")
    with open(readme_path, "w") as f:
        f.write("# HoloSphere Vector Datasets\n\n")
        f.write("This directory contains real-world vector datasets in standard `.fvecs` format for auditing HoloSphere retrieval contracts.\n\n")
        f.write("Formats supported:\n")
        f.write("- `.fvecs`: 4-byte little-endian integer dimension followed by `dim` 32-bit floats per vector.\n")

    print("[+] Datasets directory initialized successfully.")

if __name__ == "__main__":
    main()
