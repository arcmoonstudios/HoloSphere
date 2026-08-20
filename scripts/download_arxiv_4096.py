#!/usr/bin/env python3
"""
scripts/download_arxiv_4096.py

Downloads the arXiv abstract embeddings dataset from the FANNS benchmark paper:
  "Benchmarking Filtered Approximate Nearest Neighbor Search Algorithms on
   Transformer-based Embedding Vectors" (arxiv 2507.21989, ETH Zurich SPCL, 2025)

Source: https://huggingface.co/datasets/SPCL/arxiv-for-fanns-large
  - 2,725,176 vectors × 4096 dimensions
  - Embeddings of arXiv paper abstracts via stella_en_400M_v5 (L2-normalized)
  - Pre-built binary .fvecs / .ivecs files — no conversion needed

The dataset is GATED on HuggingFace. You must:
  1. Create a HuggingFace account at https://huggingface.co
  2. Visit https://huggingface.co/datasets/SPCL/arxiv-for-fanns-large and request access
  3. Generate a read token at https://huggingface.co/settings/tokens
  4. Set it as an environment variable before running:
       $env:HF_TOKEN = "hf_..."         # PowerShell
       export HF_TOKEN="hf_..."          # bash/zsh

Usage:
  python scripts/download_arxiv_4096.py [--scale large|medium|small]

  --scale large   downloads ~2.7M vectors (default, ~44 GB)
  --scale medium  downloads   100k vectors (for quick smoke-testing, ~1.6 GB)
  --scale small   downloads     1k vectors (sanity check only)
"""

import os
import sys
import struct
import argparse

DATASETS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets"
)

# Declared vector counts per scale, from the FANNS benchmark paper (Table 4 / §4).
# Used for post-download integrity verification.
DECLARED_N = {
    "large":  2_725_176,
    "medium":   100_000,
    "small":      1_000,
}

# Every dataset variant ships these files on HuggingFace.
# ground_truth.ivecs = unfiltered ANN ground truth (no filter condition)
# ground_truth_em/r/emis.ivecs = per-filter-type ground truth
DATASET_FILES = [
    "database_vectors.fvecs",
    "query_vectors.fvecs",
    "ground_truth.ivecs",
    "ground_truth_em.ivecs",
    "ground_truth_r.ivecs",
    "ground_truth_emis.ivecs",
    "database_attributes.jsonl",
    "em_query_attributes.jsonl",
    "r_query_attributes.jsonl",
    "emis_query_attributes.jsonl",
]


# ---------------------------------------------------------------------------
# fvecs / ivecs reader (header-only, for vector-count verification)
# ---------------------------------------------------------------------------

def count_vecs(path, dtype="f"):
    """Return (n_vectors, dim) from a .fvecs or .ivecs file without loading
    all vectors into memory.  dtype='f' for float32, 'i' for int32."""
    item_size = struct.calcsize(dtype)
    with open(path, "rb") as f:
        dim_bytes = f.read(4)
        if len(dim_bytes) < 4:
            raise ValueError(f"File too short to contain a valid header: {path}")
        dim = struct.unpack("<i", dim_bytes)[0]
        if dim <= 0:
            raise ValueError(f"Invalid dimension {dim} in {path}")
        record_bytes = 4 + dim * item_size
        file_size = os.path.getsize(path)
        if file_size % record_bytes != 0:
            raise ValueError(
                f"File size {file_size} is not a multiple of record size "
                f"{record_bytes} (dim={dim}) in {path}"
            )
        n = file_size // record_bytes
    return n, dim


# ---------------------------------------------------------------------------
# Main download logic
# ---------------------------------------------------------------------------

def download_arxiv_4096(scale: str = "large"):
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("[!] huggingface_hub is not installed.")
        print("    Install it with:  pip install huggingface_hub")
        sys.exit(1)

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("[i] HF_TOKEN not set -- dataset is public, proceeding unauthenticated.")
        print("    (Set HF_TOKEN if you hit rate limits or need private access.)")
        print()

    repo_id   = f"SPCL/arxiv-for-fanns-{scale}"
    out_dir   = os.path.join(DATASETS_DIR, "arxiv_4096")
    declared_n = DECLARED_N[scale]

    os.makedirs(out_dir, exist_ok=True)

    print(f"[*] Downloading arXiv-for-FANNS ({scale}) from {repo_id}")
    print(f"    Target directory : {out_dir}")
    print(f"    Declared vectors : {declared_n:,}")
    print()

    cached_paths = {}

    for filename in DATASET_FILES:
        dest = os.path.join(out_dir, filename)

        # Skip files that are already fully present in the output dir
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            print(f"[=] {filename} already present, skipping download.")
            cached_paths[filename] = dest
            continue

        print(f"[dl] {filename} ...", flush=True)
        try:
            cached = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
                token=hf_token,
                local_dir=out_dir,        # write directly into our output dir
                local_dir_use_symlinks=False,
            )
            cached_paths[filename] = cached
            size_mb = os.path.getsize(cached) / 1_048_576
            print(f"    -> {cached}  ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"[!] Failed to download {filename}: {e}")
            print("    Check that your token is valid and access has been granted.")
            sys.exit(1)

    print()
    print("[*] Verifying database_vectors.fvecs ...")
    db_path = cached_paths["database_vectors.fvecs"]
    try:
        loaded_n, loaded_dim = count_vecs(db_path, dtype="f")
    except Exception as e:
        print(f"[!] Verification failed: {e}")
        sys.exit(1)

    print(f"    Loaded  : {loaded_n:,} vectors × {loaded_dim} dimensions")
    print(f"    Declared: {declared_n:,} vectors")

    if loaded_dim != 4096:
        print(f"[!] DIMENSION MISMATCH — expected 4096, got {loaded_dim}.")
        print("    Something is wrong with the downloaded file.")
        sys.exit(1)

    if loaded_n != declared_n:
        print(
            f"[!] COUNT MISMATCH — loaded {loaded_n:,} vectors, "
            f"paper declares {declared_n:,}."
        )
        print("    The file may be truncated or the paper's count has been revised.")
        print("    Flagging as anomaly — do not use this as a benchmark baseline.")
        sys.exit(1)

    print(f"    [OK] Count and dimension match paper declaration.")

    print()
    print("[*] Verifying query_vectors.fvecs ...")
    q_path = cached_paths["query_vectors.fvecs"]
    try:
        q_n, q_dim = count_vecs(q_path, dtype="f")
    except Exception as e:
        print(f"[!] Verification failed: {e}")
        sys.exit(1)
    print(f"    Loaded  : {q_n:,} query vectors × {q_dim} dimensions")
    if q_dim != 4096:
        print(f"[!] Query dimension mismatch — expected 4096, got {q_dim}.")
        sys.exit(1)
    print(f"    [OK] Query vectors OK.")

    print()
    print("[+] Download complete.")
    print()
    print("    Dataset summary")
    print("    ---------------")
    print(f"    Location  : {out_dir}")
    print(f"    Scale     : {scale}")
    print(f"    Vectors   : {loaded_n:,} × {loaded_dim}D  (stella_en_400M_v5, L2-normalised)")
    print(f"    Source    : {repo_id}  [FANNS benchmark, arxiv 2507.21989]")
    print()
    print("    Files:")
    for filename in DATASET_FILES:
        p = os.path.join(out_dir, filename)
        if os.path.exists(p):
            size_mb = os.path.getsize(p) / 1_048_576
            print(f"      {filename:<45}  {size_mb:>10.1f} MB")
    print()
    print("    Next steps:")
    print("      - Run with PlannerDefault first to confirm natural routing past ExactScan")
    print("        at N=2.7M, D=4096 (predicted N_cross ~3,300 at complex_dim 2048 =>")
    print("        this corpus clears it by ~800x).")
    print("      - If PlannerDefault does not match RiveroAdaptive/GraphOnly in the")
    print("        routing verification block, flag it — that's a planner cost formula")
    print("        discrepancy, not a benchmark bug.")


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download arXiv-for-FANNS embeddings dataset (SPCL/ETH Zurich, 2025)"
    )
    parser.add_argument(
        "--scale",
        choices=["large", "medium", "small"],
        default="large",
        help=(
            "Dataset scale: large (~2.7M vecs, ~44 GB), "
            "medium (100k vecs, ~1.6 GB), "
            "small (1k vecs, <100 MB). Default: large"
        ),
    )
    args = parser.parse_args()
    download_arxiv_4096(scale=args.scale)


if __name__ == "__main__":
    main()
