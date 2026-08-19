#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-cohere-1m}"
K="${2:-10}"
CONTRACT="${3:-Certified}"

echo -e "\033[1;36m╔═════════════════════════════════════════════════════════════════════════════╗\033[0m"
echo -e "\033[1;36m║              HOLOSPHERE PUBLIC DATASET BENCHMARK & PROOF AUDIT              ║\033[0m"
echo -e "\033[1;36m╚═════════════════════════════════════════════════════════════════════════════╝\033[0m"

echo -e "\n\033[1;33m📊 Benchmark Parameters:\033[0m"
echo "   • Dataset:    ${DATASET}"
echo "   • Top-K:      ${K}"
echo "   • Contract:   ${CONTRACT}"

echo -e "\n\033[1;32m🚀 Executing cargo benchmark harness...\033[0m"
cargo bench --bench public_dataset_benchmark

echo -e "\n\033[1;36m✨ Public Benchmark execution completed.\033[0m"
