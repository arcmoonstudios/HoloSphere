# scripts/run_public_benchmarks.ps1
param(
    [string]$Dataset = "cohere-1m",
    [int]$K = 10,
    [string]$Contract = "Certified"
)

Write-Host "╔═════════════════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║              HOLOSPHERE PUBLIC DATASET BENCHMARK & PROOF AUDIT              ║" -ForegroundColor Cyan
Write-Host "╚═════════════════════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan

Write-Host "`n📊 Benchmark Parameters:" -ForegroundColor Yellow
Write-Host "   • Dataset:    $Dataset"
Write-Host "   • Top-K:      $K"
Write-Host "   • Contract:   $Contract"

Write-Host "`n🚀 Executing cargo benchmark harness..." -ForegroundColor Green
cargo bench --bench public_dataset_benchmark

Write-Host "`n✨ Public Benchmark execution completed." -ForegroundColor Cyan
