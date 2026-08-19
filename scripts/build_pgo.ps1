<#
.SYNOPSIS
    Builds HoloSphere with Profile-Guided Optimization (PGO) using cargo-pgo or native rustc.
.DESCRIPTION
    Automates:
    1. Installing prerequisites
    2. Building instrumented binary
    3. Exercising representative proof and benchmark workloads
    4. Merging profiles and compiling the optimized binary
.EXAMPLE
    .\scripts\build_pgo.ps1 -Workload Benchmarks
#>

param (
    [ValidateSet("cargo-pgo", "native")]
    [string]$Method = "cargo-pgo",

    [ValidateSet("Benchmarks", "Server")]
    [string]$Workload = "Benchmarks"
)

$ErrorActionPreference = "Stop"

Write-Host "🌙 HoloSphere Profile-Guided Optimization (PGO) Builder" -ForegroundColor Cyan
Write-Host "════════════════════════════════════════════════════════════" -ForegroundColor DarkGray

if ($Method -eq "cargo-pgo") {
    Write-Host "==> Checking cargo-pgo and llvm-tools-preview..." -ForegroundColor Yellow
    rustup component add llvm-tools-preview
    if (-not (Get-Command cargo-pgo -ErrorAction SilentlyContinue)) {
        Write-Host "==> Installing cargo-pgo..." -ForegroundColor Yellow
        cargo install cargo-pgo
    }

    Write-Host "==> Step 1: Building instrumented binary..." -ForegroundColor Green
    $env:RUSTFLAGS = "-C target-cpu=native"
    cargo pgo build --release

    Write-Host "==> Step 2: Running representative workload ($Workload)..." -ForegroundColor Green
    if ($Workload -eq "Benchmarks") {
        cargo pgo test --bench universal_scorecard_benchmark -- --nocapture
        cargo pgo test --bench gate_b_hierarchical_proof -- --nocapture
        cargo pgo test --bench rivero_search_scaling -- --nocapture
    } else {
        Write-Host "Starting daemon for profile collection..."
        $proc = Start-Process cargo -ArgumentList "pgo run --release --bin hnsqr_daemon" -PassThru
        Start-Sleep -Seconds 10
        Stop-Process -Id $proc.Id -Force
    }

    Write-Host "==> Step 3: Compiling optimized production binary..." -ForegroundColor Green
    cargo pgo optimize --release

    Write-Host "✅ PGO Build complete: target/release/hnsqr_daemon" -ForegroundColor Cyan
} else {
    $pgoDir = "$PSScriptRoot\..\target\pgo-data"
    if (Test-Path $pgoDir) { Remove-Item -Recurse -Force $pgoDir }
    New-Item -ItemType Directory -Path $pgoDir -Force | Out-Null

    Write-Host "==> Step 1: Building with -Cprofile-generate..." -ForegroundColor Green
    $env:RUSTFLAGS = "-Cprofile-generate=$pgoDir -Ctarget-cpu=native"
    cargo build --release --all-targets

    Write-Host "==> Step 2: Exercising benchmarks..." -ForegroundColor Green
    & "$PSScriptRoot\..\target\release\benches\universal_scorecard_benchmark.exe"
    & "$PSScriptRoot\..\target\release\benches\gate_b_hierarchical_proof.exe"

    Write-Host "==> Step 3: Merging profile data..." -ForegroundColor Green
    $sysroot = (rustc --print sysroot).Trim()
    $hostTriple = (rustc -vV | Select-String "host:").ToString().Split(":")[1].Trim()
    $llvmProfdata = "$sysroot\lib\rustlib\$hostTriple\bin\llvm-profdata.exe"

    if (-not (Test-Path $llvmProfdata)) {
        Write-Error "llvm-profdata not found at $llvmProfdata. Run 'rustup component add llvm-tools-preview'."
    }

    & $llvmProfdata merge -o "$pgoDir\merged.profdata" (Get-ChildItem "$pgoDir\*.profraw").FullName

    Write-Host "==> Step 4: Compiling with -Cprofile-use..." -ForegroundColor Green
    $env:RUSTFLAGS = "-Cprofile-use=$pgoDir\merged.profdata -Ctarget-cpu=native"
    cargo build --release

    Write-Host "✅ Native PGO Build complete: target/release/hnsqr_daemon" -ForegroundColor Cyan
}
