[CmdletBinding()]
param(
    [string]$BinaryPath = (Join-Path $PSScriptRoot "..\target\release\hnsqr_mcp_stdio.exe"),
    [string]$DataDirectory = (Join-Path $env:LOCALAPPDATA "HoloSphere\model-agent"),
    [string]$Tenant = "local-agents",
    [ValidateSet("readonly", "readwrite", "admin")]
    [string]$Role = "readwrite"
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true

$sourceBinary = (Resolve-Path -LiteralPath $BinaryPath).Path
$binaryDigest = (Get-FileHash -LiteralPath $sourceBinary -Algorithm SHA256).Hash.ToLowerInvariant()
$snapshotDirectory = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\target')).Path
$snapshotDirectory = Join-Path $snapshotDirectory 'agent-integrations'
[IO.Directory]::CreateDirectory($snapshotDirectory) | Out-Null
$binary = Join-Path $snapshotDirectory "hnsqr_mcp_stdio-$($binaryDigest.Substring(0, 16)).exe"
if (-not (Test-Path -LiteralPath $binary)) {
    Copy-Item -LiteralPath $sourceBinary -Destination $binary
} elseif ((Get-FileHash -LiteralPath $binary -Algorithm SHA256).Hash.ToLowerInvariant() -ne $binaryDigest) {
    throw "Existing agent integration snapshot failed its content-hash check: $binary"
}
$environment = @(
    "HNSQR_DATA_DIR=$DataDirectory",
    "HNSQR_MCP_TENANT=$Tenant",
    "HNSQR_MCP_ROLE=$Role"
)

# Codex versions that accept only `fast` or `flex` reject the legacy
# `service_tier = "default"` value before any MCP command can run. Removing
# only that obsolete override preserves the user's remaining configuration and
# restores Codex's built-in default selection.
$codexConfigPath = Join-Path $HOME ".codex\config.toml"
if (Test-Path -LiteralPath $codexConfigPath) {
    $codexConfig = Get-Content -Raw -LiteralPath $codexConfigPath
    $updatedCodexConfig = [regex]::Replace(
        $codexConfig,
        '(?m)^\s*service_tier\s*=\s*["'']default["'']\s*\r?\n?',
        ''
    )
    if ($updatedCodexConfig -ne $codexConfig) {
        [IO.File]::WriteAllText(
            $codexConfigPath,
            $updatedCodexConfig,
            [Text.UTF8Encoding]::new($false)
        )
    }
}

function Remove-ExistingServer {
    param([ValidateSet("codex", "claude")][string]$Client)
    $priorPreference = $PSNativeCommandUseErrorActionPreference
    $PSNativeCommandUseErrorActionPreference = $false
    try {
        switch ($Client) {
            "codex" { & $codexCommand mcp remove holosphere 1>$null 2>$null }
            "claude" { & $claudeCommand mcp remove --scope user holosphere 1>$null 2>$null }
        }
    } finally {
        $PSNativeCommandUseErrorActionPreference = $priorPreference
    }
}

function Set-AntigravityMcpServer {
    param([Parameter(Mandatory)][string]$Path)

    $directory = Split-Path -Parent $Path
    [IO.Directory]::CreateDirectory($directory) | Out-Null
    $config = if (Test-Path -LiteralPath $Path) {
        Get-Content -Raw -LiteralPath $Path | ConvertFrom-Json -AsHashtable
    } else {
        @{}
    }
    if (-not $config.ContainsKey("mcpServers")) {
        $config["mcpServers"] = @{}
    }
    $config["mcpServers"]["holosphere"] = @{
        command = $binary
        args = @()
        env = @{
            HNSQR_DATA_DIR = $DataDirectory
            HNSQR_MCP_TENANT = $Tenant
            HNSQR_MCP_ROLE = $Role
        }
        disabled = $false
    }
    $serialized = $config | ConvertTo-Json -Depth 100
    [IO.File]::WriteAllText(
        $Path,
        $serialized + [Environment]::NewLine,
        [Text.UTF8Encoding]::new($false)
    )
}

$codexCommand = (Get-Command codex.cmd -CommandType Application -ErrorAction SilentlyContinue).Source
$claudeCommand = (Get-Command claude.exe -CommandType Application -ErrorAction SilentlyContinue).Source
foreach ($client in @{
    codex = $codexCommand
    claude = $claudeCommand
}.GetEnumerator()) {
    if (-not $client.Value) {
        throw "The '$($client.Key)' native launcher is not installed or is not on PATH."
    }
}
foreach ($client in @("codex", "claude")) {
    Remove-ExistingServer $client
}

& $codexCommand mcp add `
    --env $environment[0] `
    --env $environment[1] `
    --env $environment[2] `
    holosphere -- $binary

# Gemini's supported local agent surface is Google Antigravity. Current builds
# read the global sparse config; existing older IDE installations can retain one
# of the legacy locations, so update those only when already present.
$antigravityConfigPaths = @(
    (Join-Path $HOME ".gemini\config\mcp_config.json")
)
foreach ($legacyPath in @(
    (Join-Path $HOME ".gemini\antigravity\mcp_config.json"),
    (Join-Path $HOME ".gemini\antigravity-ide\mcp_config.json")
)) {
    if (Test-Path -LiteralPath $legacyPath) {
        $antigravityConfigPaths += $legacyPath
    }
}
foreach ($path in $antigravityConfigPaths) {
    Set-AntigravityMcpServer -Path $path
}

# Antigravity CLI's fine-grained permission engine otherwise prompts for every
# MCP call. Auto-approve only HoloSphere, never the global mcp(*) namespace.
$antigravitySettingsPath = Join-Path $HOME ".gemini\antigravity-cli\settings.json"
$antigravitySettingsDirectory = Split-Path -Parent $antigravitySettingsPath
[IO.Directory]::CreateDirectory($antigravitySettingsDirectory) | Out-Null
$antigravitySettings = if (Test-Path -LiteralPath $antigravitySettingsPath) {
    Get-Content -Raw -LiteralPath $antigravitySettingsPath | ConvertFrom-Json -AsHashtable
} else {
    @{}
}
if (-not $antigravitySettings.ContainsKey("permissions")) {
    $antigravitySettings["permissions"] = @{}
}
if (-not $antigravitySettings["permissions"].ContainsKey("allow")) {
    $antigravitySettings["permissions"]["allow"] = @()
}
$antigravityAllow = @($antigravitySettings["permissions"]["allow"])
if ($antigravityAllow -notcontains "mcp(holosphere/*)") {
    $antigravitySettings["permissions"]["allow"] = @(
        $antigravityAllow + "mcp(holosphere/*)"
    )
}
# Antigravity CLI requires an explicit model provider when authenticating with an
# existing Gemini API key. Select the provider without copying the secret into this
# file; the CLI continues to read GEMINI_API_KEY from the process environment.
if (-not [string]::IsNullOrWhiteSpace($env:GEMINI_API_KEY)) {
    $antigravitySettings["modelProvider"] = "gemini"
}
[IO.File]::WriteAllText(
    $antigravitySettingsPath,
    ($antigravitySettings | ConvertTo-Json -Depth 100) + [Environment]::NewLine,
    [Text.UTF8Encoding]::new($false)
)

$claudeDefinition = @{
    type = "stdio"
    command = $binary
    args = @()
    env = @{
        HNSQR_DATA_DIR = $DataDirectory
        HNSQR_MCP_TENANT = $Tenant
        HNSQR_MCP_ROLE = $Role
    }
} | ConvertTo-Json -Depth 8 -Compress
& $claudeCommand mcp add-json --scope user holosphere $claudeDefinition

# Pre-approve only HoloSphere's MCP tools for Claude Code. This is narrower than a
# global permission bypass and lets its agent loop use retrieval and governed memory
# writeback without stopping for every call.
$claudeSettingsPath = Join-Path $HOME ".claude\settings.json"
$claudeSettingsDirectory = Split-Path -Parent $claudeSettingsPath
[IO.Directory]::CreateDirectory($claudeSettingsDirectory) | Out-Null
$settings = if (Test-Path -LiteralPath $claudeSettingsPath) {
    Get-Content -Raw -LiteralPath $claudeSettingsPath | ConvertFrom-Json -AsHashtable
} else {
    @{}
}
if (-not $settings.ContainsKey("permissions")) {
    $settings["permissions"] = @{}
}
if (-not $settings["permissions"].ContainsKey("allow")) {
    $settings["permissions"]["allow"] = @()
}
$allow = @($settings["permissions"]["allow"])
if ($allow -notcontains "mcp__holosphere__*") {
    $settings["permissions"]["allow"] = @($allow + "mcp__holosphere__*")
}
$serializedSettings = $settings | ConvertTo-Json -Depth 100
[IO.File]::WriteAllText(
    $claudeSettingsPath,
    $serializedSettings + [Environment]::NewLine,
    [Text.UTF8Encoding]::new($false)
)

[PSCustomObject]@{
    Binary = $binary
    DataDirectory = $DataDirectory
    Tenant = $Tenant
    Role = $Role
    Codex = "configured"
    Gemini = "configured through Google Antigravity"
    Claude = "configured; mcp__holosphere__* pre-approved"
}
