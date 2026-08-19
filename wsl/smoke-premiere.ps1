# Phase 3 smoke test — headless CI with exported TOML (Windows).
# Usage: .\wsl\smoke-premiere.ps1 [-Config path] [-BaselineJson path]
param(
    [string]$Config = "irix-install\iris-windows.toml",
    [int]$BootTimeoutSec = 120,
    [string]$BaselineJson = "target\release\.iris-smoke-baseline.json",
    [int]$MaxUnderruns = 50
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

Write-Host "=== IRIS Phase 3 smoke (premiere TOML) ===" -ForegroundColor Cyan

if (-not (Test-Path $Config)) {
    Write-Error "Missing config: $Config — export from iris-gui first."
}

& wsl\ensure-build.bat cli
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Starting iris headless CI..."
$iris = Start-Process -FilePath "target\release\iris.exe" -ArgumentList @("--config", $Config) -PassThru -WindowStyle Hidden
Start-Sleep -Seconds 8

Write-Host "iris-ci ping..."
& target\release\iris-ci.exe ping
if ($LASTEXITCODE -ne 0) {
    Stop-Process -Id $iris.Id -Force -ErrorAction SilentlyContinue
    Write-Error "iris-ci ping failed — is ci=true and ci_socket set in TOML?"
}

Write-Host "perf snapshot..."
$jsonPath = "target\release\.iris-smoke-run.json"
& wsl\profile.ps1 -JsonOut $jsonPath | Out-Null

if (Test-Path $jsonPath) {
    $run = Get-Content $jsonPath -Raw | ConvertFrom-Json
    if ($null -ne $run.cpal_underruns -and $run.cpal_underruns -gt $MaxUnderruns) {
        Write-Warning "cpal underruns $($run.cpal_underruns) > $MaxUnderruns"
    }
    if (-not (Test-Path $BaselineJson)) {
        Copy-Item $jsonPath $BaselineJson
        Write-Host "Saved baseline: $BaselineJson" -ForegroundColor Green
    } else {
        $base = Get-Content $BaselineJson -Raw | ConvertFrom-Json
        Write-Host "Baseline underruns: $($base.cpal_underruns)  run: $($run.cpal_underruns)"
    }
}

Write-Host "Stopping iris..."
Stop-Process -Id $iris.Id -Force -ErrorAction SilentlyContinue
Write-Host "Smoke complete." -ForegroundColor Green
