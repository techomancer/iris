# Capture host stderr + print checklist when IRIX apps quit silently.
# Usage:
#   wsl\capture-app-crash.ps1
#   wsl\capture-app-crash.ps1 -Config irix-install\iris-windows-384.toml
param(
    [string]$Config = "irix-install\iris-windows.toml",
    [string]$LogFile = "premiere-debug.log"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

Write-Host "=== IRIS silent-quit capture ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "Before reproducing, note your TOML banks line:" -ForegroundColor Yellow
if (Test-Path $Config) {
    Select-String -Path $Config -Pattern "^banks\s*=" | ForEach-Object { Write-Host "  $($_.Line)" }
} else {
    Write-Host "  (config not found: $Config)" -ForegroundColor Red
}
Write-Host ""
Write-Host "Second terminal:  telnet 127.0.0.1 8888" -ForegroundColor Yellow
Write-Host "When an app quits, run in monitor:" -ForegroundColor Yellow
Write-Host "  stop`n  status`n  regs`n  bt`n  dt 80`n  cow status`n  mc status"
Write-Host ""
Write-Host "In IRIX shell after quit:" -ForegroundColor Yellow
Write-Host "  hinv -t memory"
Write-Host "  ps -ef | grep -i <appname>"
Write-Host "  tail -50 /var/adm/SYSLOG"
Write-Host ""
Write-Host "Logging to: $LogFile" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop iris when done." -ForegroundColor Gray
Write-Host ""

& wsl\ensure-build.bat cli
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$logPath = Join-Path $root $LogFile
$header = @"
=== IRIS capture session $(Get-Date -Format o) ===
Config: $Config
Banks: $(if (Test-Path $Config) { (Select-String -Path $Config -Pattern '^banks\s*=').Line } else { 'n/a' })
===
"@

$header | Out-File -FilePath $logPath -Encoding utf8

& "target\release\iris.exe" --config $Config 2>&1 | Tee-Object -FilePath $logPath -Append
