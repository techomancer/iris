# Phase 3 verification: start headless CI iris, probe monitor + iris-ci, stop.
param(
    [string]$Config = "irix-install\iris-smoke-ci.toml",
    [int]$WarmupSec = 8
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

function Send-Monitor($cmd, [int]$timeoutSec = 5) {
    $client = New-Object System.Net.Sockets.TcpClient
    $client.Connect("127.0.0.1", 8888)
    $stream = $client.GetStream()
    $writer = New-Object System.IO.StreamWriter($stream)
    $writer.NewLine = "`n"
    $reader = New-Object System.IO.StreamReader($stream)
    $buf = New-Object System.Text.StringBuilder
    $deadline = (Get-Date).AddSeconds($timeoutSec)
    # Read until initial prompt
    while ((Get-Date) -lt $deadline -and -not ($buf.ToString() -match "> `r?`n$")) {
        if ($stream.DataAvailable) {
            $line = $reader.ReadLine()
            if ($null -eq $line) { break }
            [void]$buf.AppendLine($line)
        } else { Start-Sleep -Milliseconds 30 }
    }
    $writer.WriteLine($cmd)
    $writer.Flush()
    $deadline = (Get-Date).AddSeconds($timeoutSec)
    while ((Get-Date) -lt $deadline) {
        if ($stream.DataAvailable) {
            $line = $reader.ReadLine()
            if ($null -eq $line) { break }
            [void]$buf.AppendLine($line)
            if ($line.Trim() -eq ">") { break }
        } else { Start-Sleep -Milliseconds 30 }
    }
    $client.Close()
    # Strip ASCII art banner; keep command output
    $text = $buf.ToString()
    $idx = $text.LastIndexOf("Monitor")
    if ($idx -ge 0) {
        $text = $text.Substring($idx)
    }
    return $text.Trim()
}

Write-Host "=== Phase 3 verify ===" -ForegroundColor Cyan
if (-not (Test-Path $Config)) { throw "Missing $Config" }
if (-not (Test-Path "target\release\iris.exe")) { throw "Build iris first" }
if (-not (Test-Path "target\release\iris-ci.exe")) { throw "Build iris-ci first" }

$logOut = Join-Path $env:TEMP "iris-smoke-out.log"
$logErr = Join-Path $env:TEMP "iris-smoke-err.log"
Write-Host "Starting iris (logs: $logOut)..."
$iris = Start-Process -FilePath "target\release\iris.exe" `
    -ArgumentList @("--config", $Config) `
    -RedirectStandardOutput $logOut `
    -RedirectStandardError $logErr `
    -PassThru -WindowStyle Hidden

try {
    Write-Host "Warmup ${WarmupSec}s..."
    Start-Sleep -Seconds $WarmupSec

    if ($iris.HasExited) {
        Write-Host "--- iris stderr ---" -ForegroundColor Red
        Get-Content $logErr -ErrorAction SilentlyContinue | Select-Object -Last 40
        Write-Host "--- iris stdout ---" -ForegroundColor Red
        Get-Content $logOut -ErrorAction SilentlyContinue | Select-Object -Last 20
        throw "iris exited early (code $($iris.ExitCode))"
    }

    Write-Host "`n--- iris-ci ping ---" -ForegroundColor Yellow
    $ping = & target\release\iris-ci.exe ping 2>&1
    $ping | Write-Host
    if ($LASTEXITCODE -ne 0) { throw "iris-ci ping failed ($LASTEXITCODE)" }

    Write-Host "`n--- monitor: perf snapshot ---" -ForegroundColor Yellow
    Send-Monitor "perf snapshot" | Write-Host

    Write-Host "`n--- monitor: hal2 status ---" -ForegroundColor Yellow
    Send-Monitor "hal2 status" | Write-Host

    Write-Host "`n(note: headless CI skips REX3; use premiere CLI/GUI for rex jit status)" -ForegroundColor DarkGray

    Write-Host "`nPASS: CI socket + monitor commands OK" -ForegroundColor Green
} finally {
    if (-not $iris.HasExited) {
        Write-Host "Stopping iris (pid $($iris.Id))..."
        Stop-Process -Id $iris.Id -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 1
    }
}
