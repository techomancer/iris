# IRIS performance snapshot helper (Windows)
# Usage: .\wsl\profile.ps1 [-IrisExe path] [-MonitorHost 127.0.0.1:8888] [-JsonOut path]

param(
    [string]$IrisExe = "target\release\iris.exe",
    [string]$MonitorHost = "127.0.0.1:8888",
    [string]$JsonOut = ""
)

$ErrorActionPreference = "Stop"
$timestamp = Get-Date -Format o

function Send-Monitor($cmd) {
    try {
        $client = New-Object System.Net.Sockets.TcpClient
        $client.Connect($MonitorHost.Split(':')[0], [int]$MonitorHost.Split(':')[1])
        $stream = $client.GetStream()
        $w = New-Object System.IO.StreamWriter($stream)
        $r = New-Object System.IO.StreamReader($stream)
        $w.WriteLine($cmd)
        $w.Flush()
        Start-Sleep -Milliseconds 250
        $out = ""
        while ($stream.DataAvailable) {
            $line = $r.ReadLine()
            if ($null -eq $line) { break }
            $out += $line + "`n"
        }
        $client.Close()
        return $out.Trim()
    } catch {
        return "(monitor unavailable: $_)"
    }
}

Write-Host "=== IRIS profile snapshot ===" -ForegroundColor Cyan
Write-Host "Time: $timestamp"

$proc = Get-Process iris -ErrorAction SilentlyContinue
$cpuSec = $null
$wsMb = $null
if ($proc) {
    $cpuSec = [math]::Round($proc.CPU, 2)
    $wsMb = [math]::Round($proc.WorkingSet64/1MB, 1)
    Write-Host "`nProcess iris.exe:"
    Write-Host "  CPU (s): $cpuSec"
    Write-Host "  WS (MB): $wsMb"
} else {
    Write-Host "`niris.exe not running — start premiere bat first." -ForegroundColor Yellow
}

Write-Host "`n--- monitor: audio status ---"
$audio = Send-Monitor "audio status"
Write-Host $audio

Write-Host "`n--- monitor: rex jit status ---"
$rexJit = Send-Monitor "rex jit status"
Write-Host $rexJit

Write-Host "`n--- monitor: perf snapshot ---"
$perf = Send-Monitor "perf snapshot"
Write-Host $perf

$underruns = $null
if ($audio -match 'underruns?\s*[=:]\s*(\d+)') {
    $underruns = [int]$Matches[1]
}

$snapshot = [ordered]@{
    timestamp = $timestamp
    iris_running = [bool]$proc
    cpu_seconds = $cpuSec
    working_set_mb = $wsMb
    audio_status = $audio
    rex_jit_status = $rexJit
    perf_snapshot = $perf
    host_underruns = $underruns
}

if ($JsonOut) {
    $snapshot | ConvertTo-Json -Depth 4 | Set-Content -Encoding utf8 $JsonOut
    Write-Host "`nWrote JSON baseline: $JsonOut" -ForegroundColor Green
}

Write-Host "`nDone. Compare idle vs glxgears; underruns should stay low under load." -ForegroundColor Green
