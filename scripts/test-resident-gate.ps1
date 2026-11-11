param(
    [ValidateSet('vulkan', 'dx12')]
    [string[]] $Backend = @('vulkan', 'dx12')
)

$ErrorActionPreference = 'Stop'
$previousBackend = [Environment]::GetEnvironmentVariable('REX3_RESIDENT_BACKEND', 'Process')
$previousProfile = [Environment]::GetEnvironmentVariable('IRIS_REX_JIT_PROFILE', 'Process')
$requestedBackends = @($Backend | ForEach-Object { $_.ToLowerInvariant() })
if ($requestedBackends.Count -eq 0 -or
    @($requestedBackends | Select-Object -Unique).Count -ne $requestedBackends.Count) {
    throw 'Select at least one backend, without duplicates'
}
$profile = Join-Path 'target' "resident-gate-jit-$([Guid]::NewGuid().ToString('N')).bin"
$env:IRIS_REX_JIT_PROFILE = $profile

function Invoke-ResidentTests {
    param([string] $Filter, [int] $Expected, [bool] $Ignored)

    $testArgs = @(
        'test', '--locked', '-p', 'iris', '--lib', '--features', 'wgpu,rex-jit,rexdiag',
        $Filter, '--', '--test-threads=1'
    )
    if ($Ignored) {
        $testArgs += @('--ignored', '--nocapture')
    }

    Write-Host "Running $Filter ($Expected tests, backend=$env:REX3_RESIDENT_BACKEND)"
    # Windows PowerShell treats redirected native stderr as an error record.
    $previousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        $output = & cargo @testArgs 2>&1
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousPreference
    }
    if ($exitCode -ne 0) {
        $output | ForEach-Object { Write-Host $_ }
        throw "cargo test failed for $Filter (exit $exitCode)"
    }
    $summaries = @($output | Where-Object { "$_" -match '^test result:' })
    if ($summaries.Count -ne 1 -or
        "$($summaries[0])" -notmatch '^test result: ok\. (\d+) passed; 0 failed; (\d+) ignored;') {
        throw "Missing or ambiguous libtest summary for $Filter"
    }
    $passed = [int] $Matches[1]
    $ignoredCount = [int] $Matches[2]
    if ($passed -ne $Expected) {
        throw "$Filter executed $passed tests; expected $Expected"
    }
    if ($Ignored) {
        if ($ignoredCount -ne 0 -or @($output | Where-Object { "$_" -match '\bskipped\b' }).Count -ne 0) {
            throw "$Filter must execute every requested hardware test without skips"
        }
        $adapters = @($output | Where-Object {
            "$_" -match 'resident (test|no-touch|device-loss).*:|direct Newport scanout adapter:'
        })
        if ($adapters.Count -lt $Expected) {
            throw "$Filter reported $($adapters.Count) hardware adapters for $Expected tests"
        }
        Write-Host "Hardware adapter reports: $($adapters.Count) for $Expected tests"
        Write-Host $adapters[0]
    }
    Write-Host $summaries[0]
}

try {
    Invoke-ResidentTests 'resident_' 25 $false
    Invoke-ResidentTests 'dev::ng1::rex3_flat_fill::tests' 3 $false
    Invoke-ResidentTests 'direct_ng1_scanout_' 2 $false
    foreach ($name in $requestedBackends) {
        $env:REX3_RESIDENT_BACKEND = $name
        Invoke-ResidentTests 'resident_gpu_' 10 $true
        Invoke-ResidentTests 'dev::ng1::wgpu::direct_tests' 6 $true
    }
} finally {
    [Environment]::SetEnvironmentVariable('REX3_RESIDENT_BACKEND', $previousBackend, 'Process')
    [Environment]::SetEnvironmentVariable('IRIS_REX_JIT_PROFILE', $previousProfile, 'Process')
    if (Test-Path -LiteralPath $profile) {
        Remove-Item -LiteralPath $profile
    }
}
