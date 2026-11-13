# Give a Windows machine with no sound card a virtual one, so the real-device
# audio tests have a default output to play into and loop back (TESTING.md,
# Host audio). For a CI runner: it installs a driver, so it needs an
# administrator and changes the machine's audio devices.
#
# Installs VB-Audio's VB-CABLE, pinned by hash, from the vendor's own
# download. VB-CABLE is donationware: the vendor's licensing page asks for a
# paid licence for professional use and for redistribution. This script
# downloads from the vendor on every run and keeps nothing in the repository.
# It trusts the publisher of the driver's catalog, installs the driver with
# devcon (pinned by hash, from the sound-ci-helpers project that cubeb's CI
# also takes it from), starts Windows Audio and waits for the cable's render
# endpoint. The runner has no other sound device, so that endpoint becomes
# the default output, and the script fails if another render endpoint
# exists. Then run the tests with IRIS_AUDIO_LOOPBACK=default:default-render.
#
# Usage: audio-loopback-windows.ps1

$ErrorActionPreference = "Stop"
Set-StrictMode -Version 2.0

$driverUrl = "https://download.vb-audio.com/Download_CABLE/VBCABLE_Driver_Pack43.zip"
$driverSha256 = "66fd0a4d9f4896ff41632b7e3d53892c085c4561f53e8ae8d0f0bc10eedd1cdd"
$devconUrl = "https://raw.githubusercontent.com/LABSN/sound-ci-helpers/20a2b9bbd21fd005d8fe89933901506dba84ea4e/windows/devcon.exe"
$devconSha256 = "97cff42f8c0fe4fbdf991273159516bf78090625a933c3983ebd6f62284e329a"

$work = Join-Path ([System.IO.Path]::GetTempPath()) "iris-vbcable"
New-Item -ItemType Directory -Force -Path $work | Out-Null

function Get-Pinned([string] $Url, [string] $Sha256, [string] $Path) {
    Invoke-WebRequest -Uri $Url -OutFile $Path -MaximumRetryCount 3 -RetryIntervalSec 2
    $got = (Get-FileHash -Algorithm SHA256 -Path $Path).Hash.ToLowerInvariant()
    if ($got -ne $Sha256) {
        throw "$Url has SHA-256 $got, expected $Sha256"
    }
}

$zip = Join-Path $work "vbcable.zip"
$devcon = Join-Path $work "devcon.exe"
Get-Pinned $driverUrl $driverSha256 $zip
Get-Pinned $devconUrl $devconSha256 $devcon
$driver = Join-Path $work "driver"
Expand-Archive -LiteralPath $zip -DestinationPath $driver -Force

# Without the publisher trusted, the install waits for a prompt nobody answers.
$catalog = Join-Path $driver "vbaudio_cable64_win7.cat"
$signer = (Get-AuthenticodeSignature -FilePath $catalog).SignerCertificate
if ($null -eq $signer) {
    throw "no signer certificate on $catalog"
}
$cer = Join-Path $work "vbcable.cer"
[System.IO.File]::WriteAllBytes($cer, $signer.Export([System.Security.Cryptography.X509Certificates.X509ContentType]::Cert))
certutil.exe -addstore -f TrustedPublisher $cer
if ($LASTEXITCODE -ne 0) { throw "certutil failed: $LASTEXITCODE" }

# devcon exits 1 when it asks for a reboot, and that fails here too: the
# cable's endpoint is expected without one.
& $devcon install (Join-Path $driver "vbMmeCable64_win7.inf") VBAudioVACWDM
if ($LASTEXITCODE -ne 0) { throw "devcon install failed: $LASTEXITCODE" }

# cubeb's CI also lets all apps use a microphone before its capture tests.
# A loopback of a render device is not a microphone, so this may not be
# needed; it stays until a run shows that it is not.
$privacy = "HKLM:\SOFTWARE\Policies\Microsoft\Windows\AppPrivacy"
New-Item -Path $privacy -Force | Out-Null
New-ItemProperty -Path $privacy -Name LetAppsAccessMicrophone -Value 1 -PropertyType DWord -Force | Out-Null

Start-Service -Name AudioEndpointBuilder, Audiosrv
# Endpoints appear some seconds after the install. Windows names them after
# the driver's model, not the cable's ends: a CI run saw "Speakers (VB-Audio
# Virtual Cable)" for the render side and "CABLE Output (VB-Audio Virtual
# Cable)" for the capture side. An endpoint's instance ID says which side it
# is: {0.0.0.*} renders, {0.0.1.*} captures.
function Get-Endpoints {
    Get-PnpDevice -Class AudioEndpoint -PresentOnly -ErrorAction SilentlyContinue |
        Where-Object { $_.Status -eq "OK" }
}
$render = @()
for ($i = 0; $i -lt 60 -and $render.Count -eq 0; $i++) {
    $render = @(Get-Endpoints | Where-Object { $_.InstanceId -like "*{0.0.0.*" -and $_.FriendlyName -like "*VB-Audio Virtual Cable*" })
    if ($render.Count -eq 0) { Start-Sleep -Seconds 1 }
}
$all = Get-Endpoints | ForEach-Object { "$($_.FriendlyName)  [$($_.InstanceId)]" }
Write-Host "audio endpoints:`n$($all -join "`n")"
if ($render.Count -eq 0) {
    throw "no render endpoint of the VB-Audio Virtual Cable appeared within 60 s; endpoints found: $($all -join '; ')"
}
# The tests play into the default output. The runner has no sound device, so
# the cable is the only render endpoint and so the default; another one could
# be the default instead, and the tests would hear nothing.
$others = @(Get-Endpoints | Where-Object { $_.InstanceId -like "*{0.0.0.*" -and $_.FriendlyName -notlike "*VB-Audio Virtual Cable*" })
if ($others.Count -gt 0) {
    throw "the cable is not the only render endpoint, so it may not be the default output: $(($others | ForEach-Object FriendlyName) -join '; ')"
}
Write-Host "audio-loopback-windows: $($render[0].FriendlyName) is the only render endpoint"
