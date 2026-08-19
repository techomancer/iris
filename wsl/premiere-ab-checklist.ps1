# A/B checklist for YouTube premiere recording (Windows native IRIS).
# Usage: .\wsl\premiere-ab-checklist.ps1

Write-Host ""
Write-Host "=== IRIS Premiere A/B Checklist ===" -ForegroundColor Cyan
Write-Host ""

Write-Host "TAKE A (slow baseline — GUI CPU path)" -ForegroundColor Yellow
Write-Host "  Build:  cargo build -p iris-gui --release   (no premiere feature)"
Write-Host "  Run:    wsl\run-iris-gui-windows.bat"
Write-Host "  Expect: Low MIPS in status bar; high host CPU at idle desktop"
Write-Host ""

Write-Host "TAKE B (fast — premiere CLI, lightning + rex-jit + idle-pause)" -ForegroundColor Green
Write-Host "  Run:    wsl\run-iris-premiere.bat"
Write-Host "  Config: irix-install\iris-windows.toml (scale=1, export from GUI)"
Write-Host "  Expect: lower idle CPU with idle-pause; rex-jit speeds up REX3 draw"
Write-Host "  Optional: rebuild with --features ...,jitv2 to try the region JIT (experimental)"
Write-Host ""

Write-Host "TAKE B-alt (in-process GUI with premiere core + GL capture)" -ForegroundColor Green
Write-Host "  Run:    wsl\run-iris-gui-premiere.bat"
Write-Host "  Sets:   IRIS_GUI_GL=1, lightning+idle-pause build"
Write-Host "  Note:   Still slower than CLI OpenGL for heavy 3D"
Write-Host ""

Write-Host "Verify (telnet 127.0.0.1 8888):" -ForegroundColor Cyan
Write-Host "  perf snapshot   (Phase 3 aggregate: REX3, HAL2, affinity)"
Write-Host "  rex jit status"
Write-Host "  hal2 status   (cpal underruns should stay low; codec A uses dedicated pump thread)"
Write-Host ""
Write-Host "Benchmark helper: wsl\profile.ps1"
Write-Host "CI smoke:         wsl\smoke-premiere.ps1"
Write-Host ""

Write-Host "GUI workflow:" -ForegroundColor Cyan
Write-Host "  File -> Prepare for premiere...  (exports TOML)"
Write-Host ""
