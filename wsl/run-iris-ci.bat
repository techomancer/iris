@echo off
setlocal
cd /d "%~dp0.."
call wsl\ensure-build.bat cli
if errorlevel 1 exit /b 1
echo Starting iris in CI mode (TCP 127.0.0.1:19851) with irix-install\iris-windows.toml
echo Use: iris-ci ping   (after VM is up)
start "iris-ci" /wait target\release\iris.exe --config irix-install\iris-windows.toml
