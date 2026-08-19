@echo off
REM Native Windows IRIS CLI — mouse grab: click window, release: Right Ctrl.
REM Full performance stack: lightning + rex-jit + idle-pause.
cd /d "%~dp0.."
call "%~dp0ensure-build.bat" cli
if errorlevel 1 exit /b 1
start "" "target\release\iris.exe" --config irix-install\iris-windows.toml
