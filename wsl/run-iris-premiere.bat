@echo off
REM One-liner for YouTube / recording: full-feature iris CLI, premiere config.
cd /d "%~dp0.."
call "%~dp0ensure-build.bat" cli
if errorlevel 1 exit /b 1
echo.
echo Premiere mode: iris CLI + OpenGL. Click window to grab mouse; Right Ctrl releases.
echo Monitor: telnet 127.0.0.1 8888  ^|  rex jit status  ^|  hal2 status
echo.
"target\release\iris.exe" --config irix-install\iris-windows.toml
