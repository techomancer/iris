@echo off
REM Usage: ensure-build.bat cli|gui
REM Rebuilds when the binary or feature stamp is missing/outdated.
setlocal
cd /d "%~dp0.."

if /i "%~1"=="cli" goto :cli
if /i "%~1"=="gui" goto :gui
echo Usage: ensure-build.bat cli^|gui
exit /b 1

:cli
set BIN=target\release\iris.exe
set STAMP=target\release\.iris-cli-premiere.stamp
set WANT=lightning,rex-jit,idle-pause
set BUILD=cargo +nightly-x86_64-pc-windows-msvc build --release --bin iris --features %WANT%
goto :check

:gui
set BIN=target\release\iris-gui.exe
set STAMP=target\release\.iris-gui-premiere.stamp
if defined IRIS_GUI_WANT (set WANT=%IRIS_GUI_WANT%) else (set WANT=premiere)
if defined IRIS_GUI_FEATURES (set FEAT=%IRIS_GUI_FEATURES%) else (set FEAT=premiere)
set BUILD=cargo +nightly-x86_64-pc-windows-msvc build -p iris-gui --release --features %FEAT%
goto :check

:check
if not exist "%BIN%" goto :build
if not exist "%STAMP%" goto :build
set /p STAMPVAL=<"%STAMP%"
if not "%STAMPVAL%"=="%WANT%" goto :build
exit /b 0

:build
echo Building %BIN% (features: %WANT%)...
%BUILD%
if errorlevel 1 exit /b 1
echo %WANT%> "%STAMP%"
exit /b 0
