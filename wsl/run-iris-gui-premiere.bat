@echo off
REM iris-gui with premiere embedded core (lightning + idle-pause) + optional GL capture.
cd /d "%~dp0.."
set IRIS_GUI_FEATURES=premiere
set IRIS_GUI_STAMP=target\release\.iris-gui-premiere.stamp
set IRIS_GUI_WANT=premiere

call "%~dp0ensure-build.bat" gui
if errorlevel 1 exit /b 1

set IRIS_GUI_GL=1

echo.
echo Premiere GUI: in-process iris + OpenGL capture (IRIS_GUI_GL=1).
echo For max 3D recording use run-iris-premiere.bat (native CLI OpenGL).
echo.
start "" "target\release\iris-gui.exe"
