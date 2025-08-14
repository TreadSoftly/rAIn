@echo off
setlocal
set "HERE=%~dp0"
for %%I in ("%HERE%..\..") do set "ROOT=%%~fI"

if not defined TERM set "TERM=xterm-256color"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=UTF-8"
set "PYTHONUNBUFFERED=1"
set "FORCE_COLOR=1"
if not defined PANOPTES_NESTED_PROGRESS set "PANOPTES_NESTED_PROGRESS=1"
if not defined PANOPTES_ENABLE_OSC8 set "PANOPTES_ENABLE_OSC8=1"

call "%ROOT%\installers\run.cmd" %*
exit /b %ERRORLEVEL%
