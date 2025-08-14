@echo off
setlocal EnableExtensions
set "HERE=%~dp0"

rem --- Progress-friendly environment for the Python builder ---
if not defined TERM set "TERM=xterm-256color"
set "PYTHONUTF8=1"
set "PYTHONUNBUFFERED=1"
set "FORCE_COLOR=1"
set "PANOPTES_NESTED_PROGRESS=1"

powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%HERE%build.ps1" %*
exit /b %ERRORLEVEL%
