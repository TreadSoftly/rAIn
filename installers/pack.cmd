@echo off
setlocal
set "HERE=%~dp0"

rem Keep packaging quiet & progress-friendly for any subprocess Python
if not defined TERM set "TERM=xterm-256color"
set "PYTHONUTF8=1"
set "PYTHONUNBUFFERED=1"
set "FORCE_COLOR=1"
set "PANOPTES_NESTED_PROGRESS=1"
set "PANOPTES_PROGRESS_ACTIVE=0"

powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%HERE%pack.ps1" %*
exit /b %ERRORLEVEL%
