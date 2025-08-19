@echo off
setlocal
set "HERE=%~dp0"
if not defined TERM set "TERM=xterm-256color"
set "PYTHONUTF8=1"
set "PYTHONUNBUFFERED=1"
set "FORCE_COLOR=1"
set "PANOPTES_NESTED_PROGRESS=1"
set "PANOPTES_PROGRESS_ACTIVE=0"
call "%HERE%run.cmd" argos obb %*
exit /b %ERRORLEVEL%
