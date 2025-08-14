@echo off
setlocal EnableExtensions EnableDelayedExpansion
set "HERE=%~dp0"

set "PYTHONUTF8=1"
set "FORCE_COLOR=1"
if not defined TERM set "TERM=xterm-256color"
set "PANOPTES_PROGRESS_ACTIVE="

call "%HERE%installers\run.cmd" argos %*
exit /b %ERRORLEVEL%
