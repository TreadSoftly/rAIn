@echo off
setlocal EnableExtensions EnableDelayedExpansion
set "HERE=%~dp0"
set "PYTHONUTF8=1"
set "FORCE_COLOR=1"
if not defined TERM set "TERM=xterm-256color"
set "PANOPTES_PROGRESS_ACTIVE="
rem Do NOT set ARGOS_AUTOBUILD here; keep selection interactive by default.
call "%HERE%installers\build.cmd" %*
exit /b %ERRORLEVEL%
