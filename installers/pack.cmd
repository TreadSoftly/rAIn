@echo off
setlocal
set "HERE=%~dp0"
powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%HERE%pack.ps1" %*
exit /b %ERRORLEVEL%
