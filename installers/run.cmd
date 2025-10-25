@echo off
setlocal EnableExtensions

set "HERE=%~dp0"
for %%I in ("%HERE%..") do set "ROOT=%%~fI"

rem Prefer PowerShell 7 if available, otherwise fall back to Windows PowerShell.
set "PS_EXE=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"
if exist "%ProgramFiles%\PowerShell\7\pwsh.exe" set "PS_EXE=%ProgramFiles%\PowerShell\7\pwsh.exe"
if exist "%ProgramFiles(x86)%\PowerShell\7\pwsh.exe" set "PS_EXE=%ProgramFiles(x86)%\PowerShell\7\pwsh.exe"

"%PS_EXE%" -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%HERE%run.ps1" %*
exit /b %ERRORLEVEL%
