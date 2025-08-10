@echo off
setlocal
set "HERE=%~dp0"
for %%I in ("%HERE%..\..") do set "ROOT=%%~fI"
call "%ROOT%\installers\build.cmd" argos %*
