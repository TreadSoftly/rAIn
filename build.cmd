@echo off
setlocal
set "HERE=%~dp0"
call "%HERE%installers\build.cmd" %*
