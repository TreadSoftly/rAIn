@echo off
setlocal
set "TARGET=%~dp0"
set "TARGET=%TARGET:~0,-1%"

echo %PATH% | findstr /I /C:"%TARGET%" >NUL
if %ERRORLEVEL%==0 (
  echo Already on PATH.
  exit /b 0
)

set /p RESP=Add "%TARGET%" to your PATH (current user)? [Y/n]
if /I "%RESP%"=="N" goto :skip
if /I "%RESP%"=="No" goto :skip

for /f "tokens=2,*" %%A in ('reg query HKCU\Environment /v Path ^| find /I "Path"') do set "CUR=%%B"
if "%CUR%"=="" (set "NEW=%TARGET%") else (set "NEW=%CUR%;%TARGET%")
reg add HKCU\Environment /v Path /t REG_EXPAND_SZ /d "%NEW%" /f >NUL

echo ✅ Added. Open a new terminal to use: run / argos
exit /b 0

:skip
echo Skipped.
exit /b 0
