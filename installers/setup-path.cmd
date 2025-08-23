@echo off
setlocal EnableExtensions
set "TARGET=%~dp0"
set "TARGET=%TARGET:~0,-1%"

rem Already present (anywhere)?
echo %PATH% | findstr /I /C:"%TARGET%" >NUL
if %ERRORLEVEL%==0 (
  rem Still make sure it is FIRST in the current session
  set "PATH=%TARGET%;%PATH%"
) else (
  set "PATH=%TARGET%;%PATH%"
)

set "_QUIET=0"
if /I "%~1"=="/quiet" set "_QUIET=1"
if /I "%~1"=="-q"     set "_QUIET=1"
if /I "%~1"=="-y"     set "_QUIET=1"

if "%_QUIET%"=="0" (
  set /p RESP=Add "%TARGET%" to your PATH (current user, prepended)? [y/n]
  if /I "%RESP%"=="N"  goto :skip
  if /I "%RESP%"=="No" goto :skip
)

for /f "tokens=2,*" %%A in ('reg query HKCU\Environment /v Path ^| find /I "Path"') do set "CUR=%%B"
if "%CUR%"=="" (set "NEW=%TARGET%") else (set "NEW=%TARGET%;%CUR%")
reg add HKCU\Environment /v Path /t REG_EXPAND_SZ /d "%NEW%" /f >NUL

if "%_QUIET%"=="0" (
  echo.
  echo ✅ Added. New terminals will use your shims (higher priority than Python^*\Scripts).
  echo Commands available:
  echo   • build, all, argos
  echo   • d/detect, hm/heatmap, gj/geojson, lv/livevideo
  echo   • classify/clf, pose/pse, obb/object
)
exit /b 0

:skip
echo Skipped.
exit /b 0
