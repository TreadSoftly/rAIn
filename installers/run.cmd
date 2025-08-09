@echo off
setlocal enabledelayedexpansion
set "HERE=%~dp0"
for %%I in ("%HERE%\..") do set "ROOT=%%~fI"
set "PROJ_DIR=%ROOT%\projects"

set "proj="
set "tokens="

:parse
if "%~1"=="" goto afterparse
set "t=%~1"
if /I "%t%"=="run"      (shift & goto parse)
if /I "%t%"=="me"       (shift & goto parse)
if /I "%t%"=="argos"    (set "proj=argos" & shift & goto parse)
if /I "%t%"=="argos:run" (set "proj=argos" & shift & goto parse)
if /I "%t%"=="run:argos" (set "proj=argos" & shift & goto parse)
set "tokens=%tokens% %t%"
shift
goto parse

:afterparse
if not defined proj (
  echo %cd% | findstr /I /C:"\projects\argos" >nul && set "proj=argos"
)

if not defined proj (
  if /I "%cd%"=="%ROOT%"  (
    echo Specify the project:  run argos  ^|  argos [args]
    exit /b 2
  )
  if /I "%cd%"=="%PROJ_DIR%" (
    echo Specify the project:  run argos  ^|  argos [args]
    exit /b 2
  )
)

where python >NUL 2>&1
if %ERRORLEVEL% NEQ 0 (
  echo Python 3 not found. 1>&2
  exit /b 1
)

if /I "%proj%"=="argos" (
  python "%ROOT%\projects\argos\bootstrap.py" --ensure --yes >NUL 2>&1
  for /f "usebackq delims=" %%i in (`python "%ROOT%\projects\argos\bootstrap.py" --print-venv`) do set "VPY=%%i"
  set "PYTHONPYCACHEPREFIX=%LOCALAPPDATA%\rAIn\pycache"
  "%VPY%" -m panoptes.cli %tokens%
  exit /b %ERRORLEVEL%
)

echo Unknown project: %proj% 1>&2
exit /b 2
