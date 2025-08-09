@echo off
setlocal enabledelayedexpansion

rem -- locate repo roots
set "HERE=%~dp0"
for %%I in ("%HERE%\..") do set "ROOT=%%~fI"
set "PROJ_DIR=%ROOT%\projects"

rem -- parse args
set "proj="
set "tokens="
:parse
if "%~1"=="" goto afterparse
set "t=%~1"
if /I "%t%"=="run"       (shift & goto parse)
if /I "%t%"=="me"        (shift & goto parse)
if /I "%t%"=="argos"     (set "proj=argos" & shift & goto parse)
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

rem -- choose Python (prefer py -3)
set "PY="
where py >NUL 2>&1 && set "PY=py -3"
if not defined PY where python >NUL 2>&1 && set "PY=python"
if not defined PY (
  echo Python 3 not found. 1>&2
  exit /b 1
)

if /I "%proj%"=="argos" (
  %PY% "%ROOT%\projects\argos\bootstrap.py" --ensure --yes >NUL 2>&1

  set "VPY="
  for /f "tokens=* delims=" %%i in ('%PY% "%ROOT%\projects\argos\bootstrap.py" --print-venv') do set "VPY=%%i"
  if not defined VPY set "VPY=%PY%"

  set "PYTHONPYCACHEPREFIX=%LOCALAPPDATA%\rAIn\pycache"
  "%VPY%" -m panoptes.cli %tokens%
  exit /b %ERRORLEVEL%
)

echo Unknown project: %proj% 1>&2
exit /b 2
