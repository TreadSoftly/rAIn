@echo off
setlocal enabledelayedexpansion
set HERE=%~dp0
for %%I in ("%HERE%\..") do set ROOT=%%~fI
set CWD=%CD%

rem parse args
set PROJ=
set TOKENS=
:loop
if "%~1"=="" goto parsed
set A=%~1
for %%z in (%A%) do set LA=%%z
for %%a in (run me) do if /I "!LA!"=="%%a" shift & goto loop
for %%a in (argos argos:run run:argos) do if /I "!LA!"=="%%a" set PROJ=argos& shift & goto loop
set TOKENS=%TOKENS% "%~1"
shift
goto loop
:parsed

rem infer from CWD
echo %CWD% | findstr /I "\\projects\\argos" >NUL && set PROJ=argos

rem need explicit project at root/projects
if "%PROJ%"=="" (
  if /I "%CWD%"=="%ROOT%"  goto needproj
  if /I "%CWD%"=="%ROOT%\projects" goto needproj
)

if /I "%PROJ%"=="argos" (
  where py >NUL 2>&1 && (set "PY=py -3") || (set "PY=python")
  %PY% "%ROOT%\projects\argos\bootstrap.py" --ensure --yes >NUL 2>&1
  for /f "usebackq delims=" %%i in (`%PY% "%ROOT%\projects\argos\bootstrap.py" --print-venv`) do set "VPY=%%i"
  set "PYTHONPYCACHEPREFIX=%LOCALAPPDATA%\rAIn\pycache"
  "%VPY%" -m panoptes.cli %TOKENS%
  exit /b %ERRORLEVEL%
)

echo Unknown project: %PROJ%
exit /b 2

:needproj
echo Specify the project:  run argos  ^|  argos [args]
exit /b 2
