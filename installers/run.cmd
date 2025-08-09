@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "HERE=%~dp0"
for %%I in ("%HERE%..") do set "ROOT=%%~fI"
set "PROJ_DIR=%ROOT%\projects"

rem --- Ensure Git LFS ---
where git >NUL 2>&1
if %ERRORLEVEL% EQU 0 (
  git lfs version >NUL 2>&1
  if %ERRORLEVEL% EQU 0 (
    git -C "%ROOT%" lfs install --local >NUL 2>&1
    set "GIT_LFS_SKIP_SMUDGE=0"
    git -C "%ROOT%" lfs pull >NUL 2>&1
  ) else (
    echo ⚠️  Git LFS not installed. Large files (models) may be missing.
    echo     Install Git LFS and re-run this command.
  )
)

rem --- Parse args ---
set "PROJ="
set "SAW_BUILD=0"
set "TOKENS="

:parse
if "%~1"=="" goto afterparse

if /I "%~1"=="run"         shift & goto parse
if /I "%~1"=="me"          shift & goto parse
if /I "%~1"=="build"       set "SAW_BUILD=1" & shift & goto parse
if /I "%~1"=="package"     set "SAW_BUILD=1" & shift & goto parse
if /I "%~1"=="pack"        set "SAW_BUILD=1" & shift & goto parse
if /I "%~1"=="argos"       set "PROJ=argos" & shift & goto parse
if /I "%~1"=="argos:run"   set "PROJ=argos" & shift & goto parse
if /I "%~1"=="run:argos"   set "PROJ=argos" & shift & goto parse
if /I "%~1"=="argos:build" set "PROJ=argos" & set "SAW_BUILD=1" & shift & goto parse
if /I "%~1"=="build:argos" set "PROJ=argos" & set "SAW_BUILD=1" & shift & goto parse

set "TOKENS=%TOKENS% %1"
shift
goto parse

:afterparse
rem --- Infer project from CWD ---
echo %CD% | findstr /I "\\projects\\argos" >NUL && set "PROJ=argos"

if not defined PROJ (
  if /I "%CD%"=="%ROOT%" goto needproj
  if /I "%CD%"=="%PROJ_DIR%" goto needproj
)
goto gotproj

:needproj
echo Specify the project:  run argos  ^|  argos [args]
exit /b 2

:gotproj
if "%SAW_BUILD%"=="1" (
  if exist "%ROOT%\installers\build.cmd" (
    call "%ROOT%\installers\build.cmd" %PROJ% %TOKENS%
    exit /b %ERRORLEVEL%
  ) else (
    echo Build script not found: installers\build.cmd
    exit /b 1
  )
)

rem --- Find Python (prefer py -3, then python3/python) ---
set "PY="
where py >NUL 2>&1 && set "PY=py -3"
if not defined PY (
  for %%P in (python3.exe python.exe) do (
    where %%P >NUL 2>&1 && set "PY=%%P" && goto gotpy
  )
)
if not defined PY (
  echo Python 3 not found.
  exit /b 1
)

:gotpy
%PY% "%ROOT%\projects\argos\bootstrap.py" --ensure --yes >NUL 2>&1
for /f "usebackq delims=" %%V in (`%PY% "%ROOT%\projects\argos\bootstrap.py" --print-venv`) do set "VPY=%%V"
set "PYTHONPYCACHEPREFIX=%LOCALAPPDATA%\rAIn\pycache"
"%VPY%" -m panoptes.cli %TOKENS%
exit /b %ERRORLEVEL%
