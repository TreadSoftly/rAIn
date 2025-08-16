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

rem --- Progress-friendly environment for CLI runs ---
if not defined TERM set "TERM=xterm-256color"
set "PYTHONUTF8=1"
set "PYTHONUNBUFFERED=1"
set "FORCE_COLOR=1"
set "PANOPTES_NESTED_PROGRESS=1"
set "PANOPTES_PROGRESS_ACTIVE=0"

rem --- Parse args ---
set "PROJ="
set "SAW_BUILD=0"
set "TOKENS="
set "MODE_LIVE=0"
set "FOUND_L=0"
set "FOUND_V=0"

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

rem ---- broadened live detection ----
if /I "%~1"=="lv"         set "MODE_LIVE=1" & shift & goto parse
if /I "%~1"=="livevideo"  set "MODE_LIVE=1" & shift & goto parse
if /I "%~1"=="live"       set "MODE_LIVE=1" & shift & goto parse
if /I "%~1"=="video"      set "MODE_LIVE=1" & shift & goto parse
if /I "%~1"=="ldv"        set "MODE_LIVE=1" & shift & goto parse
if /I "%~1"=="lvd"        set "MODE_LIVE=1" & shift & goto parse
if /I "%~1"=="l"          set "FOUND_L=1"
if /I "%~1"=="v"          set "FOUND_V=1"

set "TOKENS=%TOKENS% %1"
shift
goto parse

:afterparse
if "%MODE_LIVE%"=="0" (
  if "%FOUND_L%"=="1" if "%FOUND_V%"=="1" set "MODE_LIVE=1"
)

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

rem ---- module switch ----
if "%MODE_LIVE%"=="1" ( set "PYMOD=panoptes.live.cli" ) else ( set "PYMOD=panoptes.cli" )

"%VPY%" -m %PYMOD% %TOKENS%
exit /b %ERRORLEVEL%
