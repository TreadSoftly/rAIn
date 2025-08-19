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

rem --- Parse args (preserving quotes safely) ---
set "PROJ="
set "SAW_BUILD=0"
set "MODE_LIVE=0"
set "FOUND_L=0"
set "FOUND_V=0"
set "TOKENS="
set "CRUMB="
set "OPNORM="

:parse
if "%~1"=="" goto afterparse

rem Skip wrappers
if /I "%~1"=="run"         shift & goto parse
if /I "%~1"=="me"          shift & goto parse

rem Build
if /I "%~1"=="build"       set "SAW_BUILD=1" & shift & goto parse
if /I "%~1"=="package"     set "SAW_BUILD=1" & shift & goto parse
if /I "%~1"=="pack"        set "SAW_BUILD=1" & shift & goto parse

rem Project token
if /I "%~1"=="argos"       set "PROJ=argos" & shift & goto parse
if /I "%~1"=="argos:run"   set "PROJ=argos" & shift & goto parse
if /I "%~1"=="run:argos"   set "PROJ=argos" & shift & goto parse
if /I "%~1"=="argos:build" set "PROJ=argos" & set "SAW_BUILD=1" & shift & goto parse
if /I "%~1"=="build:argos" set "PROJ=argos" & set "SAW_BUILD=1" & shift & goto parse

rem Live aliases
if /I "%~1"=="lv"         set "MODE_LIVE=1" & shift & goto parse
if /I "%~1"=="livevideo"  set "MODE_LIVE=1" & shift & goto parse
if /I "%~1"=="live"       set "MODE_LIVE=1" & shift & goto parse
if /I "%~1"=="video"      set "MODE_LIVE=1" & shift & goto parse
if /I "%~1"=="ldv"        set "MODE_LIVE=1" & shift & goto parse
if /I "%~1"=="lvd"        set "MODE_LIVE=1" & shift & goto parse

rem "crumb" tokens that can combine to mean live mode
if /I "%~1"=="l"          set "FOUND_L=1" & set "CRUMB=%CRUMB% "%~1"" & shift & goto parse
if /I "%~1"=="v"          set "FOUND_V=1" & set "CRUMB=%CRUMB% "%~1"" & shift & goto parse

rem First-op detection (don't add it to TOKENS; we'll front-load later)
if not defined OPNORM (
  if /I "%~1"=="d"          set "OPNORM=d"  & shift & goto parse
  if /I "%~1"=="detect"     set "OPNORM=d"  & shift & goto parse
  if /I "%~1"=="-d"         set "OPNORM=d"  & shift & goto parse
  if /I "%~1"=="--detect"   set "OPNORM=d"  & shift & goto parse

  if /I "%~1"=="hm"         set "OPNORM=hm" & shift & goto parse
  if /I "%~1"=="heatmap"    set "OPNORM=hm" & shift & goto parse
  if /I "%~1"=="-hm"        set "OPNORM=hm" & shift & goto parse
  if /I "%~1"=="--hm"       set "OPNORM=hm" & shift & goto parse
  if /I "%~1"=="-heatmap"   set "OPNORM=hm" & shift & goto parse
  if /I "%~1"=="--heatmap"  set "OPNORM=hm" & shift & goto parse

  if /I "%~1"=="gj"         set "OPNORM=gj" & shift & goto parse
  if /I "%~1"=="geojson"    set "OPNORM=gj" & shift & goto parse
  if /I "%~1"=="-gj"        set "OPNORM=gj" & shift & goto parse
  if /I "%~1"=="--gj"       set "OPNORM=gj" & shift & goto parse
  if /I "%~1"=="-geojson"   set "OPNORM=gj" & shift & goto parse
  if /I "%~1"=="--geojson"  set "OPNORM=gj" & shift & goto parse

  if /I "%~1"=="classify"   set "OPNORM=classify" & shift & goto parse
  if /I "%~1"=="clf"        set "OPNORM=classify" & shift & goto parse
  if /I "%~1"=="pose"       set "OPNORM=pose"     & shift & goto parse
  if /I "%~1"=="pse"        set "OPNORM=pose"     & shift & goto parse
  if /I "%~1"=="obb"        set "OPNORM=obb"      & shift & goto parse
  if /I "%~1"=="object"     set "OPNORM=obb"      & shift & goto parse
)

rem Otherwise, accumulate token preserving quotes
set "TOKENS=%TOKENS% "%~1""
shift
goto parse

:afterparse
if "%MODE_LIVE%"=="0" (
  if "%FOUND_L%"=="1" if "%FOUND_V%"=="1" (
    set "MODE_LIVE=1"
  ) else (
    rem Not live; put crumbs back in front (safe; rare case)
    set "TOKENS=%CRUMB%%TOKENS%"
  )
)

rem Infer project from cwd if not given
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
    set "TOKENS_FINAL=%TOKENS%"
    if defined OPNORM set "TOKENS_FINAL=""%OPNORM%""%TOKENS%"
    call "%ROOT%\installers\build.cmd" %PROJ% %TOKENS_FINAL%
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

rem Front-load op if we detected one
set "TOKENS_FINAL=%TOKENS%"
if defined OPNORM set "TOKENS_FINAL=""%OPNORM%""%TOKENS%"

"%VPY%" -m %PYMOD% %TOKENS_FINAL%
exit /b %ERRORLEVEL%
