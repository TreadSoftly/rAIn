@echo off
setlocal enabledelayedexpansion

set "HERE=%~dp0"
for %%I in ("%HERE%..") do set "ROOT=%%~fI"

rem --- Ensure Git LFS ---
where git >nul 2>&1
if not errorlevel 1 (
  git lfs version >nul 2>&1
  if not errorlevel 1 (
    git -C "%ROOT%" lfs install --local >nul 2>&1
    set GIT_LFS_SKIP_SMUDGE=0
    git -C "%ROOT%" lfs pull >nul 2>&1
  ) else (
    echo ⚠️  Git LFS not installed. Large files (models) may be missing.
    echo     Install Git LFS and re-run this command.
  )
)

rem --- Find Python ---
for %%P in (python3.exe python.exe) do (
  where %%P >nul 2>&1 && set "PY=%%P" && goto :gotpy
)
echo Python 3 not found.
exit /b 1

:gotpy
"%PY%" "%ROOT%\projects\argos\bootstrap.py" --ensure --yes >nul 2>&1
for /f "usebackq delims=" %%V in (`"%PY%" "%ROOT%\projects\argos\bootstrap.py" --print-venv`) do set "VPY=%%V"
set "PYTHONPYCACHEPREFIX=%LOCALAPPDATA%\rAIn\pycache"
"%VPY%" -m panoptes.cli %*
exit /b %ERRORLEVEL%
