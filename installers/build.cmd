@echo off
setlocal EnableDelayedExpansion
set "HERE=%~dp0"
for %%I in ("%HERE%\..") do set "ROOT=%%~fI"

rem Accept optional leading token from subproject launchers (e.g., "argos")
if /I "%1"=="argos" shift

where py >NUL 2>&1 && (set "PY=py -3") || (set "PY=python")

%PY% "%ROOT%\projects\argos\bootstrap.py" --ensure --yes --reinstall >NUL 2>&1
for /f "usebackq delims=" %%i in (`%PY% "%ROOT%\projects\argos\bootstrap.py" --print-venv`) do set "VPY=%%i"

"%VPY%" -m pip check
set "PYTHONPYCACHEPREFIX=%LOCALAPPDATA%\rAIn\pycache"

rem Persist PATH + update current cmd session
call "%HERE%setup-path.cmd" /quiet

"%VPY%" -m panoptes.tools.build_models %*
exit /b %ERRORLEVEL%
