[CmdletBinding()] param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Args)
$ErrorActionPreference = "Stop"

# Drop optional leading token from subproject launchers (e.g., "argos")
if ($Args.Length -gt 0 -and $Args[0] -eq 'argos') { $Args = $Args[1..($Args.Length-1)] }

$HERE = Split-Path -Parent $MyInvocation.MyCommand.Path
$ROOT = Split-Path -Parent $HERE

# Prefer 'py -3' when available (then python3/python)
$pyExe = (Get-Command py -ErrorAction SilentlyContinue)?.Source
$pyArgs = @()
if ($pyExe) { $pyArgs = @('-3') }
if (-not $pyExe) { $pyExe = (Get-Command python3 -ErrorAction SilentlyContinue)?.Source }
if (-not $pyExe) { $pyExe = (Get-Command python  -ErrorAction SilentlyContinue)?.Source }
if (-not $pyExe) { Write-Error "Python 3 not found."; exit 1 }

# Ensure venv + project
& $pyExe @pyArgs "$ROOT\projects\argos\bootstrap.py" --ensure --yes --reinstall *> $null
$vpy = & $pyExe @pyArgs "$ROOT\projects\argos\bootstrap.py" --print-venv
& $vpy -m pip check

# Keep pyc out of repo
$env:PYTHONPYCACHEPREFIX = "$env:LOCALAPPDATA\rAIn\pycache"

# Put installers/ on PATH now (current session + persisted)
& (Join-Path $HERE 'setup-path.ps1') -Quiet

# Run the build inside the venv
& $vpy -m panoptes.tools.build_models @Args
exit $LASTEXITCODE
