[CmdletBinding()] param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Args)
$ErrorActionPreference = "Stop"

# Drop optional leading token from subproject launchers (e.g., "argos")
if ($Args.Length -gt 0 -and $Args[0] -eq 'argos') { $Args = $Args[1..($Args.Length-1)] }

$HERE = Split-Path -Parent $MyInvocation.MyCommand.Path
$ROOT = Split-Path -Parent $HERE

$py = (Get-Command python3 -ErrorAction SilentlyContinue)?.Source
if (-not $py) { $py = (Get-Command python -ErrorAction SilentlyContinue)?.Source }

& $py "$ROOT\projects\argos\bootstrap.py" --ensure --yes --reinstall *> $null
$vpy = & $py "$ROOT\projects\argos\bootstrap.py" --print-venv

& $vpy -m pip check
$env:PYTHONPYCACHEPREFIX = "$env:LOCALAPPDATA\rAIn\pycache"

& $vpy -m panoptes.tools.build_models @Args
exit $LASTEXITCODE
