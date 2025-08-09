[CmdletBinding()] param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Args)
$ErrorActionPreference = "Stop"

$HERE = Split-Path -Parent $MyInvocation.MyCommand.Path
$ROOT = Split-Path -Parent $HERE

$py = (Get-Command python3 -ErrorAction SilentlyContinue)?.Source
if (-not $py) { $py = (Get-Command python -ErrorAction SilentlyContinue)?.Source }
& $py "$ROOT\projects\argos\bootstrap.py" --ensure --yes *> $null
$vpy = & $py "$ROOT\projects\argos\bootstrap.py" --print-venv
$env:PYTHONPYCACHEPREFIX = "$env:LOCALAPPDATA\rAIn\pycache"
& $vpy -m panoptes.tools.build_models @Args
exit $LASTEXITCODE
