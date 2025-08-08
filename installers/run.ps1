[CmdletBinding()] param([Parameter(ValueFromRemainingArguments=$true)][string[]]$Args)
$ErrorActionPreference = "Stop"

$HERE = Split-Path -Parent $MyInvocation.MyCommand.Path
$ROOT = Split-Path -Parent $HERE
$PROJ = $null

function ToLower($s){ $s.ToLowerInvariant() }

# Parse tokens
$tokens = @()
foreach ($a in $Args) {
  $la = ToLower $a
  if ($la -in @('run','me')) { continue }
  elseif ($la -in @('argos','argos:run','run:argos')) { $PROJ='argos' }
  else { $tokens += $a }
}

# Infer project from CWD
$cwd = (Get-Location).Path
if (-not $PROJ) {
  if ($cwd -match [Regex]::Escape("\projects\argos")) { $PROJ='argos' }
}

# Require explicit project at repo root or projects/
if (-not $PROJ) {
  if ($cwd -eq $ROOT -or $cwd -eq (Join-Path $ROOT 'projects')) {
    Write-Host "Specify the project:  run argos  |  argos [args]" -ForegroundColor Yellow
    exit 2
  }
}

if ($PROJ -eq 'argos') {
  $py = (Get-Command python -ErrorAction SilentlyContinue)?.Source
  if (-not $py) { $py = (Get-Command py -ErrorAction SilentlyContinue).Source }
  & $py "$ROOT\projects\argos\bootstrap.py" --ensure --yes | Out-Null
  $vpy = & $py "$ROOT\projects\argos\bootstrap.py" --print-venv
  $env:PYTHONPYCACHEPREFIX = Join-Path $env:LOCALAPPDATA "rAIn\pycache"
  & $vpy -m panoptes.cli @tokens
  exit $LASTEXITCODE
}

Write-Error "Unknown project: $PROJ"
exit 2
