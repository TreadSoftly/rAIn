[CmdletBinding()] param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Args)
$ErrorActionPreference = "Stop"

$HERE = Split-Path -Parent $MyInvocation.MyCommand.Path
$ROOT = Split-Path -Parent $HERE

<<<<<<< HEAD
# Parse tokens
$proj   = $null
=======
function Ensure-GitLFS {
  if (Get-Command git -ErrorAction SilentlyContinue) {
    & git lfs version *> $null
    if ($LASTEXITCODE -eq 0) {
      & git -C $ROOT lfs install --local *> $null
      $env:GIT_LFS_SKIP_SMUDGE = '0'
      & git -C $ROOT lfs pull *> $null
    }
    else {
      Write-Host "⚠️  Git LFS not installed. Large files (models) may be missing."
      Write-Host "   Install Git LFS and re-run this command."
    }
  }
}

$proj = $null
>>>>>>> c55090c (Argos: Interactive model setup, build command, and model updates)
$tokens = @()
$sawBuild = $false
foreach ($a in $Args) {
  $la = $a.ToLowerInvariant()
  if ($la -in @('run', 'me')) { continue }
  elseif ($la -in @('build', 'package', 'pack')) { $sawBuild = $true }
  elseif ($la -in @('argos', 'argos:run', 'run:argos', 'argos:build', 'build:argos')) { $proj = 'argos' }
  else { $tokens += $a }
}

$cwd = (Get-Location).Path
if (-not $proj) {
  if ($cwd -like "*\projects\argos*") { $proj = 'argos' }
}

if (-not $proj) {
  if ($cwd -eq $ROOT -or $cwd -eq (Join-Path $ROOT 'projects')) {
    Write-Host "Specify the project:  run argos  |  argos [args]" -ForegroundColor Yellow
    exit 2
  }
}

<<<<<<< HEAD
=======
Ensure-GitLFS

# Build hand-off
if ($sawBuild) {
  & (Join-Path $ROOT 'installers\build.ps1') $proj @tokens
  exit $LASTEXITCODE
}

>>>>>>> c55090c (Argos: Interactive model setup, build command, and model updates)
if ($proj -eq 'argos') {
  $py = (Get-Command python3 -ErrorAction SilentlyContinue)?.Source
  if (-not $py) { $py = (Get-Command python -ErrorAction SilentlyContinue)?.Source }
  if (-not $py) { Write-Error "Python 3 not found."; exit 1 }

  & $py "$ROOT\projects\argos\bootstrap.py" --ensure --yes *> $null
  $vpy = & $py "$ROOT\projects\argos\bootstrap.py" --print-venv
  $env:PYTHONPYCACHEPREFIX = "$env:LOCALAPPDATA\rAIn\pycache"
  & $vpy -m panoptes.cli @tokens
  exit $LASTEXITCODE
}

Write-Error "Unknown project: $proj"
exit 2
