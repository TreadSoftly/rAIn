[CmdletBinding()] param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Args)
$ErrorActionPreference = "Stop"

# Progress-friendly environment
if (-not $env:TERM) { $env:TERM = 'xterm-256color' }
$env:PYTHONUTF8 = '1'
$env:PYTHONUNBUFFERED = '1'
$env:FORCE_COLOR = '1'
$env:PANOPTES_NESTED_PROGRESS = '1'
$env:PANOPTES_PROGRESS_ACTIVE = '0'
try { $null = $PSStyle; $PSStyle.OutputRendering = 'Ansi' } catch {}

function _Here {
  if ($PSCommandPath) { return (Split-Path -Parent $PSCommandPath) }
  if ($MyInvocation.MyCommand.Path) { return (Split-Path -Parent $MyInvocation.MyCommand.Path) }
  return (Get-Location).Path
}
$HERE = _Here
$ROOT = Split-Path -Parent $HERE

function Enable-GitLFS {
  if (Get-Command git -ErrorAction SilentlyContinue) {
    & git lfs version *> $null
    if ($LASTEXITCODE -eq 0) {
      & git -C $ROOT lfs install --local *> $null
      $env:GIT_LFS_SKIP_SMUDGE = '0'
      & git -C $ROOT lfs pull *> $null
    } else {
      Write-Host "⚠️  Git LFS not installed. Large files (models) may be missing."
      Write-Host "   Install Git LFS and re-run this command."
    }
  }
}

$proj = $null
$runTests = $true
$wantSdist = $true
$wantWheel = $true
$tokens = @()

foreach ($a in $Args) {
  $la = $a.ToLowerInvariant()
  switch ($la) {
    'pack' { continue }
    'package' { continue }
    'build' { continue }
    'me' { continue }
    '--no-tests' { $runTests = $false; continue }
    '--wheel-only' { $wantSdist = $false; $wantWheel = $true; continue }
    '--sdist-only' { $wantSdist = $true; $wantWheel = $false; continue }
    'argos' { $proj = 'argos'; continue }
    'argos:pack' { $proj = 'argos'; continue }
    'pack:argos' { $proj = 'argos'; continue }
    'argos:build' { $proj = 'argos'; continue }
    'build:argos' { $proj = 'argos'; continue }
    default { $tokens += $a }
  }
}

$cwd = (Get-Location).Path
if (-not $proj) {
  if ($cwd -like "*\projects\argos*") { $proj = 'argos' }
}

if (-not $proj) {
  if ($cwd -eq $ROOT -or $cwd -eq (Join-Path $ROOT 'projects')) {
    Write-Host "Specify the project:  pack argos  |  argos pack  |  pack --no-tests" -ForegroundColor Yellow
    return
  }
}

Enable-GitLFS

if ($proj -eq 'argos') {
  # PS 5.1–safe Python detection
  $pyExe  = $null
  $pyArgs = @()
  $cmd = Get-Command py -ErrorAction SilentlyContinue
  if ($cmd) { $pyExe = $cmd.Source; $pyArgs = @('-3') }
  if (-not $pyExe) { $cmd = Get-Command python3 -ErrorAction SilentlyContinue; if ($cmd) { $pyExe = $cmd.Source } }
  if (-not $pyExe) { $cmd = Get-Command python  -ErrorAction SilentlyContinue; if ($cmd) { $pyExe = $cmd.Source } }
  if (-not $pyExe) { throw "Python 3 not found." }

  & $pyExe @pyArgs "$ROOT\projects\argos\bootstrap.py" --ensure --yes *> $null
  $vpy = & $pyExe @pyArgs "$ROOT\projects\argos\bootstrap.py" --print-venv
  $env:PYTHONPYCACHEPREFIX = "$env:LOCALAPPDATA\rAIn\pycache"

  & $vpy -m pip install --upgrade pip setuptools wheel build *> $null
  Push-Location "$ROOT\projects\argos"
  & $vpy -m pip install -e .[dev] *> $null

  if ($runTests) {
    & $vpy -m pytest -q
  }

  $buildArgs = @()
  if ($wantSdist) { $buildArgs += '--sdist' }
  if ($wantWheel) { $buildArgs += '--wheel' }
  & $vpy -m build @buildArgs

  Pop-Location

  $outDir = Join-Path $ROOT 'dist\argos'
  if (-not (Test-Path -LiteralPath $outDir)) { New-Item -ItemType Directory -Path $outDir | Out-Null }
  Get-ChildItem "$ROOT\projects\argos\dist\*" -File -ErrorAction SilentlyContinue | ForEach-Object {
    Copy-Item $_.FullName $outDir -Force
  }

  Write-Host "✅ Build complete."
  Write-Host "   • artifacts → projects\argos\dist\"
  Write-Host "   • mirror    → dist\argos\"
  return
}

throw "Unknown project: $proj"
