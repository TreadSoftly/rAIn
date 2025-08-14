[CmdletBinding()] param([Parameter(ValueFromRemainingArguments = $true)][string[]]$BuildArgs)
$ErrorActionPreference = "Stop"

# --- Progress-friendly environment ---
if (-not $env:TERM) { $env:TERM = 'xterm-256color' }
$env:PYTHONUTF8 = '1'
$env:PYTHONUNBUFFERED = '1'
$env:FORCE_COLOR = '1'
$env:PANOPTES_NESTED_PROGRESS = '1'
try { $null = $PSStyle; $PSStyle.OutputRendering = 'Ansi' } catch {}

# Accept optional leading token from subproject launchers (e.g., "argos")
if ($BuildArgs.Length -gt 0 -and $BuildArgs[0] -eq 'argos') { $BuildArgs = $BuildArgs[1..($BuildArgs.Length - 1)] }

# Robust script location (works even if pasted into a console)
function _Here {
  if ($PSCommandPath) { return (Split-Path -Parent $PSCommandPath) }
  if ($MyInvocation.MyCommand.Path) { return (Split-Path -Parent $MyInvocation.MyCommand.Path) }
  return (Get-Location).Path
}
$HERE = _Here
$ROOT = Split-Path -Parent $HERE

# PS 5.1–safe Python detection
$pyExe = $null
$pyArgs = @()
$cmd = Get-Command py -ErrorAction SilentlyContinue
if ($cmd) { $pyExe = $cmd.Source; $pyArgs = @('-3') }
if (-not $pyExe) { $cmd = Get-Command python3 -ErrorAction SilentlyContinue; if ($cmd) { $pyExe = $cmd.Source } }
if (-not $pyExe) { $cmd = Get-Command python  -ErrorAction SilentlyContinue; if ($cmd) { $pyExe = $cmd.Source } }
if (-not $pyExe) { throw "Python 3 not found." }

# Progress helpers + optional hooks  (robust import with fallbacks)
$progMod = Join-Path $HERE 'lib\progress.psm1'
if (Test-Path -LiteralPath $progMod) {
  try { Import-Module $progMod -Force -DisableNameChecking -ErrorAction Stop }
  catch { . $progMod }
}
# Last-resort: define no-op shims so the build never fails due to progress
if (-not (Get-Command Start-ProgressPhase -ErrorAction SilentlyContinue)) {
  function Start-ProgressPhase { param([string]$Activity, [int]$Total = 100) }
  function Set-Progress { param([int]$Done, [string]$Status = "") }
  function Step-Progress { param([int]$Delta = 1, [string]$Status = "") }
  function Complete-Progress { }
  function Invoke-Step { param([string]$Name, [scriptblock]$Body, [int]$Weight = 1) & $Body }
}

if (Test-Path (Join-Path $HERE 'build-hooks.ps1')) { . (Join-Path $HERE 'build-hooks.ps1') }

# ── Build phase: ONLY non-interactive steps here ─────────────────────────────
# Mark PS-driven progress as active so Python progress stays quiet during these tiny phases.
$env:PANOPTES_PROGRESS_ACTIVE = '1'
Start-ProgressPhase "Build ARGOS" -Total 3

try {
  Invoke-Step -Name "Bootstrap venv" -Weight 1 -Body {
    & $pyExe @pyArgs "$ROOT\projects\argos\bootstrap.py" --ensure --yes --reinstall > $null 2>&1
    if ($LASTEXITCODE -ne 0) { throw "bootstrap failed ($LASTEXITCODE)" }
  }

  # Resolve venv Python for subsequent steps
  $vpy = & $pyExe @pyArgs "$ROOT\projects\argos\bootstrap.py" --print-venv

  Invoke-Step -Name "Sanity check" -Weight 1 -Body {
    & $vpy -m pip check
    if ($LASTEXITCODE -ne 0) { throw "pip check failed ($LASTEXITCODE)" }
  }

  Invoke-Step -Name "Setup PATH" -Weight 1 -Body {
    & (Join-Path $HERE 'setup-path.ps1') -Quiet
  }
}
finally {
  # Always clear the progress record, even on error, to avoid "ghost" lines
  Complete-Progress
  # PS progress is done; allow Python progress UI to run.
  $env:PANOPTES_PROGRESS_ACTIVE = '0'
}

# ── Interactive model builder runs AFTER progress is cleared ────────────────
& $vpy -m panoptes.tools.build_models @BuildArgs
if ($LASTEXITCODE -ne 0) { throw "build_models failed ($LASTEXITCODE)" }

# Propagate status without exiting the host
return
