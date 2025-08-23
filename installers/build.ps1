# rAIn - Windows bootstrap (no file logging; PS 5.1+ compatible)
[CmdletBinding()] param(
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$BuildArgs
)
$ErrorActionPreference = 'Stop'

# --- Progress-friendly environment ---
if (-not $env:TERM) { $env:TERM = 'xterm-256color' }
$env:PYTHONUTF8 = '1'
$env:PYTHONUNBUFFERED = '1'
$env:FORCE_COLOR = '1'
$env:PANOPTES_NESTED_PROGRESS = '1'
$env:PIP_DISABLE_PIP_VERSION_CHECK = '1'

# Skip weight downloads during non-interactive bootstrap (interactive builder runs later)
$env:ARGOS_SKIP_WEIGHTS = '1'
# Ensure final confirm is interactive (do NOT auto-accept)
Remove-Item Env:ARGOS_ASSUME_YES -ErrorAction SilentlyContinue

# Prefer ANSI when available (safe no-op on PS 5.1)
try { $null = $PSStyle; $PSStyle.OutputRendering = 'Ansi' } catch {}

# Accept optional leading token from subproject launchers (e.g., "argos")
if ($BuildArgs.Length -gt 0 -and $BuildArgs[0] -eq 'argos') {
  if ($BuildArgs.Length -gt 1) {
    $BuildArgs = $BuildArgs[1..($BuildArgs.Length - 1)]
  }
  else {
    $BuildArgs = @()
  }
}

# Resolve script location robustly
function Resolve-Here {
  if ($PSCommandPath) { return (Split-Path -Parent $PSCommandPath) }
  if ($MyInvocation.MyCommand.Path) { return (Split-Path -Parent $MyInvocation.MyCommand.Path) }
  return (Get-Location).Path
}
$HERE = Resolve-Here
$ROOT = Split-Path -Parent $HERE

# Optional progress helpers (never fail build if missing)
$progMod = Join-Path $HERE 'lib\progress.psm1'
if (Test-Path -LiteralPath $progMod) {
  try { Import-Module $progMod -Force -DisableNameChecking -ErrorAction Stop }
  catch { . $progMod }
}
if (-not (Get-Command Start-ProgressPhase -ErrorAction SilentlyContinue)) {
  function Start-ProgressPhase { param([string]$Activity, [int]$Total = 100) }
  function Set-Progress { param([int]$Done, [string]$Status = "") }
  function Step-Progress { param([int]$Delta = 1, [string]$Status = "") }
  function Complete-Progress { }
  function Invoke-Step { param([string]$Name, [scriptblock]$Body, [int]$Weight = 1) & $Body }
}

# Optional local hooks
if (Test-Path (Join-Path $HERE 'build-hooks.ps1')) { . (Join-Path $HERE 'build-hooks.ps1') }

# ----- Python detection and optional auto-install -----
function Get-PythonCandidate {
  $c = Get-Command py -ErrorAction SilentlyContinue
  if ($c) { return @{ Exe = $c.Source; Args = @('-3') } }
  $c = Get-Command python3 -ErrorAction SilentlyContinue
  if ($c) { return @{ Exe = $c.Source; Args = @() } }
  $c = Get-Command python -ErrorAction SilentlyContinue
  if ($c) { return @{ Exe = $c.Source; Args = @() } }
  return $null
}

function Install-PythonIfMissing {
  $candidate = Get-PythonCandidate
  if ($candidate) { return $candidate }

  Write-Host 'Python 3 was not found. Attempting automatic installation...'
  $installed = $false

  # 1) winget
  $wingetCmd = Get-Command winget -ErrorAction SilentlyContinue
  if ($wingetCmd -and -not $installed) {
    try {
      Write-Host 'Installing Python via winget (silent).'
      & $wingetCmd.Source install --id Python.Python.3.12 --source winget --accept-source-agreements --accept-package-agreements --silent
      if ($LASTEXITCODE -eq 0) { $installed = $true }
    }
    catch { }
  }

  # 2) Chocolatey
  $chocoCmd = Get-Command choco -ErrorAction SilentlyContinue
  if ($chocoCmd -and -not $installed) {
    try {
      Write-Host 'Installing Python via Chocolatey (silent).'
      & $chocoCmd install python -y --no-progress
      if ($LASTEXITCODE -eq 0) { $installed = $true }
    }
    catch { }
  }

  # 3) Official installer download (silent)
  if (-not $installed) {
    try {
      $arch = if ([Environment]::Is64BitOperatingSystem) { 'amd64' } else { 'win32' }
      $url = if ($arch -eq 'amd64') {
        'https://www.python.org/ftp/python/3.12.4/python-3.12.4-amd64.exe'
      }
      else {
        'https://www.python.org/ftp/python/3.12.4/python-3.12.4.exe'
      }
      $temp = Join-Path $env:TEMP 'python-installer.exe'
      Write-Host ('Downloading Python installer from: {0}' -f $url)
      try {
        Invoke-WebRequest -Uri $url -OutFile $temp -UseBasicParsing
      }
      catch {
        Start-BitsTransfer -Source $url -Destination $temp
      }
      Write-Host 'Running Python installer (quiet).'
      Start-Process -FilePath $temp -ArgumentList '/quiet', 'InstallAllUsers=1', 'PrependPath=1', 'Include_test=0', 'SimpleInstall=1' -Wait -PassThru | Out-Null
      Remove-Item -LiteralPath $temp -Force -ErrorAction SilentlyContinue
      $installed = $true
    }
    catch {
      Write-Host 'Automatic Python installation failed.'
    }
  }

  if ($installed) {
    # Refresh PATH in this session
    $env:PATH = [Environment]::GetEnvironmentVariable('PATH', 'Machine') + ';' + [Environment]::GetEnvironmentVariable('PATH', 'User')
    Start-Sleep -Seconds 2
    $candidate = Get-PythonCandidate
    if ($candidate) { return $candidate }
  }

  throw 'Python 3 not found and automatic installation was unsuccessful.'
}

# Acquire Python
$py = Get-PythonCandidate
if (-not $py) { $py = Install-PythonIfMissing }
$pyExe = $py['Exe']
$pyArgs = $py['Args']

# ----- Build phase: ONLY non-interactive steps here -----
$env:PANOPTES_PROGRESS_ACTIVE = '1'
Start-ProgressPhase 'Build ARGOS' -Total 3
try {
  Invoke-Step -Name 'Bootstrap venv' -Weight 1 -Body {
    Write-Host 'Bootstrapping virtual environment (quiet; no log files)...'
    # Suppress both stdout/stderr to avoid NativeCommandError on harmless warnings.
    & $pyExe @pyArgs "$ROOT\projects\argos\bootstrap.py" --ensure --yes --reinstall 1>$null 2>$null
    $ec = $LASTEXITCODE
    if ($ec -ne 0) {
      Write-Host ''
      Write-Host '==== BOOTSTRAP FAILED - re-running with visible output (no files written) ===='
      $prevEAP = $ErrorActionPreference
      $ErrorActionPreference = 'Continue'
      # Merge stderr into stdout and print to console only
      & $pyExe @pyArgs "$ROOT\projects\argos\bootstrap.py" --ensure --yes --reinstall 2>&1 | Out-Host
      $ErrorActionPreference = $prevEAP
      throw ('bootstrap failed ({0})' -f $ec)
    }
  }

  # Resolve venv Python for subsequent steps (suppress stderr so PS doesn't raise NativeCommandError)
  $vpy = & $pyExe @pyArgs "$ROOT\projects\argos\bootstrap.py" --print-venv 2>$null
  $vpy = $vpy.Trim()

  Invoke-Step -Name 'Sanity check' -Weight 1 -Body {
    & $vpy -m pip check 2>$null
    if ($LASTEXITCODE -ne 0) {
      Write-Host "WARNING: 'pip check' reported issues; continuing."
    }
  }

  Invoke-Step -Name 'Setup PATH' -Weight 1 -Body {
    & (Join-Path $HERE 'setup-path.ps1') -Quiet
  }
}
finally {
  Complete-Progress
  $env:PANOPTES_PROGRESS_ACTIVE = '0'
}

# ----- Interactive model builder runs AFTER progress is cleared -----
# IMPORTANT: Do NOT auto-feed "1" unless the user explicitly opts in.
$auto = ($env:ARGOS_AUTOBUILD -eq '1')

# Avoid terminating on harmless stderr during the interactive phase
$prevEAP = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
try {
  if ($auto) {
    '1' | & $vpy -m panoptes.tools.build_models @BuildArgs
  }
  else {
    & $vpy -m panoptes.tools.build_models @BuildArgs
  }
}
finally {
  $ErrorActionPreference = $prevEAP
}

if ($LASTEXITCODE -ne 0) { throw ('build_models failed ({0})' -f $LASTEXITCODE) }
return
