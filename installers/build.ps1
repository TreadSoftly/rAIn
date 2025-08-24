[CmdletBinding()] param(
  [Parameter(ValueFromRemainingArguments = $true)][string[]]$BuildArgs
)
$ErrorActionPreference = "Stop"

# ---------------- Friendly environment ----------------
if (-not $env:TERM) { $env:TERM = 'xterm-256color' }
$env:PYTHONUTF8 = '1'
$env:PYTHONUNBUFFERED = '1'
$env:FORCE_COLOR = '1'
$env:PANOPTES_NESTED_PROGRESS = '1'
$env:PIP_DISABLE_PIP_VERSION_CHECK = '1'
# Prefer wheels only, never build from source (prevents toolchain headaches on clean machines)
$env:PIP_ONLY_BINARY = ':all:'
$env:PIP_NO_BUILD_ISOLATION = '1'

# Skip weight downloads during bootstrap; interactive fetcher runs later
$env:ARGOS_SKIP_WEIGHTS = '1'
# Ensure final confirm is interactive (do NOT auto-accept)
Remove-Item Env:ARGOS_ASSUME_YES -ErrorAction SilentlyContinue

try { $null = $PSStyle; $PSStyle.OutputRendering = 'Ansi' } catch {}

# Accept optional leading token from subproject launchers (e.g. "argos")
if ($BuildArgs.Length -gt 0 -and $BuildArgs[0] -eq 'argos') {
  $BuildArgs = $BuildArgs[1..($BuildArgs.Length - 1)]
}

# ---------------- Path helpers ----------------
function _Here {
  if ($PSCommandPath) { return (Split-Path -Parent $PSCommandPath) }
  if ($MyInvocation.MyCommand.Path) { return (Split-Path -Parent $MyInvocation.MyCommand.Path) }
  return (Get-Location).Path
}
$HERE = _Here
$ROOT = Split-Path -Parent $HERE

# ---------------- Progress helpers (robust import with fallbacks) ----------------
$progMod = Join-Path $HERE 'lib\progress.psm1'
if (Test-Path -LiteralPath $progMod) {
  try { Import-Module $progMod -Force -DisableNameChecking -ErrorAction Stop } catch { . $progMod }
}
if (-not (Get-Command Start-ProgressPhase -ErrorAction SilentlyContinue)) {
  function Start-ProgressPhase { param([string]$Activity, [int]$Total = 100) }
  function Set-Progress { param([int]$Done, [string]$Status = "") }
  function Step-Progress { param([int]$Delta = 1, [string]$Status = "") }
  function Complete-Progress { }
  function Invoke-Step { param([string]$Name, [scriptblock]$Body, [int]$Weight = 1) & $Body }
}

if (Test-Path (Join-Path $HERE 'build-hooks.ps1')) { . (Join-Path $HERE 'build-hooks.ps1') }

# ---------------- System checks ----------------
function Test-IsAdmin {
  try {
    $wi = [Security.Principal.WindowsIdentity]::GetCurrent()
    $wp = New-Object Security.Principal.WindowsPrincipal($wi)
    return $wp.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
  }
  catch { return $false }
}

function Test-VcRuntimePresent {
  if ($env:OS -ne 'Windows_NT') { return $true }
  try {
    $sys = Join-Path $env:WINDIR 'System32'
    $has1 = Test-Path -LiteralPath (Join-Path $sys 'vcruntime140.dll')
    $has2 = Test-Path -LiteralPath (Join-Path $sys 'vcruntime140_1.dll')
    $has3 = Test-Path -LiteralPath (Join-Path $sys 'msvcp140.dll')
    return ($has1 -and $has2 -and $has3)
  }
  catch { return $false }
}

function Install-VcRedistIfMissing {
  if ($env:OS -ne 'Windows_NT') { return }
  if (Test-VcRuntimePresent) { return }

  Write-Host "Installing Microsoft Visual C++ Redistributable (x64)..." -ForegroundColor Yellow
  $tmp = Join-Path $env:TEMP 'vc_redist.x64.exe'
  try {
    Invoke-WebRequest -UseBasicParsing -Uri 'https://aka.ms/vs/17/release/vc_redist.x64.exe' -OutFile $tmp
    $args = '/quiet', '/norestart'
    $p = Start-Process -FilePath $tmp -ArgumentList $args -PassThru -Wait
    if ($p.ExitCode -ne 0) { throw "vc_redist failed with code $($p.ExitCode)" }
  }
  catch {
    throw "Could not install the VC++ Redistributable automatically. Please run this script as an Administrator or install the 'Microsoft Visual C++ 2015–2022 Redistributable (x64)' manually, then try again. (Error: $($_.Exception.Message))"
  }
  finally {
    Remove-Item -LiteralPath $tmp -ErrorAction SilentlyContinue | Out-Null
  }
}

# Auto-elevate ONLY if we need VC++ and we're not admin yet
if ($env:OS -eq 'Windows_NT' -and -not (Test-VcRuntimePresent) -and -not (Test-IsAdmin)) {
  if (-not $env:RAIN_ELEVATED) {
    Write-Host "Elevation needed to install VC++ runtime. Requesting Administrator privileges..." -ForegroundColor Yellow
    $quotedScript = '"' + $PSCommandPath + '"'
    $argList = @("-ExecutionPolicy", "Bypass", "-NoProfile", "-File", $quotedScript)
    if ($BuildArgs -and $BuildArgs.Count -gt 0) { $argList += $BuildArgs }
    $env:RAIN_ELEVATED = "1"
    Start-Process -FilePath "powershell.exe" -Verb RunAs -ArgumentList $argList | Out-Null
    Exit
  }
}

# ---------------- Python resolution ----------------
$script:pyExe = $null
$script:pyArgs = @()

function Find-Python {
  $script:pyExe = $null; $script:pyArgs = @()
  $c = Get-Command py -ErrorAction SilentlyContinue
  if ($c) {
    # Try to force 64-bit interpreter via py launcher
    $script:pyExe = $c.Source
    $script:pyArgs = @('-3-64')
  }
  if (-not $script:pyExe) {
    $c = Get-Command python3 -ErrorAction SilentlyContinue
    if ($c) { $script:pyExe = $c.Source }
  }
  if (-not $script:pyExe) {
    $c = Get-Command python -ErrorAction SilentlyContinue
    if ($c) { $script:pyExe = $c.Source }
  }
  return [bool]$script:pyExe
}

function Test-PythonOk {
  if (-not $script:pyExe) { return $false }
  try {
    $v = & $script:pyExe @script:pyArgs --version 2>&1
  }
  catch { return $false }
  if ($v -match 'Python ([0-9]+)\.([0-9]+)\.([0-9]+)') {
    $maj = [int]$Matches[1]; $min = [int]$Matches[2]
    return ($maj -eq 3 -and $min -ge 9 -and $min -le 12)
  }
  return $false
}

function Get-PythonArch {
  if (-not $script:pyExe) { return "unknown" }
  try {
    $arch = & $script:pyExe @script:pyArgs -c "import platform; print(platform.architecture()[0])"
    return $arch.Trim()
  }
  catch { return "unknown" }
}

function Install-Python {
  Write-Host "Installing Python 3.12 (64-bit)..." -ForegroundColor Yellow
  if ($env:OS -eq 'Windows_NT') {
    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if ($winget) {
      & winget install --id Python.Python.3.12 -e --accept-package-agreements --accept-source-agreements
      if ($LASTEXITCODE -ne 0) { & winget install --id Python.Python.3.11 -e --accept-package-agreements --accept-source-agreements }
    }
    else {
      $arch = 'amd64'
      $ver = '3.12.10'
      $url = "https://www.python.org/ftp/python/$ver/python-$ver-$arch.exe"
      $tmp = Join-Path $env:TEMP "python-$ver-$arch.exe"
      Invoke-WebRequest -Uri $url -OutFile $tmp -UseBasicParsing
      $pyInstallArgs = "/quiet InstallAllUsers=0 PrependPath=1 Include_test=0"
      if (Test-IsAdmin) { $pyInstallArgs = "/quiet InstallAllUsers=1 PrependPath=1 Include_test=0" }
      $p = Start-Process -FilePath $tmp -ArgumentList $pyInstallArgs -Wait -PassThru
      if ($p.ExitCode -ne 0) { throw "Python installer exited with code $($p.ExitCode)." }
    }
    Start-Sleep -Seconds 2
    if (-not (Find-Python) -or -not (Test-PythonOk)) {
      throw "Python 3.9–3.12 required but not located after installation."
    }
    return
  }

  if ($env:OS -eq 'MacOS') {
    $brew = Get-Command brew -ErrorAction SilentlyContinue
    if ($brew) {
      & brew install python@3.12
      if ($LASTEXITCODE -ne 0) { & brew install python@3.11 }
      if (-not (Find-Python) -or -not (Test-PythonOk)) { throw "Python 3.9–3.12 not found after Homebrew install." }
      return
    }
    else { throw "Homebrew not found. Install Homebrew or Python 3.9–3.12 and re-run." }
  }

  if ($env:OS -ne 'Windows_NT' -and $(Get-Command uname -ErrorAction SilentlyContinue)) {
    if (Get-Command apt-get -ErrorAction SilentlyContinue) { & apt-get update; & apt-get install -y python3 }
    elseif (Get-Command dnf -ErrorAction SilentlyContinue) { & dnf install -y python3 }
    elseif (Get-Command yum -ErrorAction SilentlyContinue) { & yum install -y python3 }
    else { throw "No supported package manager found. Please install Python 3.9–3.12 and re-run." }
    if (-not (Find-Python) -or -not (Test-PythonOk)) { throw "Python 3.9–3.12 not found after package install." }
    return
  }

  throw "Unsupported OS. Please install Python 3.9–3.12 and re-run."
}

# ---------------- Main bootstrap ----------------
if (-not (Find-Python) -or -not (Test-PythonOk)) {
  Install-Python
}

# Enforce 64-bit Python for ONNX/DLL compatibility
$pyArch = Get-PythonArch
if ($pyArch -ne '64bit') {
  Write-Host "Detected $pyArch Python. Installing/using 64‑bit Python for ONNX compatibility..." -ForegroundColor Yellow
  Install-Python
  if (-not (Find-Python)) { throw "Unable to resolve Python after installation." }
}

# Ensure VC++ runtime only if missing (elevation has already been handled above if needed)
if ($env:OS -eq 'Windows_NT') { Install-VcRedistIfMissing }

# --- Bootstrap the virtual environment ---
$scriptPath = Join-Path $ROOT 'projects\argos\bootstrap.py'
& $script:pyExe @script:pyArgs $scriptPath --ensure --yes
if ($LASTEXITCODE -ne 0) { throw "bootstrap failed ($LASTEXITCODE)" }

# Resolve venv Python
$vpy = & $script:pyExe @script:pyArgs "$scriptPath" --print-venv
if (-not $vpy) { throw "could not resolve venv python" }

# Speed up Python's bytecode cache for clean machines
$env:PYTHONPYCACHEPREFIX = "$env:LOCALAPPDATA\rAIn\pycache"

# --- Launch the interactive model selector / fetcher ---
& $vpy -m panoptes.model._fetch_models @BuildArgs
if ($LASTEXITCODE -ne 0) { throw "model selector failed ($LASTEXITCODE)" }
