# installers\build.ps1 — zero‑touch bootstrap + model selector (asks smoke check once)
[CmdletBinding()]
param([Parameter(ValueFromRemainingArguments = $true)][string[]]$BuildArgs)
$ErrorActionPreference = 'Stop'

# ---------- Progress-friendly environment (pin progress to one line) ----------
if (-not $env:TERM) { $env:TERM = 'xterm-256color' }
$env:PYTHONUTF8 = '1'
$env:PYTHONUNBUFFERED = '1'
$env:FORCE_COLOR = '1'
$env:PANOPTES_NESTED_PROGRESS = '1'
$env:PANOPTES_PROGRESS_ACTIVE = '0'

# Argos/Panoptes progress: one-line, overwrite in place, no final newline
$env:ARGOS_PROGRESS_STREAM = 'stdout'
$env:ARGOS_FORCE_PLAIN_PROGRESS = '1'
$env:ARGOS_PROGRESS_TAIL = 'erase'
$env:ARGOS_PROGRESS_FINAL_NEWLINE = '0'

# Keep selector responsible for fetching weights; bootstrap should not fetch them
$env:ARGOS_SKIP_WEIGHTS = '1'
Remove-Item Env:ARGOS_ASSUME_YES -ErrorAction SilentlyContinue

# Silence pip noise that would push progress lines up
$env:PIP_PROGRESS_BAR = 'off'
$env:PIP_NO_COLOR = '1'
try {
  $pipCfg = Join-Path $env:TEMP 'pip-singleline.ini'
  @"
[global]
progress-bar = off
quiet = 1
disable-pip-version-check = true
no-color = true
"@ | Set-Content -LiteralPath $pipCfg -Encoding ASCII
  $env:PIP_CONFIG_FILE = $pipCfg
}
catch { }

try { $null = $PSStyle; $PSStyle.OutputRendering = 'Ansi' } catch {}
try { [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new() } catch {}

function _Here {
  if ($PSCommandPath) { return (Split-Path -Parent $PSCommandPath) }
  if ($MyInvocation.MyCommand.Path) { return (Split-Path -Parent $MyInvocation.MyCommand.Path) }
  return (Get-Location).Path
}
$HERE = _Here
$ROOT = Split-Path -Parent $HERE

# ---------- OS flags ----------
$OsIsWindows = ($env:OS -eq 'Windows_NT')
$OsIsMac = $false
$OsIsLinux = $false
if (-not $OsIsWindows) {
  try {
    $OsIsWindows = [System.Runtime.InteropServices.RuntimeInformation]::IsOSPlatform([System.Runtime.InteropServices.OSPlatform]::Windows)
    $OsIsLinux = [System.Runtime.InteropServices.RuntimeInformation]::IsOSPlatform([System.Runtime.InteropServices.OSPlatform]::Linux)
    $OsIsMac = [System.Runtime.InteropServices.RuntimeInformation]::IsOSPlatform([System.Runtime.InteropServices.OSPlatform]::OSX)
  }
  catch {
    $OsIsLinux = ($env:OSTYPE -like '*linux*')
    $OsIsMac = ($env:OSTYPE -like '*darwin*')
  }
} # end if (-not $OsIsWindows)
# ---------- Helpers ----------
function Invoke-Quiet {
  param([string]$Exe, [string[]]$CommandArgs)
  & $Exe @CommandArgs *>&1 | Out-Null
  return $LASTEXITCODE
}
function Find-Python {
}
function Find-Python {
  $script:pyExe = $null; $script:pyArgs = @()
  $c = Get-Command py -ErrorAction SilentlyContinue
  if ($c) { $script:pyExe = $c.Source; $script:pyArgs = @('-3') }
  if (-not $script:pyExe) { $c = Get-Command python3 -ErrorAction SilentlyContinue; if ($c) { $script:pyExe = $c.Source } }
  if (-not $script:pyExe) { $c = Get-Command python  -ErrorAction SilentlyContinue; if ($c) { $script:pyExe = $c.Source } }
  return [bool]$script:pyExe
}
function Test-PythonOk {
  if (-not $script:pyExe) { return $false }
  try { $v = & $script:pyExe @script:pyArgs --version 2>&1 } catch { return $false }
  if ($v -match 'Python ([0-9]+)\.([0-9]+)\.([0-9]+)') {
    $maj = [int]$Matches[1]; $min = [int]$Matches[2]
    return ($maj -eq 3 -and $min -ge 9 -and $min -le 12)
  }
  return $false
}
function Test-IsAdmin {
  try {
    $wi = [Security.Principal.WindowsIdentity]::GetCurrent()
    $wp = New-Object Security.Principal.WindowsPrincipal($wi)
    return $wp.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
  }
  catch { return $false }
}
function Install-Python {
  Write-Host "Python 3.9 - 3.12 not found or unsupported. Attempting to install Python..."
  if ($OsIsWindows) {
    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if ($winget) {
      & winget install --id Python.Python.3.12 -e --accept-package-agreements --accept-source-agreements
      if ($LASTEXITCODE -ne 0) { & winget install --id Python.Python.3.11 -e --accept-package-agreements --accept-source-agreements }
    }
    else {
      $arch = if ([Environment]::Is64BitOperatingSystem) { 'amd64' } else { 'win32' }
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
    if (-not (Find-Python) -or -not (Test-PythonOk)) { throw "Python 3.9 - 3.12 required but not located after installation." }
    return
  }
  if ($OsIsMac) {
    $brew = Get-Command brew -ErrorAction SilentlyContinue
    if ($brew) {
      & brew install python@3.12
      if ($LASTEXITCODE -ne 0) { & brew install python@3.11 }
      if (-not (Find-Python) -or -not (Test-PythonOk)) { throw "Python 3.9 - 3.12 required but not found after Homebrew install." }
      return
    }
    else { throw "Homebrew not found. Please install Homebrew or Python 3.9 - 3.12 and re-run." }
  }
  if ($OsIsLinux) {
    if (Get-Command apt-get -ErrorAction SilentlyContinue) { & apt-get update; & apt-get install -y python3 }
    elseif (Get-Command dnf -ErrorAction SilentlyContinue) { & dnf install -y python3 }
    elseif (Get-Command yum -ErrorAction SilentlyContinue) { & yum install -y python3 }
    else { throw "No supported package manager found. Please install Python 3.9 - 3.12 and re-run." }
    if (-not (Find-Python) -or -not (Test-PythonOk)) { throw "Python 3.9 - 3.12 required but not found after package install." }
    return
  }
  throw "Unsupported OS. Please install Python 3.9 - 3.12 and re-run."
}
function Install-PythonPackage {
  param([string]$Vpy, [string]$Package, [string]$Constraint = '')
  & $Vpy -m pip show $Package *>&1 | Out-Null
  if ($LASTEXITCODE -ne 0) {
    $pkg = if ($Constraint) { "$Package$Constraint" } else { $Package }
    & $Vpy -m pip install --no-input --quiet $pkg *>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "pip failed to install $pkg" }
  }
}

# ---------- Python ensure ----------
$script:pyExe = $null
$script:pyArgs = @()
if (-not (Find-Python) -or -not (Test-PythonOk)) { Install-Python }
# ---------- Git LFS best-effort ----------
if (Get-Command git -ErrorAction SilentlyContinue) {
  & git lfs version > $null 2>&1
  if ($LASTEXITCODE -eq 0) {
    & git -C $ROOT lfs install --local > $null 2>&1
    $env:GIT_LFS_SKIP_SMUDGE = '0'
    & git -C $ROOT lfs pull > $null 2>&1
  } # end if ($LASTEXITCODE -eq 0)
} # end if (Get-Command git ...)

# ---------- Bootstrap venv ----------
$scriptPath = Join-Path $ROOT 'projects\argos\bootstrap.py'
if ((Invoke-Quiet $script:pyExe ($script:pyArgs + $scriptPath + @('--ensure', '--yes'))) -ne 0) {
  throw "bootstrap failed"
}

# Resolve the venv python and point pycache to local appdata (keeps the tree clean)
$scriptPath = Join-Path $ROOT 'projects\argos\bootstrap.py'
if ((Invoke-Quiet $script:pyExe ($script:pyArgs + $scriptPath + @('--ensure', '--yes'))) -ne 0) {
  throw "bootstrap failed"
}

# Resolve the venv python and point pycache to local appdata (keeps the tree clean)
$vpy = & $script:pyExe @script:pyArgs "$scriptPath" --print-venv
if (-not $vpy) { throw "could not resolve venv python" }
$env:PYTHONPYCACHEPREFIX = "$env:LOCALAPPDATA\rAIn\pycache"

# ---------- Core wheels that the project assumes are present ----------
# For build-time smoke check / non-live operations we prefer headless OpenCV to keep things lean.
Install-PythonPackage -Vpy $vpy -Package 'pip'
Install-PythonPackage -Vpy $vpy -Package 'setuptools'
Install-PythonPackage -Vpy $vpy -Package 'wheel'
Install-PythonPackage -Vpy $vpy -Package 'opencv-python-headless'  # satisfy optional import to prevent stderr warnings

# ---------- Launch model selector (interactive; may prompt for smoke check) ----------
& $vpy -m panoptes.model._fetch_models @BuildArgs
if ($LASTEXITCODE -ne 0) { throw "model selector failed ($LASTEXITCODE)" }
