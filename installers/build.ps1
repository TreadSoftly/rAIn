[CmdletBinding()]
param([Parameter(ValueFromRemainingArguments = $true)][string[]]$BuildArgs)
$ErrorActionPreference = 'Stop'

# --- Progress-friendly environment ---
if (-not $env:TERM) { $env:TERM = 'xterm-256color' }
$env:PYTHONUTF8 = '1'
$env:PYTHONUNBUFFERED = '1'
$env:FORCE_COLOR = '1'
$env:PANOPTES_NESTED_PROGRESS = '1'
$env:PIP_DISABLE_PIP_VERSION_CHECK = '1'

# Force the CLI to use a single-line progress on a proper TTY stream
$env:ARGOS_PROGRESS_STREAM = 'stdout'
$env:ARGOS_FORCE_PLAIN_PROGRESS = '1'

# During bootstrap, skip weights (the selector will handle them)
$env:ARGOS_SKIP_WEIGHTS = '1'
Remove-Item Env:ARGOS_ASSUME_YES -ErrorAction SilentlyContinue

# Prefer CPU-only wheels for PyTorch in the exporter sandbox
$env:ARGOS_TORCH_VERSION = '2.4.1'
$env:ARGOS_TORCH_INDEX_URL = 'https://download.pytorch.org/whl/cpu'
# Modern opset cascade for YOLOv8/11/12 exports
$env:ARGOS_ONNX_OPSETS = '21 20 19 18 17'
# Default export image size (can be overridden by user)
if (-not $env:ARGOS_ONNX_IMGSZ) { $env:ARGOS_ONNX_IMGSZ = '640' }

# ---- Keep progress to ONE line: silence pip's noisy progress/log output ----
$env:PIP_PROGRESS_BAR = 'off'
$env:PIP_NO_COLOR = '1'

# Create a minimal, ephemeral pip config that enforces single-line friendliness
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
catch {
  Write-Host "Warning: could not create pip single-line config. Continuing with env-based settings."
}

try { $null = $PSStyle; $PSStyle.OutputRendering = 'Ansi' } catch {}
try { [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new() } catch {}

function Install-VcRedistIfMissing {
  if ($env:OS -ne 'Windows_NT') { return }
  try {
    $sys = Join-Path $env:WINDIR 'System32'
    $has1 = Test-Path -LiteralPath (Join-Path $sys 'vcruntime140.dll')
    $has2 = Test-Path -LiteralPath (Join-Path $sys 'vcruntime140_1.dll')
    $has3 = Test-Path -LiteralPath (Join-Path $sys 'msvcp140.dll')
    if ($has1 -and $has2 -and $has3) { return }
  }
  catch {}
  Write-Host "Installing Microsoft Visual C++ Redistributable (x64)..." -ForegroundColor Yellow
  $tmp = Join-Path $env:TEMP 'vc_redist.x64.exe'
  try {
    Invoke-WebRequest -UseBasicParsing -Uri 'https://aka.ms/vs/17/release/vc_redist.x64.exe' -OutFile $tmp
    $vcRedistArgs = '/quiet', '/norestart'
    $p = Start-Process -FilePath $tmp -ArgumentList $vcRedistArgs -PassThru -Wait
    if ($p.ExitCode -ne 0) { throw "vc_redist failed with code $($p.ExitCode)" }
  }
  catch {
    throw "Could not install VC++ Redistributable automatically: $($_.Exception.Message)"
  }
  finally {
    Remove-Item -LiteralPath $tmp -ErrorAction SilentlyContinue | Out-Null
  }
}

function _Here {
  if ($PSCommandPath) { return (Split-Path -Parent $PSCommandPath) }
  if ($MyInvocation.MyCommand.Path) { return (Split-Path -Parent $MyInvocation.MyCommand.Path) }
  return (Get-Location).Path
}
$HERE = _Here
$ROOT = Split-Path -Parent $HERE

# --- OS flags ---
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
}

# --- Python ensure (auto-install if needed) ---
$script:pyExe = $null
$script:pyArgs = @()

function Find-Python {
  $script:pyExe = $null; $script:pyArgs = @()
  $c = Get-Command py -ErrorAction SilentlyContinue
  if ($c) { $script:pyExe = $c.Source; $script:pyArgs = @('-3') }
  if (-not $script:pyExe) { $c = Get-Command python3 -ErrorAction SilentlyContinue; if ($c) { $script:pyExe = $c.Source; $script:pyArgs = @() } }
  if (-not $script:pyExe) { $c = Get-Command python  -ErrorAction SilentlyContinue; if ($c) { $script:pyExe = $c.Source; $script:pyArgs = @() } }
  if ($script:pyExe) { return $true }

  if ($OsIsWindows) {
    $launcherCandidates = @()
    if ($env:LOCALAPPDATA) { $launcherCandidates += Join-Path $env:LOCALAPPDATA 'Programs\Python\Launcher\py.exe' }
    if ($env:ProgramFiles) { $launcherCandidates += Join-Path $env:ProgramFiles 'Python\Launcher\py.exe' }
    if ($env:ProgramW6432) { $launcherCandidates += Join-Path $env:ProgramW6432 'Python\Launcher\py.exe' }
    if (${env:ProgramFiles(x86)}) { $launcherCandidates += Join-Path ${env:ProgramFiles(x86)} 'Python\Launcher\py.exe' }
    if ($env:SystemRoot) {
      $launcherCandidates += Join-Path $env:SystemRoot 'py.exe'
      $launcherCandidates += Join-Path $env:SystemRoot 'pyw.exe'
    }
    foreach ($candidate in $launcherCandidates) {
      if (Test-Path -LiteralPath $candidate) { $script:pyExe = $candidate; $script:pyArgs = @('-3'); return $true }
    }

    $dirCandidates = New-Object System.Collections.Generic.List[string]
    if ($env:LOCALAPPDATA) { $dirCandidates.Add((Join-Path $env:LOCALAPPDATA 'Programs\Python')) }
    foreach ($root in @($env:ProgramFiles, $env:ProgramW6432, ${env:ProgramFiles(x86)})) {
      if ($root) {
        $dirCandidates.Add($root)
        $dirCandidates.Add((Join-Path $root 'Python'))
      }
    }
    foreach ($base in $dirCandidates) {
      if (-not $base) { continue }
      if (-not (Test-Path -LiteralPath $base)) { continue }
      try {
        $pyDirs = Get-ChildItem -LiteralPath $base -Directory -ErrorAction SilentlyContinue | Where-Object { $_.Name -like 'Python3*' }
      }
      catch { continue }
      foreach ($dir in ($pyDirs | Sort-Object Name -Descending)) {
        $candidate = Join-Path $dir.FullName 'python.exe'
        if (Test-Path -LiteralPath $candidate) { $script:pyExe = $candidate; $script:pyArgs = @(); return $true }
      }
      $directCandidate = Join-Path $base 'python.exe'
      if (Test-Path -LiteralPath $directCandidate) { $script:pyExe = $directCandidate; $script:pyArgs = @(); return $true }
    }

    # Registry detection: uses InstallPath registered by the Python installer
    $regRoots = @(
      'HKLM:\SOFTWARE\Python\PythonCore',
      'HKLM:\SOFTWARE\WOW6432Node\Python\PythonCore',
      'HKCU:\SOFTWARE\Python\PythonCore'
    )
    foreach ($regRoot in $regRoots) {
      try {
        $versions = Get-ChildItem -LiteralPath $regRoot -ErrorAction SilentlyContinue | Where-Object {
          $_.PSChildName -match '^3\.(9|1[0-2])($|[^0-9])'
        }
      }
      catch { continue }
      foreach ($versionKey in ($versions | Sort-Object PSChildName -Descending)) {
        $installKeyPath = Join-Path $versionKey.PSPath 'InstallPath'
        try {
          $installKey = Get-Item -LiteralPath $installKeyPath -ErrorAction SilentlyContinue
          if (-not $installKey) { continue }
          $exePath = $installKey.GetValue('ExecutablePath')
          if ($exePath -and (Test-Path -LiteralPath $exePath)) { $script:pyExe = $exePath; $script:pyArgs = @(); return $true }
          $rootPath = $installKey.GetValue('')
          if ($rootPath) {
            $candidate = Join-Path $rootPath 'python.exe'
            if (Test-Path -LiteralPath $candidate) { $script:pyExe = $candidate; $script:pyArgs = @(); return $true }
          }
        }
        catch { continue }
      }
    }
  }

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
    $waitSeconds = 30
    $announcedWait = $false
    for ($elapsed = 0; $elapsed -lt $waitSeconds; $elapsed++) {
      if (Find-Python) {
        if (Test-PythonOk) { return }
      }
      if (-not $announcedWait) {
        Write-Host "Waiting for newly installed Python to surface..." -ForegroundColor Yellow
        $announcedWait = $true
      }
      Start-Sleep -Seconds 1
    }
    throw "Python 3.9 - 3.12 required but not located after installation."
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
    elseif (Get-Command dnf   -ErrorAction SilentlyContinue) { & dnf install -y python3 }
    elseif (Get-Command yum   -ErrorAction SilentlyContinue) { & yum install -y python3 }
    else { throw "No supported package manager found. Please install Python 3.9 - 3.12 and re-run." }
    if (-not (Find-Python) -or -not (Test-PythonOk)) { throw "Python 3.9 - 3.12 required but not found after package install." }
    return
  }
  throw "Unsupported OS. Please install Python 3.9 - 3.12 and re-run."
}
if (-not (Find-Python) -or -not (Test-PythonOk)) { Install-Python }

# Ensure VC++ runtime before anything may trigger local ONNX export (Windows only)
Install-VcRedistIfMissing

# --- Git LFS best-effort (non-fatal) ---
if (Get-Command git -ErrorAction SilentlyContinue) {
  & git lfs version > $null 2>&1
  if ($LASTEXITCODE -eq 0) {
    & git -C $ROOT lfs install --local > $null 2>&1
    $env:GIT_LFS_SKIP_SMUDGE = '0'
    & git -C $ROOT lfs pull > $null 2>&1
  }
  else {
    Write-Host "Warning: Git LFS not installed. Large files (models) may be missing."
  }
}

# --- Bootstrap venv (ensure) ---
$scriptPath = Join-Path $ROOT 'projects\argos\bootstrap.py'
& $script:pyExe @script:pyArgs $scriptPath --ensure --yes --reinstall --with-dev --extras audio
if ($LASTEXITCODE -ne 0) { throw "bootstrap failed ($LASTEXITCODE)" }

# --- Resolve venv Python ---
$vpy = & $script:pyExe @script:pyArgs "$scriptPath" --print-venv
if (-not $vpy) { throw "could not resolve venv python" }
if (-not (Test-Path -LiteralPath $vpy)) { throw "venv python missing at $vpy" }
$venvRoot = Split-Path -Parent (Split-Path -Parent $vpy)
Write-Verbose "[Argos] python: $vpy (venv=$venvRoot)"
$env:PANOPTES_VENV_ROOT = $venvRoot
$env:PYTHONPYCACHEPREFIX = Join-Path $env:LOCALAPPDATA "rAIn\pycache"

# Add installers/ to PATH (session + profile) non-interactively (parity with Bash build)
try { & (Join-Path $HERE 'setup-path.ps1') -Quiet } catch { }

# Make sure we don't accidentally keep the bootstrap 'skip' flag for the selector
Remove-Item Env:ARGOS_SKIP_WEIGHTS -ErrorAction SilentlyContinue

# --- Launch the *model selector* (user picks pack, fetches, exports, THEN smoke check prompts once) ---
$prevQuietEnv = $env:PANOPTES_BUILD_QUIET
$env:PANOPTES_BUILD_QUIET = '1'
try {
  & $vpy -m panoptes.model._fetch_models @BuildArgs
}
finally {
  if ($null -ne $prevQuietEnv) {
    $env:PANOPTES_BUILD_QUIET = $prevQuietEnv
  }
  else {
    Remove-Item Env:PANOPTES_BUILD_QUIET -ErrorAction SilentlyContinue
  }
}
$buildExit = $LASTEXITCODE
if ($buildExit -eq 0) {
  & $vpy -m compileall -q (Join-Path $ROOT 'projects\argos')
  if ($LASTEXITCODE -ne 0) { throw "compileall failed ($LASTEXITCODE)" }
  exit 0
}

# Typer Exit with code 1 means the user chose "Exit" from the selector; treat as success.
if ($buildExit -eq 1) { exit 0 }

exit $buildExit
