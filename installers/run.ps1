# installers\run.ps1 — robust launcher for Argos CLI and LiveVideo
[CmdletBinding()]
param(
  # NOTE: do NOT name this "$Args" — that would shadow the automatic variable.
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$ArgList
)
$ErrorActionPreference = "Stop"

# ---------- Progress-friendly environment for CLI runs ----------
if (-not $env:TERM) { $env:TERM = 'xterm-256color' }
$env:PYTHONUTF8 = '1'
$env:PYTHONUNBUFFERED = '1'
$env:FORCE_COLOR = '1'
$env:PANOPTES_NESTED_PROGRESS = '1'
$env:PANOPTES_PROGRESS_ACTIVE = '0'

# Argos/Panoptes single-line progress (keep Python spinners tidy)
$env:ARGOS_PROGRESS_STREAM = 'stdout'
$env:ARGOS_FORCE_PLAIN_PROGRESS = '1'
$env:ARGOS_PROGRESS_TAIL = 'erase'
$env:ARGOS_PROGRESS_FINAL_NEWLINE = '0'

# ---- Prefer DSHOW on Windows; keep MSMF disabled (works best on Win 10/11) ----
if ($env:OS -eq 'Windows_NT') {
  if (-not $env:OPENCV_VIDEOIO_PRIORITY_DSHOW) { $env:OPENCV_VIDEOIO_PRIORITY_DSHOW = '1000' }
  if (-not $env:OPENCV_VIDEOIO_PRIORITY_MSMF) { $env:OPENCV_VIDEOIO_PRIORITY_MSMF = '0' }
}

# Silence pip (prevents progress lines being interleaved)
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

# ANSI + UTF-8 out, where available
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
  catch { }
  Write-Host "Installing Microsoft Visual C++ Redistributable (x64)..." -ForegroundColor Yellow
  $tmp = Join-Path $env:TEMP 'vc_redist.x64.exe'
  try {
    Invoke-WebRequest -UseBasicParsing -Uri 'https://aka.ms/vs/17/release/vc_redist.x64.exe' -OutFile $tmp
    $installerArgs = '/quiet', '/norestart'
    $p = Start-Process -FilePath $tmp -ArgumentList $installerArgs -PassThru -Wait
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

# Approved verb: Enable- (instead of Ensure-)
function Enable-GitLfs {
  if (Get-Command git -ErrorAction SilentlyContinue) {
    & git lfs version > $null 2>&1
    if ($LASTEXITCODE -eq 0) {
      & git -C $ROOT lfs install --local > $null 2>&1
      $env:GIT_LFS_SKIP_SMUDGE = '0'
      & git -C $ROOT lfs pull > $null 2>&1
    }
    else {
      Write-Host "⚠️  Git LFS not installed. Large files (models) may be missing."
      Write-Host "   Install Git LFS and re-run this command."
    }
  }
}

# Approved verb: Install- (instead of Ensure-OpenCV)
# If launching live/video mode, install the GUI build of OpenCV; otherwise the headless build.
function Install-OpenCvPackage {
  [CmdletBinding()]
  param(
    [Parameter(Mandatory)][string]$Vpy,
    [Parameter(Mandatory)][bool]$NeedsGui
  )
  $want = if ($NeedsGui) { 'opencv-python' } else { 'opencv-python-headless' }
  $avoid = if ($NeedsGui) { 'opencv-python-headless' } else { 'opencv-python' }

  & $Vpy -m pip show $want *>&1 | Out-Null
  if ($LASTEXITCODE -ne 0) {
    # Remove conflicting build if present (these two conflict)
    & $Vpy -m pip show $avoid *>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) {
      & $Vpy -m pip uninstall -y $avoid *>&1 | Out-Null
    }
    & $Vpy -m pip install --no-input --quiet $want *>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "Failed to install $want" }
  }
}

# ---------- Parse args ----------
$proj = $null
$tokens = New-Object System.Collections.Generic.List[string]
$sawBuild = $false
$liveMode = $false
$foundL = $false
$foundV = $false

foreach ($a in ($ArgList | ForEach-Object { $_ })) {
  $la = $a.ToLowerInvariant()
  if ($la -in @('run', 'me')) { continue }
  elseif ($la -in @('build', 'package', 'pack')) { $sawBuild = $true }
  elseif ($la -in @('argos', 'argos:run', 'run:argos', 'argos:build', 'build:argos')) { $proj = 'argos' }
  elseif ($la -in @('lv', 'livevideo', 'live', 'video', 'ldv', 'lvd')) { $liveMode = $true }
  else {
    if ($la -eq 'l') { $foundL = $true }
    if ($la -eq 'v') { $foundV = $true }
    $tokens.Add($a) | Out-Null
  }
}

# Allow "l v" crumbs to imply live mode; if enabled, drop one l and one v from the argv
if (-not $liveMode -and $foundL -and $foundV) {
  $liveMode = $true
  $new = New-Object System.Collections.Generic.List[string]
  $skipL = $true; $skipV = $true
  foreach ($t in $tokens) {
    $key = $t.ToLowerInvariant()
    if ($skipL -and $key -eq 'l') { $skipL = $false; continue }
    if ($skipV -and $key -eq 'v') { $skipV = $false; continue }
    $new.Add($t) | Out-Null
  }
  $tokens = $new
}

# ---- Accept legacy flag '--no-headless' by dropping it ----
if ($tokens.Count -gt 0) {
  $filtered = New-Object System.Collections.Generic.List[string]
  foreach ($t in $tokens) {
    $k = $t.ToLowerInvariant()
    if ($k -in @('--no-headless', '-no-headless')) { continue }  # default is windowed already
    $filtered.Add($t) | Out-Null
  }
  $tokens = $filtered
}

# Infer project from CWD
$cwd = (Get-Location).Path
if (-not $proj) { if ($cwd -like "*\projects\argos*") { $proj = 'argos' } }
if (-not $proj) {
  if ($cwd -eq $ROOT -or $cwd -eq (Join-Path $ROOT 'projects')) {
    Write-Host "Specify the project:  run argos  |  argos [args]" -ForegroundColor Yellow
    exit 2
  }
}

Enable-GitLfs

# ---------- If user asked to (re)build, chain to build.ps1 ----------
if ($sawBuild) {
  & (Join-Path $ROOT 'installers\build.ps1') $proj @([string[]]$tokens)
  exit $LASTEXITCODE
}

# ---------- Python / venv ----------
# PS 5.1-safe Python detection
$pyExe = $null
$pyArgs = @()
$cmd = Get-Command py -ErrorAction SilentlyContinue
if ($cmd) { $pyExe = $cmd.Source; $pyArgs = @('-3') }
if (-not $pyExe) { $cmd = Get-Command python3 -ErrorAction SilentlyContinue; if ($cmd) { $pyExe = $cmd.Source } }
if (-not $pyExe) { $cmd = Get-Command python  -ErrorAction SilentlyContinue; if ($cmd) { $pyExe = $cmd.Source } }
if (-not $pyExe) { throw "Python 3 not found." }

# Always ensure the venv quietly to avoid stderr noise being treated as errors
$bootstrap = Join-Path $ROOT 'projects\argos\bootstrap.py'
& $pyExe @pyArgs "$bootstrap" --ensure --yes > $null 2>&1

# Resolve venv python
$vpy = & $pyExe @pyArgs "$bootstrap" --print-venv
if (-not $vpy) { throw "could not resolve venv python" }

# Keep pycache out of tree
$env:PYTHONPYCACHEPREFIX = "$env:LOCALAPPDATA\rAIn\pycache"

# Ensure VC++ runtime before any potential ONNX use on Windows
Install-VcRedistIfMissing

# ---------- Ensure OpenCV in the venv (GUI vs headless) ----------
Install-OpenCvPackage -Vpy $vpy -NeedsGui:$liveMode

# ---------- Select module and live-mode env ----------
$pyMod = if ($liveMode) { 'panoptes.live.cli' } else { 'panoptes.cli' }
if ($liveMode) {
  $env:PANOPTES_LIVE = '1'
  $env:PANOPTES_PROGRESS_TAIL = 'none'
  $env:PANOPTES_PROGRESS_FINAL_NEWLINE = '0'
  $env:PANOPTES_NESTED_PROGRESS = '0'
}

# ---------- Launch ----------
& $vpy -m $pyMod @([string[]]$tokens)
exit $LASTEXITCODE
