# installers\run.ps1 — robust launcher for Argos CLI and LiveVideo
[CmdletBinding()] param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Args)
$ErrorActionPreference = "Stop"

# ---------- Progress-friendly environment for CLI runs ----------
if (-not $env:TERM) { $env:TERM = 'xterm-256color' }
$env:PYTHONUTF8 = '1'
$env:PYTHONUNBUFFERED = '1'
$env:FORCE_COLOR = '1'
$env:PANOPTES_NESTED_PROGRESS = '1'
$env:PANOPTES_PROGRESS_ACTIVE = '0'

# Argos/Panoptes single-line progress
$env:ARGOS_PROGRESS_STREAM = 'stdout'
$env:ARGOS_FORCE_PLAIN_PROGRESS = '1'
$env:ARGOS_PROGRESS_TAIL = 'erase'
$env:ARGOS_PROGRESS_FINAL_NEWLINE = '0'

# ---- Prefer DSHOW on Windows; keep MSMF disabled (works best on Win 10/11) ----
if ($env:OS -eq 'Windows_NT') {
  if (-not $env:OPENCV_VIDEOIO_PRIORITY_DSHOW) { $env:OPENCV_VIDEOIO_PRIORITY_DSHOW = '1000' }
  if (-not $env:OPENCV_VIDEOIO_PRIORITY_MSMF) { $env:OPENCV_VIDEOIO_PRIORITY_MSMF = '0' }
}

# Silence pip (prevents progress lines from being pushed)
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

function Initialize-GitLFS {
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

function Invoke-Quiet {
  param([string]$Exe, [string[]]$ArgumentList)
  & $Exe @ArgumentList *>&1 | Out-Null
  return $LASTEXITCODE
}

# ---------- Parse args ----------
$proj = $null
$tokens = New-Object System.Collections.Generic.List[string]
$sawBuild = $false
$liveMode = $false
$foundL = $false
$foundV = $false

foreach ($a in $Args) {
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

# Allow "lv d" or "l v ..." shorthand to flip on live mode
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
    return
  }
}

Initialize-GitLFS

# ---------- If user asked to (re)build, chain to build.ps1 ----------
if ($sawBuild) {
  & (Join-Path $ROOT 'installers\build.ps1') $proj @($tokens)
  return
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
if ((Invoke-Quiet $pyExe ($pyArgs + $bootstrap + @('--ensure', '--yes'))) -ne 0) {
  throw "bootstrap ensure failed"
}

# Resolve venv python
$vpy = & $pyExe @pyArgs "$bootstrap" --print-venv
if (-not $vpy) { throw "could not resolve venv python" }

# Keep pycache out of tree
$env:PYTHONPYCACHEPREFIX = "$env:LOCALAPPDATA\rAIn\pycache"

# ---------- Install OpenCV in the venv if needed ----------
# If launching live/video mode, install the GUI build of OpenCV; otherwise the headless build.
function Install-OpenCV {
  param([string]$Vpy, [bool]$NeedsGUI)
  $want = if ($NeedsGUI) { 'opencv-python' } else { 'opencv-python-headless' }
  $avoid = if ($NeedsGUI) { 'opencv-python-headless' } else { 'opencv-python' }

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
Install-OpenCV -Vpy $vpy -NeedsGUI:$liveMode

# ---------- Token normalization to the CLI’s actual syntax ----------
# The underlying CLI expects:  <INPUT> --task <TASK> [other args]
# Users often type: "d all", "detect all", "hm all", "livevideo detect", etc.
# We translate those to:       "all --task detect|heatmap"
$taskMap = @{
  'd' = 'detect'; 'detect' = 'detect';
  'hm' = 'heatmap'; 'heatmap' = 'heatmap';
  'gj' = 'geojson'; 'geojson' = 'geojson';
  'classify' = 'classify'; 'clf' = 'classify';
  'pose' = 'pose'; 'pse' = 'pose';
  'obb' = 'obb'; 'object' = 'obb';
}

function Convert-PanoptesArgs {
  param([System.Collections.Generic.List[string]]$Tok)
  if (-not $Tok -or $Tok.Count -eq 0) { return @() }

  # If caller already provided --task/-t, respect it.
  $hasTaskFlag = $false
  foreach ($t in $Tok) {
    $k = $t.ToLowerInvariant()
    if ($k -eq '--task' -or $k -eq '-t') { $hasTaskFlag = $true; break }
  }
  if ($hasTaskFlag) { return [string[]]$Tok }

  # If first token is a task alias, pull it out.
  $task = $null
  $userInput = $null
  $rest = New-Object System.Collections.Generic.List[string]

  if ($Tok.Count -gt 0) {
    $first = $Tok[0].ToLowerInvariant()
    if ($taskMap.ContainsKey($first)) {
      $task = $taskMap[$first]
      # The next positional (if any) is the input; default to 'all'
      if ($Tok.Count -gt 1) { $userInput = $Tok[1] }
      for ($i = 2; $i -lt $Tok.Count; $i++) { $rest.Add($Tok[$i]) | Out-Null }
      if (-not $userInput) { $userInput = 'all' }
      return @($userInput, '--task', $task) + $rest
    }
  }

  # If first token looks like input and SECOND token is a task alias (e.g. "all d")
  if ($Tok.Count -gt 1) {
    $second = $Tok[1].ToLowerInvariant()
    if ($taskMap.ContainsKey($second)) {
      $task = $taskMap[$second]
      $userInput = $Tok[0]
      for ($i = 2; $i -lt $Tok.Count; $i++) { $rest.Add($Tok[$i]) | Out-Null }
      return @($userInput, '--task', $task) + $rest
    }
  }

  # No explicit task provided; leave as-is (caller may be using full CLI syntax already)
  return [string[]]$Tok
}

$tokens = [System.Collections.Generic.List[string]]$tokens
$tokens = [System.Collections.Generic.List[string]](Convert-PanoptesArgs -Tok $tokens)
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
