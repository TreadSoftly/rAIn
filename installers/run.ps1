[CmdletBinding()] param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Args)
$ErrorActionPreference = "Stop"

# --- Progress-friendly environment for CLI runs ---
if (-not $env:TERM) { $env:TERM = 'xterm-256color' }
$env:PYTHONUTF8 = '1'
$env:PYTHONUNBUFFERED = '1'
$env:FORCE_COLOR = '1'
$env:PANOPTES_NESTED_PROGRESS = '1'
$env:PANOPTES_PROGRESS_ACTIVE = '0'
try { $null = $PSStyle; $PSStyle.OutputRendering = 'Ansi' } catch {}
try { [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new() } catch {}

# Robust script location (works even if pasted into a console)
function _Here {
  if ($PSCommandPath) { return (Split-Path -Parent $PSCommandPath) }
  if ($MyInvocation.MyCommand.Path) { return (Split-Path -Parent $MyInvocation.MyCommand.Path) }
  return (Get-Location).Path
}
$HERE = _Here
$ROOT = Split-Path -Parent $HERE

function Ensure-GitLFS {
  if (Get-Command git -ErrorAction SilentlyContinue) {
    & git lfs version > $null 2>&1
    if ($LASTEXITCODE -eq 0) {
      & git -C $ROOT lfs install --local > $null 2>&1
      $env:GIT_LFS_SKIP_SMUDGE = '0'
      & git -C $ROOT lfs pull > $null 2>&1
    } else {
      Write-Host "⚠️  Git LFS not installed. Large files (models) may be missing."
      Write-Host "   Install Git LFS and re-run this command."
    }
  }
}

# Parse args
$proj = $null; $tokens = @(); $sawBuild = $false
foreach ($a in $Args) {
  $la = $a.ToLowerInvariant()
  if     ($la -in @('run','me')) { continue }
  elseif ($la -in @('build','package','pack')) { $sawBuild = $true }
  elseif ($la -in @('argos','argos:run','run:argos','argos:build','build:argos')) { $proj = 'argos' }
  else { $tokens += $a }
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

Ensure-GitLFS

# Normalize like "clip.mp4 d" -> "d clip.mp4", etc.
$opsMap = @{
  'd'='d'; 'detect'='d'; '-d'='d'; '--detect'='d';
  'hm'='hm'; 'heatmap'='hm'; '-hm'='hm'; '--hm'='hm'; '-heatmap'='hm'; '--heatmap'='hm';
  'gj'='gj'; 'geojson'='gj'; '-gj'='gj'; '--gj'='gj'; '-geojson'='gj'; '--geojson'='gj';
}
$opIdx = -1; $opNorm = $null
for ($i=0; $i -lt $tokens.Count; $i++) {
  $key = $tokens[$i].ToLowerInvariant()
  if ($opsMap.ContainsKey($key)) { $opIdx = $i; $opNorm = $opsMap[$key]; break }
}
if ($opIdx -gt 0) {
  $new = New-Object System.Collections.Generic.List[string]
  $new.Add($opNorm) | Out-Null
  for ($j=0; $j -lt $tokens.Count; $j++) { if ($j -ne $opIdx) { $new.Add($tokens[$j]) | Out-Null } }
  $tokens = $new
}

if ($sawBuild) {
  & (Join-Path $ROOT 'installers\build.ps1') $proj @tokens
  return
}

if ($proj -eq 'argos') {
  # PS 5.1-safe Python detection
  $pyExe  = $null
  $pyArgs = @()
  $cmd = Get-Command py -ErrorAction SilentlyContinue
  if ($cmd) { $pyExe = $cmd.Source; $pyArgs = @('-3') }
  if (-not $pyExe) { $cmd = Get-Command python3 -ErrorAction SilentlyContinue; if ($cmd) { $pyExe = $cmd.Source } }
  if (-not $pyExe) { $cmd = Get-Command python  -ErrorAction SilentlyContinue; if ($cmd) { $pyExe = $cmd.Source } }
  if (-not $pyExe) { throw "Python 3 not found." }

  & $pyExe @pyArgs "$ROOT\projects\argos\bootstrap.py" --ensure --yes > $null 2>&1
  $vpy = & $pyExe @pyArgs "$ROOT\projects\argos\bootstrap.py" --print-venv
  $env:PYTHONPYCACHEPREFIX = "$env:LOCALAPPDATA\rAIn\pycache"
  & $vpy -m panoptes.cli @tokens
  return
}

throw "Unknown project: $proj"
