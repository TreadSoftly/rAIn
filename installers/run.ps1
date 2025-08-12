[CmdletBinding()] param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Args)
$ErrorActionPreference = "Stop"

$HERE = Split-Path -Parent $MyInvocation.MyCommand.Path
$ROOT = Split-Path -Parent $HERE

function Ensure-GitLFS {
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

$proj = $null; $tokens = @(); $sawBuild = $false
foreach ($a in $Args) {
  $la = $a.ToLowerInvariant()
  if     ($la -in @('run','me')) { continue }
  elseif ($la -in @('build','package','pack')) { $sawBuild = $true }
  elseif ($la -in @('argos','argos:run','run:argos','argos:build','build:argos')) { $proj = 'argos' }
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

Ensure-GitLFS

# Normalize like "clip.mp4 d" -> "d clip.mp4", "all detect" -> "detect all"
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
  $build = Join-Path $ROOT 'installers\build.ps1'
  & $build $proj @tokens
  exit $LASTEXITCODE
}

if ($proj -eq 'argos') {
  $pyExe = (Get-Command py -ErrorAction SilentlyContinue)?.Source
  $pyArgs = @()
  if ($pyExe) { $pyArgs = @('-3') }
  if (-not $pyExe) { $pyExe = (Get-Command python3 -ErrorAction SilentlyContinue)?.Source }
  if (-not $pyExe) { $pyExe = (Get-Command python  -ErrorAction SilentlyContinue)?.Source }
  if (-not $pyExe) { Write-Error "Python 3 not found."; exit 1 }

  & $pyExe @pyArgs "$ROOT\projects\argos\bootstrap.py" --ensure --yes *> $null
  $vpy = & $pyExe @pyArgs "$ROOT\projects\argos\bootstrap.py" --print-venv
  $env:PYTHONPYCACHEPREFIX = "$env:LOCALAPPDATA\rAIn\pycache"
  & $vpy -m panoptes.cli @tokens
  exit $LASTEXITCODE
}

Write-Error "Unknown project: $proj"
exit 2
