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
        }
        else {
            Write-Host "⚠️  Git LFS not installed. Large files (models) may be missing."
            Write-Host "   Install Git LFS and re-run this command."
        }
    }
}

# Parse args
$proj = $null
$tokens = @()
$sawBuild = $false
foreach ($a in $Args) {
    $la = $a.ToLowerInvariant()
    if ($la -in @('run', 'me')) { continue }
    elseif ($la -in @('build', 'package', 'pack')) { $sawBuild = $true }
    elseif ($la -in @('argos', 'argos:run', 'run:argos', 'argos:build', 'build:argos')) { $proj = 'argos' }
    else { $tokens += $a }
}

# Infer project from CWD
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

# Build hand-off
if ($sawBuild) {
    $build = Join-Path $ROOT 'installers\build.ps1'
    if (Test-Path $build) {
        & $build $proj @tokens
        exit $LASTEXITCODE
    }
    else {
        Write-Error "Build script not found: installers\build.ps1"
        exit 1
    }
}

if ($proj -eq 'argos') {
    # Find Python (prefer py -3, then python3/python)
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
