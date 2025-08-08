#Requires -Version 5.1
$ErrorActionPreference = 'Stop'

$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = (Resolve-Path (Join-Path $Here '..')).Path

function Ensure-GitLFS {
    if (Get-Command git -ErrorAction SilentlyContinue) {
        & git lfs version *> $null
        if ($LASTEXITCODE -eq 0) {
            & git -C $Root lfs install --local  *> $null
            $env:GIT_LFS_SKIP_SMUDGE = '0'
            & git -C $Root lfs pull *> $null
        } else {
            Write-Host "⚠️  Git LFS not installed. Large files (models) may be missing." -ForegroundColor Yellow
            Write-Host "   Install Git LFS and re-run this command." -ForegroundColor Yellow
        }
    }
}

# Parse args to detect project "argos" (keep behavior similar to bash script)
$proj   = $null
$tokens = @()
foreach ($t in $args) {
    $lt = $t.ToLowerInvariant()
    switch ($lt) {
        'run' { continue }
        'me'  { continue }
        'argos' { $proj = 'argos'; continue }
        'argos:run' { $proj = 'argos'; continue }
        'run:argos' { $proj = 'argos'; continue }
        default { $tokens += $t }
    }
}

if (-not $proj) {
    $cwd = (Get-Location).Path
    if ($cwd -like "*\projects\argos*") { $proj = 'argos' }
}

if (-not $proj) {
    if ($cwd -eq $Root -or $cwd -eq (Join-Path $Root 'projects')) {
        Write-Error 'Specify the project:  run argos  |  argos [args]'
        exit 2
    }
}

Ensure-GitLFS

if ($proj -eq 'argos') {
    $py = (Get-Command python3 -ErrorAction SilentlyContinue)?.Source
    if (-not $py) { $py = (Get-Command python -ErrorAction SilentlyContinue)?.Source }
    if (-not $py) { Write-Error "Python 3 not found."; exit 1 }

    & $py "$Root\projects\argos\bootstrap.py" --ensure --yes *> $null
    $vpy = & $py "$Root\projects\argos\bootstrap.py" --print-venv
    $env:PYTHONPYCACHEPREFIX = "$env:LOCALAPPDATA\rAIn\pycache"
    & $vpy -m panoptes.cli @tokens
    exit $LASTEXITCODE
}

Write-Error "Unknown project: $proj"
exit 2
