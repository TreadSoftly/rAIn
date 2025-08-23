[CmdletBinding()]
param([switch]$Quiet)
$ErrorActionPreference = "Stop"

function _Here {
  if ($PSCommandPath) { return (Split-Path -Parent $PSCommandPath) }
  if ($MyInvocation.MyCommand.Path) { return (Split-Path -Parent $MyInvocation.MyCommand.Path) }
  return (Get-Location).Path
}

$HERE   = _Here
$target = $HERE.TrimEnd('\')

# Helper for case-insensitive, trailing-slash-agnostic comparison
function _Norm([string]$p) {
  if (-not $p) { return "" }
  return $p.Trim().TrimEnd('\').ToLowerInvariant()
}

# --- Put installers FIRST on the current session PATH (de-dup) ---
$parts = @($env:Path -split ';' | Where-Object { $_ })
$rest  = foreach ($p in $parts) { if (_Norm $p -ne (_Norm $target)) { $p } }
$env:Path = (($target) + ';' + ($rest -join ';')).Trim(';')

# --- Persist for the current user (de-dup + prepend) ---
$curUser = [Environment]::GetEnvironmentVariable('Path','User')
if (-not $curUser) { $curUser = '' }
$partsU = @($curUser -split ';' | Where-Object { $_ })
$restU  = foreach ($p in $partsU) { if (_Norm $p -ne (_Norm $target)) { $p } }
$newUser = (($target) + ';' + ($restU -join ';')).Trim(';')
[Environment]::SetEnvironmentVariable('Path', $newUser, 'User')

# --- Ensure future PowerShell sessions also see it early (optional) ---
if (-not (Test-Path -LiteralPath $PROFILE)) { New-Item -ItemType File -Path $PROFILE -Force | Out-Null }
$marker = '# rAIn installers on PATH (prepend)'
if (-not (Select-String -LiteralPath $PROFILE -SimpleMatch $marker -Quiet)) {
  Add-Content -Path $PROFILE -Value "`n$marker`n`$env:Path = '$target;' + `$env:Path"
}

if (-not $Quiet) {
  Write-Host "✅ Added (prepended) to PATH:" -ForegroundColor Green
  Write-Host "    $target"
  Write-Host ""
  Write-Host "This ensures 'lv', 'livevideo', etc. resolve to your repo shims instead of"
  Write-Host "global Python\Scripts stubs. Open a NEW terminal to pick up persisted PATH."
}
return
