[CmdletBinding()] param()
$ErrorActionPreference = "Stop"

$HERE = Split-Path -Parent $MyInvocation.MyCommand.Path
$target = $HERE.TrimEnd('\')

# Already on PATH?
if ($env:Path.Split(';') -contains $target) {
  Write-Host "Already on PATH."
  exit 0
}

$resp = Read-Host "Add '$target' to your PATH for this user? [Y/n]"
if (-not $resp) { $resp = 'Y' }
if ($resp.ToLower() -notin @('y', 'yes')) { Write-Host "Skipped."; exit 0 }

# Persist for the current user
$cur = [Environment]::GetEnvironmentVariable('Path', 'User')
if (-not $cur) { $cur = '' }
if ($cur.Split(';') -notcontains $target) {
  $new = (($cur.TrimEnd(';') + ';' + $target).TrimStart(';')).TrimEnd(';')
  [Environment]::SetEnvironmentVariable('Path', $new, 'User')
}

# Also add to the current session
if ($env:Path.Split(';') -notcontains $target) { $env:Path += ";$target" }

# Add to PowerShell profile for good measure
if (-not (Test-Path -LiteralPath $PROFILE)) {
  New-Item -ItemType File -Path $PROFILE -Force | Out-Null
}
Add-Content -Path $PROFILE -Value "`n# rAIn installers on PATH`n`$env:Path += ';$target'"

Write-Host "✅ Added. Open a new terminal for global effect."
