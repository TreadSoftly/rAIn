[CmdletBinding()]
param([switch]$Quiet)
$ErrorActionPreference = "Stop"

$HERE = Split-Path -Parent $MyInvocation.MyCommand.Path
$target = $HERE.TrimEnd('\')

# Already on PATH for this session?
if ($env:Path.Split(';') -contains $target) {
  if (-not $Quiet) { Write-Host "Already on PATH." }
  exit 0
}

if (-not $Quiet) {
  $resp = Read-Host "Add '$target' to your PATH for this user? [Y/n]"
  if (-not $resp) { $resp = 'Y' }
  if ($resp.ToLower() -notin @('y','yes')) { Write-Host "Skipped."; exit 0 }
}

# Persist for the current user
$cur = [Environment]::GetEnvironmentVariable('Path','User')
if (-not $cur) { $cur = '' }
if ($cur.Split(';') -notcontains $target) {
  $new = (($cur.TrimEnd(';') + ';' + $target).TrimStart(';')).TrimEnd(';')
  [Environment]::SetEnvironmentVariable('Path',$new,'User')
}

# Add to the current session immediately
if ($env:Path.Split(';') -notcontains $target) { $env:Path += ";$target" }

# Ensure it sticks for new PS sessions too
if (-not (Test-Path -LiteralPath $PROFILE)) {
  New-Item -ItemType File -Path $PROFILE -Force | Out-Null
}
$marker = '# rAIn installers on PATH'
if (-not (Select-String -LiteralPath $PROFILE -SimpleMatch $marker -Quiet)) {
  Add-Content -Path $PROFILE -Value "`n$marker`n`$env:Path += ';$target'"
}

if (-not $Quiet) { Write-Host "✅ Added. New terminals will also see the change." }
