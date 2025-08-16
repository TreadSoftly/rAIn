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

# Already on PATH for this session?
if ($env:Path.Split(';') -contains $target) {
  if (-not $Quiet) { Write-Host "Already on PATH." }
  return
}

if (-not $Quiet) {
  $resp = Read-Host "Add '$target' to your PATH for this user? [y/n]"
  if (-not $resp) { $resp = 'Y' }
  if ($resp.ToLower() -notin @('y','yes')) { Write-Host "Skipped."; return }
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
if (-not (Test-Path -LiteralPath $PROFILE)) { New-Item -ItemType File -Path $PROFILE -Force | Out-Null }
$marker = '# rAIn installers on PATH'
if (-not (Select-String -LiteralPath $PROFILE -SimpleMatch $marker -Quiet)) {
  Add-Content -Path $PROFILE -Value "`n$marker`n`$env:Path += ';$target'"
}

if (-not $Quiet) {
  Write-Host " Added. New terminals will also see the change."
  Write-Host " Commands available: build, all, argos, d/detect, hm/heatmap, gj/geojson, lv/livevideo"
}
return