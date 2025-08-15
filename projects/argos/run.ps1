[CmdletBinding()]
param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Args)
$ErrorActionPreference = "Stop"

function _Here {
  if ($PSCommandPath) { return (Split-Path -Parent $PSCommandPath) }
  if ($MyInvocation.MyCommand.Path) { return (Split-Path -Parent $MyInvocation.MyCommand.Path) }
  return (Get-Location).Path
}
$Here = _Here
$Root = (Resolve-Path (Join-Path $Here '..\..')).Path

$env:TERM = if ($env:TERM) { $env:TERM } else { 'xterm-256color' }
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'UTF-8'
$env:PYTHONUNBUFFERED = '1'
$env:FORCE_COLOR = '1'
$env:PANOPTES_NESTED_PROGRESS = if ($env:PANOPTES_NESTED_PROGRESS) { $env:PANOPTES_NESTED_PROGRESS } else { '1' }
$env:PANOPTES_ENABLE_OSC8 = if ($env:PANOPTES_ENABLE_OSC8) { $env:PANOPTES_ENABLE_OSC8 } else { '1' }

try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}
if ($PSVersionTable.PSVersion.Major -ge 7) {
  try { $PSStyle.OutputRendering = 'Ansi' } catch {}
}

# Delegate to the root installer launcher. Root script handles:
# - Git LFS warmup
# - Python detection (PS 5.1-safe)
# - Command normalization and project inference
& (Join-Path $Root 'installers\run.ps1') @Args
exit $LASTEXITCODE
