[CmdletBinding()] param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Args)
$ErrorActionPreference = "Stop"

$env:PYTHONUTF8 = "1"
$env:FORCE_COLOR = "1"
if (-not $env:TERM) { $env:TERM = "xterm-256color" }
Remove-Item Env:PANOPTES_PROGRESS_ACTIVE -ErrorAction SilentlyContinue

$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
& (Join-Path $Here 'installers\run.ps1') @Args
