$Here = Split-Path -Parent $MyInvocation.MyCommand.Path

if (-not $env:TERM) { $env:TERM = 'xterm-256color' }
$env:PYTHONUTF8 = '1'
$env:PYTHONUNBUFFERED = '1'
$env:FORCE_COLOR = '1'
$env:PANOPTES_NESTED_PROGRESS = '1'
$env:PANOPTES_PROGRESS_ACTIVE = '0'
try { $null = $PSStyle; $PSStyle.OutputRendering = 'Ansi' } catch {}

& (Join-Path $Here 'run.ps1') argos hm @args
