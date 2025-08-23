$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $env:TERM) { $env:TERM = 'xterm-256color' }
$env:PYTHONUTF8 = '1'
$env:PYTHONUNBUFFERED = '1'
$env:FORCE_COLOR = '1'
$env:PANOPTES_NESTED_PROGRESS = '1'
$env:PANOPTES_PROGRESS_ACTIVE = '0'
# Prefer DSHOW; disable MSMF
if ($env:OS -eq 'Windows_NT') {
  if (-not $env:OPENCV_VIDEOIO_PRIORITY_DSHOW) { $env:OPENCV_VIDEOIO_PRIORITY_DSHOW = '1000' }
  if (-not $env:OPENCV_VIDEOIO_PRIORITY_MSMF)  { $env:OPENCV_VIDEOIO_PRIORITY_MSMF  = '0' }
}
try { $null = $PSStyle; $PSStyle.OutputRendering = 'Ansi' } catch {}
& (Join-Path $Here 'run.ps1') argos lv @args
