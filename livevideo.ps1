$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
& (Join-Path $Here 'lv.ps1') @args