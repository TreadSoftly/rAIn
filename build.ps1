$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
& (Join-Path $Here 'installers\build.ps1') @args
