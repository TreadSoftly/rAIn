$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
& (Join-Path $Here 'installers\pack.ps1') @args
