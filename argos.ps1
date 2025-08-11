$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
& (Join-Path $Here 'installers\run.ps1') argos @args
