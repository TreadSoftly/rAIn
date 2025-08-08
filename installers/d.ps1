$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
& (Join-Path $Here 'run.ps1') argos d @args
