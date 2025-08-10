$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = (Resolve-Path (Join-Path $Here '..\..')).Path
& (Join-Path $Root 'installers\build.ps1') argos @args
