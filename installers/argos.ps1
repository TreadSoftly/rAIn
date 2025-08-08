[CmdletBinding()] param([Parameter(ValueFromRemainingArguments=$true)][string[]]$Args)
$HERE = Split-Path -Parent $MyInvocation.MyCommand.Path
& "$HERE\run.ps1" argos @Args
