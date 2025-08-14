# installers\lib\progress.psm1
# Small, safe helpers for Write-Progress. No external deps.
# Also coordinates with Python progress (PANOPTES_PROGRESS_ACTIVE).

$script:Current = @{
  Activity = "Working..."
  Status   = ""
  Total    = 100
  Done     = 0
  Start    = [DateTime]::UtcNow
  Id       = 1
}

# Track nesting so we only toggle env var at outermost scope
if (-not ($script:Depth)) { $script:Depth = 0 }
$script:_PrevActive = $null

function Enter-ProgressScope {
  if ($script:Depth -eq 0) {
    $script:_PrevActive = $env:PANOPTES_PROGRESS_ACTIVE
    $env:PANOPTES_PROGRESS_ACTIVE = '1'  # suppress Python spinners while PS owns the TTY
  }
  $script:Depth++
}

function Exit-ProgressScope {
  if ($script:Depth -gt 0) { $script:Depth-- }
  if ($script:Depth -eq 0) {
    if ($script:_PrevActive) {
      $env:PANOPTES_PROGRESS_ACTIVE = $script:_PrevActive
    }
    else {
      Remove-Item Env:PANOPTES_PROGRESS_ACTIVE -ErrorAction SilentlyContinue | Out-Null
    }
    $script:_PrevActive = $null
  }
}

function Start-ProgressPhase {
  [CmdletBinding()]
  param(
    [Parameter(Mandatory)][string]$Activity,
    [int]$Total = 100
  )
  Enter-ProgressScope
  $script:Current.Activity = $Activity
  $script:Current.Total = [Math]::Max(1, $Total)
  $script:Current.Done = 0
  $script:Current.Start = [DateTime]::UtcNow
  Write-Progress -Id $script:Current.Id -Activity $Activity -Status "0% (starting)" -PercentComplete 0
}

function Set-Progress {
  [CmdletBinding()]
  param(
    [Parameter(Mandatory)][int]$Done,
    [string]$Status = ""
  )
  $script:Current.Done = [Math]::Max(0, $Done)
  $pct = [int]([Math]::Min(100, 100.0 * $script:Current.Done / $script:Current.Total))
  if (-not $Status) { $Status = "$pct% complete" }
  Write-Progress -Id $script:Current.Id -Activity $script:Current.Activity -Status $Status -PercentComplete $pct
}

function Step-Progress {
  [CmdletBinding()]
  param(
    [int]$Delta = 1,
    [string]$Status = ""
  )
  Set-Progress -Done ($script:Current.Done + $Delta) -Status $Status
}

function Complete-Progress {
  [CmdletBinding()]
  param()
  # ensure accounting and cleanly clear the record
  $script:Current.Done = $script:Current.Total
  Write-Progress -Id $script:Current.Id -Activity $script:Current.Activity -Status "Done" -PercentComplete 100 -Completed
  # force a newline so PSReadLine doesn't "ghost" the next prompt on the progress row
  Write-Host ""
  Exit-ProgressScope
}

function Invoke-Step {
  <#
    .SYNOPSIS
      Runs a step, shows Write-Progress, bumps progress by Weight.
  #>
  [CmdletBinding()]
  param(
    [Parameter(Mandatory)][string]$Name,
    [Parameter(Mandatory)][scriptblock]$Body,
    [int]$Weight = 1
  )

  # Pre-status ping
  $cur = $script:Current
  $pct = [int]([Math]::Min(100, 100.0 * $cur.Done / $cur.Total))
  Write-Progress -Id $cur.Id -Activity $cur.Activity -Status "$pct% • $Name" -PercentComplete $pct

  try {
    & $Body
    Step-Progress -Delta $Weight -Status $Name
  }
  catch {
    # Surface the failure while keeping the progress UI consistent
    Write-Progress -Id $cur.Id -Activity $cur.Activity -Status "FAILED: $Name" -PercentComplete $pct
    throw
  }
}

# Export when imported as a module; no-op when dot-sourced
if ($MyInvocation.MyCommand.Module) {
  Export-ModuleMember -Function Start-ProgressPhase, Set-Progress, Step-Progress, Complete-Progress, Invoke-Step
}
