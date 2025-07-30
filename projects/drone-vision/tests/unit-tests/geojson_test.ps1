<#
    geojson-smoke.ps1
    Run the Drone-Vision CLI (“target”) with the *gj* task against a few
    public-domain images that contain #lat_lon fragments.  Resulting .geojson
    files are copied into projects\drone-vision\tests\results\.

    Usage:  pwsh geojson_test.ps1 (will be patching this into the unit test to
    run it automatically through pytest)
            (or simply paste the block into an interactive session)
#>

$ErrorActionPreference = 'Stop'

# ── determine the script’s root folder (works even when pasted) ───────────
if ($PSScriptRoot -and $PSScriptRoot.Trim()) {
    $scriptRoot = $PSScriptRoot
} else {
    $scriptRoot = (Get-Location).ProviderPath
}

# ── output folder ─────────────────────────────────────────────────────────
$resultsDir = Join-Path $scriptRoot 'projects\drone-vision\tests\results'
if (-not (Test-Path $resultsDir)) {
    New-Item -ItemType Directory -Path $resultsDir | Out-Null
}

# ── stable public-domain photos *with* #lat_lon fragments ─────────────────
$urls = @(
    # Golden Gate Bridge
    'https://upload.wikimedia.org/wikipedia/commons/0/0c/GoldenGateBridge-001.jpg#lat37.8199_lon-122.4783',

    # Statue of Liberty
    'https://upload.wikimedia.org/wikipedia/commons/a/a1/Statue_of_Liberty_7.jpg#lat40.6892_lon-74.0445',

    # Sydney Opera House
    'https://upload.wikimedia.org/wikipedia/commons/4/40/Sydney_Opera_House_Sails.jpg#lat-33.8568_lon151.2153.jpg'
)

# ── helper: file-stem without any URL fragment ────────────────────────────
function Get-BaseName([string]$url) {
    $tail = ($url -split '/' | Select-Object -Last 1) -replace '#.*$',''
    [IO.Path]::GetFileNameWithoutExtension($tail)
}

# ── main loop ─────────────────────────────────────────────────────────────
foreach ($u in $urls) {

    $name = Get-BaseName $u
    Write-Host ("⏳  {0,-30}" -f $name) -NoNewline

    try {
        # run CLI, capture ALL output, preserve exit code
        $raw   = & target $u gj 2>&1
        $exit  = $LASTEXITCODE
        $rawTx = $raw | Out-String

        if ($exit -eq 0 -and $rawTx -match '★ wrote (.+?\.geojson)') {
            Move-Item -LiteralPath $Matches[1]
                    -Destination (Join-Path $resultsDir "$name.geojson")
                    -Force
            Write-Host ' ✔'
        }
        elseif ($exit -eq 0 -and $rawTx.Trim().StartsWith('{')) {
            $rawTx | Out-File -Encoding UTF8
                    -FilePath (Join-Path $resultsDir "$name.geojson")
            Write-Host ' ✔ (captured)'
        }
        else {
            throw "CLI exit $exitn$rawTx"
        }

    } catch {
        Write-Host ' ✖'
        Write-Host $_.Exception.Message -ForegroundColor Red
    }
}

Write-Host "n✅  Finished. Open the files in VS Code or QGIS." -ForegroundColor Green
