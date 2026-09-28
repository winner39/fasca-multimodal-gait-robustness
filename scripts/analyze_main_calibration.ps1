param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $projectRoot "src"

& $Python -m gait_robust.analyze_main_calibration `
    --artifact-root (Join-Path $projectRoot "artifacts") `
    --output-dir (Join-Path $projectRoot "artifacts\main_calibration") `
    --bootstrap-samples 20000 `
    --bootstrap-seed 20260704 `
    --bins 10

if ($LASTEXITCODE -ne 0) {
    throw "Primary calibration analysis failed with exit code $LASTEXITCODE"
}

