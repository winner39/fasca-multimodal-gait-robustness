param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $projectRoot "src"
$outputRoot = Join-Path $projectRoot "artifacts\fault_triggered_router"
$analysisRoot = Join-Path $projectRoot "artifacts\fault_triggered_router_analysis"
New-Item -ItemType Directory -Force $outputRoot, $analysisRoot | Out-Null
$baselineRoot = Join-Path $projectRoot "artifacts\fasca_factorial_confirmatory"

& $Python -m gait_robust.evaluate_corruptions `
    --data (Join-Path $projectRoot "data\processed\windows_v2.npz") `
    --run-root (Join-Path $projectRoot "artifacts\fasca_qstructssl_mar_screen") `
    --method embracenet_fasca_qstructssl_mar_kd `
    --model-type embracenet `
    --fallback-run-root $baselineRoot `
    --fallback-method embracenet_fasca_kd `
    --fallback-model-type embracenet `
    --output-method fault_triggered_router `
    --output (Join-Path $outputRoot "seed51.csv") `
    --prediction-output (Join-Path $outputRoot "seed51_predictions.csv.gz") `
    --folds 5 `
    --seeds 51 `
    --partition-seed 20260901
if ($LASTEXITCODE -ne 0) {
    throw "Seed-51 router evaluation failed with exit code $LASTEXITCODE"
}

& $Python -m gait_robust.evaluate_corruptions `
    --data (Join-Path $projectRoot "data\processed\windows_v2.npz") `
    --run-root (Join-Path $projectRoot "artifacts\fasca_qstructssl_mar_confirm") `
    --method embracenet_fasca_qstructssl_mar_kd `
    --model-type embracenet `
    --fallback-run-root $baselineRoot `
    --fallback-method embracenet_fasca_kd `
    --fallback-model-type embracenet `
    --output-method fault_triggered_router `
    --output (Join-Path $outputRoot "seed52_53.csv") `
    --prediction-output (Join-Path $outputRoot "seed52_53_predictions.csv.gz") `
    --folds 5 `
    --seeds 52 53 `
    --partition-seed 20260901
if ($LASTEXITCODE -ne 0) {
    throw "Seed-52/53 router evaluation failed with exit code $LASTEXITCODE"
}

& $Python -m gait_robust.analyze_main_corruption_statistics `
    --predictions `
        (Join-Path $projectRoot "artifacts\fasca_architecture_corruption_stats\embracenet_fasca_predictions.csv.gz") `
        (Join-Path $outputRoot "seed51_predictions.csv.gz") `
        (Join-Path $outputRoot "seed52_53_predictions.csv.gz") `
    --reference-method fault_triggered_router `
    --output-dir $analysisRoot
if ($LASTEXITCODE -ne 0) {
    throw "Router statistics failed with exit code $LASTEXITCODE"
}

