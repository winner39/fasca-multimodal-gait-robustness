param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $projectRoot "src"
$targetRoot = Join-Path $projectRoot "artifacts\fasca_structssl_screen"
$analysisRoot = Join-Path $projectRoot "artifacts\fasca_structssl_analysis"
$corruptionRoot = Join-Path $projectRoot "artifacts\fasca_structssl_corruptions"
New-Item -ItemType Directory -Force $analysisRoot, $corruptionRoot | Out-Null

& $Python -m gait_robust.analyze_fasca_factorial `
    --prediction-roots `
        (Join-Path $projectRoot "artifacts\fasca_factorial_confirmatory") `
        (Join-Path $projectRoot "artifacts\fasca_ssl_confirmatory") `
        $targetRoot `
    --include-methods `
        rapid_embracenet_fasca_kd `
        rapid_embracenet_fasca_ssl_kd `
        rapid_embracenet_fasca_structssl_kd `
    --include-seeds 51 `
    --reference-method rapid_embracenet_fasca_structssl_kd `
    --output-dir (Join-Path $analysisRoot "missing_modalities")
if ($LASTEXITCODE -ne 0) {
    throw "Missing-modality analysis failed with exit code $LASTEXITCODE"
}

$targetPredictions = Join-Path $corruptionRoot "structssl_predictions.csv.gz"
& $Python -m gait_robust.evaluate_corruptions `
    --data (Join-Path $projectRoot "data\processed\windows_v2.npz") `
    --run-root $targetRoot `
    --method embracenet_fasca_structssl_kd `
    --model-type embracenet `
    --output (Join-Path $corruptionRoot "structssl.csv") `
    --prediction-output $targetPredictions `
    --folds 5 `
    --seeds 51 `
    --partition-seed 20260901
if ($LASTEXITCODE -ne 0) {
    throw "Corruption evaluation failed with exit code $LASTEXITCODE"
}

& $Python -m gait_robust.analyze_main_corruption_statistics `
    --predictions `
        (Join-Path $projectRoot "artifacts\fasca_architecture_corruption_stats\embracenet_fasca_predictions.csv.gz") `
        (Join-Path $projectRoot "artifacts\fasca_ssl_corruption_stats\embracenet_fasca_ssl_predictions.csv.gz") `
        $targetPredictions `
    --include-seeds 51 `
    --reference-method embracenet_fasca_structssl_kd `
    --output-dir (Join-Path $analysisRoot "corruptions")
if ($LASTEXITCODE -ne 0) {
    throw "Corruption statistics failed with exit code $LASTEXITCODE"
}

