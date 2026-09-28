param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $projectRoot "src"
$screenRoot = Join-Path $projectRoot "artifacts\fasca_qstructssl_mar_screen"
$confirmRoot = Join-Path $projectRoot "artifacts\fasca_qstructssl_mar_confirm"
$analysisRoot = Join-Path $projectRoot "artifacts\fasca_qstructssl_mar_confirm_analysis"
$corruptionRoot = Join-Path $projectRoot "artifacts\fasca_qstructssl_mar_confirm_corruptions"
New-Item -ItemType Directory -Force $analysisRoot, $corruptionRoot | Out-Null

& $Python -m gait_robust.analyze_fasca_factorial `
    --prediction-roots `
        (Join-Path $projectRoot "artifacts\fasca_factorial_confirmatory") `
        $screenRoot `
        $confirmRoot `
    --include-methods `
        rapid_embracenet_fasca_kd `
        rapid_embracenet_fasca_qstructssl_mar_kd `
    --reference-method rapid_embracenet_fasca_qstructssl_mar_kd `
    --output-dir (Join-Path $analysisRoot "missing_modalities")
if ($LASTEXITCODE -ne 0) {
    throw "Missing-modality analysis failed with exit code $LASTEXITCODE"
}

$confirmPredictions = Join-Path $corruptionRoot "confirm_predictions.csv.gz"
& $Python -m gait_robust.evaluate_corruptions `
    --data (Join-Path $projectRoot "data\processed\windows_v2.npz") `
    --run-root $confirmRoot `
    --method embracenet_fasca_qstructssl_mar_kd `
    --model-type embracenet `
    --output (Join-Path $corruptionRoot "confirm.csv") `
    --prediction-output $confirmPredictions `
    --folds 5 `
    --seeds 52 53 `
    --partition-seed 20260901
if ($LASTEXITCODE -ne 0) {
    throw "Corruption evaluation failed with exit code $LASTEXITCODE"
}

& $Python -m gait_robust.analyze_main_corruption_statistics `
    --predictions `
        (Join-Path $projectRoot "artifacts\fasca_architecture_corruption_stats\embracenet_fasca_predictions.csv.gz") `
        (Join-Path $projectRoot "artifacts\fasca_qstructssl_mar_corruptions\combined_predictions.csv.gz") `
        $confirmPredictions `
    --reference-method embracenet_fasca_qstructssl_mar_kd `
    --output-dir (Join-Path $analysisRoot "corruptions")
if ($LASTEXITCODE -ne 0) {
    throw "Corruption statistics failed with exit code $LASTEXITCODE"
}

