param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $projectRoot "src"
$dataRoot = Join-Path $projectRoot "data"
$processedRoot = Join-Path $dataRoot "processed"
$datasetRoot = Join-Path $dataRoot "multimodal-gait-dataset-1.0.0\a-multimodal-gait-dataset-of-brain-activity-muscle-activity-kinematics-and-ground-forces-in-young-adults-1.0.0\Dataset"
$completeData = Join-Path $processedRoot "windows_v2.npz"
$observedData = Join-Path $processedRoot "observed_acquisition_failures.npz"
$outputRoot = Join-Path $projectRoot "artifacts\observed_acquisition_failures"

New-Item -ItemType Directory -Force $outputRoot | Out-Null

& $Python -m gait_robust.prepare_observed_failures `
    --dataset-root $datasetRoot `
    --complete-data $completeData `
    --output $observedData
if ($LASTEXITCODE -ne 0) {
    throw "Observed-failure preparation failed with exit code $LASTEXITCODE"
}

& $Python -m gait_robust.evaluate_observed_failures `
    --complete-data $completeData `
    --observed-data $observedData `
    --baseline-root (Join-Path $projectRoot "artifacts\multimodel_confirmatory\main") `
    --fasca-root (Join-Path $projectRoot "artifacts\fasca_factorial_confirmatory") `
    --specialist-seed51-root (Join-Path $projectRoot "artifacts\fasca_qstructssl_mar_screen") `
    --specialist-confirm-root (Join-Path $projectRoot "artifacts\fasca_qstructssl_mar_confirm") `
    --output-dir $outputRoot `
    --folds 5 `
    --seeds 51 52 53 `
    --partition-seed 20260901
if ($LASTEXITCODE -ne 0) {
    throw "Observed-failure evaluation failed with exit code $LASTEXITCODE"
}

