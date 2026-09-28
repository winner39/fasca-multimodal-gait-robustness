param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $projectRoot "src"
$sslRoot = Join-Path $projectRoot "artifacts\fasca_ssl_confirmatory"
$baselineRoot = Join-Path $projectRoot "artifacts\fasca_factorial_confirmatory"
$corruptionRoot = Join-Path $projectRoot "artifacts\fasca_ssl_corruption_stats"
$analysisRoot = Join-Path $projectRoot "artifacts\fasca_ssl_analysis"
New-Item -ItemType Directory -Force $corruptionRoot, $analysisRoot | Out-Null

& $Python -m gait_robust.analyze_fasca_factorial `
    --prediction-roots $baselineRoot $sslRoot `
    --include-methods `
        rapid_embracenet_fasca_kd `
        rapid_embracenet_fasca_ssl_kd `
    --reference-method rapid_embracenet_fasca_ssl_kd `
    --output-dir (Join-Path $analysisRoot "missing_modalities")
if ($LASTEXITCODE -ne 0) {
    throw "Missing-modality analysis failed with exit code $LASTEXITCODE"
}

$sslCorruption = Join-Path $corruptionRoot "embracenet_fasca_ssl.csv"
$sslPredictions = Join-Path $corruptionRoot "embracenet_fasca_ssl_predictions.csv.gz"
& $Python -m gait_robust.evaluate_corruptions `
    --data (Join-Path $projectRoot "data\processed\windows_v2.npz") `
    --run-root $sslRoot `
    --method embracenet_fasca_ssl_kd `
    --model-type embracenet `
    --output $sslCorruption `
    --prediction-output $sslPredictions `
    --folds 5 `
    --seeds 51 52 53 `
    --partition-seed 20260901
if ($LASTEXITCODE -ne 0) {
    throw "Corruption evaluation failed with exit code $LASTEXITCODE"
}

& $Python -m gait_robust.analyze_main_corruption_statistics `
    --predictions `
        (Join-Path $projectRoot "artifacts\fasca_architecture_corruption_stats\embracenet_fasca_predictions.csv.gz") `
        $sslPredictions `
    --reference-method embracenet_fasca_ssl_kd `
    --output-dir (Join-Path $analysisRoot "corruptions")
if ($LASTEXITCODE -ne 0) {
    throw "Corruption statistics failed with exit code $LASTEXITCODE"
}

