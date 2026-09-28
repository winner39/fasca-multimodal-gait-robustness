param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $projectRoot "src"
$rawDir = Join-Path $projectRoot `
    "data\raw\gaitpdb_full\gait-in-parkinsons-disease-1.0.0"
$processed = Join-Path $projectRoot `
    "data\processed\gaitpdb_clinical_windows.npz"
$artifactRoot = Join-Path $projectRoot `
    "artifacts\gaitpdb_clinical_external"

& $Python -m gait_robust.prepare_gaitpdb `
    --raw-dir $rawDir `
    --output $processed `
    --quality-dir (Join-Path $projectRoot "artifacts\gaitpdb_data_quality")

if ($LASTEXITCODE -ne 0) {
    throw "GaitPDB preprocessing failed with exit code $LASTEXITCODE"
}

& $Python -m gait_robust.run_gaitpdb_clinical `
    --data $processed `
    --output-dir $artifactRoot `
    --split-seed 20260706 `
    --seeds 101 102 103 `
    --methods baseline fasca `
    --epochs 30 `
    --patience 6 `
    --batch-size 64 `
    --learning-rate 0.001 `
    --weight-decay 0.0001 `
    --windows-per-subject 16 `
    --device auto

if ($LASTEXITCODE -ne 0) {
    throw "GaitPDB clinical training failed with exit code $LASTEXITCODE"
}

& $Python -m gait_robust.analyze_gaitpdb_clinical `
    --predictions (Join-Path $artifactRoot "subject_predictions.csv.gz") `
    --data $processed `
    --output-dir (Join-Path $artifactRoot "analysis") `
    --bootstrap-samples 20000 `
    --bootstrap-seed 20260706

if ($LASTEXITCODE -ne 0) {
    throw "GaitPDB clinical analysis failed with exit code $LASTEXITCODE"
}

