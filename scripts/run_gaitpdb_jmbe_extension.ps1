param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $projectRoot "src"
$processed = Join-Path $projectRoot `
    "data\processed\gaitpdb_clinical_windows.npz"
$artifactRoot = Join-Path $projectRoot `
    "artifacts\gaitpdb_jmbe_extension"

& $Python -m gait_robust.run_gaitpdb_clinical `
    --data $processed `
    --output-dir $artifactRoot `
    --split-seed 20260706 `
    --seeds 101 102 103 104 105 106 107 108 109 110 `
    --methods baseline generic iid fasca `
    --epochs 30 `
    --patience 6 `
    --batch-size 64 `
    --learning-rate 0.001 `
    --weight-decay 0.0001 `
    --windows-per-subject 16 `
    --device auto

if ($LASTEXITCODE -ne 0) {
    throw "JMBE extension training failed with exit code $LASTEXITCODE"
}

& $Python -m gait_robust.analyze_gaitpdb_clinical `
    --predictions (Join-Path $artifactRoot "subject_predictions.csv.gz") `
    --data $processed `
    --output-dir (Join-Path $artifactRoot "analysis") `
    --bootstrap-samples 20000 `
    --bootstrap-seed 20260706

if ($LASTEXITCODE -ne 0) {
    throw "JMBE extension analysis failed with exit code $LASTEXITCODE"
}

& $Python -m gait_robust.benchmark_gaitpdb_model `
    --output-dir (Join-Path $artifactRoot "engineering")

if ($LASTEXITCODE -ne 0) {
    throw "Engineering benchmark failed with exit code $LASTEXITCODE"
}

