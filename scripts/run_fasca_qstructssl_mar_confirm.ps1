param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $projectRoot "src"

& $Python -m gait_robust.rapid_distill `
    --data (Join-Path $projectRoot "data\processed\windows_v2.npz") `
    --output-dir (Join-Path $projectRoot "artifacts\fasca_qstructssl_mar_confirm") `
    --teacher-root (Join-Path $projectRoot "artifacts\xtinyhar_confirmatory\distillation") `
    --methods embracenet_fasca_qstructssl_mar_kd `
    --folds 5 `
    --seeds 52 53 `
    --partition-seed 20260901 `
    --epochs-student 35 `
    --patience 8 `
    --batch-size 128 `
    --learning-rate 0.001 `
    --weight-decay 0.0001 `
    --temperature 3 `
    --alpha 0.2 `
    --augmentation-probability 0.35 `
    --severity-min 0.15 `
    --severity-max 0.75 `
    --desync-probability 0 `
    --lambda-clean 0.50 `
    --lambda-ssl 0.10 `
    --ssl-ema-decay 0.99 `
    --ssl-projector-dim 64 `
    --ssl-structured-probability 0.70 `
    --ssl-structured-severity-min 0.35 `
    --ssl-structured-severity-max 0.75 `
    --corruption-selection-weight 0.10 `
    --augmentation-profile drop_gain `
    --augmentation-warmup-epochs 5 `
    --mar-warmup-epochs 5 `
    --mar-rho 1.0 `
    --mar-weight-floor 0.25

if ($LASTEXITCODE -ne 0) {
    throw "FASCA-QStructSSL-MAR confirmation failed with exit code $LASTEXITCODE"
}

