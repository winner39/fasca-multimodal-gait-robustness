param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $projectRoot "src"
$data = Join-Path $projectRoot "data\processed\windows_v2.npz"
$teacher = Join-Path $projectRoot "artifacts\xtinyhar_confirmatory\distillation"
$structuredCurriculum = Join-Path $projectRoot "artifacts\sensor_aug_dev_v2_focus"
$structuredConstant = Join-Path $projectRoot "artifacts\fasca_no_curriculum_confirmatory"
$iidCurriculum = Join-Path $projectRoot "artifacts\fasca_iid_budget_confirmatory"
$iidConstant = Join-Path $projectRoot "artifacts\fasca_iid_no_curriculum_confirmatory"
$experimentRoot = Join-Path $projectRoot "artifacts\fasca_distribution_factorial_rapid"
$evaluationRoot = Join-Path $experimentRoot "evaluation"

function Invoke-CheckedPython {
    & $Python @args
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE"
    }
}

$common = @(
    "--data", $data,
    "--teacher-root", $teacher,
    "--folds", "5",
    "--seeds", "51", "52", "53",
    "--partition-seed", "20260901",
    "--epochs-student", "35",
    "--patience", "8",
    "--batch-size", "128",
    "--learning-rate", "0.001",
    "--weight-decay", "0.0001",
    "--temperature", "3",
    "--alpha", "0.2",
    "--augmentation-probability", "0.35",
    "--severity-min", "0.15",
    "--severity-max", "0.75",
    "--desync-probability", "0",
    "--lambda-clean", "0.50",
    "--corruption-selection-weight", "0.10"
)

# Every call is resumable: existing fold-seed predictions are reused, while a
# clean workspace trains all four cells under the same command record.
Invoke-CheckedPython -m gait_robust.rapid_distill `
    --output-dir $structuredCurriculum `
    --methods sensor_aug_kd `
    --augmentation-profile drop_gain `
    --augmentation-warmup-epochs 5 `
    @common

Invoke-CheckedPython -m gait_robust.rapid_distill `
    --output-dir $structuredConstant `
    --methods sensor_aug_no_curriculum_kd `
    --augmentation-profile drop_gain `
    --augmentation-warmup-epochs 0 `
    @common

Invoke-CheckedPython -m gait_robust.rapid_distill `
    --output-dir $iidCurriculum `
    --methods iid_aug_kd `
    --augmentation-profile iid_drop_gain `
    --augmentation-warmup-epochs 5 `
    @common

Invoke-CheckedPython -m gait_robust.rapid_distill `
    --output-dir $iidConstant `
    --methods iid_aug_kd `
    --augmentation-profile iid_drop_gain `
    --augmentation-warmup-epochs 0 `
    @common

New-Item -ItemType Directory -Force -Path $evaluationRoot | Out-Null
$cells = @(
    @{
        Name = "structured_curriculum"
        Root = $structuredCurriculum
        Method = "sensor_aug_kd"
    },
    @{
        Name = "structured_constant"
        Root = $structuredConstant
        Method = "sensor_aug_no_curriculum_kd"
    },
    @{
        Name = "iid_curriculum"
        Root = $iidCurriculum
        Method = "iid_aug_kd"
    },
    @{
        Name = "iid_constant"
        Root = $iidConstant
        Method = "iid_aug_kd"
    }
)

foreach ($cell in $cells) {
    Invoke-CheckedPython -m gait_robust.evaluate_corruptions `
        --data $data `
        --run-root $cell.Root `
        --method $cell.Method `
        --model-type rapid `
        --output-method $cell.Name `
        --output (Join-Path $evaluationRoot "$($cell.Name).csv") `
        --prediction-output (Join-Path $evaluationRoot "$($cell.Name)_predictions.csv.gz") `
        --folds 5 `
        --seeds 51 52 53 `
        --partition-seed 20260901 `
        --batch-size 128
}

Invoke-CheckedPython -m gait_robust.analyze_distribution_factorial `
    --experiment-root $experimentRoot `
    --cell structured_curriculum $structuredCurriculum sensor_aug_kd `
    --cell structured_constant $structuredConstant sensor_aug_no_curriculum_kd `
    --cell iid_curriculum $iidCurriculum iid_aug_kd `
    --cell iid_constant $iidConstant iid_aug_kd `
    --seeds 51 52 53 `
    --bootstrap-repeats 20000 `
    --bootstrap-seed 20260710

