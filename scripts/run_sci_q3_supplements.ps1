param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $projectRoot "src"
$data = Join-Path $projectRoot "data\processed\windows_v2.npz"
$teacher = Join-Path $projectRoot "artifacts\xtinyhar_confirmatory\distillation"

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

Invoke-CheckedPython -m gait_robust.rapid_distill `
    --output-dir (Join-Path $projectRoot "artifacts\fasca_factorial_confirmatory") `
    --methods moddrop_fasca_kd embracenet_fasca_kd uniform_rapid_fasca_kd `
    --augmentation-profile drop_gain `
    --augmentation-warmup-epochs 5 `
    @common

Invoke-CheckedPython -m gait_robust.rapid_distill `
    --output-dir (Join-Path $projectRoot "artifacts\fasca_no_curriculum_confirmatory") `
    --methods sensor_aug_no_curriculum_kd `
    --augmentation-profile drop_gain `
    --augmentation-warmup-epochs 0 `
    @common

Invoke-CheckedPython -m gait_robust.rapid_distill `
    --output-dir (Join-Path $projectRoot "artifacts\fasca_iid_budget_confirmatory") `
    --methods iid_aug_kd `
    --augmentation-profile iid_drop_gain `
    --augmentation-warmup-epochs 5 `
    @common

$statisticsRoot = Join-Path $projectRoot "artifacts\main_corruption_subject_stats"
$evaluationRuns = @(
    @{
        Name = "fasca"
        Root = "artifacts\sensor_aug_dev_v2_focus"
        Method = "sensor_aug_kd"
        Model = "rapid"
    },
    @{
        Name = "embracenet"
        Root = "artifacts\multimodel_confirmatory\main"
        Method = "embracenet"
        Model = "embracenet"
    },
    @{
        Name = "actionmae"
        Root = "artifacts\multimodel_confirmatory\main"
        Method = "actionmae"
        Model = "actionmae"
    }
)

foreach ($run in $evaluationRuns) {
    Invoke-CheckedPython -m gait_robust.evaluate_corruptions `
        --data $data `
        --run-root (Join-Path $projectRoot $run.Root) `
        --method $run.Method `
        --model-type $run.Model `
        --output (Join-Path $statisticsRoot "$($run.Name).csv") `
        --prediction-output (Join-Path $statisticsRoot "$($run.Name)_predictions.csv.gz") `
        --folds 5 `
        --seeds 51 52 53 `
        --partition-seed 20260901 `
        --batch-size 128
}

Invoke-CheckedPython -m gait_robust.analyze_main_corruption_statistics `
    --predictions `
        (Join-Path $statisticsRoot "fasca_predictions.csv.gz") `
        (Join-Path $statisticsRoot "embracenet_predictions.csv.gz") `
        (Join-Path $statisticsRoot "actionmae_predictions.csv.gz") `
    --reference-method sensor_aug_kd `
    --output-dir (Join-Path $statisticsRoot "analysis") `
    --bootstrap-repeats 20000 `
    --bootstrap-seed 20260904

Invoke-CheckedPython -m gait_robust.analyze_fasca_factorial `
    --prediction-roots `
        (Join-Path $projectRoot "artifacts\sensor_aug_dev_v2_focus") `
        (Join-Path $projectRoot "artifacts\fasca_factorial_confirmatory") `
        (Join-Path $projectRoot "artifacts\fasca_no_curriculum_confirmatory") `
        (Join-Path $projectRoot "artifacts\fasca_iid_budget_confirmatory") `
    --include-methods `
        rapid_sensor_aug_kd `
        rapid_moddrop_fasca_kd `
        rapid_embracenet_fasca_kd `
        rapid_uniform_rapid_fasca_kd `
        rapid_sensor_aug_no_curriculum_kd `
        rapid_iid_aug_kd `
    --reference-method rapid_sensor_aug_kd `
    --output-dir (Join-Path $projectRoot "artifacts\fasca_sci_q3_analysis") `
    --bootstrap-repeats 20000 `
    --bootstrap-seed 20260905

