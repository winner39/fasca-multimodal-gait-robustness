$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = "python"
$env:PYTHONPATH = Join-Path $Root "src"
$RunRoot = Join-Path $Root "artifacts\xtinyhar_confirmatory"
$StatusPath = Join-Path $RunRoot "pipeline-status.json"
New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null

function Write-Status {
    param(
        [string]$Stage,
        [string]$Message
    )
    @{
        stage = $Stage
        message = $Message
        updated_at = (Get-Date).ToString("o")
        partition_seed = 20260901
        folds = 5
        seeds = @(51, 52, 53)
    } | ConvertTo-Json | Set-Content -LiteralPath $StatusPath -Encoding UTF8
}

function Invoke-Python {
    param(
        [string[]]$Arguments,
        [string]$Stdout,
        [string]$Stderr
    )
    $Process = Start-Process `
        -FilePath $Python `
        -ArgumentList $Arguments `
        -WorkingDirectory $Root `
        -RedirectStandardOutput $Stdout `
        -RedirectStandardError $Stderr `
        -WindowStyle Hidden `
        -Wait `
        -PassThru
    if ($Process.ExitCode -ne 0) {
        throw "Python process exited with code $($Process.ExitCode)"
    }
}

try {
    Set-Location -LiteralPath $Root
    Write-Status "xtinyhar_baselines" "Running XTinyHAR and modality-dropout confirmatory validation"
    Invoke-Python `
        -Arguments @(
            "-m", "gait_robust.sci_cross_validate",
            "--data", "data\processed\windows_v2.npz",
            "--output-dir", "artifacts\xtinyhar_confirmatory\baselines",
            "--methods", "xtinyhar", "xtinyhar_dropout",
            "--folds", "5",
            "--seeds", "51", "52", "53",
            "--partition-seed", "20260901",
            "--epochs", "40",
            "--patience", "8"
        ) `
        -Stdout (Join-Path $RunRoot "baselines.out.log") `
        -Stderr (Join-Path $RunRoot "baselines.err.log")

    Write-Status "teacher_distillation" "Running original and selective teacher distillation confirmatory validation"
    Invoke-Python `
        -Arguments @(
            "-m", "gait_robust.xtinyhar_distill",
            "--data", "data\processed\windows_v2.npz",
            "--output-dir", "artifacts\xtinyhar_confirmatory\distillation",
            "--methods", "original_kd", "selective_kd",
            "--folds", "5",
            "--seeds", "51", "52", "53",
            "--partition-seed", "20260901",
            "--epochs-teacher", "50",
            "--epochs-student", "50",
            "--patience", "10"
        ) `
        -Stdout (Join-Path $RunRoot "distillation.out.log") `
        -Stderr (Join-Path $RunRoot "distillation.err.log")

    Write-Status "complete" "XTinyHAR confirmatory baselines and teacher distillation completed"
}
catch {
    Write-Status "failed" $_.Exception.Message
    throw
}

