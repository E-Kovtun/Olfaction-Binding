param(
    [int]$Epochs = 1500
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$Trainer = Join-Path $Root "scripts\modeling\train\train_graph_full_full.py"
$ResultDir = "results/graph/full_full/legacy/pre_v5/rerun_v2/full_full"
$RunRoot = Join-Path $Root $ResultDir
$LogDir = Join-Path $RunRoot "logs"
$CheckpointDir = Join-Path $RunRoot "checkpoints"
$StatusPath = Join-Path $RunRoot "queue_status.csv"
$EventPath = Join-Path $RunRoot "latest_event.json"

New-Item -ItemType Directory -Force -Path $LogDir, $CheckpointDir | Out-Null

$runs = @(
    @{
        Id = "01_gnn_transductive_raw_shared"
        Args = @("--arch", "gnn", "--mp_mode", "signed", "--regime", "transductive")
        Checkpoint = "gnn_signed_unentangled_boost_transductive_fold1.pt"
    },
    @{
        Id = "02_gnn_transductive_enriched_shared"
        Args = @("--arch", "gnn", "--mp_mode", "signed", "--regime", "transductive", "--transductive-exp")
        Checkpoint = "gnn_signed_transductive_exp_unentangled_boost_transductive_fold1.pt"
    },
    @{
        Id = "03_gnn_transductive_raw_disjoint"
        Args = @("--arch", "gnn", "--mp_mode", "signed", "--regime", "transductive", "--disjoint-probe-train")
        Checkpoint = "gnn_signed_disjoint_unentangled_boost_transductive_fold1.pt"
    },
    @{
        Id = "04_gnn_inductive_raw_shared"
        Args = @("--arch", "gnn", "--mp_mode", "signed", "--regime", "inductive_molecule")
        Checkpoint = "gnn_signed_unentangled_boost_inductive_molecule_fold1.pt"
    },
    @{
        Id = "05_gnn_inductive_raw_disjoint"
        Args = @("--arch", "gnn", "--mp_mode", "signed", "--regime", "inductive_molecule", "--disjoint-probe-train")
        Checkpoint = "gnn_signed_disjoint_unentangled_boost_inductive_molecule_fold1.pt"
    }
)

$statuses = @()
if (Test-Path $StatusPath) {
    $statuses = @(Import-Csv $StatusPath)
}

function Write-Event($runId, $state, $checkpoint, $message) {
    [pscustomobject]@{
        timestamp = (Get-Date).ToString("o")
        run_id = $runId
        state = $state
        checkpoint = $checkpoint
        message = $message
    } | ConvertTo-Json | Set-Content -Encoding UTF8 -LiteralPath $EventPath
}

foreach ($run in $runs) {
    $checkpointPath = Join-Path $CheckpointDir $run.Checkpoint
    if (Test-Path $checkpointPath) {
        Write-Event $run.Id "cached" $run.Checkpoint "Checkpoint already exists; skipped."
        continue
    }

    $started = Get-Date
    $logPath = Join-Path $LogDir ($run.Id + ".log")
    Write-Event $run.Id "running" $run.Checkpoint "Training started."

    $runArgs = @($run.Args) + @(
        "--fold", "1",
        "--epochs", "$Epochs",
        "--results-dir", $ResultDir,
        "--log-every", "10",
        "--plot-every", "25"
    )

    & $Python $Trainer @runArgs *> $logPath
    $exitCode = $LASTEXITCODE
    $finished = Get-Date
    $state = if ($exitCode -eq 0 -and (Test-Path $checkpointPath)) { "completed" } else { "failed" }

    $statuses += [pscustomobject]@{
        run_id = $run.Id
        state = $state
        started = $started.ToString("o")
        finished = $finished.ToString("o")
        seconds = [math]::Round(($finished - $started).TotalSeconds, 1)
        exit_code = $exitCode
        checkpoint = $run.Checkpoint
        log = (Resolve-Path $logPath).Path
    }
    $statuses | Export-Csv -NoTypeInformation -Encoding UTF8 -LiteralPath $StatusPath
    Write-Event $run.Id $state $run.Checkpoint "Training process exited with code $exitCode."

    if ($state -ne "completed") {
        exit $exitCode
    }
}

Write-Event "queue" "completed" "" "All graph rerun jobs completed."
