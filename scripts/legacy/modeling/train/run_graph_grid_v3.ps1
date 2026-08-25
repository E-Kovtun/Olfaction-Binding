param(
    [int]$Epochs = 1500,
    [int]$Throttle = 3
)

# graph_sweep_v3 — main architecture/MP-mode grid on the stable recipe.
#
#   {gnn, gat} x {pos_only, all_edges, signed} x {transductive, inductive_molecule}
#   = 12 runs (all 6 GNN first, then all 6 GAT).
#
# Same collapse-free recipe as graph_rerun_v3 (lr 1e-3, grad-clip 1.0,
# ReduceLROnPlateau; best_state restored before probe). Runs up to $Throttle
# jobs concurrently: --deterministic pins Torch to 1 thread, so N single-thread
# jobs map onto N of the 4 physical cores. A failed job does NOT abort the rest.
#
# The two gnn-signed base runs are pre-seeded from graph_rerun_v3 (identical
# config/seed/recipe) and skipped via the checkpoint-exists check.

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$Trainer = Join-Path $Root "scripts\modeling\train\train_graph_full_full.py"
$ResultDir = "results/graph/full_full/legacy/pre_v5/sweep_v3/full_full"
$RunRoot = Join-Path $Root $ResultDir
$LogDir = Join-Path $RunRoot "logs"
$CheckpointDir = Join-Path $RunRoot "checkpoints"
$StatusPath = Join-Path $RunRoot "queue_status.csv"

New-Item -ItemType Directory -Force -Path $LogDir, $CheckpointDir | Out-Null

$Stable = @(
    "--lr", "1e-3",
    "--grad-clip", "1.0",
    "--lr-scheduler",
    "--scheduler-patience", "100",
    "--scheduler-factor", "0.5",
    "--scheduler-min-lr", "1e-5",
    "--diagnostics",
    "--deterministic"
)

# GNN block first, then GAT.
$runs = @()
foreach ($arch in @("gnn", "gat")) {
    foreach ($mp in @("pos_only", "all_edges", "signed")) {
        foreach ($reg in @("transductive", "inductive_molecule")) {
            $runs += [pscustomobject]@{
                Id         = "${arch}_${mp}_${reg}"
                Args       = @("--arch", $arch, "--mp_mode", $mp, "--regime", $reg)
                Checkpoint = "${arch}_${mp}_unentangled_boost_${reg}_fold1.pt"
            }
        }
    }
}

$queue = [System.Collections.Generic.Queue[object]]::new()
$runs | ForEach-Object { $queue.Enqueue($_) }
$running = @{}
$results = @()

foreach ($run in $runs) {
    $ckptPath = Join-Path $CheckpointDir $run.Checkpoint
    if (Test-Path $ckptPath) {
        # Pre-seeded / already done: record and drop from the work queue.
        $results += [pscustomobject]@{
            run_id = $run.Id; state = "cached"; started = ""
            finished = (Get-Date).ToString("o"); seconds = 0; exit_code = 0
            checkpoint = $run.Checkpoint
        }
    }
}
$results | Export-Csv -NoTypeInformation -Encoding UTF8 -LiteralPath $StatusPath

while ($queue.Count -gt 0 -or $running.Count -gt 0) {
    while ($running.Count -lt $Throttle -and $queue.Count -gt 0) {
        $run = $queue.Dequeue()
        $ckptPath = Join-Path $CheckpointDir $run.Checkpoint
        if (Test-Path $ckptPath) { continue }   # cached (already recorded above)

        $logPath = Join-Path $LogDir ($run.Id + ".log")
        $errPath = Join-Path $LogDir ($run.Id + ".err.log")
        $runArgs = @($Trainer) + $run.Args + $Stable + @(
            "--fold", "1",
            "--epochs", "$Epochs",
            "--results-dir", $ResultDir,
            "--log-every", "10",
            "--plot-every", "25"
        )
        $p = Start-Process -FilePath $Python -ArgumentList $runArgs `
            -WorkingDirectory $Root -RedirectStandardOutput $logPath `
            -RedirectStandardError $errPath -WindowStyle Hidden -PassThru
        $running[$run.Id] = [pscustomobject]@{ proc = $p; run = $run; started = (Get-Date) }
    }

    Start-Sleep -Seconds 15

    foreach ($id in @($running.Keys)) {
        $entry = $running[$id]
        if ($entry.proc.HasExited) {
            $ckptPath = Join-Path $CheckpointDir $entry.run.Checkpoint
            # The .venv python shim makes ExitCode unreliable; the trainer writes
            # the checkpoint only on a successful finish, so presence == success.
            $ec = try { $entry.proc.ExitCode } catch { $null }
            $state = if (Test-Path $ckptPath) { "completed" } else { "failed" }
            $results += [pscustomobject]@{
                run_id = $id; state = $state
                started = $entry.started.ToString("o")
                finished = (Get-Date).ToString("o")
                seconds = [math]::Round(((Get-Date) - $entry.started).TotalSeconds, 1)
                exit_code = $ec; checkpoint = $entry.run.Checkpoint
            }
            $results | Export-Csv -NoTypeInformation -Encoding UTF8 -LiteralPath $StatusPath
            $running.Remove($id)
        }
    }
}

Write-Host "grid sweep complete: $($results.Count) runs recorded"
