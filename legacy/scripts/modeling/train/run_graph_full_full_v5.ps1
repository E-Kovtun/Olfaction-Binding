param(
    [int]$StartAt = 1,
    [int]$StopAfter = 0
)

$ErrorActionPreference = "Stop"
$ResultDir = "results/graph/full_full/v5/architecture_screen"
$LogDir = Join-Path $ResultDir "training/logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

# Three genuine LORAX folds for transductive; three independent cold-molecule
# seeds for inductive (fold 1 is only the source container in that regime).
# GNN and XGBoost always receive distinct seeds.
$RepeatSpecs = @(
    @{ Regime = "transductive";       Fold = 1; Gnn = 42; Boost = 1042 },
    @{ Regime = "transductive";       Fold = 2; Gnn = 43; Boost = 1043 },
    @{ Regime = "transductive";       Fold = 3; Gnn = 44; Boost = 1044 },
    @{ Regime = "inductive_molecule"; Fold = 1; Gnn = 42; Boost = 1042 },
    @{ Regime = "inductive_molecule"; Fold = 1; Gnn = 43; Boost = 1043 },
    @{ Regime = "inductive_molecule"; Fold = 1; Gnn = 44; Boost = 1044 }
)
$Architectures = @("gnn", "gat")
$MpModes = @("pos_only", "all_edges", "signed")

$runs = @()
foreach ($spec in $RepeatSpecs) {
    foreach ($arch in $Architectures) {
        foreach ($mp in $MpModes) {
            $runs += [pscustomobject]@{
                Regime = $spec.Regime; Fold = $spec.Fold; Arch = $arch; Mp = $mp
                GnnSeed = $spec.Gnn; BoostSeed = $spec.Boost
            }
        }
    }
}

Write-Host "full_full v5: $($runs.Count) runs; cache misses use lr=3e-3, clip=1.0; 900 epochs; last-epoch probe"
$completedThisCall = 0
for ($i = $StartAt - 1; $i -lt $runs.Count; $i++) {
    if ($StopAfter -gt 0 -and $completedThisCall -ge $StopAfter) { break }
    $r = $runs[$i]
    $stem = "$($r.Arch)_$($r.Mp)_$($r.Regime)_fold$($r.Fold)_gnn$($r.GnnSeed)_boost$($r.BoostSeed)"
    $artifact = Join-Path $ResultDir "training/checkpoints\$($r.Arch)_$($r.Mp)_unentangled_boost_$($r.Regime)_fold$($r.Fold)_gnn$($r.GnnSeed)_boost$($r.BoostSeed).pt"
    if (Test-Path -LiteralPath $artifact) {
        Write-Host "SKIP [$($i+1)/$($runs.Count)] $stem"
        continue
    }
    $log = Join-Path $LogDir "$stem.log"
    Write-Host "RUN  [$($i+1)/$($runs.Count)] $stem -> $log"
    & uv run python legacy/scripts/modeling/train/train_graph_full_full.py `
        --arch $r.Arch --mp_mode $r.Mp --regime $r.Regime --fold $r.Fold `
        --epochs 900 --lr 3e-3 --grad-clip 1.0 --device auto --seed $r.GnnSeed --boost-seed $r.BoostSeed `
        --probe-checkpoint last --results-dir (Join-Path $ResultDir "training") 2>&1 | Tee-Object -FilePath $log
    if ($LASTEXITCODE -ne 0) { throw "v5 run failed: $stem (exit $LASTEXITCODE)" }
    $completedThisCall++
}
Write-Host "v5 queue call finished; completed now: $completedThisCall"
