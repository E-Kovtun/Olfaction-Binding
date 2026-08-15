# Paper storyline — "the receptor side is underexplored" (ICLR 2027 draft plan)

Working sketch (Aug 2026, RU). Not the official doc. Thesis, section skeleton,
figure/table list, and the C9 information-criteria experiment battery.

## Central thesis (positioning via LORAX)
LORAX turbocharged the **molecule** side (LoRA over a chemical LM). We argue the
**receptor (protein) side is underexplored and underused**, though theory says it
carries no less. We (a) show this diagnostically and (b) give a **natural,
data-driven graph model that improves the receptor representation** from the
binding data itself — and the gain appears exactly where receptor-representation
quality matters most: cold (unseen) molecules.

One line: *"the molecule was already boosted with adapters; we show the receptor
is the undercultivated side, and that its representation can be improved not by a
new encoder but by binding data via a graph."*

We already have partial receptor-side evidence: ESM variants give ~equal boosting;
one-hot nearly matches ESM; ESM reads mostly as receptor identity + a weak
similarity prior. Missing: (1) head-to-head of protein embedding sources, (2)
information criteria of the receptor representation before/after our graph (=C9,
the killer experiment).

## Narrative arc (6 beats)
1. Olfactory binding is a special task — two-stage, rich history, but the standard
   binding toolkit is unavailable (no OR structures, weakly studied domain).
2. What's done instead — "domain foundation model + experimental data"; some works
   claim biological *mimicry* of the olfactory system, not just metrics.
3. The asymmetry — molecule side actively improved (LORAX etc.), receptor taken as
   a frozen off-the-shelf descriptor. That's the gap.
4. Our move — a graph that does not replace the receptor with a new encoder but
   refines it from binding data; naturally, visibly data-driven. Plus we tidy the
   evaluation (trivial boosting as an interpretable base; drop ensembling for a
   clean cls comparison on a common footing).
5. Evidence — inductive (cold-molecule) success on M2OR, robust across several
   molecule-embedding sources; fix the best (cite LORAX) and run everything on it,
   incl. transductive; transfer to Carey/Hallem; then the protein block: multiple
   protein-embedding sources + before/after representation analysis.
6. Closing — the gain is isolated in the protein side, and info criteria show the
   graph makes the receptor representation richer for the task → the receptor side
   really was underused, and it can be fixed with data.

## Section skeleton (ICLR, ~9 pp)
1. Introduction — hook (no structures → structural methods out); gap (receptor
   underexplored); contributions (diagnosis / graph refiner / clean protocol /
   isolated cold-molecule gain + info confirmation / insect transfer); teaser fig.
2. Background & Related Work — two-stage task + history; why structural/docking is
   out; the "foundation model + data" paradigm; competitors (ProSmith, LORAX,
   MolOR, Hladiš) each tagged by which side is active vs frozen; the biology-mimicry
   claim; bridge: receptor is passive everywhere → our angle.
3. Data & Evaluation regimes — M2OR; Carey/Hallem (continuous); transductive vs
   inductive-molecule; why cold-molecule is the real olfactory task; split-validity
   caveat → Supp.
4. Method — bipartite signed graph refining the receptor node from its ligand
   profile; interpretable boosting reader; molecule selection/quantile → Supp;
   scoring feature = [refined receptor ‖ raw molecule] (key to isolation).
5. Experimental setup — frozen encoders, graph training, fixed head, seeds/repeats,
   metrics; reproducible & short.
6. Protocol — trivial boosting as interpretable base (reads receptor identity);
   drop ensemble weighting for a per-source cls comparison on a common footing
   (same encoders/splits/head); rationale = attribution over mixtures.
7. Results — 7.1 M2OR cold-molecule + robustness over molecule sources, fix best
   (cite LORAX), + transductive; 7.2 Carey/Hallem transfer; 7.3 protein side:
   multiple protein sources head-to-head + before/after info criteria (C9).
8. Analysis — isolation (cls+mol vs prot+mol) + where/why it helps map + tie to C9.
9. Discussion/limits/future — data/target bottleneck; refinement is a lever not a
   silver bullet; molecule-side training, geometry, PCM pretraining, richer data.
10. Conclusion.
Supp — quantile×criterion sweeps per dataset; split validity; full per-regime
tables; the one-hot/frequency floor; competitor re-implementation details.

## Evidence ledger (claim → status → how)
C1 structural methods N/A — have (argument) — Related Work
C2 trivial boost strong, reads receptor identity — HAVE — protein-variants, one-hot vs ESM
C3 cold-molecule is the task; base beaten only by graph — HAVE — inductive head-to-head (5 seeds)
C4 gain isolated to protein refinement (cls+mol vs prot+mol) — HAVE — same series
C5 robust across several molecule embeddings — PARTIAL — need molecule-source sweep
C6 competitors don't beat base on cold molecule (common footing) — HAVE — re-impl ProSmith/LORAX/MolOR/Hladiš
C7 transfer to Carey/Hallem — IN PROGRESS — ourind runs, regression
C8 protein side underexplored: sources ≈ / underused — PARTIAL — need protein-source head-to-head
C9 graph actually IMPROVES the protein representation (info before/after) — TODO — the key new experiment
C10 method behaves sensibly across proteins — TODO — per-protein slice
Most important open = C9: isolation (C4) says "gain comes from protein refinement";
info criteria say "the representation became more informative" — different claims;
together they make the paper.

## Figures & tables (sections 7–8)
See chat message of this session for the annotated axis-by-axis list. Summary:
- Teaser Fig: receptor geometry before/after graph (phylo-colored vs function-colored).
- T1 M2OR cold-molecule head-to-head (methods × metrics); our cls+mol wins.
- T2 robustness across molecule-embedding sources (source × metric); gain holds.
- T3 transductive M2OR (base = ceiling; honest).
- F2 Carey/Hallem per-fold regression vs methods + naive floor.
- T4/F3 protein-embedding sources head-to-head; method behavior across proteins.
- F4 (C9 core) before/after info-criteria panel (profile recoverability, functional-
  kernel alignment, cold-molecule probe, phylo→function distance shift).
- F5 isolation bar (cls+mol vs prot+mol) + where-it-helps map (inductive vs transductive).
- Supp: quantile×criterion sweeps; split validity; the floor.

## C9 — information criteria before/after the graph (experiment battery)
Goal: show the graph-refined receptor rep is MORE informative for binding than raw
ESM — and localize that to cold-molecule transfer. Expect several nulls; the signal
should concentrate in profile-recoverability + functional-geometry + cold-transfer.

Shared objects (all on TRAIN edges; transfer probes use held-out molecules):
- R_raw (mean ESM-1b per receptor), R_ref (graph-refined z_prot per receptor).
- Controls: R_pca (PCA of R_raw to dim(R_ref)); R_rand (untrained/random-init graph);
  R_shuf (refined on label-permuted edges).
- K_func = cosine over signed train binding columns (functional sim);
  K_seq = ESM cosine / sequence identity (phylogenetic sim).

Battery (ordered cheap→expensive, most→least likely to fire):
1. Geometry shift phylo→function (CHEAP, do first). corr(dist_R, dist_seq) vs
   corr(dist_R, dist_func) for R_raw vs R_ref. Expect raw≈phylo, ref shifts toward
   function. kNN purity: fraction of a receptor's k-NN sharing ligand sets, raw vs ref.
2. Kernel-target alignment / dependence. KTA, CKA (linear+RBF), HSIC between receptor
   Gram and K_func (and K_seq) for raw vs ref. Expect ↑ alignment with K_func.
3. Ligand-profile recoverability (most direct). Decoder probe: from R[receptor]
   predict its binding profile over HELD-OUT molecules (multi-label). ref should
   recover profile better than raw. "the rep now knows its ligands."
4. Task-predictivity probes. Reader on [R[receptor] ‖ molecule] → binding, cold-
   molecule test; LINEAR (accessibility) + MLP + tree; raw vs ref vs controls.
   Localize transductive (expect ~no gain / drop) vs inductive (expect gain).
5. Information budget / compression. Effective dim (participation ratio, PCA
   spectrum) raw vs ref; identity retention (linear reconstruct R_raw from R_ref),
   residual = added functional component; information-plane (identity-info vs
   binding-info) before/after.
6. Mutual information with label (EXPLORATORY). MI(R[receptor]; binding | molecule)
   via KSG kNN-MI on probe residuals, or MINE/InfoNCE lower bounds; cheap proxy =
   MI(R, discretized binding profile).
Controls decide reality: gain must survive vs R_pca (not mere compression), and
die under R_shuf / R_rand (it's the learned functional structure). Trained-vs-random
also answers the open "does the GNN learn structure at all".

Expected headline pattern: generic predictivity may not rise (or fall in
transductive); signal concentrates in (3) profile recoverability, (1)/(2)
functional-geometry, (4) cold-transfer — triangulating the thesis.

## Reviewer attacks → where we stand
- "graph leaks test label" → isolation C4 + MP train-only + C9 on train reps.
- "gain within noise" → 5 seeds, ranks, sign test; honest effect size; direction+reproducibility.
- "why not ensemble" → attribution (compare representations, not mixtures).
- "underexplored = you just didn't try" → C8 (sources ≈) + C2 (identity) + C9 (positive info evidence).
