# Research landscape — OR–odorant protein–molecule interaction

> Living notes / "landscape" to return to. Captures every forward direction we
> discussed, mine and the user's, **with the user's comments inline**.
> Status as of this writing; not a commitment, a map.

## Goal & evaluation modes

- **Goal:** a competent **protein–molecule interaction** model for olfaction — not a
  single target number. Reference baseline `ESM-mean ‖ GIN → XGBoost`; numbers are
  reference points, not the objective.
- The human OR repertoire is **fixed/known** (~400) → cold-**receptor** generalization
  (`group_receptor`) is **not really the olfactory task** → de-emphasized.
- Relevant modes: **group_molecule** (new odorants, known receptors — **our main bet**)
  and **stratified** (matrix completion; other papers' default). Reference numbers:
  group_molecule ~0.78 AUROC, stratified ~0.88; marginal `prod` ~0.75.

## Two ceilings that recur across the whole map

1. **Identity / cold-receptor wall.** Receptor side ≈ an identity key; models read
   "which receptor" more than "what it binds". Mainly bites the de-emphasized
   `group_receptor` regime (less relevant since the OR repertoire is fixed/known), but
   the deeper point — that ESM encodes identity, not interaction — still shapes 1a/2b.
2. **binding ≠ activation.** Docking/affinity data predict *binding*; M2OR labels are
   *activation* (agonism). Hits 2a, 3a/3f. Mitigated by active-state/differential
   docking (3c/3d/3e) or by switching the training label to functional (4a, 2b).

---

## 1. Change the model on M2OR (beat concat+tree directly)

- **1a. Cross-attention interaction head on FROZEN features.** per-residue ESM ×
  molecule atoms/substructures → attention → binding head; encoders frozen. Sharp
  test of "interaction vs concat" on the same features. *Feasible on CPU now.*
  **[user: this is very close to the MolOR paper.]**
- **1b. End-to-end fine-tune encoders** (LoRA on ESM + trainable molecule encoder) +
  interaction head. Highest ceiling, classic way to beat a frozen-feature baseline;
  overfitting risk on sparse M2OR; **needs GPU.**
  **[user: this is close to LORAX.]**
- **1d. Upgrade the MOLECULE side** (replace frozen GIN with a trainable molecular
  encoder). Receptor side is saturated; headroom likely lives on the molecule
  (random-receptor floor on group_molecule = 0.556). *(user referred to this as
  "1c".)* **[user: I like this — we struggled with the protein for so long, maybe
  it's time to deal with the molecule.]**

---

## 2. Pretrain → fine-tune (transfer). Differ by CORPUS.

> **[user, applies to ALL of point 2]:** test each in TWO variations —
> **(i) zero-shot** (pretrained model applied to M2OR with NO fine-tuning) and
> **(ii) fine-tuned on M2OR**. Maybe zero-shot transfer is already decent — worth
> knowing how much the pretraining alone buys before paying for fine-tuning.

### 2a. Build our OWN **geometry-based interaction model** (geometry is the whole point)

**[user: the earlier phrasing buried this — the idea is literally to create a
geometry-based receptor↔ligand interaction model; that it may be hard / low-yield
is a separate question.]**

Core idea — make 3D geometry first-class, end to end:
- **Geometric inputs.** Receptor pocket in 3D (AF2 or, where it exists, experimental
  active-state, e.g. OR51E2): pocket-residue atoms / side chains / a pocket point
  cloud. Ligand in 3D: a conformer or a docked pose in that pocket.
- **Geometric representation.** Explicit pocket-atom ↔ ligand-atom **distances /
  contacts**; or an equivariant encoding (E(3)/SE(3)-equivariant GNN, GVP, or a
  3D-grid CNN over the complex à la gnina, or AlphaFold-style pair representation
  with distance bias). The interaction module *consumes geometry*, not pooled
  vectors — this is the difference from everything we tried under concat.
- **Pretraining signal.** Validated structural binding data (**PDBbind / FEP /
  DUD-E-class**) — affinity and/or pose quality — so the module learns "how a 3D
  pocket and a 3D ligand couple".
- **Then (two variations, per user):** (i) zero-shot to OR+odorant (build a pose
  first), (ii) fine-tune on M2OR (geometric inputs from AF2 receptor + generated/
  docked odorant poses; labels = M2OR activation).

*Caveats (acknowledged, separate from whether the idea is right):* structural data
are **OOD** for GPCR/OR (PDBbind ≈ soluble enzymes); the **binding≠activation**
ceiling still applies; hard and possibly low-yield. *Needs GPU.*

### 2b. Sequence-based PCM / DTI (NO geometry)

Pretrain **ESM (sequence) + molecule encoder + interaction head** on a large **GPCR
activity** corpus (**GLASS / Papyrus / ChEMBL-GPCR**) → then M2OR. No structures
needed — sidesteps the structure scarcity entirely; same domain (GPCR), millions of
pairs. This is mature ground (proteochemometrics / DeepDTA / GraphDTA / ESM-DTI).
**[user: this no-geometry path is genuinely much more feasible.]**
*Two variations:* zero-shot transfer vs fine-tuned on M2OR.
*Ceilings:* cold-target (new receptor) + OR-OOD (ORs are the divergent edge of class A).

> **2a vs 2b in one line:** same "pretrain→finetune", but 2a uses scarce **structural,
> geometric** data (hard, OOD, but real geometry), 2b uses abundant **sequence-
> activity** data (feasible, no geometry).

---

## 3. Structural / physics "teacher" (the AF/docking direction) — a fidelity ladder

> **[user: leave point 3 without comments for now.]**
> Status: paused at the positive control. Vina v1.2.7 + AF2 pockets pipeline built
> and isolated under `legacy/experiments/struct_interaction/`. Positive control (8 ORs,
> per-receptor AUROC of −affinity vs M2OR label): mean ~0.57, Stouffer p=0.012 —
> a *weak/honest-but-noisy* teacher; only OR1A1 individually clear.

- **3a. Zero-shot docking → feature** in XGBoost (our crude version; ~0.57, orthogonal
  to identity).
- **3b. Global Stage-0** — scale per-receptor docking enrichment to 30–40 receptors
  for a population estimate with CI. *(paused)*
- **3c. Dock into the ACTIVE-state structure** instead of apo-AF2 (OR51E2 has one).
  Fixes the conformation half of binding≠activation, not the endpoint half.
- **3d. Differential docking ΔΔG (active − inactive)** — theory-grounded efficacy
  proxy (two-state model: agonist prefers the active pocket).
- **3e. MD / enhanced sampling** — estimate the active-state population shift a
  ligand induces (TM6, NPxxY); mechanistic, expensive.
- **3f. Use a high-quality pretrained scorer** (gnina CNN / co-folding confidence)
  instead of crude Vina, as feature or backbone. *Needs GPU/Linux/WSL.*

---

## 4. Attack the bottleneck — data & endpoint

> **[user: will look at 4 and 5 later.]**

- **4a. OR mutagenesis data** (single residue → Δactivation): direct *activation*
  endpoint, **phylogeny-decoupled** — the cleanest test of "does the pocket matter".
  Data mining, not compute.
- **4b. Pool deorphanization datasets** — more ORs/molecules, denser overlap; eases
  the sparse / saturated-target problem.
- **4c. Active-state OR structures as they appear** (OR51E2 family) — opens the
  structural path for a handful of ORs.

## 5. Learning setup / evaluation (orthogonal; honesty + sometimes metrics)

- **5a. PU-learning / better negatives** — M2OR "negatives" are biased
  "tested-negative", not true zeros; proper PU framing may lift **AUPRC**.
- **5b. Evaluation discipline** — **group_molecule** as the primary mode (+ stratified);
  group_receptor de-emphasized (OR repertoire is fixed/known). Report **AUPRC + gap to
  marginal baselines** (not just AUROC); **group-level bootstrap** CIs (over
  receptors/molecules, not pairs).

---

### Cross-cutting open question

GPU/Linux availability gates 1b, 2a, 2b(heavy), 3f. CPU-feasible now: 1a, 1d(light),
3a, and all of 5. Resolve this to pick the first concrete experiment.
