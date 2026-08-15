# Slide plan — internship results talk (whole-project overview)

Prepared Aug 2026. Audience = smart but outside; ~18–20 slides / 20–25 min; one
deck for both talks. Through-line: *a strong simple baseline; an honest map of
where it's a ceiling and where it isn't; and the one place we truly beat it — not
by imitating chemistry, but by using the relationships in the data.*

Rules: one "hero" number per slide max; conclusions phrased fresh ("it turned out
that…"), not templated. Load-bearing slides: 7, 11, 16. Emotional peaks: 2 (hook),
15 (payoff). If cut to ~15 min: collapse 8, 12, 17 to one line each; move 20 to backup.

## Act I — The problem & why it's hard
1. **Title.** "Can we predict smell? Odorant–receptor binding models." Name, team, internship.
2. **Smell as a computation (hook).** Least-understood sense; vision/hearing digitized, smell barely. ~400 human receptor types × tens of thousands of odorants. Smell = a *combination* of receptor responses (a chord, not a note). Show: molecule → subset of receptors → pattern = percept.
3. **The concrete task.** Given a receptor (protein sequence) + an odorant (molecule): do they bind / does it activate? A special case of protein–ligand binding. Value: fragrance/flavor design, "digital nose", understanding perception.
4. **Why it's hard (data slide).** Sparse, noisy; mostly negatives; receptors narrowly tuned; few positives per receptor. In practice the molecule is almost always **new** (repertoire fixed, odorants novel) — the "cold molecule" case, which defines what "good" means. Show: nearly-empty receptor×molecule matrix.

## Act II — Approach & what we learned about the data
5. **Digitizing protein & molecule.** Frozen pretrained encoders: a protein language model (analogy: an LLM that "reads" amino-acid sequences) → vector; molecule → fingerprint vector. Off-the-shelf.
6. **The surprisingly strong baseline.** Concatenate the two vectors → gradient-boosted trees (a decade-old method). It's the yardstick everything is measured against; hard to beat. Optional single anchor number as reference.
7. **Three different exams (key slide).** A score means nothing without "tested how". (a) fill matrix gaps (both seen); (b) **new molecule**, known receptors; (c) new receptor. We argue (b) is the real olfactory task (same nose, new smells) and judge models mainly by it. Redefines "success". Show: three matrix mini-pictures with different held-out test.
8. **What the baseline actually reads.** It mostly recognizes *which receptor* (its family / "kin"), not the fine pocket chemistry. A diagnosis, not a failure: on seen receptors identity is nearly enough (hard to beat there); on truly novel receptors there's little to lean on but weak similarity. Analogy: recognizing a person by face, not by what they can do.

## Act III — Our contribution: model interaction & the niche
9. **What to attack.** Concat+tree can't model the actual receptor↔ligand *interaction*, and encoders are frozen. Two ways past it: model interaction, or use the structure of the data's connections. We took the second. Show: two forks — attention over interaction / graph of connections.
10. **The graph idea.** Build a "who-binds-what" graph; let a receptor's representation be refined by the molecules known to (not) bind it. Intuition: "a receptor is known by the company it keeps" — its binding profile, not just its sequence. Show: bipartite receptor–molecule graph, one receptor's neighbours highlighted.
11. **Where the graph wins (our niche).** Exactly on cold molecules: refining the receptor with its binding profile carries transferable signal absent from the raw embedding. The one place we consistently beat the strong baseline. Honest: small, and isolated to the protein side — but real and reproducible. Anchor: "a point or two where everything else stands still."
12. **Where the graph does NOT help (honest map).** In matrix-completion the baseline is a ceiling; the graph's compression only loses information. We sell the exact boundary where relational modeling helps vs hurts, not a universal win. Show: simple traffic-light table.

## Act IV — Benchmarking the field
13. **Who we compete with.** Recent specialized models: attention between protein & molecule; adapters over a molecular language model; graphs with the receptor "built in". One plain line each; some are fresh top-conference papers.
14. **Why fair comparison is its own work.** Published numbers are often incomparable: different protein encoders, splits, test protocols. We re-implemented competitors on a common footing (same inputs, same splits, same evaluation) to separate architecture from "lucky setup". Analogy: same track, not stopwatches from different stadiums.
15. **The comparison result (understated payoff).** On cold molecules no fancy published model beats plain boosting; the only thing that does is our graph refinement. Say it calmly: once everyone is leveled, complexity alone didn't pay off — one relational trick did.

## Act V — Rigor as a contribution & the limits
16. **Evaluation is a minefield (methodology slide).** Half the work is not fooling ourselves or the field: a "cold" split that was really a chain of near-identical molecules (a homologous series); metrics inflated by the test's positive rate; unfair tuned-vs-untuned comparisons; huge variance across random splits on small cold tests. Value: we know which loud results to trust. Analogy: easy to "break a record" if you quietly shorten the track.
17. **The real bottleneck is the data, not just the model.** Narrow receptor tuning + sparsity cap what any method can extract with current representations. So the next moves are about data and physics, not architecture. Show: one big thesis line.

## Act VI — Where next & wrap
18. **How to continue.** One line each: a trainable *molecule*-side encoder (protein side is saturated, molecule has headroom); structure/geometry — docking, 3D pocket vs ligand atoms, to catch the chemistry now blurred; protein–chemical models pretrained on large bioactivity data (zero-shot then fine-tune); richer/continuous data (insect response panels — mosquito, fly — a second front with response *strength*, not yes/no); building honest cold splits. Frame: "the map shows where to dig."
19. **What the internship established (synthesis, not cliché).** 3–4 clean lines: a strong baseline built and *understood*; a narrow but real niche where data-connections give what embeddings can't; the field re-assembled on a fair common platform; a sober map of the ceiling — set by the data, not the model. One closing thesis line (see through-line).
20. **Backup / appendix (for Q&A).** Exact per-regime numbers; split definitions; per-competitor details; the insect second front; the crude one-hot/frequency "floor" as the bottom of the scale.

Delivery notes: spend time on 7, 11, 16; 2 and 15 are the emotional beats. To reach
~15 min, collapse 8/12/17 to a line and drop 20 into backup.
