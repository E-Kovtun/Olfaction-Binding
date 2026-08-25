# notes/

Prose that does not belong in a docstring: what we decided, why, and what is still
open. Everything here is dated and written to be re-read months later.

| note | what it is | status |
|---|---|---|
| `paper_storyline.md` | thesis, section skeleton, figure/table list, the C9 experiment battery | **current** — the working plan for the paper |
| `research_landscape.md` | every forward direction discussed, with comments inline | **living** |
| `multi_source_boosting_ensemble.md` | design of the ensembler: why symmetric sources replaced the fixed ProSmith/LORAX solo/pair/triplet scheme | current — describes live code |
| `presentation_slide_plan.md` | slide plan for the results talk (Aug 2026) | done, kept for provenance |

Notes belonging to closed lines moved to [`../legacy/notes/`](../legacy/notes/)
(the 900-epoch training study and the site-MIL/attention screen). They describe
code that now lives under `legacy/scripts/`.

Two things deliberately live outside this folder:

* **Numeric results** belong in `results/` and are read through
  `scripts/analysis/summarize_runs.py`. A note that quotes a number goes stale;
  a note that says *why we measured it that way* does not.
* **The manuscript** is not in this repository (see `.gitignore`).
