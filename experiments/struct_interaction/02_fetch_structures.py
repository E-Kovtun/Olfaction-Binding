"""Stage-0-pre: fetch AF2 structures for positive-control ORs.

Resolves a small set of well-characterized OR gene names -> reviewed human
UniProt accession (via UniProt REST, NOT the unreliable uniprot_id column),
downloads the AlphaFold DB model, and checks whether the UniProt sequence
matches a receptor present in M2OR (so we can reuse its activation labels).

Output (experiments/struct_interaction/data/structures/):
    AF-<ACC>-F1-model_v4.pdb      one per receptor
    structures_manifest.csv       gene, acc, len, m2or_match, m2or_identity, n_pos_m2or

    experiments/struct_interaction/.venv/Scripts/python 02_fetch_structures.py
"""
from __future__ import annotations
import pathlib, sys, io, difflib
import pandas as pd
import requests

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE
while not (ROOT / "pyproject.toml").exists():
    ROOT = ROOT.parent
OUT = HERE / "data" / "structures"
OUT.mkdir(parents=True, exist_ok=True)

# well-characterized ORs with strong, literature-known agonists
POS_CONTROL_GENES = ["OR51E2", "OR51E1", "OR1A1", "OR2W1", "OR7D4", "OR1G1", "OR10G4", "OR2J2"]

UNIPROT = "https://rest.uniprot.org/uniprotkb/search"
AFDB_API = "https://alphafold.ebi.ac.uk/api/prediction/{acc}"


def af_pdb_url(acc: str):
    """Resolve the current AF2 pdb download URL via the AFDB API."""
    r = requests.get(AFDB_API.format(acc=acc), timeout=30)
    if r.status_code != 200:
        return None
    js = r.json()
    if not js:
        return None
    return js[0].get("pdbUrl") or js[0].get("cifUrl")


def resolve(gene: str):
    q = f"gene_exact:{gene} AND organism_id:9606 AND reviewed:true"
    r = requests.get(UNIPROT, params={"query": q, "fields": "accession,sequence",
                                       "format": "tsv", "size": 1}, timeout=30)
    r.raise_for_status()
    lines = r.text.strip().splitlines()
    if len(lines) < 2:
        return None, None
    acc, seq = lines[1].split("\t")
    return acc, seq


def main():
    cur = pd.read_csv(ROOT / "data" / "processed" / "pairs_curated.csv")
    cur_recs = list(cur["receptor"].unique())
    pos_count = cur[cur["label"] == 1].groupby("receptor")["inchikey"].nunique().to_dict()

    rows = []
    for gene in POS_CONTROL_GENES:
        try:
            acc, seq = resolve(gene)
        except Exception as e:
            print(f"{gene}: UniProt error {e}"); continue
        if not acc:
            print(f"{gene}: no reviewed human accession"); continue

        # download AF2 model via AFDB API
        pdb_path = OUT / f"AF-{acc}-F1.pdb"
        if not pdb_path.exists():
            url = af_pdb_url(acc)
            if not url:
                print(f"{gene} ({acc}): no AF2 model in AFDB"); continue
            rr = requests.get(url, timeout=60)
            if rr.status_code != 200:
                print(f"{gene} ({acc}): download failed (HTTP {rr.status_code})"); continue
            pdb_path.write_text(rr.text)

        # match to an M2OR receptor (exact, else best identity)
        if seq in cur_recs:
            match, ident = seq, 1.0
        else:
            match = max(cur_recs, key=lambda s: difflib.SequenceMatcher(None, seq, s).ratio())
            ident = difflib.SequenceMatcher(None, seq, match).ratio()
        npos = pos_count.get(match, 0) if ident > 0.95 else 0

        rows.append({"gene": gene, "acc": acc, "len": len(seq),
                     "m2or_identity": round(ident, 3),
                     "m2or_match": ident > 0.95, "n_pos_m2or": npos})
        print(f"{gene:<8} {acc}  len={len(seq)}  M2OR id={ident:.3f}  "
              f"{'MATCH' if ident > 0.95 else 'no-match'}  pos_in_M2OR={npos}")

    if not rows:
        print("\nНи одной структуры не скачано — проверь сеть/AFDB."); return
    man = pd.DataFrame(rows)
    man.to_csv(OUT.parent / "structures_manifest.csv", index=False)
    print(f"\n{man['m2or_match'].sum()}/{len(man)} structures map to an M2OR receptor with labels")
    print(f"-> {OUT}")
    print(f"-> {OUT.parent / 'structures_manifest.csv'}")


if __name__ == "__main__":
    main()
