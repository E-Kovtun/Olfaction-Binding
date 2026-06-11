"""Online completion of fields missing from the raw M2OR dump.

Two gaps in ``M2OR_20230428.csv``:
  * ``Sequence`` is inline for only ~75% of rows  -> fill via Gene ID -> UniProt.
  * ``canonicalSMILES`` is empty for ~99% of rows  -> fill via InChIKey -> PubChem.

Both are NETWORK calls and are cached to disk so they run once. Keep them
optional: the offline pipeline still works on the inline 75% of sequences.
"""
from __future__ import annotations
import json, time, pathlib, urllib.parse, urllib.request

UA = {"User-Agent": "orbind/0.0 (research; contact: you@example.com)"}


def _get(url: str, timeout: int = 30) -> str | None:
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode("utf-8", "replace")
    except Exception:
        return None


def _load_cache(p: pathlib.Path) -> dict:
    return json.loads(p.read_text()) if p.exists() else {}


def _save_cache(p: pathlib.Path, d: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d))


def gene_to_sequence(genes, cache="data/processed/gene2seq.json", sleep=0.34) -> dict:
    """Map human gene symbol (e.g. 'OR10S1') -> canonical UniProt protein sequence."""
    cache = pathlib.Path(cache)
    out = _load_cache(cache)
    base = "https://rest.uniprot.org/uniprotkb/search"
    for g in sorted(set(genes)):
        if g in out:
            continue
        q = f'(gene:{g}) AND (organism_id:9606) AND (reviewed:true)'
        url = f"{base}?query={urllib.parse.quote(q)}&format=fasta&size=1"
        fasta = _get(url)
        if not fasta:  # fall back to unreviewed (most ORs are TrEMBL)
            q2 = f'(gene:{g}) AND (organism_id:9606)'
            fasta = _get(f"{base}?query={urllib.parse.quote(q2)}&format=fasta&size=1")
        seq = "".join(fasta.splitlines()[1:]) if fasta and fasta.startswith(">") else None
        out[g] = seq
        _save_cache(cache, out)
        time.sleep(sleep)
    return out


def inchikey_to_smiles(keys, cache="data/processed/inchikey2smiles.json", sleep=0.25) -> dict:
    """Map InChIKey -> canonical SMILES via PubChem PUG REST."""
    cache = pathlib.Path(cache)
    out = _load_cache(cache)
    base = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey"
    for k in sorted(set(keys)):
        if k in out:
            continue
        txt = _get(f"{base}/{urllib.parse.quote(k)}/property/CanonicalSMILES/TXT")
        out[k] = txt.strip().splitlines()[0] if txt and txt.strip() else None
        _save_cache(cache, out)
        time.sleep(sleep)
    return out
