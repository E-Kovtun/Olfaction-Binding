"""Script 0 — download the FULL official M2OR export (with SMILES).

The GitHub flat dump (M2OR_20230428.csv) ships WITHOUT SMILES. The real database
export, served by the M2OR web app at /export-db, is a relational ZIP that
includes SMILES inline plus mixture / mutation / species flags. This endpoint is
a Laravel form: GET the homepage for a CSRF `_token` + session cookie, then POST.

Output: data/raw/M2OR.zip  (≈7.5 MB), containing
    pairs.csv  main_compounds.csv  main_receptors.csv  experiments.csv
    species.csv  assays.csv  references.csv  ...

Usage:  uv run python scripts/00_download_m2or.py
"""
import re, pathlib, http.cookiejar, urllib.request, urllib.parse

HOME = "https://m2or.chemsensim.fr/"
EXPORT = "https://m2or.chemsensim.fr/export-db"
UA = {"User-Agent": "orbind/0.0 (research)"}


def main(out="data/raw/M2OR.zip"):
    jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))

    # 1. GET homepage -> session cookie + CSRF token
    html = opener.open(urllib.request.Request(HOME, headers=UA), timeout=60).read().decode("utf-8", "replace")
    m = re.search(r'name="_token"\s+value="([^"]+)"', html)
    if not m:
        raise RuntimeError("CSRF _token not found on M2OR homepage")
    token = m.group(1)

    # 2. POST /export-db with token (cookies carried by opener) -> ZIP
    data = urllib.parse.urlencode({"_token": token}).encode()
    resp = opener.open(urllib.request.Request(EXPORT, data=data, headers=UA), timeout=600)
    blob = resp.read()

    out = pathlib.Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(blob)
    print(f"downloaded {len(blob)/1e6:.1f} MB -> {out}")
    import zipfile
    print("contents:", zipfile.ZipFile(out).namelist())


if __name__ == "__main__":
    main()
