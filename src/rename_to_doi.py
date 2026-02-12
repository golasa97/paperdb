#!/usr/bin/env python3
"""
Rename PDFs to DOI-based filenames via Crossref.
Can be used as CLI or imported by the web application.
"""

from __future__ import annotations

import argparse
import csv
import re
import time
from pathlib import Path
from typing import Optional, Tuple

import requests

CROSSREF_WORKS = "https://api.crossref.org/works"


def slugify_doi(doi: str) -> str:
    doi = doi.strip()
    doi = doi.replace("https://doi.org/", "").replace("http://doi.org/", "")
    doi = doi.replace("doi:", "").strip()
    doi = doi.replace("/", "_")
    doi = re.sub(r"[^a-zA-Z0-9._-]+", "_", doi)
    return f"{doi}.pdf"


def unsluggify_doi(filename: str) -> Optional[str]:
    """Reverse slugify: extract DOI from a slugified filename."""
    name = filename
    if name.lower().endswith('.pdf'):
        name = name[:-4]
    m = re.match(r'(10\.\d+)_(.*)', name)
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    return None


def parse_filename_hint(filename: str) -> Tuple[str, Optional[int], Optional[str]]:
    stem = Path(filename).stem
    m = re.search(r"_(19\d{2}|20\d{2})_(.+)$", stem)
    if not m:
        title_hint = stem.replace("-", " ").replace("_", " ").strip()
        return title_hint, None, None
    year = int(m.group(1))
    journal_hint = m.group(2).replace("-", " ").replace("_", " ").strip()
    title_part = stem[: m.start()]
    title_hint = title_part.replace("-", " ").replace("_", " ").strip() or stem
    return title_hint, year, journal_hint


def item_year(it) -> Optional[int]:
    for key in ("published-print", "published-online", "issued", "created"):
        parts = it.get(key, {}).get("date-parts")
        if parts and parts[0] and isinstance(parts[0][0], int):
            return parts[0][0]
    return None


def crossref_best_match(
    title_hint: str,
    year: Optional[int] = None,
    journal_hint: Optional[str] = None,
    mailto: Optional[str] = None,
    timeout_s: int = 30,
) -> Tuple[Optional[str], float, str, Optional[dict]]:
    """
    Returns (doi, score, notes, metadata_dict).
    metadata_dict has title, authors, year, journal for preview.
    """
    headers = {"User-Agent": f"doi-renamer/1.1 (mailto:{mailto})" if mailto else "doi-renamer/1.1"}
    params = {"query.title": title_hint, "rows": 5}
    if mailto:
        params["mailto"] = mailto
    if journal_hint:
        params["query.container-title"] = journal_hint

    r = requests.get(CROSSREF_WORKS, params=params, headers=headers, timeout=timeout_s)
    r.raise_for_status()
    items = r.json().get("message", {}).get("items", [])
    if not items:
        return None, 0.0, "no results", None

    want_toks = set(re.findall(r"[a-z0-9]+", title_hint.lower()))

    best_doi = None
    best_score = -1.0
    best_notes = "no match"
    best_meta = None

    for it in items:
        doi = it.get("DOI")
        if not doi:
            continue
        cr_title = " ".join(it.get("title", [])).lower()
        got_toks = set(re.findall(r"[a-z0-9]+", cr_title))
        score = 0.0
        if want_toks:
            overlap = len(want_toks & got_toks) / max(1, len(want_toks))
            score += 0.60 * overlap
        y = item_year(it)
        if year and y:
            if y == year:
                score += 0.25
            elif abs(y - year) <= 1:
                score += 0.15
        container = " ".join(it.get("container-title", [])).lower()
        if journal_hint:
            jw = journal_hint.lower()
            if jw and (jw in container or container in jw):
                score += 0.15
        if score > best_score:
            best_score = score
            best_doi = doi
            best_notes = f"year={y}, container={container[:70]}"
            # Build metadata
            authors = []
            for a in it.get("author", []):
                if "given" in a and "family" in a:
                    authors.append(f"{a['given']} {a['family']}")
                elif "family" in a:
                    authors.append(a["family"])
                elif "name" in a:
                    authors.append(a["name"])
            best_meta = {
                "title": " ".join(it.get("title", [])),
                "authors": authors,
                "year": y,
                "journal": " ".join(it.get("container-title", [])),
                "doi": doi,
            }

    best_score = max(0.0, min(1.0, best_score))
    return best_doi, best_score, best_notes, best_meta


def resolve_pdf(pdf_path: Path, mailto: Optional[str] = None) -> dict:
    """
    Resolve a single PDF to its best DOI match.
    Returns dict with: filename, doi, score, notes, metadata, new_filename.
    """
    title_hint, year, journal_hint = parse_filename_hint(pdf_path.name)
    try:
        doi, score, notes, meta = crossref_best_match(
            title_hint=title_hint, year=year,
            journal_hint=journal_hint, mailto=mailto,
        )
    except Exception as e:
        return {
            "filename": pdf_path.name, "doi": None, "score": 0,
            "notes": f"error: {e}", "metadata": None, "new_filename": None,
        }

    return {
        "filename": pdf_path.name,
        "doi": doi,
        "score": round(score, 3) if score else 0,
        "notes": notes,
        "metadata": meta,
        "new_filename": slugify_doi(doi) if doi else None,
    }


# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="manual_import")
    ap.add_argument("--mailto", default=None)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--min-score", type=float, default=0.45)
    ap.add_argument("--sleep", type=float, default=0.2)
    args = ap.parse_args()

    folder = Path(args.dir).expanduser().resolve()
    if not folder.exists() or not folder.is_dir():
        raise SystemExit(f"ERROR: not a directory: {folder}")

    pdfs = sorted(p for p in folder.glob("*.pdf") if p.is_file())
    if not pdfs:
        print(f"No PDFs found in {folder}")
        return

    out_map = folder / "doi_rename_map.tsv"
    with out_map.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["old_name", "new_name", "doi", "score", "notes", "action"])
        for old_path in pdfs:
            result = resolve_pdf(old_path, mailto=args.mailto)
            doi = result["doi"]
            score = result["score"]
            new_name = result["new_filename"]
            notes = result["notes"]

            if not doi:
                print(f"SKIP (no DOI): {old_path.name}")
                w.writerow([old_path.name, "", "", f"{score:.3f}", notes, "skip"])
                time.sleep(args.sleep)
                continue

            new_path = folder / new_name
            if score < args.min_score:
                print(f"REVIEW (low score {score:.2f}): {old_path.name} -> {new_name}")
                w.writerow([old_path.name, new_name, doi, f"{score:.3f}", f"LOW_SCORE: {notes}", "review"])
                time.sleep(args.sleep)
                continue

            if new_path.exists() and new_path.resolve() != old_path.resolve():
                print(f"SKIP (target exists): {old_path.name} -> {new_name}")
                w.writerow([old_path.name, new_name, doi, f"{score:.3f}", "target exists", "skip"])
                time.sleep(args.sleep)
                continue

            if args.apply:
                old_path.rename(new_path)
                action = "renamed"
                print(f"RENAMED: {old_path.name} -> {new_name} (score={score:.2f})")
            else:
                action = "dry-run"
                print(f"WOULD RENAME: {old_path.name} -> {new_name} (score={score:.2f})")

            w.writerow([old_path.name, new_name, doi, f"{score:.3f}", notes, action])
            time.sleep(args.sleep)

    print(f"\nWrote mapping: {out_map}")


if __name__ == "__main__":
    main()
