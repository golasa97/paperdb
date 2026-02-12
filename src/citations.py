"""
Citations and references module using the OpenAlex API (free, no key required).
Supports:
 - Fetching papers that cite a given DOI
 - Fetching a paper's own references
 - Author h-index enrichment
"""

import requests
import time
from typing import Optional

OPENALEX_API = "https://api.openalex.org"
RATE_LIMIT_DELAY = 0.1
REQUEST_TIMEOUT = 90  # seconds – some large queries are slow


def _clean_doi(doi: str) -> str:
    clean = doi.strip()
    for prefix in ["https://doi.org/", "http://doi.org/", "doi:"]:
        if clean.lower().startswith(prefix.lower()):
            clean = clean[len(prefix):]
    return clean


def _extract_authors(authorships: list) -> list[dict]:
    authors = []
    for auth in authorships:
        ao = auth.get("author", {})
        insts = auth.get("institutions", [])
        authors.append({
            "name": ao.get("display_name", "Unknown"),
            "orcid": ao.get("orcid"),
            "openalex_id": ao.get("id"),
            "institution": insts[0].get("display_name") if insts else None,
        })
    return authors


def _reconstruct_abstract(inv: dict) -> str:
    if not inv:
        return ""
    positions = {}
    for word, pos_list in inv.items():
        for pos in pos_list:
            positions[pos] = word
    if not positions:
        return ""
    return " ".join(positions.get(i, "") for i in range(max(positions.keys()) + 1))


def _parse_work(work: dict) -> dict:
    doi = work.get("doi", "")
    if doi and doi.startswith("https://doi.org/"):
        doi = doi[16:]

    loc = work.get("primary_location") or {}
    src = loc.get("source") or {}

    authors = _extract_authors(work.get("authorships", []))
    oa = work.get("open_access") or {}

    return {
        "doi": doi,
        "title": work.get("title") or "Untitled",
        "authors": [a["name"] for a in authors],
        "authors_detail": authors,
        "abstract": _reconstruct_abstract(work.get("abstract_inverted_index")),
        "journal": src.get("display_name"),
        "issn": src.get("issn_l"),
        "year": work.get("publication_year"),
        "publication_date": work.get("publication_date"),
        "cited_by_count": work.get("cited_by_count", 0),
        "referenced_works_count": len(work.get("referenced_works", [])),
        "is_oa": oa.get("is_oa", False),
        "oa_url": oa.get("oa_url"),
        "openalex_id": work.get("id"),
        "type": work.get("type"),
    }


# ---------- resolve DOI -> OpenAlex ID ----------

def _resolve_openalex_id(doi: str, email: Optional[str] = None) -> Optional[str]:
    params = {"mailto": email} if email else {}
    try:
        r = requests.get(f"{OPENALEX_API}/works/https://doi.org/{doi}",
                         params=params, timeout=REQUEST_TIMEOUT)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json().get("id")
    except Exception as e:
        print(f"Error resolving OpenAlex ID for {doi}: {e}")
        return None


def _resolve_openalex_work(doi: str, email: Optional[str] = None) -> Optional[dict]:
    """Get full work object for a DOI."""
    params = {"mailto": email} if email else {}
    try:
        r = requests.get(f"{OPENALEX_API}/works/https://doi.org/{doi}",
                         params=params, timeout=REQUEST_TIMEOUT)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"Error resolving work for {doi}: {e}")
        return None


# ---------- citing works ----------

def fetch_citing_works(doi: str, email: Optional[str] = None,
                       per_page: int = 200, max_results: int = 1000) -> dict:
    clean = _clean_doi(doi)
    oa_id = _resolve_openalex_id(clean, email=email)
    if not oa_id:
        return {"cited_doi": clean, "total_count": 0, "results": []}

    params = {"filter": f"cites:{oa_id}", "per_page": min(per_page, 200), "cursor": "*"}
    if email:
        params["mailto"] = email

    all_results = []
    total_count = 0
    while len(all_results) < max_results:
        try:
            resp = requests.get(f"{OPENALEX_API}/works", params=params, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            total_count = data.get("meta", {}).get("count", 0)
            works = data.get("results", [])
            if not works:
                break
            for w in works:
                all_results.append(_parse_work(w))
                if len(all_results) >= max_results:
                    break
            nc = data.get("meta", {}).get("next_cursor")
            if not nc or len(works) < per_page:
                break
            params["cursor"] = nc
            time.sleep(RATE_LIMIT_DELAY)
        except Exception as e:
            print(f"Error fetching citations for {doi}: {e}")
            break

    return {"cited_doi": clean, "total_count": total_count, "results": all_results}


# ---------- references (works cited BY a paper) ----------

def fetch_references(doi: str, email: Optional[str] = None,
                     max_results: int = 500) -> dict:
    """
    Fetch the works referenced BY a given DOI.
    OpenAlex stores referenced_works as a list of OpenAlex IDs in the work object.
    We batch-fetch those IDs to get full metadata.
    """
    clean = _clean_doi(doi)
    work = _resolve_openalex_work(clean, email=email)
    if not work:
        return {"source_doi": clean, "total_count": 0, "results": []}

    ref_ids = work.get("referenced_works", [])
    if not ref_ids:
        return {"source_doi": clean, "total_count": 0, "results": []}

    ref_ids = ref_ids[:max_results]
    total = len(ref_ids)

    # Batch fetch using OpenAlex filter: openalex_id in (id1|id2|...)
    # OpenAlex supports pipe-separated IDs, up to ~50 per call
    all_results = []
    batch_size = 50
    for i in range(0, len(ref_ids), batch_size):
        batch = ref_ids[i:i + batch_size]
        id_filter = "|".join(batch)
        params = {"filter": f"openalex_id:{id_filter}", "per_page": len(batch)}
        if email:
            params["mailto"] = email
        try:
            resp = requests.get(f"{OPENALEX_API}/works", params=params, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            for w in data.get("results", []):
                all_results.append(_parse_work(w))
            time.sleep(RATE_LIMIT_DELAY)
        except Exception as e:
            print(f"Error fetching references batch for {doi}: {e}")

    return {"source_doi": clean, "total_count": total, "results": all_results}


# ---------- author enrichment ----------

def fetch_author_metrics(oa_author_id: str, email: Optional[str] = None) -> Optional[dict]:
    params = {"mailto": email} if email else {}
    try:
        r = requests.get(oa_author_id, params=params, timeout=30)
        r.raise_for_status()
        d = r.json()
        s = d.get("summary_stats", {})
        return {
            "name": d.get("display_name"), "h_index": s.get("h_index", 0),
            "i10_index": s.get("i10_index", 0),
            "cited_by_count": d.get("cited_by_count", 0),
            "works_count": d.get("works_count", 0),
        }
    except Exception:
        return None


def enrich_with_author_metrics(works: list[dict], email: Optional[str] = None,
                               max_authors: int = 50) -> list[dict]:
    cache = {}
    seen = set()
    for w in works:
        det = w.get("authors_detail", [])
        if det:
            oid = det[0].get("openalex_id")
            if oid and oid not in seen and len(seen) < max_authors:
                seen.add(oid)
    for oid in seen:
        m = fetch_author_metrics(oid, email=email)
        cache[oid] = m.get("h_index", 0) if m else 0
        time.sleep(RATE_LIMIT_DELAY)
    for w in works:
        det = w.get("authors_detail", [])
        oid = det[0].get("openalex_id") if det else None
        w["first_author_h_index"] = cache.get(oid) if oid else None
    return works
