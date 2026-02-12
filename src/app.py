#!/usr/bin/env python3
"""
Flask web application for the Paper Search library.
"""

import io
import json
import os
import re
import shutil
import zipfile
from pathlib import Path

from flask import Flask, render_template, request, jsonify, send_file, Response

from database import (
    init_db, search_fts, search_semantic, get_all_papers,
    get_paper_by_doi, get_stats, get_journals, get_years, get_doc_types,
    check_dois_exist, find_similar,
    insert_paper, delete_paper, update_paper_field,
    create_list, get_all_lists, get_list, delete_list,
    add_papers_to_list, remove_papers_from_list,
    get_setting, set_setting, get_all_settings, get_all_dois,
    insert_chunks, delete_chunks_for_paper, delete_all_chunks,
    get_papers_needing_chunks, count_chunk_stats, count_embeddings_by_dims,
)
from embeddings import (
    get_embedding, get_client, create_search_text, create_paper_chunks,
    get_backend, set_backend, backend_info, get_embedding_dims,
)
from citations import fetch_citing_works, fetch_references, enrich_with_author_metrics
from rename_to_doi import resolve_pdf, slugify_doi, parse_filename_hint, crossref_best_match

app = Flask(__name__)

UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


# ---------- helpers ----------

def _get_email():
    return get_setting("polite_email", os.environ.get("POLITE_EMAIL", ""))


def _get_pdf_base_dir():
    return get_setting("pdf_base_dir", os.environ.get("PDF_BASE_DIR", ""))


def _embeddings_available():
    """Check if any embedding backend is usable."""
    b = get_backend()
    if b == "openai":
        key = get_setting("openai_api_key", os.environ.get("OPENAI_API_KEY", ""))
        return bool(key)
    elif b == "ollama":
        try:
            import requests as _r
            url = get_setting("ollama_url", "http://localhost:11434")
            return _r.get(f"{url}/api/tags", timeout=2).ok
        except Exception:
            return False
    elif b == "sbert":
        try:
            import sentence_transformers  # noqa: F401
            return True
        except ImportError:
            return False
    return False


def _get_openai_key():
    return get_setting("openai_api_key", os.environ.get("OPENAI_API_KEY", ""))


def _resolve_pdf_path(paper: dict) -> Path | None:
    """Find the actual PDF on disk for a paper."""
    stored = paper.get('pdf_path')
    if not stored:
        return None
    stored_path = Path(stored)
    pdf_base = _get_pdf_base_dir()
    if pdf_base:
        p = Path(pdf_base) / stored_path.name
        if p.exists():
            return p
        matches = list(Path(pdf_base).glob(f"**/{stored_path.name}"))
        if matches:
            return matches[0]
    if stored_path.exists():
        return stored_path
    return None


def _make_bibtex_entry(paper: dict) -> str:
    """Generate a BibTeX entry for a single paper."""
    doi = paper.get('doi', '')
    key = re.sub(r'[^a-zA-Z0-9]', '_', doi)[:40]
    authors = paper.get('authors', [])
    if isinstance(authors, str):
        authors = json.loads(authors) if authors else []
    author_str = " and ".join(authors) if authors else "Unknown"
    dtype = paper.get('doc_type', 'article')
    bib_type = {
        'article': 'article', 'book': 'book', 'textbook': 'book',
        'techreport': 'techreport', 'manual': 'manual',
        'thesis': 'phdthesis', 'presentation': 'misc', 'inproceedings': 'inproceedings',
    }.get(dtype, 'article')

    lines = [f"@{bib_type}{{{key},"]
    if paper.get('title'):
        lines.append(f"  title = {{{paper['title']}}},")
    lines.append(f"  author = {{{author_str}}},")
    if paper.get('year'):
        lines.append(f"  year = {{{paper['year']}}},")
    if paper.get('journal'):
        field = 'booktitle' if bib_type == 'inproceedings' else 'journal'
        lines.append(f"  {field} = {{{paper['journal']}}},")
    if doi:
        lines.append(f"  doi = {{{doi}}},")
    lines.append("}")
    return "\n".join(lines)


# ============================================================
# PAGES
# ============================================================

@app.route("/")
def index():
    stats = get_stats()
    return render_template("index.html", stats=stats,
                           embeddings_available=_embeddings_available())


# ============================================================
# SEARCH
# ============================================================

@app.route("/search")
def search():
    query = request.args.get("q", "").strip()
    search_type = request.args.get("type", "keyword")
    doc_type = request.args.get("doc_type", "all")
    limit = min(int(request.args.get("limit", 50)), 200)

    if not query:
        return jsonify({"results": [], "query": query, "type": search_type})

    results = []
    if search_type == "keyword":
        try:
            results = search_fts(query, limit=limit, doc_type=doc_type if doc_type != "all" else None)
        except Exception:
            try:
                results = search_fts('"' + query.replace('"', '""') + '"', limit=limit,
                                     doc_type=doc_type if doc_type != "all" else None)
            except Exception:
                results = []

    elif search_type == "semantic" and _embeddings_available():
        try:
            qe = get_embedding(query)
            results = search_semantic(qe, limit=limit,
                                      doc_type=doc_type if doc_type != "all" else None)
        except Exception as e:
            print(f"Semantic search error: {e}")

    elif search_type == "hybrid" and _embeddings_available():
        try:
            kw = search_fts(query, limit=limit, doc_type=doc_type if doc_type != "all" else None)
            kw_dois = {r['doi']: i for i, r in enumerate(kw)}
            qe = get_embedding(query)
            sem = search_semantic(qe, limit=limit, doc_type=doc_type if doc_type != "all" else None)
            sem_dois = {r['doi']: i for i, r in enumerate(sem)}
            combined = {}
            all_r = {}
            k = 60
            for d, rank in kw_dois.items():
                combined[d] = combined.get(d, 0) + 1/(k+rank)
                all_r[d] = kw[rank]
            for d, rank in sem_dois.items():
                combined[d] = combined.get(d, 0) + 1/(k+rank)
                if d not in all_r:
                    all_r[d] = sem[rank]
            sorted_d = sorted(combined, key=lambda x: combined[x], reverse=True)
            results = [all_r[d] for d in sorted_d[:limit]]
            for r in results:
                r['score'] = combined[r['doi']]
        except Exception as e:
            print(f"Hybrid search error: {e}")
            try:
                results = search_fts(query, limit=limit)
            except Exception:
                results = []
    else:
        try:
            results = search_fts(query, limit=limit)
        except Exception:
            results = []

    for r in results:
        r.pop('embedding', None)
        r.pop('full_text', None)

    return jsonify({"results": results, "query": query, "type": search_type, "count": len(results)})


# ============================================================
# PAPER DETAIL / PDF / DELETE / UPDATE
# ============================================================

@app.route("/paper/<path:doi>")
def paper_detail(doi):
    paper = get_paper_by_doi(doi)
    if not paper:
        return jsonify({"error": "Paper not found"}), 404
    paper.pop('embedding', None)
    paper.pop('full_text', None)
    return jsonify(paper)


@app.route("/pdf/<path:doi>")
def serve_pdf(doi):
    paper = get_paper_by_doi(doi)
    if not paper or not paper.get('pdf_path'):
        return "PDF not found", 404
    pdf_path = _resolve_pdf_path(paper)
    if not pdf_path:
        return f"PDF file not found on disk", 404
    return send_file(pdf_path, mimetype='application/pdf')


@app.route("/api/paper/<path:doi>", methods=["DELETE"])
def api_delete_paper(doi):
    ok = delete_paper(doi)
    return jsonify({"deleted": ok, "doi": doi})


@app.route("/api/paper/<path:doi>", methods=["PATCH"])
def api_update_paper(doi):
    data = request.get_json(silent=True) or {}
    updated = {}
    for field in ("doc_type", "title", "journal", "year"):
        if field in data:
            update_paper_field(doi, field, data[field])
            updated[field] = data[field]
    return jsonify({"updated": updated, "doi": doi})


# ============================================================
# CITATIONS / REFERENCES / SIMILAR
# ============================================================

@app.route("/citations/<path:doi>")
def citations(doi):
    enrich = request.args.get("enrich", "false").lower() == "true"
    max_r = min(int(request.args.get("max_results", 500)), 2000)
    try:
        result = fetch_citing_works(doi, email=_get_email(), max_results=max_r)
        c = result["results"]
        if enrich and c:
            c = enrich_with_author_metrics(c, email=_get_email(), max_authors=min(len(c), 100))
        return jsonify({"cited_doi": result["cited_doi"], "total_count": result["total_count"],
                        "fetched_count": len(c), "results": c})
    except Exception as e:
        return jsonify({"error": str(e), "cited_doi": doi, "total_count": 0, "fetched_count": 0, "results": []}), 500


@app.route("/references/<path:doi>")
def references(doi):
    max_r = min(int(request.args.get("max_results", 500)), 2000)
    try:
        result = fetch_references(doi, email=_get_email(), max_results=max_r)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e), "source_doi": doi, "total_count": 0, "results": []}), 500


@app.route("/api/similar/<path:doi>")
def api_similar(doi):
    limit = min(int(request.args.get("limit", 20)), 100)
    results = find_similar(doi, limit=limit)
    for r in results:
        r.pop('embedding', None)
        r.pop('full_text', None)
    return jsonify({"doi": doi, "count": len(results), "results": results})


# ============================================================
# CHECK DOIS
# ============================================================

@app.route("/api/check-dois", methods=["POST"])
def api_check_dois():
    data = request.get_json(silent=True) or {}
    dois = data.get("dois", [])[:2000]
    return jsonify({"results": check_dois_exist(dois) if dois else {}})


# ============================================================
# LISTS
# ============================================================

@app.route("/api/lists", methods=["GET"])
def api_lists():
    return jsonify({"lists": get_all_lists()})


@app.route("/api/lists", methods=["POST"])
def api_create_list():
    data = request.get_json(silent=True) or {}
    name = data.get("name", "").strip()
    if not name:
        return jsonify({"error": "Name required"}), 400
    lid = create_list(name, data.get("description", ""))
    return jsonify({"id": lid, "name": name})


@app.route("/api/lists/<int:list_id>", methods=["GET"])
def api_get_list(list_id):
    lst = get_list(list_id)
    if not lst:
        return jsonify({"error": "List not found"}), 404
    return jsonify(lst)


@app.route("/api/lists/<int:list_id>", methods=["DELETE"])
def api_delete_list(list_id):
    return jsonify({"deleted": delete_list(list_id)})


@app.route("/api/lists/<int:list_id>/papers", methods=["POST"])
def api_add_to_list(list_id):
    data = request.get_json(silent=True) or {}
    dois = data.get("dois", [])
    added = add_papers_to_list(list_id, dois)
    return jsonify({"added": added})


@app.route("/api/lists/<int:list_id>/papers", methods=["DELETE"])
def api_remove_from_list(list_id):
    data = request.get_json(silent=True) or {}
    dois = data.get("dois", [])
    removed = remove_papers_from_list(list_id, dois)
    return jsonify({"removed": removed})


@app.route("/api/lists/<int:list_id>/export/bibtex")
def api_export_bibtex(list_id):
    lst = get_list(list_id)
    if not lst:
        return jsonify({"error": "Not found"}), 404
    entries = [_make_bibtex_entry(p) for p in lst['papers']]
    bib = "\n\n".join(entries)
    return Response(bib, mimetype="text/plain",
                    headers={"Content-Disposition": f"attachment; filename={lst['name']}.bib"})


@app.route("/api/lists/<int:list_id>/export/dois")
def api_export_dois(list_id):
    lst = get_list(list_id)
    if not lst:
        return jsonify({"error": "Not found"}), 404
    text = "\n".join(p['doi'] for p in lst['papers'] if p.get('doi'))
    return Response(text, mimetype="text/plain",
                    headers={"Content-Disposition": f"attachment; filename={lst['name']}_dois.txt"})


@app.route("/api/lists/<int:list_id>/export/zip")
def api_export_zip(list_id):
    lst = get_list(list_id)
    if not lst:
        return jsonify({"error": "Not found"}), 404
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        for p in lst['papers']:
            pdf_path = _resolve_pdf_path(p)
            if pdf_path and pdf_path.exists():
                zf.write(pdf_path, pdf_path.name)
    buf.seek(0)
    return send_file(buf, mimetype='application/zip', as_attachment=True,
                     download_name=f"{lst['name']}.zip")


# ============================================================
# IMPORT
# ============================================================

@app.route("/api/import/upload", methods=["POST"])
def api_import_upload():
    """Accept uploaded PDFs and save to uploads dir."""
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files"}), 400
    saved = []
    for f in files:
        if f.filename and f.filename.lower().endswith('.pdf'):
            dest = UPLOAD_DIR / f.filename
            f.save(dest)
            saved.append(f.filename)
    return jsonify({"uploaded": saved, "count": len(saved)})


@app.route("/api/import/resolve", methods=["POST"])
def api_import_resolve():
    """Resolve uploaded PDFs to DOIs via CrossRef."""
    data = request.get_json(silent=True) or {}
    filenames = data.get("filenames", [])
    email = _get_email()
    results = []
    for fn in filenames:
        path = UPLOAD_DIR / fn
        if path.exists():
            info = resolve_pdf(path, mailto=email or None)
            results.append(info)
    return jsonify({"results": results})


@app.route("/api/import/confirm", methods=["POST"])
def api_import_confirm():
    """Confirm import: rename file to DOI-based name, move to library, ingest."""
    data = request.get_json(silent=True) or {}
    items = data.get("items", [])
    pdf_base = _get_pdf_base_dir()
    if not pdf_base:
        return jsonify({"error": "PDF base directory not configured in settings"}), 400

    email = _get_email()
    ingested = []
    errors = []

    for item in items:
        fn = item.get("filename")
        doi = item.get("doi")
        doc_type = item.get("doc_type", "article")
        if not fn or not doi:
            errors.append({"filename": fn, "error": "Missing filename or DOI"})
            continue
        src = UPLOAD_DIR / fn
        if not src.exists():
            errors.append({"filename": fn, "error": "File not found in uploads"})
            continue
        new_name = slugify_doi(doi)
        dest = Path(pdf_base) / new_name
        try:
            shutil.move(str(src), str(dest))
        except Exception as e:
            errors.append({"filename": fn, "error": f"Move failed: {e}"})
            continue

        # Fetch metadata from CrossRef
        from crossref import fetch_metadata
        meta = fetch_metadata(doi, email=email) or {}

        # Extract fulltext
        try:
            from pdf_extract import extract_text
            fulltext = extract_text(dest)
        except Exception:
            fulltext = None

        insert_paper(
            doi=doi, title=meta.get('title'), authors=meta.get('authors'),
            abstract=meta.get('abstract'), journal=meta.get('journal'),
            year=meta.get('year'), keywords=meta.get('keywords'),
            pdf_path=str(dest), full_text=fulltext, doc_type=doc_type,
        )

        # Generate chunk embeddings if backend is available
        if _embeddings_available():
            try:
                paper = get_paper_by_doi(doi)
                if paper and (paper.get('abstract') or paper.get('title')):
                    chunk_texts = create_paper_chunks(paper)
                    if chunk_texts:
                        chunk_rows = []
                        faiss_pairs = []
                        for cidx, text in chunk_texts:
                            emb = get_embedding(text)
                            chunk_rows.append((cidx, text, emb))
                            faiss_pairs.append((cidx, emb))
                        insert_chunks(doi, chunk_rows)
                        # Add to FAISS index incrementally
                        try:
                            from faiss_index import add_paper_chunks as faiss_add
                            faiss_add(doi, faiss_pairs)
                        except Exception:
                            pass
            except Exception as e:
                print(f"Embedding error for {doi}: {e}")

        ingested.append({"doi": doi, "filename": new_name})

    return jsonify({"ingested": ingested, "errors": errors})


@app.route("/api/import/delete", methods=["POST"])
def api_import_delete():
    """Delete an uploaded file without ingesting."""
    data = request.get_json(silent=True) or {}
    fn = data.get("filename", "")
    path = UPLOAD_DIR / fn
    if path.exists() and path.parent.resolve() == UPLOAD_DIR.resolve():
        path.unlink()
        return jsonify({"deleted": True})
    return jsonify({"deleted": False})


# ============================================================
# CENSUS – detect new PDFs not yet in DB
# ============================================================

@app.route("/api/census")
def api_census():
    """Scan PDF directory for files not in the database."""
    pdf_base = _get_pdf_base_dir()
    if not pdf_base or not Path(pdf_base).exists():
        return jsonify({"error": "PDF base directory not configured or does not exist", "new_files": []})

    from ingest import extract_doi_from_filename
    known = get_all_dois()
    new_files = []
    for p in Path(pdf_base).glob("**/*.pdf"):
        doi = extract_doi_from_filename(p.name)
        if doi and doi not in known:
            new_files.append({"filename": p.name, "doi": doi, "path": str(p)})
    return jsonify({"new_files": new_files, "count": len(new_files),
                    "scanned_dir": pdf_base, "known_count": len(known)})


@app.route("/api/census/ingest", methods=["POST"])
def api_census_ingest():
    """Ingest selected DOIs from census scan."""
    data = request.get_json(silent=True) or {}
    items = data.get("items", [])
    email = _get_email()
    ingested = []
    errors = []

    for item in items:
        doi = item.get("doi")
        fpath = item.get("path")
        doc_type = item.get("doc_type", "article")
        if not doi or not fpath:
            continue
        try:
            from crossref import fetch_metadata
            meta = fetch_metadata(doi, email=email) or {}
            fulltext = None
            try:
                from pdf_extract import extract_text
                fulltext = extract_text(fpath)
            except Exception:
                pass
            insert_paper(
                doi=doi, title=meta.get('title'), authors=meta.get('authors'),
                abstract=meta.get('abstract'), journal=meta.get('journal'),
                year=meta.get('year'), keywords=meta.get('keywords'),
                pdf_path=fpath, full_text=fulltext, doc_type=doc_type,
            )
            # Chunk embedding
            if _embeddings_available():
                try:
                    paper = get_paper_by_doi(doi)
                    if paper and (paper.get('abstract') or paper.get('title')):
                        chunk_texts = create_paper_chunks(paper)
                        if chunk_texts:
                            chunk_rows = []
                            faiss_pairs = []
                            for cidx, text in chunk_texts:
                                emb = get_embedding(text)
                                chunk_rows.append((cidx, text, emb))
                                faiss_pairs.append((cidx, emb))
                            insert_chunks(doi, chunk_rows)
                            try:
                                from faiss_index import add_paper_chunks as faiss_add
                                faiss_add(doi, faiss_pairs)
                            except Exception:
                                pass
                except Exception:
                    pass
            ingested.append(doi)
        except Exception as e:
            errors.append({"doi": doi, "error": str(e)})

    return jsonify({"ingested": ingested, "errors": errors})


# ============================================================
# SETTINGS
# ============================================================

@app.route("/api/settings", methods=["GET"])
def api_get_settings():
    s = get_all_settings()
    # Mask the API key
    if 'openai_api_key' in s and s['openai_api_key']:
        s['openai_api_key_set'] = True
        s['openai_api_key'] = "sk-..." + s['openai_api_key'][-4:]
    else:
        s['openai_api_key_set'] = False
        s['openai_api_key'] = ""
    # Include backend info
    s['backend_info'] = backend_info()
    s['embeddings_available'] = _embeddings_available()
    # FAISS status
    try:
        from faiss_index import is_available as faiss_ok, get_index, needs_rebuild
        s['faiss_available'] = faiss_ok()
        idx, meta = get_index()
        s['faiss_vectors'] = idx.ntotal if idx else 0
        s['faiss_needs_rebuild'] = needs_rebuild()
    except Exception:
        s['faiss_available'] = False
        s['faiss_vectors'] = 0
        s['faiss_needs_rebuild'] = True
    # Chunk stats
    try:
        cstats = count_chunk_stats()
        s['chunk_stats'] = cstats
    except Exception:
        s['chunk_stats'] = {'total_chunks': 0, 'chunks_with_embedding': 0, 'papers_with_chunks': 0}
    return jsonify(s)


@app.route("/api/settings", methods=["PUT"])
def api_put_settings():
    data = request.get_json(silent=True) or {}
    allowed = {"polite_email", "pdf_base_dir", "openai_api_key", "auto_embed",
               "default_doc_type", "embedding_backend", "ollama_url", "ollama_model",
               "sbert_model", "openai_model"}
    for k, v in data.items():
        if k in allowed:
            if k == "openai_api_key" and v.startswith("sk-..."):
                continue
            set_setting(k, str(v))
    # If backend changed, reset the cached backend
    if "embedding_backend" in data:
        try:
            set_backend(data["embedding_backend"])
        except ValueError:
            pass
    return jsonify({"saved": True})


@app.route("/api/faiss/rebuild", methods=["POST"])
def api_faiss_rebuild():
    """Rebuild the FAISS index from all embeddings in the database."""
    try:
        from faiss_index import build_index, is_available
        if not is_available():
            return jsonify({"error": "faiss-cpu not installed. Run: pip install faiss-cpu"}), 400
        build_index(force=True)
        return jsonify({"ok": True, "message": "FAISS index rebuilt"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/reembed", methods=["POST"])
def api_reembed():
    """
    Re-embed papers using the current backend with chunked embeddings.
    only_missing=true: embed papers with no chunks OR wrong dimensions.
    only_missing=false: clear ALL chunks and re-embed everything.
    """
    data = request.get_json(silent=True) or {}
    only_missing = data.get("only_missing", True)

    if not _embeddings_available():
        return jsonify({"error": "No embedding backend available"}), 400

    if only_missing:
        expected_dims = get_embedding_dims()
        papers = get_papers_needing_chunks(expected_dims)
    else:
        # Clear all chunks first for full re-embed
        delete_all_chunks()
        from database import get_connection
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT id, doi, title, authors, abstract, journal, year, keywords, full_text
            FROM papers WHERE title IS NOT NULL OR abstract IS NOT NULL
        """)
        papers = []
        for r in cur.fetchall():
            p = dict(r)
            if isinstance(p.get('authors'), str):
                try: p['authors'] = json.loads(p['authors'])
                except Exception: p['authors'] = []
            if isinstance(p.get('keywords'), str):
                try: p['keywords'] = json.loads(p['keywords'])
                except Exception: p['keywords'] = []
            papers.append(p)
        conn.close()

    if not papers:
        return jsonify({"embedded": 0, "chunks": 0, "message": "Nothing to embed"})

    embedded, total_chunks, errors = 0, 0, 0
    for paper in papers:
        # Ensure authors/keywords are lists
        if isinstance(paper.get('authors'), str):
            try: paper['authors'] = json.loads(paper['authors'])
            except Exception: paper['authors'] = []
        if isinstance(paper.get('keywords'), str):
            try: paper['keywords'] = json.loads(paper['keywords'])
            except Exception: paper['keywords'] = []

        chunk_texts = create_paper_chunks(paper)
        if not chunk_texts:
            continue
        try:
            delete_chunks_for_paper(paper['doi'])
            chunk_rows = []
            for cidx, text in chunk_texts:
                emb = get_embedding(text)
                chunk_rows.append((cidx, text, emb))
            insert_chunks(paper['doi'], chunk_rows)
            embedded += 1
            total_chunks += len(chunk_rows)
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"Embed error {paper['doi']}: {e}")

    # Rebuild FAISS index after bulk embedding
    try:
        from faiss_index import build_index, is_available
        if is_available():
            build_index(force=True)
    except Exception:
        pass

    return jsonify({
        "embedded": embedded, "chunks": total_chunks,
        "errors": errors, "total": len(papers),
    })


# ============================================================
# BROWSE / STATS / FILTERS
# ============================================================

@app.route("/browse")
def browse():
    page = int(request.args.get("page", 1))
    per_page = min(int(request.args.get("per_page", 50)), 200)
    papers = get_all_papers(limit=per_page, offset=(page-1)*per_page)
    for p in papers:
        p.pop('embedding', None)
        p.pop('full_text', None)
    stats = get_stats()
    return jsonify({"papers": papers, "page": page, "per_page": per_page,
                    "total": stats['total_papers'],
                    "total_pages": (stats['total_papers']+per_page-1)//per_page})


@app.route("/stats")
def stats():
    return jsonify({"stats": get_stats(), "journals": get_journals()[:50], "years": get_years()})


@app.route("/api/filters")
def filters():
    return jsonify({"journals": get_journals(), "years": get_years(), "doc_types": get_doc_types()})


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Paper Search Web Server")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--pdf-dir", type=str)
    parser.add_argument("--email", type=str)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    init_db()

    # Build FAISS index from chunks if stale or missing
    try:
        from faiss_index import is_available, needs_rebuild, build_index
        if is_available() and needs_rebuild():
            print("Building FAISS index from chunk embeddings...")
            build_index()
    except Exception as e:
        print(f"FAISS index: {e}")

    if args.pdf_dir:
        set_setting("pdf_base_dir", args.pdf_dir)
        print(f"PDF base directory: {args.pdf_dir}")
    if args.email:
        set_setting("polite_email", args.email)
        print(f"Polite pool email: {args.email}")

    print(f"Starting Paper Search on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)
