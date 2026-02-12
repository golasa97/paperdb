#!/usr/bin/env python3
"""
Main ingestion script for processing PDFs and building the search database.

Usage:
    python ingest.py /path/to/pdfs [--email your@email.com]
    python ingest.py --embeddings-only   # Chunk+embed papers needing it
    python ingest.py --reembed           # Clear all chunks and re-embed everything
    python ingest.py --status            # Print embedding/chunk status
"""

import argparse
import os
import re
from pathlib import Path
from urllib.parse import unquote

from tqdm import tqdm

from database import (
    init_db, insert_paper, get_paper_by_doi, get_stats,
    get_papers_needing_chunks, count_embeddings_by_dims,
    count_chunk_stats, insert_chunks, delete_all_chunks,
    delete_chunks_for_paper,
)
from crossref import fetch_metadata_batch
from pdf_extract import extract_text_batch
from embeddings import (
    get_embedding, create_paper_chunks, get_backend, backend_info,
    get_embedding_dims,
)


def extract_doi_from_filename(filename: str) -> str | None:
    name = filename
    if name.lower().endswith('.pdf'):
        name = name[:-4]
    name = unquote(name)
    all_underscore_match = re.match(r'^(10)_(\d{4,5})_(.+)$', name)
    if all_underscore_match:
        prefix = f"{all_underscore_match.group(1)}.{all_underscore_match.group(2)}"
        suffix = all_underscore_match.group(3).replace('_', '.')
        return f"{prefix}/{suffix}"
    for encoded in ['_', '--', '∕', '⁄']:
        pattern = rf'(10\.\d+){re.escape(encoded)}(.+)'
        match = re.match(pattern, name)
        if match:
            name = f"{match.group(1)}/{match.group(2)}"
            break
    if re.match(r'10\.\d+/.+', name):
        return name
    return None


def scan_pdf_directory(directory: Path) -> dict[str, Path]:
    pdf_files = {}
    for pdf_path in directory.glob("**/*.pdf"):
        doi = extract_doi_from_filename(pdf_path.name)
        if doi:
            pdf_files[doi] = pdf_path
        else:
            print(f"Warning: Could not extract DOI from filename: {pdf_path.name}")
    return pdf_files


def embed_paper_chunks(paper: dict) -> int:
    """
    Create chunks for a paper, embed each chunk, and store in DB.
    Returns number of chunks embedded.
    """
    chunk_texts = create_paper_chunks(paper)
    if not chunk_texts:
        return 0

    chunk_rows = []
    for chunk_idx, text in chunk_texts:
        emb = get_embedding(text)
        chunk_rows.append((chunk_idx, text, emb))

    insert_chunks(paper['doi'], chunk_rows)
    return len(chunk_rows)


def process_chunk(dois, pdf_files, email, skip_fulltext, skip_embeddings,
                  pdf_workers) -> dict:
    stats = {'metadata': 0, 'fulltext': 0, 'embedded': 0}
    metadata_results = fetch_metadata_batch(dois, email=email, progress=False)
    stats['metadata'] = sum(1 for m in metadata_results.values() if m is not None)
    fulltext_results = {}
    if not skip_fulltext:
        paths = [str(pdf_files[doi]) for doi in dois]
        raw = extract_text_batch(paths, max_workers=pdf_workers, progress=False)
        p2d = {str(v): k for k, v in pdf_files.items() if k in dois}
        fulltext_results = {p2d.get(p, p): t for p, t in raw.items()}
        stats['fulltext'] = sum(1 for t in fulltext_results.values() if t)
    for doi in dois:
        meta = metadata_results.get(doi, {}) or {}
        insert_paper(
            doi=doi, title=meta.get('title'), authors=meta.get('authors'),
            abstract=meta.get('abstract'), journal=meta.get('journal'),
            year=meta.get('year'), keywords=meta.get('keywords'),
            pdf_path=str(pdf_files[doi]), full_text=fulltext_results.get(doi),
        )
    if not skip_embeddings:
        for doi in dois:
            paper = get_paper_by_doi(doi)
            if paper and (paper.get('abstract') or paper.get('title')):
                try:
                    n = embed_paper_chunks(paper)
                    stats['embedded'] += n
                except Exception as e:
                    print(f"Error embedding {doi}: {e}")
    return stats


def _print_embedding_status():
    """Print diagnostic info about current embedding state."""
    info = backend_info()
    print(f"\nEmbedding backend: {info['backend']}")
    print(f"Model: {info['model']}")
    print(f"Target dimensions: {info['dims']}")

    cstats = count_chunk_stats()
    if cstats['total_chunks'] > 0:
        print(f"\nChunk embeddings in database:")
        print(f"  Papers with chunks: {cstats['papers_with_chunks']:>7,}")
        print(f"  Total chunks:       {cstats['total_chunks']:>7,}")

        dims_count = count_embeddings_by_dims()
        if dims_count:
            print(f"\n  Chunks by dimension:")
            for dims, count in sorted(dims_count.items()):
                marker = " ← current model" if dims == info['dims'] else ""
                print(f"    {dims:>5}d: {count:>7,} chunks{marker}")
    else:
        print("\nNo chunk embeddings in database yet.")


def run_embeddings_only(batch_size=50):
    """Generate chunk embeddings for papers that need them."""
    info = backend_info()
    expected_dims = get_embedding_dims()

    _print_embedding_status()

    papers = get_papers_needing_chunks(expected_dims)

    if not papers:
        print(f"\nAll papers already have {expected_dims}d chunk embeddings. Nothing to do.")
        return 0

    print(f"\n{len(papers)} papers need chunk embedding with {info['backend']} ({info['model']}).")

    total_chunks, errors = 0, 0
    for paper in tqdm(papers, desc="Chunking & embedding"):
        try:
            # Delete old chunks for this paper if any (dimension mismatch case)
            delete_chunks_for_paper(paper['doi'])
            n = embed_paper_chunks(paper)
            total_chunks += n
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"Error: {paper['doi']}: {e}")
            if errors == 6:
                print("(suppressing further error messages)")

    print(f"\nDone. {total_chunks} chunks from {len(papers) - errors} papers, {errors} errors.")

    # Rebuild FAISS index
    try:
        from faiss_index import build_index, is_available
        if is_available():
            print("Rebuilding FAISS index...")
            build_index(force=True)
    except Exception:
        pass

    return total_chunks


def run_reembed_all(batch_size=50):
    """Clear ALL chunk embeddings and re-embed everything."""
    info = backend_info()
    _print_embedding_status()

    from database import get_connection
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) as c FROM papers WHERE title IS NOT NULL OR abstract IS NOT NULL")
    total = cur.fetchone()['c']
    conn.close()

    print(f"\nThis will re-embed {total} papers using {info['backend']} ({info['model']}).")
    print("Clearing all existing chunk embeddings...")
    delete_all_chunks()

    print("Starting re-embed...\n")
    return run_embeddings_only(batch_size=batch_size)


def main():
    parser = argparse.ArgumentParser(description="Ingest PDFs into paper search database")
    parser.add_argument("pdf_dir", type=Path, nargs='?', default=None)
    parser.add_argument("--email", type=str)
    parser.add_argument("--skip-embeddings", action="store_true")
    parser.add_argument("--skip-fulltext", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--pdf-workers", type=int, default=8)
    parser.add_argument("--embeddings-only", action="store_true",
                        help="Generate chunk embeddings for papers missing them or with wrong dimensions.")
    parser.add_argument("--reembed", action="store_true",
                        help="Clear ALL chunk embeddings and re-embed everything with current backend.")
    parser.add_argument("--status", action="store_true",
                        help="Print embedding status and exit.")
    args = parser.parse_args()

    print("Initializing database...")
    init_db()

    if args.status:
        s = get_stats()
        print(f"DB: {s['total_papers']} papers, {s['with_chunks']} with chunk embeddings "
              f"({s['total_chunks']} total chunks).")
        _print_embedding_status()
        return 0

    if args.reembed:
        before = get_stats()
        print(f"DB: {before['total_papers']} papers, {before['with_chunks']} with chunks.")
        run_reembed_all(batch_size=args.chunk_size)
        after = get_stats()
        print(f"Now: {after['with_chunks']} papers with chunks ({after['total_chunks']} chunks).")
        return 0

    if args.embeddings_only:
        before = get_stats()
        print(f"DB: {before['total_papers']} papers, {before['with_chunks']} with chunks.")
        run_embeddings_only(batch_size=args.chunk_size)
        after = get_stats()
        print(f"Now: {after['with_chunks']} papers with chunks ({after['total_chunks']} chunks).")
        return 0

    if args.pdf_dir is None:
        parser.error("pdf_dir required unless --embeddings-only, --reembed, or --status")
    if not args.pdf_dir.exists():
        print(f"Error: not found: {args.pdf_dir}")
        return 1

    print(f"Scanning {args.pdf_dir}...")
    pdf_files = scan_pdf_directory(args.pdf_dir)
    print(f"Found {len(pdf_files)} PDFs with valid DOIs")

    if args.limit:
        dois = list(pdf_files.keys())[:args.limit]
        pdf_files = {d: pdf_files[d] for d in dois}

    if not args.force:
        new = [d for d in pdf_files if not get_paper_by_doi(d)]
        if len(new) < len(pdf_files):
            print(f"Skipping {len(pdf_files)-len(new)} already processed")
            pdf_files = {d: pdf_files[d] for d in new}

    if not pdf_files:
        print("No new papers to process")
        return 0

    # Check if embedding backend is available
    embedding_ok = True
    if not args.skip_embeddings:
        try:
            test = get_embedding("test")
            print(f"Embedding backend: {backend_info()['backend']} ({backend_info()['model']})")
        except Exception as e:
            print(f"Warning: Embedding backend not available ({e}), skipping embeddings")
            embedding_ok = False

    dois = list(pdf_files.keys())
    chunks = [dois[i:i+args.chunk_size] for i in range(0, len(dois), args.chunk_size)]
    print(f"\nProcessing {len(dois)} papers in {len(chunks)} chunks")

    total = {'metadata': 0, 'fulltext': 0, 'embedded': 0}
    for i, chunk in enumerate(tqdm(chunks, desc="Processing")):
        cs = process_chunk(chunk, pdf_files, args.email, args.skip_fulltext,
                           args.skip_embeddings or not embedding_ok,
                           args.pdf_workers)
        for k in total:
            total[k] += cs[k]
        if (i+1) % 5 == 0:
            tqdm.write(f"  [{get_stats()['total_papers']} in DB]")

    s = get_stats()
    print(f"\n{'='*50}")
    print(f"Total: {s['total_papers']} | Abstracts: {s['with_abstract']} | "
          f"Chunks: {s['total_chunks']} | Fulltext: {s['with_fulltext']}")

    # Rebuild FAISS index if we generated embeddings
    if total['embedded'] > 0:
        try:
            from faiss_index import build_index, is_available
            if is_available():
                print("Rebuilding FAISS index...")
                build_index(force=True)
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    exit(main())
