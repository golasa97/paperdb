"""
Database module for paper metadata storage and search.
Uses SQLite with FTS5 for full-text search.
Includes lists, settings, and document-type support.
"""

import sqlite3
import json
from pathlib import Path
from typing import Optional
import numpy as np

DB_PATH = Path(__file__).parent / "papers.db"


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


# ---------- schema ----------

def init_db():
    """Initialize / migrate the database schema."""
    conn = get_connection()
    cur = conn.cursor()

    # Main papers table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS papers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doi TEXT UNIQUE NOT NULL,
            title TEXT,
            authors TEXT,          -- JSON array
            abstract TEXT,
            journal TEXT,
            year INTEGER,
            keywords TEXT,         -- JSON array
            pdf_path TEXT,
            full_text TEXT,
            embedding BLOB,        -- numpy float32 bytes
            doc_type TEXT DEFAULT 'article',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Migration: add doc_type if missing
    cols = {r[1] for r in cur.execute("PRAGMA table_info(papers)").fetchall()}
    if "doc_type" not in cols:
        cur.execute("ALTER TABLE papers ADD COLUMN doc_type TEXT DEFAULT 'article'")

    # FTS5
    cur.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS papers_fts USING fts5(
            doi, title, authors, abstract, keywords, full_text,
            content='papers', content_rowid='id'
        )
    """)

    # Triggers
    cur.execute("""
        CREATE TRIGGER IF NOT EXISTS papers_ai AFTER INSERT ON papers BEGIN
            INSERT INTO papers_fts(rowid, doi, title, authors, abstract, keywords, full_text)
            VALUES (new.id, new.doi, new.title, new.authors, new.abstract, new.keywords, new.full_text);
        END
    """)
    cur.execute("""
        CREATE TRIGGER IF NOT EXISTS papers_ad AFTER DELETE ON papers BEGIN
            INSERT INTO papers_fts(papers_fts, rowid, doi, title, authors, abstract, keywords, full_text)
            VALUES ('delete', old.id, old.doi, old.title, old.authors, old.abstract, old.keywords, old.full_text);
        END
    """)
    cur.execute("""
        CREATE TRIGGER IF NOT EXISTS papers_au AFTER UPDATE ON papers BEGIN
            INSERT INTO papers_fts(papers_fts, rowid, doi, title, authors, abstract, keywords, full_text)
            VALUES ('delete', old.id, old.doi, old.title, old.authors, old.abstract, old.keywords, old.full_text);
            INSERT INTO papers_fts(rowid, doi, title, authors, abstract, keywords, full_text)
            VALUES (new.id, new.doi, new.title, new.authors, new.abstract, new.keywords, new.full_text);
        END
    """)

    cur.execute("CREATE INDEX IF NOT EXISTS idx_papers_year ON papers(year)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_papers_journal ON papers(journal)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_papers_doc_type ON papers(doc_type)")

    # ---- Lists ----
    cur.execute("""
        CREATE TABLE IF NOT EXISTS lists (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS list_papers (
            list_id INTEGER NOT NULL,
            doi TEXT NOT NULL,
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (list_id, doi),
            FOREIGN KEY (list_id) REFERENCES lists(id) ON DELETE CASCADE
        )
    """)

    # ---- Settings ----
    cur.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)

    # ---- Paper chunks (for chunked embeddings) ----
    cur.execute("""
        CREATE TABLE IF NOT EXISTS paper_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doi TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            chunk_text TEXT,
            embedding BLOB,
            UNIQUE(doi, chunk_index),
            FOREIGN KEY (doi) REFERENCES papers(doi) ON DELETE CASCADE
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doi ON paper_chunks(doi)")

    conn.commit()
    conn.close()


# ---------- papers CRUD ----------

def insert_paper(
    doi: str,
    title: Optional[str] = None,
    authors: Optional[list] = None,
    abstract: Optional[str] = None,
    journal: Optional[str] = None,
    year: Optional[int] = None,
    keywords: Optional[list] = None,
    pdf_path: Optional[str] = None,
    full_text: Optional[str] = None,
    embedding: Optional[np.ndarray] = None,
    doc_type: Optional[str] = None,
) -> int:
    conn = get_connection()
    cur = conn.cursor()
    blob = embedding.tobytes() if embedding is not None else None
    cur.execute("""
        INSERT INTO papers (doi,title,authors,abstract,journal,year,keywords,pdf_path,full_text,embedding,doc_type)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(doi) DO UPDATE SET
            title     = COALESCE(excluded.title, title),
            authors   = COALESCE(excluded.authors, authors),
            abstract  = COALESCE(excluded.abstract, abstract),
            journal   = COALESCE(excluded.journal, journal),
            year      = COALESCE(excluded.year, year),
            keywords  = COALESCE(excluded.keywords, keywords),
            pdf_path  = COALESCE(excluded.pdf_path, pdf_path),
            full_text = COALESCE(excluded.full_text, full_text),
            embedding = COALESCE(excluded.embedding, embedding),
            doc_type  = COALESCE(excluded.doc_type, doc_type)
    """, (
        doi, title,
        json.dumps(authors) if authors else None,
        abstract, journal, year,
        json.dumps(keywords) if keywords else None,
        pdf_path, full_text, blob, doc_type,
    ))
    pid = cur.lastrowid
    conn.commit()
    conn.close()
    return pid


def delete_paper(doi: str) -> bool:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM list_papers WHERE doi = ?", (doi,))
    cur.execute("DELETE FROM paper_chunks WHERE doi = ?", (doi,))
    cur.execute("DELETE FROM papers WHERE doi = ?", (doi,))
    deleted = cur.rowcount > 0
    conn.commit()
    conn.close()
    return deleted


def update_paper_field(doi: str, field: str, value) -> bool:
    allowed = {"doc_type", "title", "authors", "abstract", "journal", "year", "keywords"}
    if field not in allowed:
        return False
    conn = get_connection()
    if field in ("authors", "keywords") and isinstance(value, list):
        value = json.dumps(value)
    conn.execute(f"UPDATE papers SET {field} = ? WHERE doi = ?", (value, doi))
    conn.commit()
    conn.close()
    return True


# ---------- search ----------

def _parse_row(r: dict) -> dict:
    r['authors'] = json.loads(r['authors']) if r.get('authors') else []
    r['keywords'] = json.loads(r['keywords']) if r.get('keywords') else []
    return r


def search_fts(query: str, limit: int = 50, doc_type: Optional[str] = None) -> list[dict]:
    conn = get_connection()
    cur = conn.cursor()
    if doc_type and doc_type != "all":
        cur.execute("""
            SELECT p.*, bm25(papers_fts) as score
            FROM papers_fts
            JOIN papers p ON papers_fts.rowid = p.id
            WHERE papers_fts MATCH ? AND p.doc_type = ?
            ORDER BY score LIMIT ?
        """, (query, doc_type, limit))
    else:
        cur.execute("""
            SELECT p.*, bm25(papers_fts) as score
            FROM papers_fts
            JOIN papers p ON papers_fts.rowid = p.id
            WHERE papers_fts MATCH ?
            ORDER BY score LIMIT ?
        """, (query, limit))
    results = [_parse_row(dict(row)) for row in cur.fetchall()]
    conn.close()
    return results


def search_semantic(query_embedding: np.ndarray, limit: int = 50, doc_type: Optional[str] = None) -> list[dict]:
    # Try FAISS first (indexes chunks, returns paper-level aggregated results)
    try:
        from faiss_index import search as faiss_search, is_available as faiss_ok
        if faiss_ok():
            hits = faiss_search(query_embedding, k=limit * 5)  # over-fetch for aggregation
            if hits:
                results = []
                for doi, score in hits:
                    p = get_paper_by_doi(doi)
                    if not p:
                        continue
                    if doc_type and doc_type != "all" and p.get("doc_type") != doc_type:
                        continue
                    p['score'] = score
                    p.pop('embedding', None)
                    results.append(p)
                    if len(results) >= limit:
                        break
                return results
    except Exception:
        pass

    # Brute-force fallback against paper_chunks
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT doi, chunk_index, embedding FROM paper_chunks WHERE embedding IS NOT NULL")
    rows = cur.fetchall()
    conn.close()
    if not rows:
        return []

    qn = np.linalg.norm(query_embedding)
    if qn == 0:
        return []

    # Score each chunk, aggregate max per paper
    paper_scores: dict[str, float] = {}
    for row in rows:
        emb = np.frombuffer(row['embedding'], dtype=np.float32)
        if emb.shape[0] != query_embedding.shape[0]:
            continue
        score = float(np.dot(query_embedding, emb) / (qn * np.linalg.norm(emb)))
        if row['doi'] not in paper_scores or score > paper_scores[row['doi']]:
            paper_scores[row['doi']] = score

    sorted_dois = sorted(paper_scores, key=lambda d: paper_scores[d], reverse=True)
    results = []
    for doi in sorted_dois[:limit * 2]:
        p = get_paper_by_doi(doi)
        if not p:
            continue
        if doc_type and doc_type != "all" and p.get("doc_type") != doc_type:
            continue
        p['score'] = paper_scores[doi]
        p.pop('embedding', None)
        results.append(p)
        if len(results) >= limit:
            break
    return results


def find_similar(doi: str, limit: int = 20) -> list[dict]:
    """Find papers with chunk embeddings most similar to the given paper."""
    # Get the best chunk embedding for this paper (chunk 0 = metadata summary)
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT embedding FROM paper_chunks
        WHERE doi = ? AND embedding IS NOT NULL
        ORDER BY chunk_index LIMIT 1
    """, (doi,))
    row = cur.fetchone()
    if not row:
        conn.close()
        return []
    target = np.frombuffer(row['embedding'], dtype=np.float32)
    conn.close()

    # Try FAISS (returns paper-level aggregated results)
    try:
        from faiss_index import search as faiss_search, is_available as faiss_ok
        if faiss_ok():
            hits = faiss_search(target, k=limit + 5)
            results = []
            for hit_doi, score in hits:
                if hit_doi == doi:
                    continue
                p = get_paper_by_doi(hit_doi)
                if not p:
                    continue
                p['score'] = score
                p.pop('embedding', None)
                results.append(p)
                if len(results) >= limit:
                    break
            return results
    except Exception:
        pass

    # Brute-force fallback against chunks
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT doi, embedding FROM paper_chunks WHERE doi != ? AND embedding IS NOT NULL", (doi,))
    rows = cur.fetchall()
    conn.close()
    if not rows:
        return []

    tn = np.linalg.norm(target)
    if tn == 0:
        return []

    paper_scores: dict[str, float] = {}
    for r in rows:
        emb = np.frombuffer(r['embedding'], dtype=np.float32)
        if emb.shape[0] != target.shape[0]:
            continue
        score = float(np.dot(target, emb) / (tn * np.linalg.norm(emb)))
        if r['doi'] not in paper_scores or score > paper_scores[r['doi']]:
            paper_scores[r['doi']] = score

    sorted_dois = sorted(paper_scores, key=lambda d: paper_scores[d], reverse=True)
    results = []
    for d in sorted_dois[:limit]:
        p = get_paper_by_doi(d)
        if p:
            p['score'] = paper_scores[d]
            p.pop('embedding', None)
            results.append(p)
    return results


# ---------- getters ----------

def get_all_papers(limit: int = 1000, offset: int = 0) -> list[dict]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM papers ORDER BY year DESC, title LIMIT ? OFFSET ?", (limit, offset))
    results = [_parse_row(dict(r)) for r in cur.fetchall()]
    conn.close()
    return results


def get_paper_by_doi(doi: str) -> Optional[dict]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM papers WHERE doi = ?", (doi,))
    row = cur.fetchone()
    conn.close()
    return _parse_row(dict(row)) if row else None


def check_dois_exist(dois: list[str]) -> dict[str, bool]:
    if not dois:
        return {}
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("CREATE TEMP TABLE IF NOT EXISTS _ck (doi TEXT)")
    cur.execute("DELETE FROM _ck")
    for i in range(0, len(dois), 500):
        cur.executemany("INSERT INTO _ck (doi) VALUES (?)", [(d,) for d in dois[i:i+500]])
    cur.execute("SELECT c.doi, CASE WHEN p.doi IS NOT NULL THEN 1 ELSE 0 END AS e FROM _ck c LEFT JOIN papers p ON c.doi=p.doi")
    result = {r['doi']: bool(r['e']) for r in cur.fetchall()}
    cur.execute("DROP TABLE IF EXISTS _ck")
    conn.close()
    for d in dois:
        result.setdefault(d, False)
    return result


def get_papers_missing_embeddings() -> list[dict]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT id,doi,title,authors,abstract,journal,year,keywords
        FROM papers
        WHERE embedding IS NULL AND (title IS NOT NULL OR abstract IS NOT NULL)
        ORDER BY id
    """)
    results = [_parse_row(dict(r)) for r in cur.fetchall()]
    conn.close()
    return results


def get_papers_wrong_embedding_dims(expected_dims: int) -> list[dict]:
    """Find papers whose embedding dimension doesn't match the expected size.
    Embeddings are stored as raw float32, so byte length = dims * 4."""
    expected_bytes = expected_dims * 4
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT id,doi,title,authors,abstract,journal,year,keywords
        FROM papers
        WHERE embedding IS NOT NULL
          AND LENGTH(embedding) != ?
          AND (title IS NOT NULL OR abstract IS NOT NULL)
        ORDER BY id
    """, (expected_bytes,))
    results = [_parse_row(dict(r)) for r in cur.fetchall()]
    conn.close()
    return results


def count_embeddings_by_dims() -> dict[int, int]:
    """Return a dict mapping embedding dimension -> count of papers with that dim."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT LENGTH(embedding)/4 as dims, COUNT(*) as cnt
        FROM paper_chunks
        WHERE embedding IS NOT NULL
        GROUP BY dims
        ORDER BY cnt DESC
    """)
    results = {row['dims']: row['cnt'] for row in cur.fetchall()}
    conn.close()
    return results


# ---------- paper_chunks CRUD ----------

def insert_chunks(doi: str, chunks: list[tuple[int, str, Optional[np.ndarray]]]):
    """
    Insert or replace chunks for a paper.
    chunks: list of (chunk_index, chunk_text, embedding_or_None)
    """
    conn = get_connection()
    cur = conn.cursor()
    for chunk_index, chunk_text, embedding in chunks:
        blob = embedding.tobytes() if embedding is not None else None
        cur.execute("""
            INSERT INTO paper_chunks (doi, chunk_index, chunk_text, embedding)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(doi, chunk_index) DO UPDATE SET
                chunk_text = excluded.chunk_text,
                embedding = excluded.embedding
        """, (doi, chunk_index, chunk_text, blob))
    conn.commit()
    conn.close()


def delete_chunks_for_paper(doi: str):
    """Remove all chunks for a paper."""
    conn = get_connection()
    conn.execute("DELETE FROM paper_chunks WHERE doi = ?", (doi,))
    conn.commit()
    conn.close()


def delete_all_chunks():
    """Remove all chunks (for full re-embed)."""
    conn = get_connection()
    conn.execute("DELETE FROM paper_chunks")
    conn.commit()
    conn.close()


def get_chunks_for_paper(doi: str) -> list[dict]:
    """Return all chunks for a paper, ordered by chunk_index."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM paper_chunks WHERE doi = ? ORDER BY chunk_index", (doi,))
    results = [dict(r) for r in cur.fetchall()]
    conn.close()
    return results


def get_papers_needing_chunks(expected_dims: int) -> list[dict]:
    """
    Find papers that need (re-)chunking:
    - Papers with no chunks at all
    - Papers whose chunks have wrong embedding dimensions
    Returns paper metadata rows.
    """
    expected_bytes = expected_dims * 4
    conn = get_connection()
    cur = conn.cursor()

    # Papers with no chunks at all
    cur.execute("""
        SELECT p.id, p.doi, p.title, p.authors, p.abstract, p.journal,
               p.year, p.keywords, p.full_text
        FROM papers p
        LEFT JOIN paper_chunks pc ON p.doi = pc.doi
        WHERE pc.id IS NULL
          AND (p.title IS NOT NULL OR p.abstract IS NOT NULL)
        ORDER BY p.id
    """)
    no_chunks = [_parse_row(dict(r)) for r in cur.fetchall()]

    # Papers whose chunks have wrong dimensions
    cur.execute("""
        SELECT DISTINCT p.id, p.doi, p.title, p.authors, p.abstract, p.journal,
               p.year, p.keywords, p.full_text
        FROM papers p
        JOIN paper_chunks pc ON p.doi = pc.doi
        WHERE pc.embedding IS NOT NULL AND LENGTH(pc.embedding) != ?
        ORDER BY p.id
    """, (expected_bytes,))
    wrong_dims = [_parse_row(dict(r)) for r in cur.fetchall()]

    conn.close()

    # Deduplicate
    seen = set()
    result = []
    for p in no_chunks + wrong_dims:
        if p['doi'] not in seen:
            seen.add(p['doi'])
            result.append(p)
    return result


def count_chunk_stats() -> dict:
    """Return chunk-level statistics."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) as c FROM paper_chunks")
    total_chunks = cur.fetchone()['c']
    cur.execute("SELECT COUNT(*) as c FROM paper_chunks WHERE embedding IS NOT NULL")
    with_emb = cur.fetchone()['c']
    cur.execute("SELECT COUNT(DISTINCT doi) as c FROM paper_chunks")
    papers_chunked = cur.fetchone()['c']
    conn.close()
    return {
        'total_chunks': total_chunks,
        'chunks_with_embedding': with_emb,
        'papers_with_chunks': papers_chunked,
    }


def get_stats() -> dict:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) as c FROM papers")
    total = cur.fetchone()['c']
    cur.execute("SELECT COUNT(*) as c FROM papers WHERE abstract IS NOT NULL")
    wa = cur.fetchone()['c']
    cur.execute("SELECT COUNT(*) as c FROM papers WHERE embedding IS NOT NULL")
    we = cur.fetchone()['c']
    cur.execute("SELECT COUNT(*) as c FROM papers WHERE full_text IS NOT NULL")
    wf = cur.fetchone()['c']
    cur.execute("SELECT MIN(year) as mn, MAX(year) as mx FROM papers WHERE year IS NOT NULL")
    yr = cur.fetchone()
    # Chunk stats
    cur.execute("SELECT COUNT(DISTINCT doi) as c FROM paper_chunks WHERE embedding IS NOT NULL")
    papers_chunked = cur.fetchone()['c']
    cur.execute("SELECT COUNT(*) as c FROM paper_chunks WHERE embedding IS NOT NULL")
    total_chunks = cur.fetchone()['c']
    conn.close()
    return {
        'total_papers': total, 'with_abstract': wa,
        'with_embedding': we, 'with_fulltext': wf,
        'with_chunks': papers_chunked,
        'total_chunks': total_chunks,
        'year_range': (yr['mn'], yr['mx']) if yr['mn'] else None,
    }


def get_journals() -> list[tuple]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT journal, COUNT(*) as c FROM papers WHERE journal IS NOT NULL GROUP BY journal ORDER BY c DESC")
    r = [(row['journal'], row['c']) for row in cur.fetchall()]
    conn.close()
    return r


def get_years() -> list[tuple]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT year, COUNT(*) as c FROM papers WHERE year IS NOT NULL GROUP BY year ORDER BY year DESC")
    r = [(row['year'], row['c']) for row in cur.fetchall()]
    conn.close()
    return r


def get_doc_types() -> list[tuple]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT doc_type, COUNT(*) as c FROM papers WHERE doc_type IS NOT NULL GROUP BY doc_type ORDER BY c DESC")
    r = [(row['doc_type'], row['c']) for row in cur.fetchall()]
    conn.close()
    return r


# ---------- lists ----------

def create_list(name: str, description: str = "") -> int:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("INSERT INTO lists (name, description) VALUES (?,?)", (name, description))
    lid = cur.lastrowid
    conn.commit()
    conn.close()
    return lid


def get_all_lists() -> list[dict]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT l.*, COUNT(lp.doi) as paper_count
        FROM lists l LEFT JOIN list_papers lp ON l.id = lp.list_id
        GROUP BY l.id ORDER BY l.created_at DESC
    """)
    r = [dict(row) for row in cur.fetchall()]
    conn.close()
    return r


def get_list(list_id: int) -> Optional[dict]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM lists WHERE id = ?", (list_id,))
    lst = cur.fetchone()
    if not lst:
        conn.close()
        return None
    lst = dict(lst)
    cur.execute("""
        SELECT p.doi, p.title, p.authors, p.journal, p.year, p.doc_type, p.pdf_path, p.abstract, p.keywords
        FROM list_papers lp JOIN papers p ON lp.doi = p.doi
        WHERE lp.list_id = ? ORDER BY lp.added_at DESC
    """, (list_id,))
    lst['papers'] = [_parse_row(dict(r)) for r in cur.fetchall()]
    conn.close()
    return lst


def delete_list(list_id: int) -> bool:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM lists WHERE id = ?", (list_id,))
    ok = cur.rowcount > 0
    conn.commit()
    conn.close()
    return ok


def add_papers_to_list(list_id: int, dois: list[str]) -> int:
    conn = get_connection()
    cur = conn.cursor()
    added = 0
    for d in dois:
        try:
            cur.execute("INSERT OR IGNORE INTO list_papers (list_id, doi) VALUES (?,?)", (list_id, d))
            added += cur.rowcount
        except Exception:
            pass
    conn.commit()
    conn.close()
    return added


def remove_papers_from_list(list_id: int, dois: list[str]) -> int:
    conn = get_connection()
    cur = conn.cursor()
    removed = 0
    for d in dois:
        cur.execute("DELETE FROM list_papers WHERE list_id = ? AND doi = ?", (list_id, d))
        removed += cur.rowcount
    conn.commit()
    conn.close()
    return removed


# ---------- settings ----------

def get_setting(key: str, default: str = "") -> str:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT value FROM settings WHERE key = ?", (key,))
    row = cur.fetchone()
    conn.close()
    return row['value'] if row else default


def set_setting(key: str, value: str):
    conn = get_connection()
    conn.execute("INSERT INTO settings (key,value) VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))
    conn.commit()
    conn.close()


def get_all_settings() -> dict:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT key, value FROM settings")
    r = {row['key']: row['value'] for row in cur.fetchall()}
    conn.close()
    return r


# ---------- census ----------

def get_all_dois() -> set[str]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT doi FROM papers")
    r = {row['doi'] for row in cur.fetchall()}
    conn.close()
    return r


if __name__ == "__main__":
    init_db()
    print(f"Database initialized at {DB_PATH}")
