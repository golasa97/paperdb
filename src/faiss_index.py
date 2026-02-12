"""
FAISS index for fast vector search over paper chunks.

Maintains a FAISS index over the paper_chunks table. Each chunk gets its
own row in the index. Search results are aggregated to paper level by
taking the maximum chunk score per paper.

The index is stored as two files next to papers.db:
  - papers.faiss       (the FAISS index)
  - papers.faiss.meta  (JSON mapping FAISS row -> [doi, chunk_index])
"""

import json
import numpy as np
from pathlib import Path
from typing import Optional

# Try to import faiss; if missing, we fall back to brute-force
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

INDEX_PATH = Path(__file__).parent / "papers.faiss"
META_PATH = Path(__file__).parent / "papers.faiss.meta"

# Cached in-memory
_index = None
_meta = None  # list of [doi, chunk_index] pairs, aligned with FAISS rows


def is_available() -> bool:
    return FAISS_AVAILABLE


def _load_index():
    """Load index + meta from disk into memory."""
    global _index, _meta
    if not FAISS_AVAILABLE:
        return
    if INDEX_PATH.exists() and META_PATH.exists():
        _index = faiss.read_index(str(INDEX_PATH))
        _meta = json.loads(META_PATH.read_text())
    else:
        _index = None
        _meta = None


def get_index():
    global _index, _meta
    if _index is None:
        _load_index()
    return _index, _meta


def build_index(force: bool = False):
    """
    (Re)build the FAISS index from all chunk embeddings in paper_chunks.

    Uses IndexFlatIP (exact inner product on L2-normalized vectors = cosine
    similarity). For collections up to ~2M chunks this is fast enough
    (sub-20ms queries). For larger collections, consider IndexIVFFlat.
    """
    global _index, _meta

    if not FAISS_AVAILABLE:
        print("faiss not installed — skipping index build")
        return

    from database import get_connection
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT doi, chunk_index, embedding
        FROM paper_chunks
        WHERE embedding IS NOT NULL
        ORDER BY doi, chunk_index
    """)
    rows = cur.fetchall()
    conn.close()

    if not rows:
        print("No chunk embeddings found in database")
        _index = None
        _meta = None
        return

    # Detect dimension from first embedding
    first_emb = np.frombuffer(rows[0]['embedding'], dtype=np.float32)
    dim = first_emb.shape[0]

    # Build matrix
    meta_list = []
    vectors = []
    skipped = 0
    for row in rows:
        emb = np.frombuffer(row['embedding'], dtype=np.float32)
        if emb.shape[0] != dim:
            skipped += 1
            continue
        # L2-normalize for cosine similarity via inner product
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        vectors.append(emb)
        meta_list.append([row['doi'], row['chunk_index']])

    if not vectors:
        print("No valid chunk embeddings after filtering")
        return

    matrix = np.vstack(vectors).astype(np.float32)

    # IndexFlatIP = exact inner product (= cosine sim on normalized vectors)
    index = faiss.IndexFlatIP(dim)
    index.add(matrix)

    # Save to disk
    faiss.write_index(index, str(INDEX_PATH))
    META_PATH.write_text(json.dumps(meta_list))

    _index = index
    _meta = meta_list

    n_papers = len(set(m[0] for m in meta_list))
    print(f"FAISS index built: {len(meta_list)} chunks from {n_papers} papers, dim={dim}"
          f"{f', skipped {skipped} dim mismatches' if skipped else ''}")


def search(query_embedding: np.ndarray, k: int = 50) -> list[tuple[str, float]]:
    """
    Search the FAISS index for papers matching the query.

    Searches against all chunks, then aggregates to paper level by taking
    the maximum chunk score per paper.

    Args:
        query_embedding: The query vector (same dimension as index).
        k: Number of *papers* to return.

    Returns:
        List of (doi, max_similarity_score) tuples, sorted by descending score.
    """
    index, meta = get_index()
    if index is None or meta is None:
        return []

    # L2-normalize query
    qn = np.linalg.norm(query_embedding)
    if qn > 0:
        query_embedding = query_embedding / qn

    q = query_embedding.reshape(1, -1).astype(np.float32)

    # Check dimension match
    if q.shape[1] != index.d:
        return []

    # Fetch more chunks than needed since multiple chunks per paper
    fetch_k = min(k * 10, index.ntotal)
    scores, indices = index.search(q, fetch_k)

    # Aggregate: max score per paper
    paper_scores: dict[str, float] = {}
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(meta):
            continue
        doi = meta[idx][0]
        score_f = float(score)
        if doi not in paper_scores or score_f > paper_scores[doi]:
            paper_scores[doi] = score_f

    # Sort by score and return top k papers
    sorted_papers = sorted(paper_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_papers[:k]


def add_paper_chunks(doi: str, chunk_embeddings: list[tuple[int, np.ndarray]]):
    """
    Incrementally add chunk vectors for a paper.
    chunk_embeddings: list of (chunk_index, embedding) pairs.
    For large batches, prefer build_index().
    """
    global _index, _meta
    index, meta = get_index()

    if index is None:
        build_index()
        return

    if not chunk_embeddings:
        return

    dim = index.d
    vectors = []
    new_meta = []
    for chunk_idx, emb in chunk_embeddings:
        if emb.shape[0] != dim:
            continue
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        vectors.append(emb)
        new_meta.append([doi, chunk_idx])

    if not vectors:
        return

    matrix = np.vstack(vectors).astype(np.float32)
    index.add(matrix)
    meta.extend(new_meta)

    faiss.write_index(index, str(INDEX_PATH))
    META_PATH.write_text(json.dumps(meta))

    _index = index
    _meta = meta


def remove_paper(doi: str):
    """
    FAISS flat indexes don't support removal.
    Rebuild the index when papers are deleted.
    """
    build_index()


def needs_rebuild() -> bool:
    """Check if the index is stale (different count than DB)."""
    index, meta = get_index()
    if index is None:
        return True
    try:
        from database import count_chunk_stats
        stats = count_chunk_stats()
        return stats['chunks_with_embedding'] != index.ntotal
    except Exception:
        return True


if __name__ == "__main__":
    if not FAISS_AVAILABLE:
        print("faiss-cpu not installed. Install with: pip install faiss-cpu")
    else:
        build_index()
