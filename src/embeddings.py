"""
Embeddings module supporting multiple backends:
  - openai:  OpenAI API (text-embedding-3-small, 1536 dims)
  - ollama:  Local Ollama server — supports:
      nomic-embed-text (768d), qwen3-embedding:0.6b (1024d),
      qwen3-embedding:4b (2560d), qwen3-embedding:8b (4096d), and others
  - sbert:   sentence-transformers — supports:
      BAAI/bge-large-en-v1.5 (1024d), Qwen/Qwen3-Embedding-0.6B (1024d), etc.
  - mlx:     Apple MLX embeddings (local, macOS/Apple Silicon)

The active backend is determined by (in order):
  1. Explicit set_backend() call
  2. The 'embedding_backend' key in the settings DB
  3. Auto-detection: tries ollama -> mlx -> sbert -> openai

Dimensions are auto-detected from the first embedding call, or looked up
from the KNOWN_DIMS table for common models.
"""

import os
import numpy as np
import requests as _requests
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from pathlib import Path

# --------------- backend state ---------------

_backend = None        # 'openai' | 'ollama' | 'sbert' | 'mlx'
_sbert_model = None    # lazy-loaded SentenceTransformer instance
_mlx_model = None      # lazy-loaded MLX embedding model instance
_openai_client = None  # lazy-loaded OpenAI client
_ollama_session = None  # lazy-loaded requests session for keep-alive

BACKEND_DEFAULTS = {
    "openai": {"model": "text-embedding-3-small", "dims": 1536},
    "ollama": {"model": "nomic-embed-text",       "dims": 768, "url": "http://localhost:11434"},
    "sbert":  {"model": "BAAI/bge-large-en-v1.5", "dims": 1024},
    "mlx":    {"model": "mlx-community/bge-small-en-v1.5-4bit", "dims": 384},
}

# Known model → default dimension mapping.
# Used to predict dims before the first embedding call.
# Models not listed here will auto-detect dims on first use.
KNOWN_DIMS = {
    # Ollama models
    "nomic-embed-text":          768,
    "mxbai-embed-large":         1024,
    "all-minilm":                384,
    "snowflake-arctic-embed":    1024,
    "qwen3-embedding":           1024,   # default tag is 0.6b
    "qwen3-embedding:0.6b":      1024,
    "qwen3-embedding:4b":        2560,
    "qwen3-embedding:8b":        4096,
    # sentence-transformers / HuggingFace models
    "BAAI/bge-large-en-v1.5":    1024,
    "BAAI/bge-base-en-v1.5":     768,
    "BAAI/bge-small-en-v1.5":    384,
    "Qwen/Qwen3-Embedding-0.6B": 1024,
    "Qwen/Qwen3-Embedding-4B":   2560,
    "Qwen/Qwen3-Embedding-8B":   4096,
    # MLX community models
    "mlx-community/bge-small-en-v1.5-4bit": 384,
    "mlx-community/bge-base-en-v1.5-4bit": 768,
    "mlx-community/bge-large-en-v1.5-4bit": 1024,
    "mlx-community/Qwen3-Embedding-0.6B-4bit": 1024,
    "mlx-community/Qwen3-Embedding-4B-4bit": 2560,
    "mlx-community/Qwen3-Embedding-8B-4bit": 4096,
    # OpenAI models
    "text-embedding-3-small":    1536,
    "text-embedding-3-large":    3072,
    "text-embedding-ada-002":    1536,
}

# Cache: once we see the actual dim from an embedding call, store it here
_detected_dims: Optional[int] = None


def _read_setting(key: str, default: str = "") -> str:
    """Read a setting from the DB without importing database at module level."""
    try:
        from database import get_setting
        return get_setting(key, default)
    except Exception:
        return default


def _detect_backend() -> str:
    """Auto-detect which backend is available."""
    saved = _read_setting("embedding_backend", "")
    if saved in ("openai", "ollama", "sbert", "mlx"):
        return saved

    # Try Ollama
    try:
        url = _read_setting("ollama_url", BACKEND_DEFAULTS["ollama"]["url"])
        r = _requests.get(f"{url}/api/tags", timeout=2)
        if r.ok:
            return "ollama"
    except Exception:
        pass

    # Try MLX embeddings (best on Apple Silicon)
    try:
        import mlx_embeddings  # noqa: F401
        return "mlx"
    except ImportError:
        pass

    # Try sentence-transformers
    try:
        import sentence_transformers  # noqa: F401
        return "sbert"
    except ImportError:
        pass

    # Fall back to OpenAI
    key = _read_setting("openai_api_key", os.environ.get("OPENAI_API_KEY", ""))
    if key:
        return "openai"

    return "ollama"


def get_backend() -> str:
    global _backend
    if _backend is None:
        _backend = _detect_backend()
    return _backend


def set_backend(name: str):
    """Explicitly set the backend."""
    global _backend, _sbert_model, _mlx_model, _openai_client, _ollama_session, _detected_dims
    if name not in ("openai", "ollama", "sbert", "mlx"):
        raise ValueError(f"Unknown backend: {name}")
    _backend = name
    _sbert_model = None
    _mlx_model = None
    _openai_client = None
    _ollama_session = None
    _detected_dims = None


def _get_model_name() -> str:
    """Return the active model name for the current backend."""
    b = get_backend()
    return _read_setting(f"{b}_model", BACKEND_DEFAULTS[b]["model"])


def get_embedding_dims() -> int:
    """
    Return the dimension of the active model.
    Priority: explicit override → detected from actual call → known lookup → backend default.
    """
    # 1. Explicit override in settings
    custom = _read_setting("embedding_dims", "")
    if custom:
        try:
            return int(custom)
        except ValueError:
            pass

    # 2. Detected from a real embedding call
    if _detected_dims is not None:
        return _detected_dims

    # 3. Known model lookup
    model = _get_model_name()
    if model in KNOWN_DIMS:
        return KNOWN_DIMS[model]

    # 4. Backend default
    b = get_backend()
    return BACKEND_DEFAULTS[b]["dims"]


def backend_info() -> dict:
    b = get_backend()
    return {
        "backend": b,
        "model": _get_model_name(),
        "dims": get_embedding_dims(),
    }


# --------------- OpenAI ---------------

def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        key = _read_setting("openai_api_key", os.environ.get("OPENAI_API_KEY", ""))
        if not key:
            raise ValueError("OpenAI API key not configured")
        _openai_client = OpenAI(api_key=key)
    return _openai_client


def _embed_openai(text: str) -> np.ndarray:
    client = _get_openai_client()
    model = _read_setting("openai_model", BACKEND_DEFAULTS["openai"]["model"])
    if len(text) > 30000:
        text = text[:30000]
    resp = client.embeddings.create(model=model, input=text)
    return np.array(resp.data[0].embedding, dtype=np.float32)


def _embed_openai_batch(texts: list[str]) -> list[np.ndarray]:
    client = _get_openai_client()
    model = _read_setting("openai_model", BACKEND_DEFAULTS["openai"]["model"])
    processed = [t[:30000] for t in texts]
    resp = client.embeddings.create(model=model, input=processed)
    return [np.array(item.embedding, dtype=np.float32) for item in resp.data]


def _get_ollama_session():
    global _ollama_session
    if _ollama_session is None:
        _ollama_session = _requests.Session()
    return _ollama_session


def _ollama_embed_payload(text_or_texts):
    model = _read_setting("ollama_model", BACKEND_DEFAULTS["ollama"]["model"])
    payload = {"model": model, "input": text_or_texts}
    keep_alive = _read_setting("ollama_keep_alive", "")
    if keep_alive:
        payload["keep_alive"] = keep_alive
    truncate = _read_setting("ollama_truncate", "")
    if truncate.lower() in {"1", "true", "yes", "on"}:
        payload["truncate"] = True
    return payload


def _ollama_post(path: str, payload: dict, timeout: int):
    url = _read_setting("ollama_url", BACKEND_DEFAULTS["ollama"]["url"])
    retries_raw = _read_setting("ollama_retries", "2").strip()
    retries = int(retries_raw) if retries_raw.isdigit() else 2
    session = _get_ollama_session()
    last_err = None
    for _ in range(max(1, retries + 1)):
        try:
            resp = session.post(f"{url}{path}", json=payload, timeout=timeout)
            resp.raise_for_status()
            return resp
        except Exception as e:
            last_err = e
    if last_err is not None:
        raise last_err
    raise RuntimeError("Ollama request failed")


# --------------- Ollama ---------------

def _embed_ollama(text: str) -> np.ndarray:
    if len(text) > 30000:
        text = text[:30000]
    single_timeout_raw = _read_setting("ollama_timeout_single", "120").strip()
    single_timeout = int(single_timeout_raw) if single_timeout_raw.isdigit() else 120

    # Try newer /api/embed first (required for qwen3-embedding, supports batch)
    try:
        resp = _ollama_post("/api/embed", _ollama_embed_payload(text), timeout=single_timeout)
        data = resp.json()
        embs = data.get("embeddings", [])
        if embs:
            return np.array(embs[0], dtype=np.float32)
    except Exception:
        pass

    # Fall back to older /api/embeddings (for older Ollama versions)
    model = _read_setting("ollama_model", BACKEND_DEFAULTS["ollama"]["model"])
    payload = {"model": model, "prompt": text}
    keep_alive = _read_setting("ollama_keep_alive", "")
    if keep_alive:
        payload["keep_alive"] = keep_alive
    resp = _ollama_post("/api/embeddings", payload, timeout=single_timeout)
    emb = resp.json().get("embedding", [])
    return np.array(emb, dtype=np.float32)


def _embed_ollama_batch(texts: list[str]) -> list[np.ndarray]:
    """Batch embed via /api/embed (supports array input natively)."""
    processed = [t[:30000] for t in texts]
    batch_timeout_raw = _read_setting("ollama_timeout_batch", "600").strip()
    batch_timeout = int(batch_timeout_raw) if batch_timeout_raw.isdigit() else 600
    try:
        resp = _ollama_post("/api/embed", _ollama_embed_payload(processed), timeout=batch_timeout)
        data = resp.json()
        embs = data.get("embeddings", [])
        if embs and len(embs) == len(texts):
            return [np.array(e, dtype=np.float32) for e in embs]
    except Exception:
        pass

    # Fallback: parallel single-item embedding calls to reduce wall-clock time.
    workers_raw = _read_setting("ollama_fallback_workers", "4").strip()
    workers = int(workers_raw) if workers_raw.isdigit() else 4
    workers = max(1, min(workers, len(processed)))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_embed_ollama, processed))


# --------------- sentence-transformers ---------------

def _get_mlx_model():
    """Load an MLX embedding model with broad compatibility across package versions."""
    global _mlx_model
    if _mlx_model is None:
        import mlx_embeddings
        model_name = _read_setting("mlx_model", BACKEND_DEFAULTS["mlx"]["model"])

        if hasattr(mlx_embeddings, "EmbeddingModel"):
            _mlx_model = mlx_embeddings.EmbeddingModel.from_registry(model_name)
        elif hasattr(mlx_embeddings, "load"):
            _mlx_model = mlx_embeddings.load(model_name)
        elif hasattr(mlx_embeddings, "load_model"):
            _mlx_model = mlx_embeddings.load_model(model_name)
        else:
            raise RuntimeError(
                "Unsupported mlx_embeddings package API. "
                "Expected one of: EmbeddingModel, load(), or load_model()."
            )
    return _mlx_model


def _mlx_encode(texts: list[str]):
    """Call whichever encode/embed API the installed mlx_embeddings package exposes."""
    model = _get_mlx_model()

    if hasattr(model, "encode"):
        return model.encode(texts)
    if hasattr(model, "embed"):
        return model.embed(texts)
    if callable(model):
        return model(texts)

    raise RuntimeError("MLX model object does not provide encode/embed/callable interface")


def _normalize_mlx_embeddings(raw, expected_count: int) -> list[np.ndarray]:
    """Normalize potential return shapes from mlx_embeddings into list[np.ndarray]."""
    if isinstance(raw, dict):
        if "embeddings" in raw:
            raw = raw["embeddings"]
        elif "embedding" in raw:
            raw = [raw["embedding"]]

    arr = np.asarray(raw, dtype=np.float32)
    if arr.ndim == 1:
        return [arr]
    if arr.ndim == 2:
        return [row.astype(np.float32) for row in arr]

    raise RuntimeError(f"Unexpected MLX embedding output shape: {arr.shape}")


def _embed_mlx(text: str) -> np.ndarray:
    if len(text) > 30000:
        text = text[:30000]
    return _normalize_mlx_embeddings(_mlx_encode([text]), expected_count=1)[0]


def _embed_mlx_batch(texts: list[str]) -> list[np.ndarray]:
    processed = [t[:30000] for t in texts]
    embs = _normalize_mlx_embeddings(_mlx_encode(processed), expected_count=len(processed))
    if len(embs) != len(processed):
        raise RuntimeError(
            f"MLX backend returned {len(embs)} embeddings for {len(processed)} inputs"
        )
    return embs


# --------------- sentence-transformers ---------------

def _get_sbert_model():
    global _sbert_model
    if _sbert_model is None:
        import torch
        from sentence_transformers import SentenceTransformer
        model_name = _read_setting("sbert_model", BACKEND_DEFAULTS["sbert"]["model"])
        preferred_device = _read_setting("sbert_device", "auto").strip().lower()
        if preferred_device in {"cuda", "mps", "cpu"}:
            device = preferred_device
        elif preferred_device.startswith("cuda:"):
            device = preferred_device
        else:
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        _sbert_model = SentenceTransformer(model_name, device=device)
    return _sbert_model


def _embed_sbert(text: str) -> np.ndarray:
    model = _get_sbert_model()
    if len(text) > 30000:
        text = text[:30000]
    return model.encode(text, normalize_embeddings=True).astype(np.float32)


def _embed_sbert_batch(texts: list[str]) -> list[np.ndarray]:
    model = _get_sbert_model()
    processed = [t[:30000] for t in texts]
    configured_batch_size = _read_setting("sbert_batch_size", "").strip()
    if configured_batch_size.isdigit() and int(configured_batch_size) > 0:
        batch_size = int(configured_batch_size)
    else:
        # Keep CPU memory stable while increasing default GPU throughput.
        device_name = str(getattr(model, "device", "cpu"))
        batch_size = 128 if device_name.startswith(("cuda", "mps")) else 32

    embs = model.encode(
        processed,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=batch_size,
        convert_to_numpy=True,
    )
    return [e.astype(np.float32) for e in embs]


# --------------- unified interface ---------------

def get_client():
    """Legacy compatibility — returns OpenAI client or None for local backends."""
    b = get_backend()
    if b == "openai":
        return _get_openai_client()
    return None


def get_embedding(text: str, client=None) -> np.ndarray:
    """Get embedding using the active backend. `client` param kept for compat."""
    global _detected_dims
    b = get_backend()
    if b == "openai":
        emb = _embed_openai(text)
    elif b == "ollama":
        emb = _embed_ollama(text)
    elif b == "sbert":
        emb = _embed_sbert(text)
    elif b == "mlx":
        emb = _embed_mlx(text)
    else:
        raise ValueError(f"Unknown backend: {b}")
    # Auto-detect dims from the first real call
    if _detected_dims is None:
        _detected_dims = emb.shape[0]
    return emb


def get_embeddings_batch(texts: list[str], batch_size: int = 100,
                         progress: bool = True) -> list[np.ndarray]:
    """Get embeddings for multiple texts."""
    global _detected_dims
    if not texts:
        return []

    b = get_backend()
    if b == "openai":
        all_embs = []
        from tqdm import tqdm as _tqdm
        batches = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]
        it = _tqdm(batches, desc="Embedding") if progress else batches
        for batch in it:
            all_embs.extend(_embed_openai_batch(batch))
        if all_embs and _detected_dims is None:
            _detected_dims = all_embs[0].shape[0]
        return all_embs
    elif b == "ollama":
        from tqdm import tqdm as _tqdm
        all_embs = []
        batch_n_raw = _read_setting("ollama_batch_size", "512").strip()
        batch_n = int(batch_n_raw) if batch_n_raw.isdigit() else 512
        batch_n = max(1, batch_n)
        batches = [texts[i:i+batch_n] for i in range(0, len(texts), batch_n)]
        it = _tqdm(batches, desc="Embedding (Ollama)") if progress else batches
        for batch in it:
            all_embs.extend(_embed_ollama_batch(batch))
        if all_embs and _detected_dims is None:
            _detected_dims = all_embs[0].shape[0]
        return all_embs
    elif b == "sbert":
        embs = _embed_sbert_batch(texts)
        if embs and _detected_dims is None:
            _detected_dims = embs[0].shape[0]
        return embs
    elif b == "mlx":
        embs = _embed_mlx_batch(texts)
        if embs and _detected_dims is None:
            _detected_dims = embs[0].shape[0]
        return embs
    raise ValueError(f"Unknown backend: {b}")


def create_search_text(paper: dict) -> str:
    """Create combined text for embedding from paper metadata (legacy single-embedding)."""
    parts = []
    if paper.get('title'):
        parts.append(f"Title: {paper['title']}")
    if paper.get('abstract'):
        parts.append(f"Abstract: {paper['abstract']}")
    if paper.get('keywords'):
        kw = paper['keywords']
        if isinstance(kw, list):
            parts.append(f"Keywords: {', '.join(kw)}")
        else:
            parts.append(f"Keywords: {kw}")
    if paper.get('full_text'):
        parts.append(f"Content: {paper['full_text'][:2000]}")
    return "\n\n".join(parts)


# --------------- text chunking ---------------

import re as _re


def _make_metadata_prefix(paper: dict) -> str:
    """Short metadata header prepended to each chunk for context."""
    parts = []
    if paper.get('title'):
        parts.append(paper['title'])
    authors = paper.get('authors', [])
    if authors:
        if isinstance(authors, list):
            parts.append(', '.join(authors[:3]) + (' et al.' if len(authors) > 3 else ''))
        else:
            parts.append(str(authors))
    jy = []
    if paper.get('journal'):
        jy.append(paper['journal'])
    if paper.get('year'):
        jy.append(str(paper['year']))
    if jy:
        parts.append(' '.join(jy))
    return ' | '.join(parts)


def chunk_text(text: str, target_tokens: int = 600, min_tokens: int = 400,
               max_tokens: int = 800, overlap_tokens: int = 100) -> list[str]:
    """
    Split text into overlapping chunks, splitting on sentence boundaries.

    Token counts are approximated as chars/4 (reasonable for English).
    Returns a list of chunk strings.
    """
    if not text or not text.strip():
        return []

    # Approximate chars per token
    CPT = 4
    target_chars = target_tokens * CPT
    min_chars = min_tokens * CPT
    max_chars = max_tokens * CPT
    overlap_chars = overlap_tokens * CPT

    # Split into sentences. Handle common abbreviations and decimal numbers
    # by splitting on sentence-ending punctuation followed by space+uppercase or newline.
    sentences = _re.split(r'(?<=[.!?])\s+(?=[A-Z\d"\(])', text)
    # Also split on double newlines (paragraph breaks)
    expanded = []
    for s in sentences:
        parts = s.split('\n\n')
        expanded.extend(p.strip() for p in parts if p.strip())
    sentences = expanded

    if not sentences:
        return [text[:max_chars]] if len(text) > min_chars else [text]

    chunks = []
    current_sents = []
    current_len = 0

    for sent in sentences:
        sent_len = len(sent) + 1  # +1 for joining space

        # If adding this sentence exceeds max, flush
        if current_len + sent_len > max_chars and current_sents:
            chunks.append(' '.join(current_sents))

            # Build overlap from end of current chunk
            overlap_sents = []
            overlap_len = 0
            for s in reversed(current_sents):
                if overlap_len + len(s) + 1 > overlap_chars:
                    break
                overlap_sents.insert(0, s)
                overlap_len += len(s) + 1

            current_sents = overlap_sents
            current_len = overlap_len

        current_sents.append(sent)
        current_len += sent_len

    # Flush remaining
    if current_sents:
        last_text = ' '.join(current_sents)
        if len(last_text) < min_chars and chunks:
            # Too small — merge with previous chunk
            chunks[-1] = chunks[-1] + ' ' + last_text
        else:
            chunks.append(last_text)

    return chunks


def create_paper_chunks(paper: dict) -> list[tuple[int, str]]:
    """
    Create indexed text chunks for a paper.

    Returns list of (chunk_index, text) tuples:
      - chunk 0: metadata chunk (title, abstract, keywords)
      - chunks 1..N: full-text chunks with metadata prefix (if full_text available)

    Each text chunk is ready to be embedded directly.
    """
    prefix = _make_metadata_prefix(paper)
    chunks = []

    # Chunk 0: metadata/summary (always present if paper has title or abstract)
    meta_parts = []
    if paper.get('title'):
        meta_parts.append(f"Title: {paper['title']}")
    if paper.get('abstract'):
        meta_parts.append(f"Abstract: {paper['abstract']}")
    if paper.get('keywords'):
        kw = paper['keywords']
        if isinstance(kw, list):
            meta_parts.append(f"Keywords: {', '.join(kw)}")
        elif kw:
            meta_parts.append(f"Keywords: {kw}")
    if meta_parts:
        chunks.append((0, '\n'.join(meta_parts)))

    # Chunks 1..N: full-text chunks with short prefix
    if paper.get('full_text'):
        text_chunks = chunk_text(paper['full_text'])
        for i, chunk in enumerate(text_chunks):
            chunks.append((i + 1, f"[{prefix}]\n{chunk}"))

    return chunks


if __name__ == "__main__":
    test = "Probability tables for unresolved resonance region self-shielding"
    print(f"Backend: {get_backend()}")
    print(f"Info: {backend_info()}")
    try:
        emb = get_embedding(test)
        print(f"Shape: {emb.shape}, Norm: {np.linalg.norm(emb):.4f}")
    except Exception as e:
        print(f"Error: {e}")
