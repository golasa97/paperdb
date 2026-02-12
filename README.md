# Paper Search

A self-hosted academic paper library with full-text search, semantic search, citation tracking, and reference management. Runs entirely locally — no cloud services required.

## What You Need

- **Python 3.10+** (3.11 or 3.12 recommended)
- **~500MB disk** for Ollama + the embedding model
- Your PDF library (files named as DOIs, e.g., `10.1103_PhysRevC.108.034615.pdf`)
- Optionally, a pre-built `papers.db` database file

## Quick Start (Automated)

The install script handles everything: Python venv, pip packages, Ollama, the embedding model, FAISS, and a systemd service.

```bash
# Make the installer executable
chmod +x install.sh

# Run it (interactive — prompts for paths)
sudo ./install.sh

# Or non-interactive with defaults
sudo ./install.sh --defaults
```

The installer will:
1. Install system dependencies (python3, pip, curl)
2. Create a `papersearch` system user
3. Copy app files to `/opt/paper_search`
4. Create a Python virtual environment with all dependencies
5. Install Ollama and prompt you to choose an embedding model
6. Install `faiss-cpu` for fast vector search
7. Initialize the database and configure the embedding backend
8. Create and start a systemd service

After it finishes, open `http://localhost:5000` in a browser.

## Quick Start (Manual)

If you prefer to set things up yourself or aren't on systemd:

### 1. Python environment

```bash
cd paper_search/src
python3 -m venv venv
source venv/bin/activate
pip install flask requests PyMuPDF numpy tqdm faiss-cpu
```

### 2. Install Ollama

```bash
# Linux
curl -fsSL https://ollama.com/install.sh | sh

# macOS
brew install ollama

# Start the server
ollama serve          # runs in foreground
# or on systemd Linux:
sudo systemctl start ollama
```

### 3. Pull the embedding model

```bash
ollama pull nomic-embed-text
```

Verify it works:
```bash
curl http://localhost:11434/api/embeddings \
  -d '{"model":"nomic-embed-text","prompt":"test"}'
```

You should get back JSON with an `"embedding"` array.

### 4. Initialize the database

```bash
python database.py
```

If you were given a `papers.db` file, place it in the `paper_search/` directory (next to `install.sh`). The installer will copy it to the install location. The schema auto-migrates on startup.

### 5. Ingest your PDFs (if starting fresh)

```bash
python ingest.py /path/to/your/pdfs --email you@example.com
```

This scans the directory for PDFs with DOI-based filenames, fetches metadata from CrossRef, extracts full text, and generates chunk embeddings. Each paper's text is split into overlapping 400-800 token chunks, and each chunk is embedded separately for fine-grained semantic search. For a large library this can take hours on the first run. It saves progress after every paper, so it's safe to interrupt and resume.

If you already have a `papers.db` but need to (re-)embed with the current model (e.g., after switching from OpenAI to Ollama, or changing models):

```bash
python ingest.py --embeddings-only  # auto-detects missing/mismatched chunks
python ingest.py --reembed          # nuke all chunks and re-embed from scratch
python ingest.py --status           # check current embedding state
```

### 6. Start the server

```bash
python app.py --port 5000 --pdf-dir /path/to/your/pdfs --email you@example.com
```

Open `http://localhost:5000`.

## Running as a Service (systemd)

If you didn't use the install script, create the unit file manually:

```ini
# /etc/systemd/system/paper_search.service
[Unit]
Description=Paper Search Web Application
After=network.target ollama.service
Wants=ollama.service

[Service]
Type=simple
User=youruser
WorkingDirectory=/path/to/paper_search
ExecStart=/path/to/paper_search/venv/bin/python app.py --host 0.0.0.0 --port 5000
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Then:
```bash
sudo systemctl daemon-reload
sudo systemctl enable paper_search
sudo systemctl start paper_search

# Check status
systemctl status paper_search

# View logs
journalctl -u paper_search -f
```

**Important**: Make sure Ollama's service starts before paper_search. The `Wants=ollama.service` and `After=ollama.service` lines handle this if Ollama was installed via its official installer (which creates `ollama.service`).

## Server Flags

```
python app.py [OPTIONS]
```

| Flag | Default | Description |
|---|---|---|
| `--port` | `5000` | HTTP port |
| `--host` | `0.0.0.0` | Bind address (`0.0.0.0` for LAN access, `127.0.0.1` for local only) |
| `--pdf-dir` | — | Path to your PDF library. Saved to settings DB on startup. |
| `--email` | — | Email for CrossRef/OpenAlex polite pool (faster rate limits). Saved to settings DB. |
| `--debug` | off | Enable Flask debug mode (auto-reload on code changes). **Not for production.** |

The `--pdf-dir` and `--email` flags write to the settings database on startup. After the first run, these are persisted and you don't need to pass them again — but they can also be changed from the web UI's Settings tab at any time.

## Ingestion Flags

```
python ingest.py [pdf_dir] [OPTIONS]
```

| Flag | Default | Description |
|---|---|---|
| `pdf_dir` | — | Directory containing PDFs (not needed with `--embeddings-only`) |
| `--email` | — | Email for CrossRef polite pool |
| `--skip-embeddings` | off | Skip generating embeddings (useful if no backend configured yet) |
| `--skip-fulltext` | off | Skip PDF text extraction |
| `--limit N` | — | Process only N papers (for testing) |
| `--force` | off | Reprocess papers already in the database |
| `--chunk-size N` | `50` | Papers per processing batch |
| `--pdf-workers N` | `8` | Parallel workers for PDF text extraction |
| `--embeddings-only` | off | Generate embeddings for papers that are missing them **or have wrong dimensions** (e.g., after switching models). Detects mismatches automatically. |
| `--reembed` | off | Clear ALL embeddings and re-embed everything with the current backend. Use when switching embedding models. |
| `--status` | off | Print embedding status (backend, model, dimension breakdown) and exit. |

### Common workflows

```bash
# First-time ingest of a big library
python ingest.py /data/papers --email me@uni.edu --chunk-size 100

# Check what embedding state looks like
python ingest.py --status

# After switching to a new model (e.g., qwen3-embedding:8b):
# This detects the 160K old 1536d embeddings and re-embeds them at 4096d
python ingest.py --embeddings-only

# Or nuke all embeddings and start fresh with the current backend
python ingest.py --reembed

# Test with a few papers
python ingest.py /data/papers --limit 10

# Re-ingest everything (metadata + text + embeddings)
python ingest.py /data/papers --force --email me@uni.edu
```

## Features

### Search
- **Keyword search** — SQLite FTS5 full-text search across titles, abstracts, authors, keywords, and full text
- **Semantic search** — Each paper's full text is split into overlapping 400-800 token chunks and embedded separately. Searches match against all chunks, then aggregate to paper level by best-matching chunk. This means a query about a specific method will find papers that discuss it anywhere in the text, not just in the abstract.
- **Hybrid search** — Reciprocal rank fusion of keyword + semantic results
- **Document type filter** — Filter by article, book, technical report, manual, thesis, presentation, conference paper

### Citation Explorer
Click **Citing** on any paper to see who cites it, or **References** to see what it cites. Data comes from [OpenAlex](https://openalex.org/) (free, no key needed).

- Sort by year, citation count, reference count, author h-index, journal, library status
- Filter by keyword across all fields
- Each result is color-coded: green = in your library, red = not in your library
- Select papers and export DOIs to a file, or add them to a list

### Similar Papers
Click **Similar** to find papers with the closest embeddings in your library. Requires embeddings to be generated.

### Lists
Create named lists of papers from search results or the citation/reference panels. Each list supports:
- **Export BibTeX** — Download as `.bib` file or copy to clipboard
- **Export DOIs** — Download as text file
- **Download PDFs** — Zip of all PDFs in the list
- **Delete list** — Removes the list, not the papers

### Import
Drag-and-drop PDFs into the Import tab. The app:
1. Uploads the file
2. Resolves it to a DOI via CrossRef (shows title, authors, confidence score)
3. On confirm: renames to DOI format, moves to your library, fetches metadata, extracts text, generates embedding

### Census
Click the **Census** button to scan your PDF directory for files not yet in the database. Select which to ingest.

### Settings (Web UI)
All configuration is in the Settings tab:
- **Embedding backend**: Ollama / sentence-transformers / OpenAI
- **Model names and URLs** for each backend
- **FAISS index status** with rebuild button
- **Embed missing** / **Re-embed all** buttons
- **Email** for polite API pool
- **PDF directory** path
- **Auto-embed on import** toggle
- **Default document type** for new imports

## Embedding Backends

| Backend | Install | Speed (CPU) | Quality | Dims |
|---|---|---|---|---|
| **Ollama** | Install Ollama + `ollama pull <model>` | ~50 papers/sec | Very good → Best | varies |
| **sentence-transformers** | `pip install sentence-transformers` | ~30 papers/sec | Very good → Best | varies |
| **OpenAI** | `pip install openai` + API key | ~200 papers/sec | Best | 1536 |

### Recommended Ollama models

| Model | Pull command | Dims | Size | Notes |
|---|---|---|---|---|
| `nomic-embed-text` | `ollama pull nomic-embed-text` | 768 | 274 MB | Fast, solid quality. Good default. |
| `qwen3-embedding:0.6b` | `ollama pull qwen3-embedding:0.6b` | 1024 | 490 MB | Good quality, still fast. |
| `qwen3-embedding:4b` | `ollama pull qwen3-embedding:4b` | 2560 | 2.7 GB | High quality. Good if you have ≥8GB RAM. |
| `qwen3-embedding:8b` | `ollama pull qwen3-embedding:8b` | 4096 | 5.2 GB | **#1 on MTEB multilingual** (score 70.58). Best local option. Needs ≥12GB RAM. |
| `mxbai-embed-large` | `ollama pull mxbai-embed-large` | 1024 | 670 MB | Good alternative to nomic. |

### Recommended sentence-transformers models

| Model | HuggingFace ID | Dims | Notes |
|---|---|---|---|
| `bge-large-en-v1.5` | `BAAI/bge-large-en-v1.5` | 1024 | English-focused, good for technical papers |
| `Qwen3-Embedding-0.6B` | `Qwen/Qwen3-Embedding-0.6B` | 1024 | Multilingual, 32K context |
| `Qwen3-Embedding-4B` | `Qwen/Qwen3-Embedding-4B` | 2560 | Higher quality, needs more RAM |
| `Qwen3-Embedding-8B` | `Qwen/Qwen3-Embedding-8B` | 4096 | Best quality, needs GPU |

### Choosing a model

For a nuclear engineering paper library on a typical workstation:
- **8-16 GB RAM, no GPU**: Use `nomic-embed-text` or `qwen3-embedding:0.6b` via Ollama.
- **16-32 GB RAM**: Use `qwen3-embedding:4b` — meaningfully better retrieval quality.
- **32+ GB RAM or GPU**: Use `qwen3-embedding:8b` — state-of-the-art.
- **Cloud / pay-per-use**: OpenAI `text-embedding-3-small` is fast and excellent.

**Switching backends** requires re-embedding your library because the dimensions differ. Use `python ingest.py --embeddings-only` or the Settings tab's "Re-embed All" button. Dimension mismatches are handled gracefully — the app just won't return semantic results until you re-embed.

## FAISS Vector Index

Without FAISS, semantic search scans every chunk embedding in Python (O(N) with numpy). This works for small libraries but gets slow at 100K+ chunks.

With FAISS (`pip install faiss-cpu`), the app builds an index file (`papers.faiss`) over all chunk embeddings. Searches run in ~5-20ms regardless of collection size. Quality is identical — same cosine similarity math, just using optimized C++ with SIMD. Multiple matching chunks from the same paper are aggregated by taking the highest chunk score.

The index auto-builds on server startup if stale. You can also rebuild it from the Settings tab or via the API:

```bash
curl -X POST http://localhost:5000/api/faiss/rebuild
```

**RAM note:** The FAISS index lives in memory. For 160K papers with ~10 chunks each (1.6M chunks) at 1024 dimensions, expect ~6.5 GB of RAM for the index. Systems running larger embedding models (2560d, 4096d) will use proportionally more.

## File Layout

```
paper_search/
├── install.sh              # Automated installer (run from here)
├── README.md               # This file
├── papers.db               # SQLite database (place here to share pre-built DB)
├── papers.faiss            # FAISS index (optional, to share pre-built index)
├── papers.faiss.meta       # FAISS DOI mapping
└── src/                    # Python source code
    ├── app.py              # Flask web server (all routes)
    ├── database.py         # SQLite schema, queries, lists, settings
    ├── embeddings.py       # Multi-backend embedding (Ollama/sbert/OpenAI)
    ├── faiss_index.py      # FAISS index build/search/update
    ├── citations.py        # OpenAlex citation & reference lookup
    ├── crossref.py         # CrossRef metadata fetching
    ├── pdf_extract.py      # PyMuPDF text extraction
    ├── ingest.py           # CLI for bulk PDF ingestion
    ├── rename_to_doi.py    # PDF → DOI resolution via CrossRef
    ├── requirements.txt    # Python dependencies
    ├── templates/
    │   └── index.html      # Web UI (single-page app)
    └── uploads/            # Temporary upload staging area
```

After installation, the deployed copy at `/opt/paper_search` (or wherever you installed) is flat — all `.py` files, `templates/`, `venv/`, and `papers.db` live at the same level.

## API Endpoints

All endpoints are JSON unless noted.

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Web UI |
| `GET` | `/search?q=...&type=keyword&doc_type=all` | Search papers |
| `GET` | `/paper/<doi>` | Paper detail |
| `GET` | `/pdf/<doi>` | Serve PDF file |
| `PATCH` | `/api/paper/<doi>` | Update paper fields (doc_type, title, etc.) |
| `DELETE` | `/api/paper/<doi>` | Delete a paper |
| `GET` | `/citations/<doi>` | Papers that cite this DOI |
| `GET` | `/references/<doi>` | Papers referenced by this DOI |
| `GET` | `/api/similar/<doi>` | Semantically similar papers |
| `POST` | `/api/check-dois` | Check which DOIs exist in DB |
| `GET` | `/api/lists` | List all lists |
| `POST` | `/api/lists` | Create a list |
| `GET` | `/api/lists/<id>` | Get list with papers |
| `DELETE` | `/api/lists/<id>` | Delete a list |
| `POST` | `/api/lists/<id>/papers` | Add papers to list |
| `DELETE` | `/api/lists/<id>/papers` | Remove papers from list |
| `GET` | `/api/lists/<id>/export/bibtex` | Download .bib file |
| `GET` | `/api/lists/<id>/export/dois` | Download DOI list |
| `GET` | `/api/lists/<id>/export/zip` | Download PDFs as zip |
| `POST` | `/api/import/upload` | Upload PDFs |
| `POST` | `/api/import/resolve` | Resolve uploads to DOIs |
| `POST` | `/api/import/confirm` | Confirm import |
| `POST` | `/api/import/delete` | Delete uploaded file |
| `GET` | `/api/census` | Scan for new PDFs |
| `POST` | `/api/census/ingest` | Ingest census results |
| `GET` | `/api/settings` | Get settings |
| `PUT` | `/api/settings` | Update settings |
| `POST` | `/api/faiss/rebuild` | Rebuild FAISS index |
| `POST` | `/api/reembed` | Re-embed papers |
| `GET` | `/browse` | Paginated paper list |
| `GET` | `/stats` | Database statistics |
| `GET` | `/api/filters` | Available filter values |

## Troubleshooting

**"Ollama not available" in Settings**
- Is Ollama running? `systemctl status ollama` or `curl http://localhost:11434/api/tags`
- If not: `sudo systemctl start ollama` or `ollama serve`
- Is `nomic-embed-text` pulled? `ollama list` should show it

**Semantic search returns no results**
- Check that papers have chunk embeddings: run `python ingest.py --status` to see the breakdown
- If 0 chunks: run `python ingest.py --embeddings-only` or click "Embed Missing Papers" in Settings
- If you switched models, the old chunks have wrong dimensions. `--embeddings-only` auto-detects this and re-embeds them, or use `--reembed` to start fresh
- Papers without full text will only have a metadata chunk (title+abstract) — they still work for search but full-text papers get better coverage

**FAISS "needs rebuild" or search is slow**
- Install faiss-cpu: `pip install faiss-cpu`
- Click "Rebuild Index" in Settings, or restart the server (auto-rebuilds on startup)

**Import says "PDF base directory not configured"**
- Go to Settings, set the PDF Base Directory to where your library lives

**Citation lookup fails for some DOIs**
- DOIs with special characters (parentheses, semicolons) are resolved via OpenAlex work ID first. If OpenAlex doesn't know the paper, it won't find citations.
- Very old or obscure papers may not be in OpenAlex.

**Census finds no new files**
- Census scans the PDF base directory for files whose DOI (extracted from the filename) is not in the database. If your PDFs aren't named as DOIs, use the Import tab instead.

## Giving This to Someone Else

To share your library, give them:

1. The entire `paper_search/` directory (which contains `install.sh`, `README.md`, and `src/`)
2. Your `papers.db` file — place it in `paper_search/` next to `install.sh`
3. Your PDF directory
4. Optionally, `papers.faiss` and `papers.faiss.meta` (saves them rebuilding the index) — also place next to `install.sh`

They run `sudo ./install.sh`, pick their embedding model, point it at the PDF directory, and they're up. The installer will prompt them to choose from the available Ollama models and will detect if existing chunk embeddings need to be re-generated for the new model.

If you give them the `.db` file with chunks from a different model (e.g., OpenAI 1536-dim vs Ollama 1024-dim), the installer detects the mismatch and offers to re-embed. The app handles dimension mismatches gracefully in the meantime (just shows no semantic results until re-embedded).
