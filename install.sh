#!/usr/bin/env bash
set -euo pipefail

# ============================================================
#  Paper Database — Automated Installer
#
#  Expected layout (install.sh lives above the source):
#
#    paperdb/
#    ├── install.sh          ← you are here
#    ├── README.md
#    ├── papers.db           ← optional, if sharing a pre-built DB
#    └── src/
#        ├── app.py
#        ├── database.py
#        ├── embeddings.py
#        ├── ...
#        └── templates/
#            └── index.html
#
#  Usage:
#    chmod +x install.sh
#    sudo ./install.sh              # interactive
#    sudo ./install.sh --defaults   # non-interactive, uses defaults
#
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$SCRIPT_DIR/src"

# ---------- defaults ----------
DEFAULT_INSTALL_DIR="/opt/paperdb"
DEFAULT_PDF_DIR="/mnt/storage/library/"
DEFAULT_PORT=5000
DEFAULT_EMAIL="me@uni.edu"
DEFAULT_USER="paperdb"
DEFAULT_MODEL="qwen3-embedding:0.6b"
DEFAULT_BACKEND="ollama"
DEFAULT_MLX_MODEL="mlx-community/bge-small-en-v1.5-4bit"

# ---------- colors ----------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; BOLD='\033[1m'; NC='\033[0m'

info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()   { echo -e "${RED}[ERROR]${NC} $*"; }
die()   { err "$*"; exit 1; }

# ---------- preflight ----------
if [[ $EUID -ne 0 ]]; then
    die "This script must be run as root (sudo ./install.sh)"
fi

if [[ ! -d "$SRC_DIR" ]]; then
    die "Source directory not found at $SRC_DIR — install.sh must be one level above src/"
fi

# ---------- parse args ----------
INTERACTIVE=true
for arg in "$@"; do
    case $arg in
        --defaults) INTERACTIVE=false ;;
        --help|-h)
            echo "Usage: sudo ./install.sh [--defaults]"
            echo "  --defaults   Use default paths and model, no prompts"
            exit 0 ;;
    esac
done

ask() {
    local var=$1 prompt=$2 default=$3
    if $INTERACTIVE; then
        read -rp "$(echo -e "${BLUE}?${NC} ${prompt} [${default}]: ")" input
        eval "$var=\"${input:-$default}\""
    else
        eval "$var=\"$default\""
    fi
}

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║       Paper Search — Installer           ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════╝${NC}"
echo ""

# ============================================================
# Gather configuration
# ============================================================

ask INSTALL_DIR  "Installation directory"                "$DEFAULT_INSTALL_DIR"
ask PDF_DIR      "PDF library directory (or blank)"      "$DEFAULT_PDF_DIR"
ask PORT         "Web server port"                       "$DEFAULT_PORT"
ask EMAIL        "Email for CrossRef/OpenAlex (optional)" "$DEFAULT_EMAIL"
ask SVC_USER     "System user to run the service"        "$DEFAULT_USER"

# ---------- embedding backend + model selection ----------
ask EMBEDDING_BACKEND "Embedding backend (ollama/mlx)" "$DEFAULT_BACKEND"
EMBEDDING_BACKEND="${EMBEDDING_BACKEND,,}"
if [[ "$EMBEDDING_BACKEND" != "ollama" && "$EMBEDDING_BACKEND" != "mlx" ]]; then
    EMBEDDING_BACKEND="$DEFAULT_BACKEND"
fi

OLLAMA_MODEL="$DEFAULT_MODEL"
MLX_MODEL="$DEFAULT_MLX_MODEL"
if $INTERACTIVE && [[ "$EMBEDDING_BACKEND" == "ollama" ]]; then
    echo ""
    echo -e "${BOLD}Select an Ollama embedding model:${NC}"
    echo ""
    echo "  1) nomic-embed-text        768d    274 MB   Fast, good quality"
    echo "  2) qwen3-embedding:0.6b   1024d    490 MB   Good quality, still fast (default)"
    echo "  3) qwen3-embedding:4b     2560d    2.7 GB   High quality (needs ≥8 GB RAM)"
    echo "  4) qwen3-embedding:8b     4096d    5.2 GB   Best quality — MTEB #1 (needs ≥12 GB RAM)"
    echo "  5) mxbai-embed-large      1024d    670 MB   Good alternative"
    echo "  6) all-minilm              384d     46 MB   Tiny/fast, lower quality"
    echo "  7) Custom (enter model name manually)"
    echo ""
    read -rp "$(echo -e "${BLUE}?${NC} Choice [2]: ")" model_choice
    model_choice="${model_choice:-2}"

    case "$model_choice" in
        1) OLLAMA_MODEL="nomic-embed-text" ;;
        2) OLLAMA_MODEL="qwen3-embedding:0.6b" ;;
        3) OLLAMA_MODEL="qwen3-embedding:4b" ;;
        4) OLLAMA_MODEL="qwen3-embedding:8b" ;;
        5) OLLAMA_MODEL="mxbai-embed-large" ;;
        6) OLLAMA_MODEL="all-minilm" ;;
        7)
            read -rp "$(echo -e "${BLUE}?${NC} Ollama model name: ")" OLLAMA_MODEL
            if [[ -z "$OLLAMA_MODEL" ]]; then
                OLLAMA_MODEL="$DEFAULT_MODEL"
                warn "No model entered, using default: $DEFAULT_MODEL"
            fi
            ;;
        *) OLLAMA_MODEL="qwen3-embedding:0.6b" ;;
    esac
fi

if $INTERACTIVE && [[ "$EMBEDDING_BACKEND" == "mlx" ]]; then
    echo ""
    echo -e "${BOLD}Select an MLX embedding model:${NC}"
    echo ""
    echo "  1) mlx-community/bge-small-en-v1.5-4bit      384d   Fastest"
    echo "  2) mlx-community/bge-base-en-v1.5-4bit       768d   Balanced"
    echo "  3) mlx-community/bge-large-en-v1.5-4bit     1024d   Best BGE quality"
    echo "  4) mlx-community/Qwen3-Embedding-0.6B-4bit  1024d   Multilingual"
    echo "  5) mlx-community/Qwen3-Embedding-4B-4bit    2560d   Higher quality"
    echo "  6) mlx-community/Qwen3-Embedding-8B-4bit    4096d   Highest quality"
    echo "  7) Custom (enter model name manually)"
    echo ""
    read -rp "$(echo -e "${BLUE}?${NC} Choice [1]: ")" mlx_choice
    mlx_choice="${mlx_choice:-1}"

    case "$mlx_choice" in
        1) MLX_MODEL="mlx-community/bge-small-en-v1.5-4bit" ;;
        2) MLX_MODEL="mlx-community/bge-base-en-v1.5-4bit" ;;
        3) MLX_MODEL="mlx-community/bge-large-en-v1.5-4bit" ;;
        4) MLX_MODEL="mlx-community/Qwen3-Embedding-0.6B-4bit" ;;
        5) MLX_MODEL="mlx-community/Qwen3-Embedding-4B-4bit" ;;
        6) MLX_MODEL="mlx-community/Qwen3-Embedding-8B-4bit" ;;
        7)
            read -rp "$(echo -e "${BLUE}?${NC} MLX model name: ")" MLX_MODEL
            if [[ -z "$MLX_MODEL" ]]; then
                MLX_MODEL="$DEFAULT_MLX_MODEL"
                warn "No model entered, using default: $DEFAULT_MLX_MODEL"
            fi
            ;;
        *) MLX_MODEL="$DEFAULT_MLX_MODEL" ;;
    esac
fi

if [[ "$EMBEDDING_BACKEND" == "mlx" ]]; then
    EMBEDDING_MODEL="$MLX_MODEL"
else
    EMBEDDING_MODEL="$OLLAMA_MODEL"
fi

# ---------- summary ----------
echo ""
info "Configuration:"
info "  Install dir : $INSTALL_DIR"
info "  PDF dir     : ${PDF_DIR:-<not set>}"
info "  Port        : $PORT"
info "  Email       : ${EMAIL:-<not set>}"
info "  Service user: $SVC_USER"
info "  Backend     : $EMBEDDING_BACKEND"
info "  Embed model : $EMBEDDING_MODEL"
echo ""

if $INTERACTIVE; then
    read -rp "$(echo -e "${YELLOW}Proceed? [Y/n]: ${NC}")" yn
    [[ "${yn,,}" == "n" ]] && die "Aborted."
fi

# ============================================================
# 1. System dependencies
# ============================================================
info "Installing system dependencies..."

if command -v apt-get &>/dev/null; then
    apt-get update -qq
    apt-get install -y -qq python3 python3-venv python3-pip curl >/dev/null
    ok "apt packages installed"
elif command -v dnf &>/dev/null; then
    dnf install -y -q python3 python3-pip curl >/dev/null
    ok "dnf packages installed"
elif command -v pacman &>/dev/null; then
    pacman -Sy --noconfirm python python-pip curl >/dev/null
    ok "pacman packages installed"
else
    warn "Could not detect package manager — make sure python3, pip, and curl are installed"
fi

# ============================================================
# 2. Create service user
# ============================================================
if ! id "$SVC_USER" &>/dev/null; then
    info "Creating system user '$SVC_USER'..."
    useradd --system --shell /usr/sbin/nologin --home-dir "$INSTALL_DIR" "$SVC_USER" 2>/dev/null || true
    ok "User '$SVC_USER' created"
else
    ok "User '$SVC_USER' already exists"
fi

# ============================================================
# 3. Copy application files
# ============================================================
info "Installing application to $INSTALL_DIR..."

mkdir -p "$INSTALL_DIR"

# Copy Python source from src/
for f in app.py database.py embeddings.py citations.py crossref.py \
         pdf_extract.py ingest.py rename_to_doi.py faiss_index.py \
         requirements.txt; do
    if [[ -f "$SRC_DIR/$f" ]]; then
        cp "$SRC_DIR/$f" "$INSTALL_DIR/"
    else
        warn "Missing source file: src/$f"
    fi
done

# Templates
mkdir -p "$INSTALL_DIR/templates"
if [[ -f "$SRC_DIR/templates/index.html" ]]; then
    cp "$SRC_DIR/templates/index.html" "$INSTALL_DIR/templates/"
fi

# Uploads staging directory
mkdir -p "$INSTALL_DIR/uploads"

# Copy database if present (alongside install.sh or in src/)
for loc in "$SCRIPT_DIR" "$SRC_DIR"; do
    if [[ -f "$loc/papers.db" ]]; then
        info "Copying existing papers.db from $loc..."
        cp "$loc/papers.db" "$INSTALL_DIR/"
        ok "Database copied"
        break
    fi
done

# Copy FAISS index if present
for loc in "$SCRIPT_DIR" "$SRC_DIR"; do
    for f in papers.faiss papers.faiss.meta; do
        if [[ -f "$loc/$f" ]]; then
            cp "$loc/$f" "$INSTALL_DIR/"
        fi
    done
done

ok "Application files installed"

# ============================================================
# 4. Python virtual environment
# ============================================================
info "Setting up Python virtual environment..."

python3 -m venv "$INSTALL_DIR/venv"
source "$INSTALL_DIR/venv/bin/activate"

pip install --upgrade pip -q
pip install flask requests PyMuPDF numpy tqdm -q
pip install faiss-cpu -q 2>/dev/null || warn "faiss-cpu install failed — will use brute-force search (still works, just slower for large libraries)"

ok "Python environment ready"
deactivate

if [[ "$EMBEDDING_BACKEND" == "ollama" ]]; then
# ============================================================
# 5. Install Ollama
# ============================================================
info "Installing Ollama..."

if command -v ollama &>/dev/null; then
    ok "Ollama already installed: $(ollama --version 2>/dev/null || echo 'unknown version')"
else
    curl -fsSL https://ollama.com/install.sh | sh
    ok "Ollama installed"
fi

# Start Ollama service (the official installer creates its own systemd unit)
if systemctl list-unit-files 2>/dev/null | grep -q ollama; then
    systemctl enable ollama
    systemctl start ollama
    sleep 3
    ok "Ollama service started"
else
    warn "Ollama systemd unit not found — you may need to run 'ollama serve' manually"
fi

# Quick connectivity check
if curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
    ok "Ollama is responding on localhost:11434"
else
    warn "Ollama doesn't seem to be responding — check 'systemctl status ollama'"
fi

# Pull the selected embedding model
info "Pulling embedding model: $OLLAMA_MODEL ..."
ollama pull "$OLLAMA_MODEL"
ok "Model '$OLLAMA_MODEL' ready"

# Verify the model works
info "Verifying embedding model..."
EMBED_TEST=$(curl -sf http://localhost:11434/api/embed \
    -d "{\"model\":\"$OLLAMA_MODEL\",\"input\":\"test\"}" 2>/dev/null || echo "")
if echo "$EMBED_TEST" | grep -q '"embeddings"'; then
    ok "Embedding model is working"
else
    # Try older endpoint as fallback
    EMBED_TEST=$(curl -sf http://localhost:11434/api/embeddings \
        -d "{\"model\":\"$OLLAMA_MODEL\",\"prompt\":\"test\"}" 2>/dev/null || echo "")
    if echo "$EMBED_TEST" | grep -q '"embedding"'; then
        ok "Embedding model is working (legacy API)"
    else
        warn "Could not verify embedding model — it may still be loading. Check after install."
    fi
fi
else
    info "Installing MLX dependencies in virtual environment..."
    source "$INSTALL_DIR/venv/bin/activate"
    pip install mlx mlx-embeddings -q || warn "MLX packages failed to install; you can install manually later with: pip install mlx mlx-embeddings"
    python3 -c "import mlx_embeddings" >/dev/null 2>&1 && ok "mlx-embeddings import check passed" || warn "mlx-embeddings import check failed"
    deactivate
fi

# ============================================================
# 6. Initialize database & settings
# ============================================================
info "Initializing database and settings..."

source "$INSTALL_DIR/venv/bin/activate"
cd "$INSTALL_DIR"

python3 -c "
from database import init_db, set_setting
init_db()
set_setting('embedding_backend', '${EMBEDDING_BACKEND}')
if '${EMBEDDING_BACKEND}' == 'ollama':
    set_setting('ollama_url', 'http://localhost:11434')
    set_setting('ollama_model', '${OLLAMA_MODEL}')
elif '${EMBEDDING_BACKEND}' == 'mlx':
    set_setting('mlx_model', '${MLX_MODEL}')
pdf_dir = '${PDF_DIR}'
if pdf_dir:
    set_setting('pdf_base_dir', pdf_dir)
email = '${EMAIL}'
if email:
    set_setting('polite_email', email)
set_setting('auto_embed', 'yes')
print('Database initialized, settings saved')
"

ok "Database ready"

# ============================================================
# 6b. Check for embedding dimension mismatch
# ============================================================
info "Checking existing embeddings for compatibility..."

# Run a Python check that detects mismatches and reports status
EMBED_CHECK=$(python3 -c "
from database import get_stats, count_chunk_stats, count_embeddings_by_dims
from embeddings import backend_info, get_embedding_dims

info = backend_info()
dims = get_embedding_dims()
stats = get_stats()
cstats = count_chunk_stats()

total = stats['total_papers']
has_chunks = cstats['papers_with_chunks']
total_chunks = cstats['total_chunks']
needs = total - has_chunks if has_chunks > 0 else total

# Check for dimension mismatch in existing chunks
dim_counts = count_embeddings_by_dims()
wrong_dims = sum(cnt for d, cnt in dim_counts.items() if d != dims)

if total == 0:
    print('STATUS:empty')
elif has_chunks == 0 and total > 0:
    print(f'STATUS:no_chunks:{total}')
elif wrong_dims > 0:
    print(f'STATUS:wrong_dims:{wrong_dims}:{has_chunks}:{dims}')
elif needs > 0:
    print(f'STATUS:partial:{needs}:{has_chunks}')
else:
    print(f'STATUS:ok:{has_chunks}:{total_chunks}')
" 2>/dev/null || echo "STATUS:error")

deactivate

case "$EMBED_CHECK" in
    STATUS:empty)
        ok "No papers in database — nothing to embed yet."
        ;;
    STATUS:ok:*)
        CHUNKED=$(echo "$EMBED_CHECK" | cut -d: -f3)
        CHUNKS=$(echo "$EMBED_CHECK" | cut -d: -f4)
        ok "All $CHUNKED papers have chunk embeddings ($CHUNKS chunks). Dimensions match."
        ;;
    STATUS:no_chunks:*)
        COUNT=$(echo "$EMBED_CHECK" | cut -d: -f3)
        warn "$COUNT papers found but none have chunk embeddings for the current model."
        if $INTERACTIVE; then
            echo ""
            echo -e "  ${BOLD}Papers need to be chunked and embedded with $EMBEDDING_MODEL.${NC}"
            echo "  This will split each paper's text into 400-800 token chunks"
            echo "  and generate an embedding for each chunk."
            echo ""
            echo "  For $COUNT papers, this typically takes:"
            echo "    • qwen3-embedding:0.6b — ~2-5 hours on CPU"
            echo "    • nomic-embed-text     — ~1-3 hours on CPU"
            echo "    • qwen3-embedding:8b   — ~8-20 hours on CPU (or ~1-3h on GPU)"
            echo ""
            echo "  Options:"
            echo "    1) Skip for now — start the server and embed later"
            echo "    2) Start embedding now (runs in foreground, Ctrl-C safe)"
            echo ""
            read -rp "$(echo -e "${BLUE}?${NC} Choice [1]: ")" embed_choice
            embed_choice="${embed_choice:-1}"
            if [[ "$embed_choice" == "2" ]]; then
                info "Starting chunk embedding — this will take a while..."
                source "$INSTALL_DIR/venv/bin/activate"
                cd "$INSTALL_DIR"
                python3 ingest.py --embeddings-only || warn "Embedding encountered errors (partial progress saved)"
                deactivate
            else
                info "Skipping embedding for now."
                echo "  Run later with:"
                echo "    sudo -u $SVC_USER $INSTALL_DIR/venv/bin/python $INSTALL_DIR/ingest.py --embeddings-only"
            fi
        else
            info "Run embedding later: sudo -u $SVC_USER $INSTALL_DIR/venv/bin/python $INSTALL_DIR/ingest.py --embeddings-only"
        fi
        ;;
    STATUS:wrong_dims:*)
        WRONG=$(echo "$EMBED_CHECK" | cut -d: -f3)
        HAS=$(echo "$EMBED_CHECK" | cut -d: -f4)
        TDIMS=$(echo "$EMBED_CHECK" | cut -d: -f5)
        warn "$WRONG of $HAS chunk embeddings have wrong dimensions (need ${TDIMS}d for $EMBEDDING_MODEL)."
        if $INTERACTIVE; then
            echo ""
            echo -e "  ${BOLD}Existing embeddings were created with a different model.${NC}"
            echo "  They need to be re-embedded to work with $EMBEDDING_MODEL."
            echo ""
            echo "  Options:"
            echo "    1) Skip for now — start the server and re-embed later"
            echo "    2) Re-embed now (runs in foreground, Ctrl-C safe)"
            echo ""
            read -rp "$(echo -e "${BLUE}?${NC} Choice [1]: ")" reembed_choice
            reembed_choice="${reembed_choice:-1}"
            if [[ "$reembed_choice" == "2" ]]; then
                info "Starting re-embed — this will take a while..."
                source "$INSTALL_DIR/venv/bin/activate"
                cd "$INSTALL_DIR"
                python3 ingest.py --reembed || warn "Re-embedding encountered errors (partial progress saved)"
                deactivate
            else
                info "Skipping re-embed for now."
                echo "  Run later with:"
                echo "    sudo -u $SVC_USER $INSTALL_DIR/venv/bin/python $INSTALL_DIR/ingest.py --embeddings-only"
                echo "  (This auto-detects dimension mismatches and re-embeds only what's needed.)"
            fi
        else
            info "Run re-embed later: sudo -u $SVC_USER $INSTALL_DIR/venv/bin/python $INSTALL_DIR/ingest.py --embeddings-only"
        fi
        ;;
    STATUS:partial:*)
        NEEDS=$(echo "$EMBED_CHECK" | cut -d: -f3)
        HAS=$(echo "$EMBED_CHECK" | cut -d: -f4)
        warn "$NEEDS papers are missing chunk embeddings ($HAS already done)."
        if $INTERACTIVE; then
            echo ""
            echo "  Options:"
            echo "    1) Skip for now — embed later"
            echo "    2) Embed missing papers now"
            echo ""
            read -rp "$(echo -e "${BLUE}?${NC} Choice [1]: ")" partial_choice
            partial_choice="${partial_choice:-1}"
            if [[ "$partial_choice" == "2" ]]; then
                info "Embedding missing papers..."
                source "$INSTALL_DIR/venv/bin/activate"
                cd "$INSTALL_DIR"
                python3 ingest.py --embeddings-only || warn "Embedding encountered errors"
                deactivate
            else
                info "Skipping — run later with: sudo -u $SVC_USER $INSTALL_DIR/venv/bin/python $INSTALL_DIR/ingest.py --embeddings-only"
            fi
        else
            info "Run later: sudo -u $SVC_USER $INSTALL_DIR/venv/bin/python $INSTALL_DIR/ingest.py --embeddings-only"
        fi
        ;;
    *)
        warn "Could not check embedding status — check after install."
        ;;
esac

# Build FAISS index if there are embeddings
source "$INSTALL_DIR/venv/bin/activate"
cd "$INSTALL_DIR"
python3 -c "
try:
    from faiss_index import build_index, is_available
    if is_available():
        build_index()
except Exception as e:
    print(f'FAISS index skipped: {e}')
"
deactivate

# ============================================================
# 7. Set ownership
# ============================================================
chown -R "$SVC_USER:$SVC_USER" "$INSTALL_DIR"

# If PDF_DIR is set, make sure the service user can read it
if [[ -n "$PDF_DIR" ]] && [[ -d "$PDF_DIR" ]]; then
    PDF_GROUP=$(stat -c '%G' "$PDF_DIR" 2>/dev/null || stat -f '%Sg' "$PDF_DIR" 2>/dev/null || echo "")
    if [[ -n "$PDF_GROUP" ]] && [[ "$PDF_GROUP" != "root" ]]; then
        usermod -aG "$PDF_GROUP" "$SVC_USER" 2>/dev/null || true
        info "Added $SVC_USER to group '$PDF_GROUP' for PDF directory access"
    fi
fi

# ============================================================
# 8. Create systemd service
# ============================================================
info "Creating systemd service..."

UNIT_OLLAMA_AFTER=""
UNIT_OLLAMA_WANTS=""
if [[ "$EMBEDDING_BACKEND" == "ollama" ]]; then
    UNIT_OLLAMA_AFTER=" ollama.service"
    UNIT_OLLAMA_WANTS="ollama.service"
fi

cat > /etc/systemd/system/paperdb.service << UNIT
[Unit]
Description=Paper Search Web Application
After=network.target${UNIT_OLLAMA_AFTER}
${UNIT_OLLAMA_WANTS:+Wants=${UNIT_OLLAMA_WANTS}}

[Service]
Type=simple
User=$SVC_USER
Group=$SVC_USER
WorkingDirectory=$INSTALL_DIR
ExecStart=$INSTALL_DIR/venv/bin/python app.py --host 0.0.0.0 --port $PORT
Restart=on-failure
RestartSec=5

# Hardening
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ReadWritePaths=$INSTALL_DIR
${PDF_DIR:+ReadOnlyPaths=$PDF_DIR}

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable paperdb
systemctl start paperdb

ok "Service installed and started"

# ============================================================
# 9. Verify
# ============================================================
echo ""
info "Waiting for server to start..."
sleep 3

if curl -sf "http://localhost:$PORT/" >/dev/null 2>&1; then
    ok "Paper Search is running at http://localhost:$PORT"
else
    warn "Server may still be starting — check: systemctl status paperdb"
fi

# ============================================================
# Done
# ============================================================
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  Installation complete!                              ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════╝${NC}"
echo ""
echo "  Web UI:    http://localhost:$PORT"
echo "  Backend:   $EMBEDDING_BACKEND"
echo "  Model:     $EMBEDDING_MODEL"
echo "  Logs:      journalctl -u paperdb -f"
echo "  Status:    systemctl status paperdb"
echo "  Restart:   systemctl restart paperdb"
echo "  Stop:      systemctl stop paperdb"
echo ""
echo "  Config:    $INSTALL_DIR   (source + DB + venv)"
echo "  Settings:  Use the web UI Settings tab"
echo ""
if [[ -z "$PDF_DIR" ]]; then
    echo -e "  ${YELLOW}NOTE: No PDF directory was set. Configure it in${NC}"
    echo -e "  ${YELLOW}the Settings tab or re-run the installer.${NC}"
    echo ""
fi
echo "  To generate/update chunk embeddings:"
echo "    sudo -u $SVC_USER $INSTALL_DIR/venv/bin/python $INSTALL_DIR/ingest.py --embeddings-only"
echo ""
echo "  To check embedding status:"
echo "    sudo -u $SVC_USER $INSTALL_DIR/venv/bin/python $INSTALL_DIR/ingest.py --status"
echo ""
