#!/usr/bin/env bash
# One-command start for macOS/Linux (no Docker required).
# Everything stays inside this project folder.
set -uo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
command -v "$PYTHON" >/dev/null 2>&1 || { echo "[start] Python 3.10+ is required on PATH"; exit 1; }

if [ ! -x .venv/bin/python ] && [ ! -x .venv/Scripts/python.exe ]; then
    echo "[start] creating virtualenv (.venv)..."
    "$PYTHON" -m venv .venv || exit 1
fi
if [ -x .venv/bin/python ]; then VPY=.venv/bin/python; else VPY=.venv/Scripts/python.exe; fi

if [ ! -f .venv/.deps-installed ]; then
    echo "[start] installing dependencies..."
    PIP_CACHE_DIR="$PWD/.runtime/pip-cache" "$VPY" -m pip install --upgrade pip -q || exit 1
    PIP_CACHE_DIR="$PWD/.runtime/pip-cache" "$VPY" -m pip install -q -r requirements.txt || exit 1
    echo ok > .venv/.deps-installed
fi

# A local .env (e.g. a hosted model such as Google Gemini) overrides the local
# Ollama setup entirely: the app reads it, the script stays out of the way.
if [ -f .env ]; then
    echo "[start] .env found - using its LLM configuration; skipping local Ollama setup."
    exec "$VPY" -m uvicorn app.main:app --host 0.0.0.0 --port 8000
fi

# --- ensure the local LLM backend (Ollama, portable, project-local) ---------
export LLM_MODEL="${LLM_MODEL:-llama3.2:3b}"
export OLLAMA_HOST="127.0.0.1:11434"
export OLLAMA_MODELS="$PWD/.runtime/ollama-models"
mkdir -p .runtime

OLLAMA_BIN="$(command -v ollama 2>/dev/null || true)"
if [ -z "$OLLAMA_BIN" ] && [ -x .runtime/ollama/ollama ]; then
    OLLAMA_BIN="$PWD/.runtime/ollama/ollama"
fi

if [ -z "$OLLAMA_BIN" ]; then
    echo "[start] downloading portable Ollama into .runtime (one time)..."
    case "$(uname -s)/$(uname -m)" in
        Linux/x86_64)  URL=https://ollama.com/download/ollama-linux-amd64.tgz; ARCH=tgz ;;
        Linux/arm64)   URL=https://ollama.com/download/ollama-linux-arm64.tgz; ARCH=tgz ;;
        Darwin/*)      URL=https://ollama.com/download/ollama-darwin.zip; ARCH=zip ;;
        *) URL=""; ARCH="" ;;
    esac
    if [ -n "$URL" ]; then
        mkdir -p .runtime/ollama-dl
        curl -L --retry 3 -o .runtime/ollama-dl/ollama."$ARCH" "$URL" || true
        case "$ARCH" in
            tgz) mkdir -p .runtime/ollama && tar -xzf .runtime/ollama-dl/ollama.tgz -C .runtime/ollama ;;
            zip) mkdir -p .runtime/ollama && unzip -q -o .runtime/ollama-dl/ollama.zip -d .runtime/ollama
                 OLLAMA_BIN="$PWD/.runtime/ollama/Ollama.app/Contents/Resources/ollama" ;;
        esac
        [ -x "$OLLAMA_BIN" ] || OLLAMA_BIN="$PWD/.runtime/ollama/ollama"
        [ -x "$OLLAMA_BIN" ] || OLLAMA_BIN=""
    fi
fi

if [ -n "$OLLAMA_BIN" ]; then
    if ! curl -s -o /dev/null "http://127.0.0.1:11434/" 2>/dev/null; then
        echo "[start] starting Ollama server..."
        ("$OLLAMA_BIN" serve > .runtime/ollama-serve.log 2>&1 &)
        sleep 4
    fi
    if ! "$OLLAMA_BIN" list 2>/dev/null | grep -q "$LLM_MODEL"; then
        echo "[start] pulling $LLM_MODEL (~2 GB, one time)..."
        "$OLLAMA_BIN" pull "$LLM_MODEL" || { echo "[start] model pull failed"; OLLAMA_BIN=""; }
    fi
fi

if [ -z "$OLLAMA_BIN" ] || ! curl -s -o /dev/null "http://127.0.0.1:11434/" 2>/dev/null; then
    echo "[start] WARNING: local LLM unavailable. Starting with the deterministic"
    echo "[start] extractive fallback (LLM_PROVIDER=extractive). See README."
    export LLM_PROVIDER=extractive
else
    export LLM_PROVIDER=ollama
fi

echo
echo "[start] Commute Policy Assistant on http://127.0.0.1:8000  (Ctrl+C to stop)"
echo "[start] token:   $VPY -m app.cli issue-token --company acme --role employee"
echo
exec "$VPY" -m uvicorn app.main:app --host 0.0.0.0 --port 8000
