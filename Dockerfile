# Commute Policy Assistant — API service
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY data/policies/ data/policies/
COPY eval/cases.jsonl eval/cases.jsonl

# Persistent state (SQLite catalog, Chroma vectors, JWT secret, model cache)
# lives in /data; mount a volume here to survive restarts.
ENV DATA_DIR=/data \
    POLICIES_DIR=/srv/app/data/policies \
    OLLAMA_BASE_URL=http://ollama:11434

RUN mkdir -p /data
EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3)"

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
