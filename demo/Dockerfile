# Optional: container image for the DEMO APP (FastAPI + stanza NER + embeddings).
# The demo app is CPU-only — the GPU is used solely by the separate vLLM server.
# Recommended for a first deployment is the native venv path (see DEPLOYMENT.md);
# use this image only if you want full containerization / docker-compose startup.
#
# Build from the REPOSITORY ROOT so the whole project is in the build context:
#   docker build -f demo/Dockerfile -t georelating-demo .
FROM python:3.12-slim

# System build deps (a few wheels compile from source, e.g. the `jq` binding)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git libgomp1 curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/models/hf \
    STANZA_RESOURCES_DIR=/models/stanza

# Install Python deps first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project
COPY . .

# Pre-download the local ML models so the first request is fast and needs no
# network (stanza English NER + the sentence-transformers few-shot embedder).
RUN python -c "import stanza; stanza.download('en', processors='tokenize,ner')" && \
    python -c "from sentence_transformers import SentenceTransformer; \
               SentenceTransformer('sentence-transformers/all-mpnet-base-v2')"

EXPOSE 8123
# The pipeline calls git rev-parse at import time; mark the repo dir safe.
RUN git config --global --add safe.directory /app || true

CMD ["uvicorn", "demo.server:app", "--host", "0.0.0.0", "--port", "8123"]
