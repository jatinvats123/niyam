# API image: FastAPI + web UI, with the e5 model and passage embeddings baked in so start-up is fast.
# Build: docker build -t niyam .
# Run:   docker run -p 8000:8000 --env-file .env -v ./data/raw/pdf:/app/data/raw/pdf:ro niyam
# (the PDF volume is optional: without it, search and answers work but PDF links return 404)
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FASTEMBED_CACHE_PATH=/models \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1

# Create the user and its writable dirs first: a `chown -R` after the model download would copy 2 GB
# into a new layer.
RUN useradd --create-home niyam && mkdir /models && chown niyam /models
WORKDIR /app
COPY pyproject.toml README.md ./
COPY niyam ./niyam
# Editable, so niyam/ stays under /app and finds data/ next to it (ROOT is the package's parent).
RUN pip install -e .

COPY --chown=niyam data/processed ./data/processed
USER niyam
# Download the e5 model and load (or, if no cache was copied in, compute) the passage embeddings.
# fastembed copies the 2 GB model file out of the Hugging Face blob store; hard-link the copy back to its
# blob so the layer stores it once. The second, offline load proves the image needs no network to start.
RUN python -c "from niyam.search import DenseIndex, load_chunks; DenseIndex(load_chunks(), 'e5')"     && python -c "import filecmp, os, pathlib; blobs = [b for b in pathlib.Path('/models/blobs').rglob('*') if b.is_file()]; [(f.unlink(), os.link(b, f)) for f in pathlib.Path('/models').glob('models--*/**/*') if f.is_file()  for b in blobs[:] if b.stat().st_size == f.stat().st_size > 10**8 and filecmp.cmp(b, f, shallow=False)  and not blobs.remove(b)]"     && du -sh /models     && HF_HUB_OFFLINE=1 python -c "from niyam.search import DenseIndex, load_chunks; DenseIndex(load_chunks(), 'e5')"

ENV HF_HUB_OFFLINE=1
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "niyam.api:app", "--host", "0.0.0.0", "--port", "8000"]
