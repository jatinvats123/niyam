# Hugging Face Space (Docker SDK, free "CPU basic"): Niyam's API and web UI on port 7860.
# The Space repo holds only this file and README.md. The code and the exported index (data/index/*.npz) come
# from GitHub at build time, so no binary file has to go through the Space's git.
# Search-only until a GEMINI_API_KEY secret is added in the Space settings; then /ask streams cited answers.
# Local test: docker build -t niyam-space --build-arg NIYAM_REF=main space/ && docker run -p 7860:7860 niyam-space
FROM python:3.12-slim

# The commit of github.com/jatinvats123/niyam to build: a branch, tag or commit SHA.
ARG NIYAM_REF=main

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FASTEMBED_CACHE_PATH=/models \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    NIYAM_INDEX=file \
    NIYAM_PRELOAD=1 \
    NIYAM_ASK_RATE_LIMIT=10/minute,100/day

# Spaces run the container as uid 1000. Create that user and the model dir before the download: a later
# `chown -R` would copy the 2 GB model into another layer.
RUN useradd --create-home --uid 1000 user && mkdir /models && chown user /models
WORKDIR /app
ADD https://github.com/jatinvats123/niyam/archive/${NIYAM_REF}.tar.gz /tmp/niyam.tar.gz
# Editable, so niyam/ stays under /app and finds data/ next to it (ROOT is the package's parent).
RUN tar -xzf /tmp/niyam.tar.gz --strip-components=1 -C /app && rm /tmp/niyam.tar.gz && pip install -e .

USER user
# Download e5 and load the exported index. fastembed copies the 2 GB model file out of the Hugging Face blob
# store; hard-link the copy back to its blob so the layer stores it once. The offline load proves the
# container needs no network to start.
RUN python -c "from niyam.search import Retriever; Retriever()" \
    && python -c "import filecmp, os, pathlib; blobs = [b for b in pathlib.Path('/models/blobs').rglob('*') if b.is_file()]; [(f.unlink(), os.link(b, f)) for f in pathlib.Path('/models').glob('models--*/**/*') if f.is_file()  for b in blobs[:] if b.stat().st_size == f.stat().st_size > 10**8 and filecmp.cmp(b, f, shallow=False)  and not blobs.remove(b)]" \
    && du -sh /models \
    && HF_HUB_OFFLINE=1 python -c "from niyam.search import Retriever; print(len(Retriever().dense.chunks), 'passages')"

ENV HF_HUB_OFFLINE=1
EXPOSE 7860
# --proxy-headers: the Space's proxy passes the visitor's address in X-Forwarded-For, which the /ask rate limit keys on.
CMD ["uvicorn", "niyam.api:app", "--host", "0.0.0.0", "--port", "7860", "--proxy-headers", "--forwarded-allow-ips", "*"]
