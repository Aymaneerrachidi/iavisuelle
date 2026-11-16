FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
ARG INSTALL_EXTRAS=ml

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip && pip install ".[${INSTALL_EXTRAS}]"

COPY configs ./configs
COPY data/README.md ./data/README.md
RUN mkdir -p /app/artifacts /app/data/processed /app/data/uploads

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health')"

CMD ["uvicorn", "vlm_lab.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

