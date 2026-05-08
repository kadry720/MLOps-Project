FROM python:3.11.9-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN addgroup --system app && \
    adduser --system --ingroup app --home /app app

COPY requirements-serving.txt .
RUN python -m pip install --upgrade pip==24.0 && \
    python -m pip install --no-cache-dir -r requirements-serving.txt

COPY src ./src
COPY configs ./configs

RUN mkdir -p /app/models /app/data/splits && \
    chown -R app:app /app

USER app

EXPOSE 8000
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import json, urllib.request; data = json.load(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5)); raise SystemExit(0 if data.get('status') == 'ok' else 1)"

CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
