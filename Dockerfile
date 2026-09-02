FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# PennyLane Lightning and XGBoost wheels require the OpenMP runtime.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt \
    && python -m pip check

# Small 2–4 qubit state vectors are slower when OpenMP fans out to every host CPU.
ENV OMP_NUM_THREADS=1

COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY dev-dashboard/ ./dev-dashboard/
COPY artifacts/ ./artifacts/

EXPOSE 8000 8501

HEALTHCHECK --interval=10s --timeout=4s --start-period=120s --retries=12 \
  CMD python -c "import json,urllib.request; p=json.load(urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=3)); raise SystemExit(0 if p.get('models_loaded') else 1)"

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
