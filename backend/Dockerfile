FROM python:3.11-slim

WORKDIR /app

# System deps for scipy / XGBoost / torch C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgomp1 wget curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# 1. Install CPU-only torch wheel first (avoids downloading the CUDA variant
#    that requirements.txt would pull by default).
RUN pip install --no-cache-dir \
    torch==2.3.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# 2. Install everything else normally — torch is already satisfied above so
#    pip won't re-download it, and all transitive deps (click, filelock, …)
#    are resolved cleanly.
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app
ENV DATA_DIR=/data/nslkdd
ENV MODEL_DIR=/models/aercf

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s \
  CMD curl -f http://localhost:8000/status || exit 1

CMD ["python", "-m", "uvicorn", "api.main:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
