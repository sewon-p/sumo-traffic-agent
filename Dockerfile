FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SUMO_SERVER_SKIP_VENV_REEXEC=1 \
    SUMO_HOME=/usr/share/sumo

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    sumo \
    sumo-tools \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# Persistent data directories
RUN mkdir -p /app/data /app/output /app/traffic_data

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/api/status')" || exit 1

CMD ["python", "server.py"]
