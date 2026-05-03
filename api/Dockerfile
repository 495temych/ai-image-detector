FROM python:3.11-slim

# opencv-headless requires these system libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps before copying code so this layer is cached
COPY api/requirements.txt api/requirements.txt
RUN pip install --no-cache-dir -r api/requirements.txt

COPY api/ api/

EXPOSE 8000

# download_artifacts.py skips files that already exist, so this is safe to run every start
CMD ["sh", "-c", "python -m api.download_artifacts && uvicorn api.main:app --host 0.0.0.0 --port 8000"]
