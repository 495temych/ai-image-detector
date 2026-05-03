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

# Copy source + model weights (weights are baked in — no credentials needed at runtime)
COPY api/ api/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
