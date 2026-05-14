FROM python:3.11-slim

# opencv-headless requires these system libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first so this layer is cached
# (context = ./api/, so paths are relative to that directory)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source + model weights into /app/api/ so the module path
# `api.main` (used in the CMD) resolves correctly from WORKDIR /app
COPY . /app/api/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
