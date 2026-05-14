# AI Image Detector — Local UI

Node.js frontend that connects to the FastAPI backend and lets you upload images to check if they are real or AI-generated. Includes a GradCAM heatmap view and a feedback system that saves corrections for future model retraining.

---

## Prerequisites

- [Node.js](https://nodejs.org/) v18+
- Python API running locally on port 8000 (see root README)

---

## Setup

### 1. Start the Python API

From the **repo root**:

```bash
# First time only — download model artifacts
cp api/.env.example api/.env
# fill in DAGSHUB_KEY_ID and DAGSHUB_S3, then:
python -m api.download_artifacts

# Start the API
uvicorn api.main:app --port 8000
```

### 2. Start the UI

```bash
cd ui
npm install
npm start
```

Open **http://localhost:3000**

---

## Usage

1. Drop or paste an image into the upload zone
2. The model returns a verdict — **Real** or **AI-generated** — with a confidence score
3. Toggle **GradCAM** to see which parts of the image the model focused on
4. Click **Yes / No** to confirm or correct the prediction — your feedback is saved locally

---

## Feedback data

User feedback is stored in `ui/feedback/`:

```
feedback/
├── feedback.jsonl      ← one JSON entry per submission
└── images/             ← uploaded images saved with UUID filenames
```

Each entry looks like:

```json
{
  "id": "uuid",
  "timestamp": "2026-05-02T16:00:00.000Z",
  "image_path": "feedback/images/uuid.jpg",
  "image_original_name": "photo.jpg",
  "model_label": "fake",
  "model_confidence": 0.94,
  "model_version": "efficientnet_v1",
  "run_id": "2561d86a...",
  "user_correct": false,
  "true_label": "real"
}
```

Only incorrect predictions are useful for retraining — entries where `user_correct` is `false` are the ones to focus on.

---

## Stack

| Layer | Tech |
|---|---|
| Server | Node.js + Express |
| API proxy | http-proxy-middleware |
| File uploads | multer |
| Fonts | Fraunces + Inter + JetBrains Mono |
