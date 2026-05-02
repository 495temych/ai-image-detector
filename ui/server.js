const express = require('express');
const { createProxyMiddleware } = require('http-proxy-middleware');
const multer = require('multer');
const { v4: uuidv4 } = require('uuid');
const path = require('path');
const fs = require('fs');

const app = express();
const PORT = 3000;
const API_URL = 'http://localhost:8000';

const FEEDBACK_DIR = path.join(__dirname, 'feedback');
const IMAGES_DIR   = path.join(FEEDBACK_DIR, 'images');
const LOG_FILE     = path.join(FEEDBACK_DIR, 'feedback.jsonl');

fs.mkdirSync(IMAGES_DIR, { recursive: true });

const storage = multer.diskStorage({
  destination: IMAGES_DIR,
  filename: (req, file, cb) => cb(null, `${uuidv4()}${path.extname(file.originalname) || '.jpg'}`),
});
const upload = multer({ storage, limits: { fileSize: 10 * 1024 * 1024 } });

app.use('/api', createProxyMiddleware({
  target: API_URL,
  changeOrigin: true,
  pathRewrite: { '^/api': '' },
}));

app.use(express.static(path.join(__dirname, 'public')));

app.post('/feedback', upload.single('image'), (req, res) => {
  const { model_label, model_confidence, model_version, run_id, user_correct, true_label } = req.body;

  const entry = {
    id: uuidv4(),
    timestamp: new Date().toISOString(),
    image_path: req.file ? path.relative(__dirname, req.file.path) : null,
    image_original_name: req.file?.originalname || null,
    model_label,
    model_confidence: parseFloat(model_confidence),
    model_version,
    run_id,
    user_correct: user_correct === 'true',
    true_label: user_correct === 'true' ? model_label : true_label,
  };

  fs.appendFileSync(LOG_FILE, JSON.stringify(entry) + '\n', 'utf8');
  res.json({ ok: true, id: entry.id });
});

app.listen(PORT, () => {
  console.log(`AI Image Detector UI → http://localhost:${PORT}`);
  console.log(`Feedback log         → ${LOG_FILE}`);
});
