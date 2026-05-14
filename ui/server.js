const express = require('express');
const { createProxyMiddleware } = require('http-proxy-middleware');
const multer = require('multer');
const { v4: uuidv4 } = require('uuid');
const path = require('path');
const fs = require('fs');

const app = express();
const PORT = 3000;
const API_URL = process.env.API_URL || 'http://localhost:8000';

// picture_samples lives one level up from ui/ locally; in Docker it's copied alongside server.js
const SAMPLES_DIR = process.env.SAMPLES_DIR || path.join(__dirname, '..', 'picture_samples');

// ── Challenge: in-memory session store ──
const sessions = new Map(); // session_id → Array<SessionItem>

function pickRandom(arr, n) {
  const copy = [...arr];
  for (let i = copy.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  return copy.slice(0, n);
}

function readSampleFiles(label) {
  const dir = path.join(SAMPLES_DIR, label);
  return fs.readdirSync(dir).filter(f => /\.(jpe?g|png|webp)$/i.test(f));
}

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

// ── Challenge routes ──

// GET /challenge/session
// Picks 5 fake + 5 real, runs all 10 /predict-explain in parallel, caches results.
// Returns {session_id, images:[{id,src}]} — no labels, no model results exposed yet.
app.get('/challenge/session', async (req, res) => {
  try {
    const fakeFiles = pickRandom(readSampleFiles('fake'), 5);
    const realFiles = pickRandom(readSampleFiles('real'), 5);
    const entries = [
      ...fakeFiles.map(f => ({ file: f, label: 'fake' })),
      ...realFiles.map(f => ({ file: f, label: 'real' })),
    ];
    // shuffle entries
    for (let i = entries.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [entries[i], entries[j]] = [entries[j], entries[i]];
    }

    const sessionId = uuidv4();

    // Fire all predict-explain calls in parallel
    const items = await Promise.all(entries.map(async ({ file, label }, idx) => {
      const filePath = path.join(SAMPLES_DIR, label, file);
      const fileBuffer = fs.readFileSync(filePath);
      const blob = new Blob([fileBuffer], { type: 'image/jpeg' });
      const form = new FormData();
      form.append('file', blob, file);
      const r = await fetch(`${API_URL}/predict-explain`, { method: 'POST', body: form });
      const data = await r.json();
      return {
        id: idx,
        src: `/challenge/image/${sessionId}/${idx}`,
        filePath,
        true_label: label,
        model_label: data.label,
        model_confidence: data.confidence,
        gradcam_b64: data.gradcam_base64 || null,
      };
    }));

    sessions.set(sessionId, items);
    res.json({ session_id: sessionId, images: items.map(({ id, src }) => ({ id, src })) });
  } catch (err) {
    console.error('challenge/session error:', err);
    res.status(500).json({ error: err.message });
  }
});

// GET /challenge/image/:session_id/:index
// Serves the image file without leaking its true label in the URL.
app.get('/challenge/image/:session_id/:index', (req, res) => {
  const session = sessions.get(req.params.session_id);
  if (!session) return res.status(404).json({ error: 'Session not found' });
  const item = session[parseInt(req.params.index, 10)];
  if (!item) return res.status(404).json({ error: 'Image not found' });
  res.sendFile(item.filePath);
});

// POST /challenge/submit
// Receives {session_id, answers:[...]} and returns full comparison + summary.
// Logs the session to feedback/sessions.jsonl for later analysis.
app.post('/challenge/submit', express.json(), (req, res) => {
  const { session_id, answers } = req.body;
  const session = sessions.get(session_id);
  if (!session) return res.status(404).json({ error: 'Session not found or expired' });

  const results = session.map((item, i) => ({
    id:               item.id,
    image_num:        i + 1,
    src:              item.src,
    true_label:       item.true_label,
    model_label:      item.model_label,
    model_confidence: item.model_confidence,
    gradcam_b64:      item.gradcam_b64,
    user_answer:      answers[i],
    human_correct:    answers[i] === item.true_label,
    model_correct:    item.model_label === item.true_label,
  }));

  const yourAcc  = results.filter(r => r.human_correct).length / results.length;
  const modelAcc = results.filter(r => r.model_correct).length / results.length;

  const summary = {
    your_accuracy:  yourAcc,
    model_accuracy: modelAcc,
    both_fooled:    results.filter(r => !r.human_correct && !r.model_correct).map(r => r.image_num),
    you_beat_model: results.filter(r =>  r.human_correct && !r.model_correct).map(r => r.image_num),
    verdict: yourAcc > modelAcc ? 'You win this round'
           : yourAcc < modelAcc ? 'Model wins this round'
           : 'Tie',
  };

  // Log session result for later analysis (best-effort — never block the response)
  try {
    const logEntry = {
      session_id,
      timestamp: new Date().toISOString(),
      summary,
      results: results.map(({ id, image_num, true_label, model_label, model_confidence, user_answer, human_correct, model_correct }) =>
        ({ id, image_num, true_label, model_label, model_confidence, user_answer, human_correct, model_correct })
      ),
    };
    fs.appendFileSync(
      path.join(FEEDBACK_DIR, 'sessions.jsonl'),
      JSON.stringify(logEntry) + '\n',
      'utf8'
    );
  } catch (logErr) {
    console.warn('Session log write failed (non-fatal):', logErr.message);
  }

  sessions.delete(session_id); // single-use
  res.json({ session_id, results, summary });
});

app.listen(PORT, () => {
  console.log(`AI Image Detector UI → http://localhost:${PORT}`);
  console.log(`Feedback log         → ${LOG_FILE}`);
});
