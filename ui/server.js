const express = require('express');
const { createProxyMiddleware } = require('http-proxy-middleware');
const multer = require('multer');
const { v4: uuidv4 } = require('uuid');
const path = require('path');
const fs = require('fs');
const {
  saveSession, getImageStatsMany, getGlobalStats,
  getRecentSessions, getImageExtendedMany, seedIfEmpty,
  getKPIs, getDriftIndicators, getConfidenceDistribution, getTrend,
} = require('./db');

// Seed analytics data on first run so Challenge mode has community stats
// immediately — runs synchronously before the server begins accepting requests.
seedIfEmpty();

const app = express();
const PORT = 3000;
const API_URL = process.env.API_URL || 'http://localhost:8000';

// image_samples lives one level up from server.js both locally (ui/../) and in Docker (/app/../)
const SAMPLES_DIR = path.join(__dirname, '..', 'image_samples');

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

function getImagePool(label) {
  const dir = path.join(SAMPLES_DIR, label);
  return fs.readdirSync(dir)
    .filter(f => f.endsWith('.jpg') || f.endsWith('.png'))
    .map(f => ({ id: `${label}/${f}`, true_label: label, file: f, label }));
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

// GET /challenge/image/:label/:file — serve benchmark image by label and filename
app.get('/challenge/image/:label/:file', (req, res) => {
  const { label, file } = req.params;
  if (!['fake', 'real'].includes(label)) return res.status(400).end();
  res.sendFile(path.join(SAMPLES_DIR, label, file));
});

// GET /challenge/session
// Picks 5 fake + 5 real from the filesystem pool, runs all 10 /predict-explain in parallel,
// caches results in memory.  Returns {session_id, images:[{id,src}]} — answers not exposed yet.
app.get('/challenge/session', async (req, res) => {
  try {
    const fakePool = getImagePool('fake');
    const realPool = getImagePool('real');
    const entries = [
      ...pickRandom(fakePool, 5),
      ...pickRandom(realPool, 5),
    ].sort(() => Math.random() - 0.5);

    const sessionId = uuidv4();

    // Fire all predict-explain calls in parallel
    const items = await Promise.all(entries.map(async (entry, idx) => {
      const filePath = path.join(SAMPLES_DIR, entry.label, entry.file);
      const fileBuffer = fs.readFileSync(filePath);
      const blob = new Blob([fileBuffer], { type: 'image/jpeg' });
      const form = new FormData();
      form.append('file', blob, entry.file);
      const r = await fetch(`${API_URL}/predict-explain`, { method: 'POST', body: form });
      const data = await r.json();
      return {
        id: idx,
        image_id: `${entry.label}/${entry.file}`,
        src: `/challenge/image/${entry.label}/${entry.file}`,
        filePath,
        true_label: entry.label,
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

// POST /challenge/submit
// Receives {session_id, answers:[...]} and returns full comparison + summary.
// Logs the session to feedback/sessions.jsonl for later analysis.
app.post('/challenge/submit', express.json(), (req, res) => {
  const { session_id, answers } = req.body;
  const session = sessions.get(session_id);
  if (!session) return res.status(404).json({ error: 'Session not found or expired' });

  const results = session.map((item, i) => ({
    id:               item.id,
    image_id:         item.image_id,
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

  // ── Persist to SQLite (best-effort) ──
  try {
    saveSession(session_id, results.map(r => ({
      image_id:      r.image_id,
      true_label:    r.true_label,
      user_answer:   r.user_answer,
      model_label:   r.model_label,
      model_conf:    r.model_confidence,
      human_correct: r.human_correct,
      model_correct: r.model_correct,
    })));
  } catch (dbErr) {
    console.warn('DB save failed (non-fatal):', dbErr.message);
  }

  // ── Community + global stats (queried after save so this session is counted) ──
  const community = getImageStatsMany(results.map(r => r.image_id));
  const extended  = getImageExtendedMany(results.map(r => r.image_id));
  const global    = getGlobalStats();
  const trend     = getRecentSessions(20);

  // Find hardest image in this session by community human confusion rate
  const hardest = results.reduce((worst, r) => {
    const stats = community[r.image_id];
    if (!worst || stats.human_confusion_pct > community[worst.image_id].human_confusion_pct)
      return r;
    return worst;
  }, null);

  // ── JSONL log (best-effort) ──
  try {
    const logEntry = {
      session_id,
      timestamp: new Date().toISOString(),
      summary,
      results: results.map(({ id, image_id, image_num, true_label, model_label, model_confidence, user_answer, human_correct, model_correct }) =>
        ({ id, image_id, image_num, true_label, model_label, model_confidence, user_answer, human_correct, model_correct })
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
  res.json({ session_id, results, summary, community, extended, global, trend, hardest_image_id: hardest?.image_id });
});

// ── Dashboard ──
app.get('/dashboard/stats', (req, res) => {
  try {
    res.json({
      kpis:      getKPIs(),
      drift:     getDriftIndicators(),
      conf_dist: getConfidenceDistribution(),
      trend:     getTrend(),
      model_info: {
        version:  'efficientnet_v1',
        run_id:   '2561d86a3f22495e91b2cc7d3d1d3497',
        auc:      0.962,
        test_acc: 0.896,
      },
    });
  } catch (err) {
    console.error('dashboard/stats error:', err);
    res.status(500).json({ error: err.message });
  }
});

app.listen(PORT, () => {
  console.log(`AI Image Detector UI → http://localhost:${PORT}`);
  console.log(`Feedback log         → ${LOG_FILE}`);
});
