const API_BASE = '/api';

// Elements
const dropZone      = document.getElementById('drop-zone');
const fileInput     = document.getElementById('file-input');
const browseBtn     = document.getElementById('browse-btn');
const dzIdle        = document.getElementById('dz-idle');
const dzPreview     = document.getElementById('dz-preview');
const previewImg    = document.getElementById('preview-img');
const imgOverlay    = document.getElementById('img-overlay');
const gradcamImg    = document.getElementById('gradcam-img');
const imgControls   = document.getElementById('img-controls');
const btnOriginal   = document.getElementById('btn-original');
const btnGradcam    = document.getElementById('btn-gradcam');
const belowImage    = document.getElementById('below-image');
const resetBtn      = document.getElementById('reset-btn');
const filenameEl    = document.getElementById('filename');
const resultEmpty   = document.getElementById('result-empty');
const resultCard    = document.getElementById('result-card');
const analyzingState= document.getElementById('analyzing-state');
const verdictIcon   = document.getElementById('verdict-icon');
const verdictLabel  = document.getElementById('verdict-label');
const verdictSub    = document.getElementById('verdict-sub');
const gaugeFill     = document.getElementById('gauge-fill');
const confNum       = document.getElementById('conf-num');
const confDesc      = document.getElementById('conf-desc');
const confBarFill   = document.getElementById('conf-bar-fill');
const metaModel     = document.getElementById('meta-model');
const metaRun       = document.getElementById('meta-run');
const rawReadout    = document.getElementById('raw-readout');
const feedbackPrompt  = document.getElementById('feedback-prompt');
const feedbackDone    = document.getElementById('feedback-done');
const feedbackDoneText= document.getElementById('feedback-done-text');
const fbYes           = document.getElementById('fb-yes');
const fbNo            = document.getElementById('fb-no');
const toast         = document.getElementById('toast');
const toastText     = document.getElementById('toast-text');
const toastClose    = document.getElementById('toast-close');
const apiDot        = document.getElementById('api-dot');
const apiStatus     = document.getElementById('api-status');

// ── Health check ──
async function checkHealth() {
  try {
    const r = await fetch(`${API_BASE}/health`);
    if (r.ok) {
      apiDot.className = 'dot ok';
      apiStatus.textContent = 'API online';
    } else throw new Error();
  } catch {
    apiDot.className = 'dot err';
    apiStatus.textContent = 'API offline';
  }
}
checkHealth();

// ── Drag & drop ──
dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});
['dragleave', 'dragend'].forEach(ev =>
  dropZone.addEventListener(ev, () => dropZone.classList.remove('drag-over'))
);
dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file?.type.startsWith('image/')) handleFile(file);
  else showToast('Please drop a valid image file.', true);
});

// ── Click to browse ──
dropZone.addEventListener('click', () => fileInput.click());
browseBtn.addEventListener('click', (e) => { e.stopPropagation(); fileInput.click(); });
dropZone.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') fileInput.click(); });
fileInput.addEventListener('change', () => { if (fileInput.files[0]) handleFile(fileInput.files[0]); });

// ── Paste support ──
document.addEventListener('paste', (e) => {
  const file = [...e.clipboardData.items]
    .find(i => i.type.startsWith('image/'))
    ?.getAsFile();
  if (file) handleFile(file);
});

// ── GradCAM toggle ──
btnOriginal.addEventListener('click', () => {
  btnOriginal.classList.add('active');
  btnGradcam.classList.remove('active');
  imgOverlay.classList.remove('visible');
});
btnGradcam.addEventListener('click', () => {
  btnGradcam.classList.add('active');
  btnOriginal.classList.remove('active');
  imgOverlay.classList.add('visible');
});

resetBtn.addEventListener('click', reset);
toastClose.addEventListener('click', () => toast.hidden = true);

// ── File handling ──
function handleFile(file) {
  if (file.size > 10 * 1024 * 1024) {
    showToast('File exceeds 10 MB limit.', true);
    return;
  }

  const url = URL.createObjectURL(file);
  previewImg.src = url;
  previewImg.onload = () => URL.revokeObjectURL(url);

  currentFile = file;
  currentResult = null;
  filenameEl.textContent = file.name || '';
  dzIdle.hidden = true;
  dzPreview.hidden = false;
  belowImage.hidden = false;
  imgControls.hidden = true;
  imgOverlay.classList.remove('visible');
  btnOriginal.classList.add('active');
  btnGradcam.classList.remove('active');

  resultEmpty.hidden = true;
  resultCard.hidden = true;
  analyzingState.hidden = false;

  classify(file);
}

async function classify(file) {
  const fd = new FormData();
  fd.append('file', file);

  try {
    const r = await fetch(`${API_BASE}/predict-explain`, { method: 'POST', body: fd });
    if (!r.ok) {
      const err = await r.json().catch(() => ({ detail: r.statusText }));
      throw new Error(err.detail || `HTTP ${r.status}`);
    }
    renderResult(await r.json());
  } catch (err) {
    analyzingState.hidden = true;
    resultEmpty.hidden = false;
    showToast(`Analysis failed: ${err.message}`, true);
  }
}

function renderResult(data) {
  const { label, confidence, gradcam_base64, model_version, run_id } = data;
  const isReal = label === 'real';
  const pct = Math.round(confidence * 100);

  // Card class
  resultCard.className = `result-card ${label}`;

  // Verdict
  verdictIcon.textContent = isReal ? '✓' : '✕';
  verdictLabel.textContent = isReal ? 'Real image' : 'AI-generated';
  verdictSub.textContent = isReal
    ? 'This image appears to be a genuine photograph.'
    : 'This image was likely generated by an AI model.';

  // Gauge — circumference = 2π×50 ≈ 314
  const circumference = 314;
  const offset = circumference - (pct / 100) * circumference;
  setTimeout(() => {
    gaugeFill.style.strokeDashoffset = offset;
    confBarFill.style.width = `${pct}%`;
    animateNumber(confNum, 0, pct, 900);
  }, 80);

  confDesc.textContent = confidenceLabel(confidence);

  // Meta
  metaModel.textContent = model_version || 'efficientnet_v1';
  metaRun.textContent = run_id ? run_id.slice(0, 16) + '…' : '—';

  // Raw JSON
  rawReadout.textContent = JSON.stringify({ label, confidence, model_version, run_id }, null, 2);

  // GradCAM
  if (gradcam_base64) {
    gradcamImg.src = `data:image/png;base64,${gradcam_base64}`;
    imgControls.hidden = false;
  }

  currentResult = data;

  // Reset feedback state
  feedbackPrompt.hidden = false;
  feedbackDone.hidden = true;

  analyzingState.hidden = true;
  resultCard.hidden = false;
}

function confidenceLabel(c) {
  if (c >= 0.95) return 'Very high confidence';
  if (c >= 0.80) return 'High confidence';
  if (c >= 0.65) return 'Moderate confidence';
  return 'Low confidence — borderline case';
}

function animateNumber(el, from, to, duration) {
  const start = performance.now();
  function step(now) {
    const t = Math.min((now - start) / duration, 1);
    const ease = 1 - Math.pow(1 - t, 3);
    el.textContent = Math.round(from + (to - from) * ease);
    if (t < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

function reset() {
  dzIdle.hidden = false;
  dzPreview.hidden = true;
  belowImage.hidden = true;
  imgControls.hidden = true;
  imgOverlay.classList.remove('visible');
  previewImg.src = '';
  gradcamImg.src = '';
  fileInput.value = '';
  filenameEl.textContent = '';

  resultCard.hidden = true;
  analyzingState.hidden = true;
  resultEmpty.hidden = false;

  gaugeFill.style.strokeDashoffset = 314;
  confBarFill.style.width = '0';
  confNum.textContent = '0';
  currentFile = null;
  currentResult = null;
  toast.hidden = true;
}

// ── Feedback ──
let currentFile = null;
let currentResult = null;

fbYes.addEventListener('click', () => submitFeedback(true, currentResult.label));
fbNo.addEventListener('click', () => {
  const trueLabel = currentResult.label === 'real' ? 'fake' : 'real';
  submitFeedback(false, trueLabel);
});

async function submitFeedback(correct, trueLabel) {
  const fd = new FormData();
  if (currentFile) fd.append('image', currentFile);
  fd.append('model_label',      currentResult.label);
  fd.append('model_confidence', currentResult.confidence);
  fd.append('model_version',    currentResult.model_version || '');
  fd.append('run_id',           currentResult.run_id || '');
  fd.append('user_correct',     correct);
  fd.append('true_label',       trueLabel);

  try {
    await fetch('/feedback', { method: 'POST', body: fd });
  } catch { /* best-effort */ }

  feedbackPrompt.hidden = true;
  feedbackDone.hidden = false;
  feedbackDoneText.textContent = correct
    ? 'Thanks! Confirmed as correct.'
    : `Thanks! Saved as "${trueLabel}" for retraining.`;
}

let toastTimer;
function showToast(msg, isError = false) {
  toastText.textContent = msg;
  toast.className = `toast${isError ? ' is-error' : ''}`;
  toast.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { toast.hidden = true; }, 5000);
}
