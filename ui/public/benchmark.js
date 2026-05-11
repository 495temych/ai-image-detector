// ── Elements ──
const screenStart  = document.getElementById('screen-start');
const screenPlay   = document.getElementById('screen-play');
const screenReveal = document.getElementById('screen-reveal');
const btnStart     = document.getElementById('btn-start');
const btnRestart   = document.getElementById('btn-restart');
const btnReal      = document.getElementById('btn-real');
const btnFake      = document.getElementById('btn-fake');
const gameImg      = document.getElementById('game-img');
const progressFill = document.getElementById('progress-fill');
const progressLabel= document.getElementById('progress-label');
const scoreHuman   = document.getElementById('score-human');
const scoreModel   = document.getElementById('score-model');
const bmVerdict    = document.getElementById('bm-verdict');
const bmResults    = document.getElementById('bm-results');
const apiDot       = document.getElementById('api-dot');
const apiStatus    = document.getElementById('api-status');

// ── State ──
let session    = null;   // { session_id, images: [{id, src}] }
let answers    = [];     // user answers in order
let currentIdx = 0;

// ── Health check ──
async function checkHealth() {
  try {
    const r = await fetch('/api/health');
    if (r.ok) { apiDot.className = 'dot ok'; apiStatus.textContent = 'API online'; }
    else throw new Error();
  } catch {
    apiDot.className = 'dot err'; apiStatus.textContent = 'API offline';
  }
}
checkHealth();

// ── Screen switching ──
function showScreen(id) {
  [screenStart, screenPlay, screenReveal].forEach(s => { s.hidden = s.id !== id; });
}

// ── Start ──
btnStart.addEventListener('click', startSession);
btnRestart.addEventListener('click', () => showScreen('screen-start'));

async function startSession() {
  btnStart.disabled = true;
  btnStart.textContent = 'Loading…';
  try {
    const r = await fetch('/challenge/session');
    if (!r.ok) throw new Error(`Server error ${r.status}`);
    session    = await r.json();
    answers    = [];
    currentIdx = 0;
    showScreen('screen-play');
    loadImage(currentIdx);
  } catch (err) {
    btnStart.disabled = false;
    btnStart.textContent = 'Start Challenge';
    alert(`Could not start: ${err.message}\n\nMake sure the API is running.`);
  }
}

// ── Game loop ──
function loadImage(idx) {
  const { src } = session.images[idx];
  const total   = session.images.length;

  progressLabel.textContent = `${idx + 1} / ${total}`;
  progressFill.style.width  = `${(idx / total) * 100}%`;

  btnReal.disabled = true;
  btnFake.disabled = true;
  btnReal.classList.remove('chosen');
  btnFake.classList.remove('chosen');
  gameImg.classList.add('loading');

  const img = new Image();
  img.onload = () => {
    gameImg.src = src;
    gameImg.classList.remove('loading');
    btnReal.disabled = false;
    btnFake.disabled = false;
  };
  img.onerror = () => {
    gameImg.src = src; // show broken-image rather than staying blank
    gameImg.classList.remove('loading');
    btnReal.disabled = false;
    btnFake.disabled = false;
  };
  img.src = src;
}

function recordAnswer(answer) {
  btnReal.disabled = true;
  btnFake.disabled = true;
  (answer === 'real' ? btnReal : btnFake).classList.add('chosen');
  answers.push(answer);

  setTimeout(() => {
    currentIdx++;
    if (currentIdx < session.images.length) {
      loadImage(currentIdx);
    } else {
      progressFill.style.width = '100%';
      submitAnswers();
    }
  }, 380);
}

btnReal.addEventListener('click', () => recordAnswer('real'));
btnFake.addEventListener('click', () => recordAnswer('fake'));

// ── Submit ──
async function submitAnswers() {
  showScreen('screen-reveal');
  scoreHuman.textContent = '…';
  scoreModel.textContent = '…';
  bmVerdict.textContent  = '';
  bmResults.innerHTML    = '<p class="bm-loading">Crunching results…</p>';

  try {
    const r = await fetch('/challenge/submit', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ session_id: session.session_id, answers }),
    });
    if (!r.ok) throw new Error(`Server error ${r.status}`);
    renderReveal(await r.json());
  } catch (err) {
    bmResults.innerHTML = `<p class="bm-error">Error loading results: ${err.message}</p>`;
  }
}

// ── Reveal ──
function renderReveal({ results }) {
  const humanScore = results.filter(r => r.human_correct).length;
  const modelScore = results.filter(r => r.model_correct).length;

  scoreHuman.textContent = `${humanScore}/10`;
  scoreModel.textContent = `${modelScore}/10`;

  if      (humanScore > modelScore) bmVerdict.textContent = 'You beat the model! 🎉';
  else if (humanScore === modelScore) bmVerdict.textContent = "It's a tie! 🤝";
  else                                bmVerdict.textContent = 'The model beat you 🤖';

  bmResults.innerHTML = '';

  const tick  = `<svg width="13" height="13" viewBox="0 0 13 13" fill="none"><path d="M2 6.5l3 3L11 3" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>`;
  const cross = `<svg width="13" height="13" viewBox="0 0 13 13" fill="none"><path d="M2 2l9 9M11 2L2 11" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>`;

  const labelText = l => l === 'real' ? 'Real' : 'AI-generated';

  results.forEach((item, i) => {
    const status = item.human_correct && item.model_correct   ? 'both-right'
      :           !item.human_correct && item.model_correct   ? 'model-only'
      :            item.human_correct && !item.model_correct  ? 'human-only'
      :                                                         'both-wrong';

    const statusLabel = {
      'both-right': 'Both right',
      'model-only': 'Model only',
      'human-only': 'You only',
      'both-wrong': 'Both wrong',
    }[status];

    const card = document.createElement('div');
    card.className = `bm-card bm-card--${status}`;
    card.style.animationDelay = `${i * 0.04}s`;
    card.innerHTML = `
      <div class="bm-card__images">
        <img class="bm-card__thumb"   src="${item.src}" alt="Image ${i + 1}">
        ${item.gradcam_b64
          ? `<img class="bm-card__gradcam" src="data:image/png;base64,${item.gradcam_b64}" alt="GradCAM">`
          : ''}
      </div>
      <div class="bm-card__info">
        <div class="bm-card__num">#${i + 1}</div>
        <div class="bm-card__truth">Ground truth: <strong>${labelText(item.true_label)}</strong></div>
        <div class="bm-card__row ${item.human_correct ? 'correct' : 'wrong'}">
          ${item.human_correct ? tick : cross}
          <span>You: ${labelText(item.user_answer)}</span>
        </div>
        <div class="bm-card__row ${item.model_correct ? 'correct' : 'wrong'}">
          ${item.model_correct ? tick : cross}
          <span>Model: ${labelText(item.model_label)}
            <span class="bm-conf">${Math.round(item.model_confidence * 100)}%</span>
          </span>
        </div>
        <div class="bm-card__status bm-card__status--${status}">${statusLabel}</div>
      </div>`;
    bmResults.appendChild(card);
  });
}
