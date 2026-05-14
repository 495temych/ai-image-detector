// ── Elements ──
const screenStart   = document.getElementById('screen-start');
const screenPlay    = document.getElementById('screen-play');
const screenReveal  = document.getElementById('screen-reveal');
const btnStart      = document.getElementById('btn-start');
const btnRestart    = document.getElementById('btn-restart');
const btnReal       = document.getElementById('btn-real');
const btnFake       = document.getElementById('btn-fake');
const gameImg       = document.getElementById('game-img');
const progressFill  = document.getElementById('progress-fill');
const progressLabel = document.getElementById('progress-label');
const sumYourAcc    = document.getElementById('sum-your-acc');
const sumModelAcc   = document.getElementById('sum-model-acc');
const sumBothFooled = document.getElementById('sum-both-fooled');
const sumYouBeat    = document.getElementById('sum-you-beat');
const sumVerdict    = document.getElementById('sum-verdict');
const sumGlobalHuman = document.getElementById('sum-global-human');
const sumGlobalModel = document.getElementById('sum-global-model');
const bmResults     = document.getElementById('bm-results');
const apiDot        = document.getElementById('api-dot');
const apiStatus     = document.getElementById('api-status');

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
  sumYourAcc.textContent    = '…';
  sumModelAcc.textContent   = '…';
  sumBothFooled.textContent = '…';
  sumYouBeat.textContent    = '…';
  sumVerdict.textContent    = '…';
  sumVerdict.className      = '';
  sumGlobalHuman.textContent = '…';
  sumGlobalModel.textContent = '…';
  bmResults.innerHTML        = '<p class="bm-loading">Crunching results…</p>';

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

// ── Reveal helpers ──
function computeSummary(results) {
  const yourAcc  = results.filter(r => r.human_correct).length / results.length;
  const modelAcc = results.filter(r => r.model_correct).length / results.length;
  // image_num is 1-based from the new server; fall back to index+1 for old server
  const num = r => r.image_num ?? (r.id + 1);
  return {
    your_accuracy:  yourAcc,
    model_accuracy: modelAcc,
    both_fooled:    results.filter(r => !r.human_correct && !r.model_correct).map(num),
    you_beat_model: results.filter(r =>  r.human_correct && !r.model_correct).map(num),
    verdict: yourAcc > modelAcc ? 'You win this round'
           : yourAcc < modelAcc ? 'Model wins this round'
           : 'Tie',
  };
}

// ── Reveal ──
function renderReveal(data) {
  const { results } = data;
  const community = data.community ?? {};
  const global    = data.global    ?? null;

  // Use server-computed summary if present; compute client-side otherwise
  const summary = data.summary ?? computeSummary(results);

  const pct       = v => `${Math.round(v * 100)}%`;
  const imageList = nums => nums.length === 0
    ? 'None'
    : `${nums.length} (${nums.map(n => `Image ${n}`).join(', ')})`;

  // ── 1. Hero score line ──
  document.getElementById('hero-you').textContent   = `You ${pct(summary.your_accuracy)}`;
  document.getElementById('hero-model').textContent = `Model ${pct(summary.model_accuracy)}`;

  const heroVerdict = document.getElementById('hero-verdict');
  heroVerdict.textContent = summary.verdict;
  heroVerdict.className   = summary.verdict.startsWith('You win')    ? 'verdict-win'
                          : summary.verdict.startsWith('Model wins') ? 'verdict-loss'
                          : 'verdict-tie';

  if (global?.avg_human_acc_pct != null) {
    const diff   = Math.round(summary.your_accuracy * 100) - global.avg_human_acc_pct;
    const vsLine = diff > 0
      ? `You scored ${diff}% above the community average of ${global.avg_human_acc_pct}%`
      : diff < 0
      ? `You scored ${Math.abs(diff)}% below the community average of ${global.avg_human_acc_pct}%`
      : `You matched the community average of ${global.avg_human_acc_pct}%`;
    document.getElementById('hero-vs-community').textContent = vsLine;
  }

  // ── 2. Insight cards ──
  // Sort by community human accuracy — hardest (lowest) first
  const withStats = results.filter(r => r.image_id && community[r.image_id]?.total_plays > 0);
  if (withStats.length > 0) {
    const sorted  = [...withStats].sort(
      (a, b) => community[a.image_id].human_acc_pct - community[b.image_id].human_acc_pct
    );
    const hardest = sorted[0];
    const easiest = sorted[sorted.length - 1];

    document.getElementById('ins-hardest-img').src           = `/challenge/image/${hardest.image_id}`;
    document.getElementById('ins-hardest-human').textContent = `Humans ${community[hardest.image_id].human_acc_pct}%`;
    document.getElementById('ins-hardest-model').textContent = `Model ${community[hardest.image_id].model_acc_pct}%`;
    document.getElementById('ins-hardest-sub').textContent   =
      `Fools ${community[hardest.image_id].human_confusion_pct}% of players`;

    document.getElementById('ins-easiest-img').src           = `/challenge/image/${easiest.image_id}`;
    document.getElementById('ins-easiest-human').textContent = `Humans ${community[easiest.image_id].human_acc_pct}%`;
    document.getElementById('ins-easiest-model').textContent = `Model ${community[easiest.image_id].model_acc_pct}%`;
  }

  // Overconfident model — wrong AND confidence > 75%
  const overconfident = results.find(r => !r.model_correct && r.model_confidence > 0.75);
  const insOverconfident = document.getElementById('ins-overconfident');
  if (overconfident) {
    insOverconfident.hidden = false;
    document.getElementById('ins-overconfident-img').src          = `/challenge/image/${overconfident.image_id}`;
    document.getElementById('ins-overconfident-conf').textContent =
      `${Math.round(overconfident.model_confidence * 100)}% confident · wrong`;
  } else {
    insOverconfident.hidden = true;
  }

  // ── 3. Summary table ──
  sumYourAcc.textContent    = pct(summary.your_accuracy);
  sumModelAcc.textContent   = pct(summary.model_accuracy);
  sumBothFooled.textContent = imageList(summary.both_fooled);
  sumYouBeat.textContent    = summary.you_beat_model.length === 0
    ? 'None'
    : `${summary.you_beat_model.length} image${summary.you_beat_model.length > 1 ? 's' : ''} (${summary.you_beat_model.map(n => `Image ${n}`).join(', ')})`;

  sumVerdict.textContent = summary.verdict;
  sumVerdict.className   = summary.verdict.startsWith('You win')    ? 'verdict-win'
                         : summary.verdict.startsWith('Model wins') ? 'verdict-loss'
                         : 'verdict-tie';

  if (global) {
    sumGlobalHuman.textContent = `${global.avg_human_acc_pct ?? '—'}%`;
    sumGlobalModel.textContent = `${global.avg_model_acc_pct ?? '—'}%`;
  } else {
    sumGlobalHuman.textContent = '—';
    sumGlobalModel.textContent = '—';
  }

  // ── 4. Per-image cards (full GradCAM view) ──
  bmResults.innerHTML = '';

  const tick  = `<svg width="13" height="13" viewBox="0 0 13 13" fill="none"><path d="M2 6.5l3 3L11 3" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>`;
  const cross = `<svg width="13" height="13" viewBox="0 0 13 13" fill="none"><path d="M2 2l9 9M11 2L2 11" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>`;
  const labelText = l => l === 'real' ? 'Real' : 'AI-generated';

  results.forEach((item, i) => {
    const imageNum = item.image_num ?? (i + 1);
    const status   = item.human_correct && item.model_correct  ? 'both-right'
      :             !item.human_correct && item.model_correct  ? 'model-only'
      :              item.human_correct && !item.model_correct ? 'human-only'
      :                                                          'both-wrong';

    const statusLabel = {
      'both-right': 'Both right',
      'model-only': 'Model only',
      'human-only': 'You only',
      'both-wrong': 'Both wrong',
    }[status];

    const stats = item.image_id ? community[item.image_id] : null;
    const communityBar = stats?.total_plays > 0 ? `
      <div class="community-bar">
        <span class="comm-plays">${stats.total_plays} plays</span>
        <span class="comm-human">Humans ${stats.human_acc_pct}%</span>
        <span class="comm-model">Model ${stats.model_acc_pct}%</span>
      </div>` : '';

    const card = document.createElement('div');
    card.className = `bm-card bm-card--${status}`;
    card.style.animationDelay = `${i * 0.04}s`;
    card.innerHTML = `
      <div class="bm-card__images">
        <img class="bm-card__thumb"   src="${item.src}" alt="Image ${imageNum}">
        ${item.gradcam_b64
          ? `<img class="bm-card__gradcam" src="data:image/png;base64,${item.gradcam_b64}" alt="GradCAM">`
          : ''}
      </div>
      <div class="bm-card__info">
        <div class="bm-card__num">Image ${imageNum}</div>
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
        ${communityBar}
      </div>`;
    bmResults.appendChild(card);
  });
}
