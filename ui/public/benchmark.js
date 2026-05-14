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
const sumYourAcc     = document.getElementById('sum-your-acc');
const sumModelAcc    = document.getElementById('sum-model-acc');
const sumBothFooled  = document.getElementById('sum-both-fooled');
const sumYouBeat     = document.getElementById('sum-you-beat');
const sumVerdict     = document.getElementById('sum-verdict');
const sumGlobalHuman = document.getElementById('sum-global-human');
const sumGlobalModel = document.getElementById('sum-global-model');
const bmCards        = document.getElementById('bm-cards');
const apiDot         = document.getElementById('api-dot');
const apiStatus      = document.getElementById('api-status');

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
  // Reset summary table cells
  sumYourAcc.textContent     = '…';
  sumModelAcc.textContent    = '…';
  sumBothFooled.textContent  = '…';
  sumYouBeat.textContent     = '…';
  sumVerdict.textContent     = '…';
  sumVerdict.className       = '';
  sumGlobalHuman.textContent = '…';
  sumGlobalModel.textContent = '…';
  // Clear dynamically-rendered sections
  document.getElementById('bm-hero').innerHTML        = '';
  document.getElementById('bm-insight-row').innerHTML = '';
  document.getElementById('bm-analytics').hidden      = true;
  bmCards.innerHTML = '<p class="bm-loading">Crunching results…</p>';

  try {
    const r = await fetch('/challenge/submit', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ session_id: session.session_id, answers }),
    });
    if (!r.ok) throw new Error(`Server error ${r.status}`);
    renderReveal(await r.json());
  } catch (err) {
    bmCards.innerHTML = `<p class="bm-error">Error loading results: ${err.message}</p>`;
  }
}

// ── Reveal helpers ──
function computeSummary(results) {
  const yourAcc  = results.filter(r => r.human_correct).length / results.length;
  const modelAcc = results.filter(r => r.model_correct).length / results.length;
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

// ── 1. Hero ──
function renderHero(summary, global) {
  const you   = Math.round(summary.your_accuracy * 100);
  const model = Math.round(summary.model_accuracy * 100);
  const diff  = global?.avg_human_acc_pct != null
    ? you - global.avg_human_acc_pct
    : null;

  const verdictEmoji = you > model ? '🏆' : you === model ? '🤝' : '🤖';
  const verdictClass = summary.verdict.startsWith('You win')    ? 'verdict-win'
                     : summary.verdict.startsWith('Model wins') ? 'verdict-loss'
                     : 'verdict-tie';

  const communityLine = diff === null ? ''
    : diff > 0
      ? `You scored ${diff}% above the community average of ${global.avg_human_acc_pct}% — nice round`
    : diff < 0
      ? `You scored ${Math.abs(diff)}% below the community average of ${global.avg_human_acc_pct}% — try again`
    : `You matched the community average of ${global.avg_human_acc_pct}% exactly`;

  document.getElementById('bm-hero').innerHTML = `
    <div class="hero-score">
      <span class="you">You ${you}%</span>
      <span class="vs">vs</span>
      <span class="mdl">Model ${model}%</span>
    </div>
    <p class="hero-verdict ${verdictClass}">${verdictEmoji} ${summary.verdict}</p>
    ${communityLine ? `<p class="hero-community">${communityLine}</p>` : ''}
  `;
}

// ── 2. Insight cards ──
function pickCaption(stats, userCorrect) {
  if (!stats || stats.total_plays < 5)
    return {
      main:    `Only ${stats?.total_plays ?? 0} plays so far`,
      explain: 'Stats will stabilise as more players see this image',
    };
  if (stats.model_acc_pct < 55 && stats.human_acc_pct < 55)
    return {
      main:    `Stumps everyone — humans ${stats.human_acc_pct}%, model ${stats.model_acc_pct}%`,
      explain: 'Both humans and the model struggle here — prime retraining candidate',
    };
  if (stats.model_acc_pct > 85 && stats.human_acc_pct < 45)
    return {
      main:    `Model sees it clearly, most humans don't`,
      explain: 'The model found a pattern humans miss — check the GradCAM below',
    };
  if (stats.human_acc_pct > 85 && stats.model_acc_pct < 60)
    return {
      main:    `Humans outperform the model on this one`,
      explain: 'A case where human intuition beats the algorithm',
    };
  if (stats.overconf_rate > 30)
    return {
      main:    `Model is overconfident and wrong ${stats.overconf_rate}% of the time`,
      explain: 'High confidence + wrong answer — the most dangerous failure mode',
    };
  if (userCorrect && stats.human_acc_pct < 40)
    return {
      main:    `You spotted it — only ${stats.human_acc_pct}% of players do`,
      explain: 'Above-average catch — this image fools most people',
    };
  if (!userCorrect && stats.human_acc_pct > 75)
    return {
      main:    `${stats.human_acc_pct}% of players got this — a well-known tell`,
      explain: 'Most players identify this correctly — you can too next round',
    };
  return {
    main:    `${stats.human_acc_pct}% human accuracy across ${stats.total_plays} plays`,
    explain: stats.model_acc_pct > stats.human_acc_pct
      ? 'Model has the edge on this image'
      : 'Humans and model are roughly matched here',
  };
}

function renderInsightCards(results, extended, community) {
  const withStats = results.map(r => ({
    ...r,
    s:  extended[r.image_id],
    cs: community[r.image_id],
  }));

  const hardest   = [...withStats].sort((a, b) =>
    (a.cs?.human_acc_pct ?? 100) - (b.cs?.human_acc_pct ?? 100))[0];
  const easiest   = [...withStats].sort((a, b) =>
    (b.cs?.human_acc_pct ?? 0)   - (a.cs?.human_acc_pct ?? 0))[0];
  const contested = [...withStats].sort((a, b) =>
    Math.abs(50 - (a.cs?.human_acc_pct ?? 100)) -
    Math.abs(50 - (b.cs?.human_acc_pct ?? 100)))[0];

  const slots = [
    { emoji: '🔥', label: 'HARDEST THIS SESSION',  r: hardest },
    { emoji: '⚖️', label: 'MOST CONTESTED',         r: contested },
    { emoji: '✅', label: 'EASIEST THIS SESSION',   r: easiest },
  ];

  document.getElementById('bm-insight-row').innerHTML = slots.map(({ emoji, label, r }) => {
    const { main, explain } = pickCaption(r.s, r.human_correct);
    return `
      <div class="insight-card">
        <p class="ins-label">${emoji} ${label}</p>
        <img class="ins-thumb" src="/challenge/image/${r.image_id}" alt="">
        <div class="ins-pills">
          <span class="comm-human">Humans ${r.cs?.human_acc_pct ?? '—'}%</span>
          <span class="comm-model">Model ${r.cs?.model_acc_pct ?? '—'}%</span>
        </div>
        <p class="ins-caption">${main}</p>
        <p class="ins-explain">${explain}</p>
      </div>`;
  }).join('');
}

// ── 3. Summary table ──
function renderSummaryTable(summary, global) {
  const pct       = v => `${Math.round(v * 100)}%`;
  const imageList = nums => nums.length === 0
    ? 'None'
    : `${nums.length} (${nums.map(n => `Image ${n}`).join(', ')})`;

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
}

// ── 4. Trend chart ──
function renderChart(trend) {
  const el = document.getElementById('bm-trend-chart');
  if (!el || !trend?.length) return;

  // Destroy previous instance if re-rendering
  if (typeof Chart !== 'undefined') Chart.getChart(el)?.destroy();
  else return; // Chart.js not yet loaded

  const avgHuman = Math.round(
    trend.reduce((s, r) => s + r.human_score / r.total * 100, 0) / trend.length
  );

  new Chart(el, {
    type: 'line',
    data: {
      labels: trend.map((_, i) => `S${i + 1}`),
      datasets: [
        {
          label:           'Human accuracy',
          data:            trend.map(s => Math.round(s.human_score / s.total * 100)),
          borderColor:     '#378ADD',
          backgroundColor: 'rgba(55,138,221,.08)',
          tension: 0.35, pointRadius: 3, fill: true,
        },
        {
          label:           'Model accuracy',
          data:            trend.map(s => Math.round(s.model_score / s.total * 100)),
          borderColor:     '#1D9E75',
          backgroundColor: 'rgba(29,158,117,.06)',
          tension: 0.35, pointRadius: 3, fill: true, borderDash: [5, 3],
        },
        {
          label:       'Community avg',
          data:        trend.map(() => avgHuman),
          borderColor: 'rgba(226,75,74,.4)',
          borderDash:  [2, 4], pointRadius: 0, fill: false,
        },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        y: {
          min: 0, max: 100,
          ticks: { callback: v => v + '%', font: { size: 11 } },
          grid:  { color: 'rgba(128,128,128,.1)' },
        },
        x: {
          ticks: { font: { size: 10 }, autoSkip: true, maxTicksLimit: 10 },
          grid:  { display: false },
        },
      },
    },
  });

  document.getElementById('bm-analytics').hidden = false;
}

// ── 5. Per-image cards ──
function renderPerImageCards(results, community) {
  bmCards.innerHTML = '';

  const tick      = `<svg width="13" height="13" viewBox="0 0 13 13" fill="none"><path d="M2 6.5l3 3L11 3" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>`;
  const cross     = `<svg width="13" height="13" viewBox="0 0 13 13" fill="none"><path d="M2 2l9 9M11 2L2 11" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>`;
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
    bmCards.appendChild(card);
  });
}

// ── Reveal coordinator ──
function renderReveal(data) {
  const summary   = data.summary   ?? computeSummary(data.results);
  const community = data.community ?? {};
  const extended  = data.extended  ?? {};
  const global    = data.global    ?? null;

  renderHero(summary, global);
  renderInsightCards(data.results, extended, community);
  renderSummaryTable(summary, global);
  renderChart(data.trend);
  renderPerImageCards(data.results, community);
}
