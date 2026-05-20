'use strict';

const PILL_OK   = '<span class="pill pill-ok">Healthy</span>';
const PILL_WARN = '<span class="pill pill-warn">Warning</span>';

function pill(status) {
  return status === 'ok' ? PILL_OK : PILL_WARN;
}

function healthRow(label, value, status, explain) {
  return `
    <div class="health-row">
      <div>
        <span class="h-label">${label}</span>
        ${explain ? `<span class="h-explain">${explain}</span>` : ''}
      </div>
      <span style="display:flex;align-items:center;gap:8px">
        <span class="h-val">${value}</span>
        ${status ? pill(status) : ''}
      </span>
    </div>`;
}

function infoRow(label, value) {
  return `
    <div class="health-row">
      <span class="h-label">${label}</span>
      <span class="h-val">${value}</span>
    </div>`;
}

async function loadDashboard() {
  const res  = await fetch('/dashboard/stats');
  const data = await res.json();
  const { kpis, drift, conf_dist, trend, model_info } = data;

  // KPIs
  document.getElementById('kpi-sessions').textContent = kpis.total_sessions;
  document.getElementById('kpi-model').textContent    = kpis.model_acc_pct + '%';
  document.getElementById('kpi-human').textContent    = kpis.human_acc_pct + '%';
  document.getElementById('kpi-gap').textContent      = kpis.gap_pct + '%';

  // Health indicators
  document.getElementById('health-table').innerHTML = [
    healthRow(
      'Mean model confidence',
      drift.mean_conf_pct + '%',
      drift.mean_conf_status,
      'Below 72% signals the model is uncertain — possible distribution shift'
    ),
    healthRow(
      'Low-confidence predictions (<65%)',
      drift.low_conf_rate_pct + '%',
      drift.low_conf_status,
      'Above 25% means many images are near the decision boundary'
    ),
    healthRow(
      'Both-fooled rate',
      drift.both_fooled_pct + '%',
      drift.both_fooled_status,
      'Images that fool both human and model — above 15% indicates hard new content'
    ),
    healthRow(
      'Drift status',
      drift.drift_detected ? 'Drift detected — review recommended' : 'No drift detected',
      drift.drift_detected ? 'warn' : 'ok',
      null
    ),
  ].join('');

  // Model info
  document.getElementById('model-table').innerHTML = [
    infoRow('Model version',   model_info.version),
    infoRow('MLflow run ID',   `<span style="font-family:var(--font-mono);font-size:12px">${model_info.run_id}</span>`),
    infoRow('Test accuracy',   (model_info.test_acc * 100).toFixed(1) + '%'),
    infoRow('AUC',             model_info.auc.toFixed(3)),
    infoRow('Registry status', '<span class="pill pill-info">Champion</span>'),
  ].join('');

  // Confidence histogram
  new Chart(document.getElementById('conf-chart'), {
    type: 'bar',
    data: {
      labels: ['<60%', '60–70%', '70–80%', '80–90%', '90–100%'],
      datasets: [{
        data: conf_dist,
        backgroundColor: conf_dist.map((_, i) =>
          i < 2 ? 'rgba(216,90,48,.7)' : 'rgba(55,138,221,.7)'
        ),
        borderRadius: 3,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: ctx => ` ${ctx.raw} predictions` } },
      },
      scales: {
        x: { ticks: { font: { size: 10 } }, grid: { display: false } },
        y: { ticks: { font: { size: 10 } }, grid: { color: 'rgba(128,128,128,.1)' } },
      },
    },
  });

  // Accuracy trend line chart
  new Chart(document.getElementById('trend-chart'), {
    type: 'line',
    data: {
      labels: trend.map((_, i) => 'S' + (i + 1)),
      datasets: [
        {
          label: 'Human',
          data: trend.map(s => Math.round(s.human_score / s.total * 100)),
          borderColor: '#378ADD',
          backgroundColor: 'rgba(55,138,221,.08)',
          tension: .35,
          pointRadius: 2,
          fill: true,
        },
        {
          label: 'Model',
          data: trend.map(s => Math.round(s.model_score / s.total * 100)),
          borderColor: '#1D9E75',
          backgroundColor: 'rgba(29,158,117,.06)',
          tension: .35,
          pointRadius: 2,
          fill: true,
          borderDash: [4, 3],
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      layout: { padding: { top: 10 } },
      scales: {
        y: {
          min: 0,
          max: 110,
          ticks: {
            callback: v => v > 100 ? '' : v + '%',
            font: { size: 10 },
            stepSize: 20,
          },
          grid: { color: 'rgba(128,128,128,.1)' },
        },
        x: {
          ticks: { font: { size: 9 }, autoSkip: true, maxTicksLimit: 8 },
          grid: { display: false },
        },
      },
    },
  });
}

loadDashboard().catch(console.error);
