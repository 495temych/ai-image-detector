'use strict';

// Populates sessions.db with 120 synthetic sessions.
// Run once from repo root: node ui/scripts/seed_db.js
//
// Accuracy model:
//   model: base 90%, degrades on hard images
//   human: base 65%, degrades more steeply on hard images
// Each image gets a fixed random difficulty so per-image stats are consistent.

const path   = require('path');
const fs     = require('fs');
const { v4: uuidv4 } = require('uuid');

// Resolve db relative to this script's location
const { saveSession, getImageStats, getGlobalStats } = require(
  path.join(__dirname, '..', 'db')
);

const SAMPLES_DIR = path.join(__dirname, '..', '..', 'image_samples');

function loadPool(label) {
  return fs.readdirSync(path.join(SAMPLES_DIR, label))
    .filter(f => /\.(jpg|jpeg|png)$/i.test(f))
    .sort()
    .map(f => ({ image_id: `${label}/${f}`, true_label: label }));
}

const FAKE_POOL = loadPool('fake');
const REAL_POOL = loadPool('real');
const ALL       = [...FAKE_POOL, ...REAL_POOL];

console.log(`Pool: ${FAKE_POOL.length} fake + ${REAL_POOL.length} real`);

// Fixed per-image difficulty (0 = easy, 1 = hard)
const difficulty = {};
for (const img of ALL) {
  difficulty[img.image_id] = Math.random();
}

function sample(arr, n) {
  return [...arr].sort(() => Math.random() - 0.5).slice(0, n);
}

function simulateAnswer(true_label, diff, isModel) {
  const base = isModel ? 0.90 : 0.65;
  const drop = isModel ? 0.40 : 0.60;
  const correct = Math.random() < (base - diff * drop);
  return correct ? true_label : (true_label === 'fake' ? 'real' : 'fake');
}

const N_SESSIONS = 120;

for (let i = 0; i < N_SESSIONS; i++) {
  const session_id = uuidv4();
  const images = [
    ...sample(FAKE_POOL, 5),
    ...sample(REAL_POOL, 5),
  ].sort(() => Math.random() - 0.5);

  const results = images.map(img => {
    const diff        = difficulty[img.image_id];
    const user_answer = simulateAnswer(img.true_label, diff, false);
    const model_label = simulateAnswer(img.true_label, diff, true);
    return {
      image_id:      img.image_id,
      true_label:    img.true_label,
      user_answer,
      model_label,
      model_conf:    parseFloat((0.55 + Math.random() * 0.44).toFixed(3)),
      human_correct: user_answer  === img.true_label,
      model_correct: model_label  === img.true_label,
    };
  });

  saveSession(session_id, results);
}

console.log(`\nSeeded ${N_SESSIONS} sessions.`);
console.log('Global stats:', getGlobalStats());
console.log('Sample image stats (fake/0001.jpg):', getImageStats('fake/0001.jpg'));
