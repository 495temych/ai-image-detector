'use strict';

const Database = require('better-sqlite3');
const path     = require('path');
const fs       = require('fs');

const DATA_DIR = path.join(__dirname, 'data');
fs.mkdirSync(DATA_DIR, { recursive: true });

const db = new Database(path.join(DATA_DIR, 'sessions.db'));

db.pragma('journal_mode = WAL');   // safe for concurrent reads

db.exec(`
  CREATE TABLE IF NOT EXISTS sessions (
    session_id   TEXT    PRIMARY KEY,
    played_at    TEXT    NOT NULL,
    human_score  INTEGER NOT NULL,
    model_score  INTEGER NOT NULL,
    total        INTEGER NOT NULL DEFAULT 10
  );

  CREATE TABLE IF NOT EXISTS session_results (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id     TEXT    NOT NULL REFERENCES sessions(session_id),
    image_id       TEXT    NOT NULL,
    true_label     TEXT    NOT NULL,
    user_answer    TEXT    NOT NULL,
    model_label    TEXT    NOT NULL,
    model_conf     REAL    NOT NULL,
    human_correct  INTEGER NOT NULL,
    model_correct  INTEGER NOT NULL,
    played_at      TEXT    NOT NULL,
    user_annotation TEXT   DEFAULT NULL
    -- user_annotation: reserved for future click-to-mark region data
    -- JSON shape when used: {clicks: [{x, y}], submitted_at: ISO string}
    -- Do not remove this column.
  );

  CREATE INDEX IF NOT EXISTS idx_results_image_id
    ON session_results(image_id);

  CREATE INDEX IF NOT EXISTS idx_results_session_id
    ON session_results(session_id);
`);

// ── Writes ────────────────────────────────────────────────────────

const _insertSession = db.prepare(`
  INSERT OR IGNORE INTO sessions
    (session_id, played_at, human_score, model_score, total)
  VALUES (?, datetime('now'), ?, ?, ?)
`);

const _insertResult = db.prepare(`
  INSERT INTO session_results
    (session_id, image_id, true_label, user_answer,
     model_label, model_conf, human_correct, model_correct, played_at)
  VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
`);

function saveSession(session_id, results) {
  const human_score = results.filter(r => r.human_correct).length;
  const model_score = results.filter(r => r.model_correct).length;

  db.transaction(() => {
    _insertSession.run(session_id, human_score, model_score, results.length);
    for (const r of results) {
      _insertResult.run(
        session_id,
        r.image_id,
        r.true_label,
        r.user_answer,
        r.model_label,
        r.model_conf,
        r.human_correct ? 1 : 0,
        r.model_correct ? 1 : 0,
      );
    }
  })();
}

// ── Reads ─────────────────────────────────────────────────────────

const _imageStats = db.prepare(`
  SELECT
    COUNT(*)                                 AS total_plays,
    ROUND(AVG(human_correct) * 100)          AS human_acc_pct,
    ROUND(AVG(model_correct) * 100)          AS model_acc_pct,
    ROUND((1.0 - AVG(human_correct)) * 100)  AS human_confusion_pct
  FROM session_results
  WHERE image_id = ?
`);

function getImageStats(image_id) {
  return _imageStats.get(image_id);
}

function getImageStatsMany(image_ids) {
  const stats = {};
  for (const id of image_ids) stats[id] = _imageStats.get(id);
  return stats;
}

const _globalStats = db.prepare(`
  SELECT
    COUNT(DISTINCT session_id)          AS total_sessions,
    ROUND(AVG(human_score) * 10)        AS avg_human_acc_pct,
    ROUND(AVG(model_score) * 10)        AS avg_model_acc_pct
  FROM sessions
`);

function getGlobalStats() {
  return _globalStats.get();
}

module.exports = { saveSession, getImageStats, getImageStatsMany, getGlobalStats };
