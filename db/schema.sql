-- Applied idempotently by db.prompt_store.init_db(). Grows by phase; every
-- statement must be safe to re-run.

-- Phase 0: versioned per-agent prompts. Version rows are append-only; the
-- "current" pointer is the only thing that moves (instant rollback).
CREATE TABLE IF NOT EXISTS agent_prompts (
  agent_id   TEXT NOT NULL,
  user_id    TEXT NOT NULL,
  version    INT  NOT NULL,
  prompt     TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (agent_id, user_id, version)
);

CREATE TABLE IF NOT EXISTS agent_prompt_current (
  agent_id        TEXT NOT NULL,
  user_id         TEXT NOT NULL,
  current_version INT  NOT NULL,
  PRIMARY KEY (agent_id, user_id)
);

-- Phase 0: per-node traces. The four-part version stamp
-- (agent_id, agent_version, prompt_version, manifest_version) is what makes
-- bugs reproducible while the agent population grows underneath you.
CREATE TABLE IF NOT EXISTS traces (
  id               BIGSERIAL PRIMARY KEY,
  ts               TIMESTAMPTZ NOT NULL DEFAULT now(),
  user_id          TEXT,
  node             TEXT NOT NULL,
  event            TEXT NOT NULL,
  agent_id         TEXT,
  agent_version    INT,
  prompt_version   INT,
  manifest_version INT,
  payload          JSONB
);

CREATE INDEX IF NOT EXISTS traces_agent_ts ON traces (agent_id, ts);

-- Phase 2: every no_match_found decision, with the rejected candidates and
-- their scores. Dual purpose: the Agent Maker's generation backlog, and seed
-- data for the Phase 5 regression eval.
CREATE TABLE IF NOT EXISTS capability_gaps (
  id               BIGSERIAL PRIMARY KEY,
  ts               TIMESTAMPTZ NOT NULL DEFAULT now(),
  user_id          TEXT NOT NULL,
  request          TEXT NOT NULL,
  rejected_matches JSONB NOT NULL
);
