-- Reify SQLite schema, version 1.
--
-- Plain SQL is truth (FIRST_PRINCIPLES.md commandment 1).
-- Any user can `sqlite3 ~/.reify/store.db` and read everything in this file.
-- Pydantic models in src/reify/core/types.py are a typed *view* over this
-- schema, not the other way around.
--
-- The file is loaded verbatim by SQLiteStore.init(); statements are
-- idempotent so calling init() repeatedly is safe.

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;
PRAGMA synchronous = NORMAL;

-- --------------------------------------------------------------------------
-- Schema version
-- --------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS schema_version (
    version    INTEGER PRIMARY KEY,
    applied_at TEXT    NOT NULL
);

-- --------------------------------------------------------------------------
-- Identities + bonds (Identity Bond Graph; full subsystem lands in v0.5)
-- --------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS identities (
    id            TEXT PRIMARY KEY,
    kind          TEXT NOT NULL CHECK (kind IN ('user', 'device', 'anonymous')),
    label         TEXT,
    created_at    TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS identity_bonds (
    from_identity      TEXT NOT NULL,
    to_identity        TEXT NOT NULL,
    confidence         REAL NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
    rationale          TEXT NOT NULL DEFAULT '',
    formed_at          TEXT NOT NULL,
    last_reinforced_at TEXT,
    PRIMARY KEY (from_identity, to_identity),
    FOREIGN KEY (from_identity) REFERENCES identities(id) ON DELETE CASCADE,
    FOREIGN KEY (to_identity)   REFERENCES identities(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_bonds_from ON identity_bonds(from_identity);

-- --------------------------------------------------------------------------
-- Memories
--
-- One physical table holds all five Pydantic networks. The `network` column
-- discriminates. Common fields are columns (for cheap, inspectable queries);
-- network-specific fields live in `payload_json` (rehydrated by Pydantic on
-- read). This is the "two memory types in v0" implementation per
-- FIRST_PRINCIPLES.md commandment 2: the type system has five networks; the
-- storage has one table.
-- --------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS memories (
    id              TEXT PRIMARY KEY,
    network         TEXT NOT NULL CHECK (network IN (
        'fact', 'experience', 'observation', 'belief', 'inference'
    )),
    content         TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    valid_at        TEXT NOT NULL,
    valid_until     TEXT,
    confidence      REAL NOT NULL DEFAULT 1.0,
    identity_id     TEXT,
    tags_json       TEXT NOT NULL DEFAULT '[]',
    metadata_json   TEXT NOT NULL DEFAULT '{}',
    payload_json    TEXT NOT NULL DEFAULT '{}',
    soft_deleted_at TEXT
    -- Note: no FK on identity_id. In v0 callers may write memories tagged
    -- with bare identity strings before the Identity Bond Graph subsystem
    -- (v0.5) populates the `identities` table. Integrity is enforced at the
    -- IBG layer, not at the memory layer.
);

CREATE INDEX IF NOT EXISTS idx_memories_network    ON memories(network);
CREATE INDEX IF NOT EXISTS idx_memories_identity   ON memories(identity_id);
CREATE INDEX IF NOT EXISTS idx_memories_valid_at   ON memories(valid_at);
CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at);
CREATE INDEX IF NOT EXISTS idx_memories_alive
    ON memories(created_at) WHERE soft_deleted_at IS NULL;

-- --------------------------------------------------------------------------
-- Provenance
--
-- Append-only chain of custody. Pulled out of the memory row so that the
-- common `SELECT memory_id, source FROM provenance` query is cheap and
-- inspectable.
-- --------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS provenance (
    rowid        INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id    TEXT NOT NULL,
    source       TEXT NOT NULL,
    session_id   TEXT,
    actor        TEXT NOT NULL CHECK (actor IN ('user', 'assistant', 'system', 'extractor')),
    raw_event_id TEXT,
    recorded_at  TEXT NOT NULL,
    FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_provenance_memory ON provenance(memory_id);
CREATE INDEX IF NOT EXISTS idx_provenance_source ON provenance(source);

-- --------------------------------------------------------------------------
-- Bitemporal Transition Graph
--
-- First-class change events linking memories. Distinguishes Reify's BTG from
-- Zep's bitemporal model: where Zep stores `valid_at`/`valid_until` ranges,
-- Reify additionally stores *the change itself* with evidence trails.
-- --------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS transitions (
    id             TEXT PRIMARY KEY,
    kind           TEXT NOT NULL,
    from_memory_id TEXT,
    to_memory_id   TEXT,
    at             TEXT NOT NULL,
    recorded_at    TEXT NOT NULL,
    confidence     REAL NOT NULL DEFAULT 1.0,
    summary        TEXT NOT NULL DEFAULT '',
    identity_id    TEXT,
    evidence_json  TEXT NOT NULL DEFAULT '[]',
    FOREIGN KEY (from_memory_id) REFERENCES memories(id)   ON DELETE SET NULL,
    FOREIGN KEY (to_memory_id)   REFERENCES memories(id)   ON DELETE SET NULL,
    FOREIGN KEY (identity_id)    REFERENCES identities(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_transitions_at   ON transitions(at);
CREATE INDEX IF NOT EXISTS idx_transitions_from ON transitions(from_memory_id);
CREATE INDEX IF NOT EXISTS idx_transitions_to   ON transitions(to_memory_id);
CREATE INDEX IF NOT EXISTS idx_transitions_kind ON transitions(kind);

-- --------------------------------------------------------------------------
-- Embeddings
--
-- Decoupled from memories so that a record can be re-embedded with a newer
-- model without rewriting the memory itself. (model, dim) is part of the
-- primary key so multiple coexisting embedders are supported.
-- --------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS embeddings (
    memory_id  TEXT    NOT NULL,
    model      TEXT    NOT NULL,
    dim        INTEGER NOT NULL,
    vector     BLOB    NOT NULL,
    created_at TEXT    NOT NULL,
    PRIMARY KEY (memory_id, model),
    FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_embeddings_model ON embeddings(model);

-- --------------------------------------------------------------------------
-- Active Reconsolidation queue
--
-- Durable so that ARW survives a crash; the foreground write path enqueues
-- here and returns immediately. Background workers dequeue, classify, and
-- produce inferences/transitions. (FIRST_PRINCIPLES commandment 3.)
-- --------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS pending_arw (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id    TEXT    NOT NULL,
    enqueued_at  TEXT    NOT NULL,
    attempts     INTEGER NOT NULL DEFAULT 0,
    last_error   TEXT,
    FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_pending_arw_enqueued ON pending_arw(enqueued_at);

-- --------------------------------------------------------------------------
-- Entity graph (v0.3 hybrid retrieval)
--
-- A lightweight, deterministic entity index materialized at write time. The
-- graph is built from the Reify Pydantic memories themselves (Facts have
-- explicit subject/object; Experience/Observation/Belief/Inference have free
-- text scanned by `reify.retrieve.entities.extract_entities`). Entities are
-- canonicalized to ``kind:slug`` (e.g. ``person:arun``, ``role:staff_engineer``)
-- so that surface variations (``Arun``, ``arun``) collapse to one node.
--
-- The engine stays useful with this table empty: graph-walk is one optional
-- recall stage among five (FIRST_PRINCIPLES.md commandment 4 \u2014 graceful
-- degradation). Populating it is foreground but post-durable: a failure here
-- never invalidates the underlying memory commit.
-- --------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS entities (
    id            TEXT PRIMARY KEY,
    label         TEXT NOT NULL,
    kind          TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_entities_kind  ON entities(kind);
CREATE INDEX IF NOT EXISTS idx_entities_label ON entities(label);

CREATE TABLE IF NOT EXISTS entity_mentions (
    memory_id  TEXT    NOT NULL,
    entity_id  TEXT    NOT NULL,
    span_start INTEGER NOT NULL DEFAULT 0,
    span_end   INTEGER NOT NULL DEFAULT 0,
    surface    TEXT    NOT NULL DEFAULT '',
    confidence REAL    NOT NULL DEFAULT 1.0,
    role       TEXT    NOT NULL DEFAULT 'mention',  -- 'subject' | 'object' | 'mention'
    PRIMARY KEY (memory_id, entity_id, span_start),
    FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE,
    FOREIGN KEY (entity_id) REFERENCES entities(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_entity_mentions_entity ON entity_mentions(entity_id);
CREATE INDEX IF NOT EXISTS idx_entity_mentions_memory ON entity_mentions(memory_id);

CREATE TABLE IF NOT EXISTS entity_edges (
    from_entity  TEXT NOT NULL,
    to_entity    TEXT NOT NULL,
    relation     TEXT NOT NULL,            -- 'co_occurs' | 'works_at' | 'located_in' | predicate
    weight       REAL NOT NULL DEFAULT 1.0,
    last_seen_at TEXT NOT NULL,
    PRIMARY KEY (from_entity, to_entity, relation),
    FOREIGN KEY (from_entity) REFERENCES entities(id) ON DELETE CASCADE,
    FOREIGN KEY (to_entity)   REFERENCES entities(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_entity_edges_from ON entity_edges(from_entity);
CREATE INDEX IF NOT EXISTS idx_entity_edges_to   ON entity_edges(to_entity);
