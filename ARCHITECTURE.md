# Reify Architecture

This document is the design of record. Every module in `src/reify/` traces back to a section here. If a component cannot be justified by something in this document, it should not exist.

## 1. Problem Statement

The current generation of agent memory systems (Mem0, Letta, Zep, Hindsight, Mastra Observational Memory, MIRIX, Cognee) shares a common ceiling. On LongMemEval, the gold-standard benchmark, the best-published systems (Mastra OM at 94.87%, Hindsight at 91.4%) hit the same wall on **multi-session synthesis** — questions that require combining evidence from many separated sessions. Both authors publish this ceiling in their own limitations sections.

The 2026 community survey *Memory for Autonomous LLM Agents: Mechanisms, Evaluation, and Emerging Frontiers* enumerates ten open challenges (section 9). The four most tractable and most economically valuable, in our judgement, are:

- **9.1 Principled consolidation** — current systems oscillate between hoarding and amnesia.
- **9.2 Causally grounded retrieval** — semantic similarity answers "what looks like this?" not "what caused this?"
- *(implicit, surfaced in benchmarks)* **Multi-session synthesis** — the 87% ceiling.
- **Cross-session identity resolution** — every system assumes a stable `user_id`.

Reify is the architecture aimed at these four.

## 1a. Operating Constraints

The architecture below is the *implementation* of the constitutional rules in [FIRST_PRINCIPLES.md](./FIRST_PRINCIPLES.md). Whenever this document conflicts with that one, that one wins. The binding rules in summary:

* Plain SQL is truth; Pydantic is a typed view.
* Two memory types in v0 (`Episode`, `Claim`); the five-network model in section 3 is the *aspiration*, surfaced in the type system but not all five are implemented in v0.
* Async-first, LLM-optional writes — the user-facing write path never waits on an LLM.
* Forget by default — every claim has a TTL.
* Recall is metadata-first — embeddings are one channel among several, not the primary one.
* Graceful degradation everywhere — every subsystem works with reduced fidelity when LLM, embedder, or network is absent.
* User-value metrics gate releases alongside benchmarks (see [ROADMAP.md](./ROADMAP.md)).
* Single library + thin adapters — each adapter (MCP, REST, OpenAI-proxy) is < 200 LOC.

The rest of this document is read against those rules. If a section appears to violate one, it is the section that is wrong.

## 2. Core Insight

> Move synthesis from read-time to write-time and consolidation-time.

When new evidence arrives, the system actively retrieves related existing memories, has the LLM classify the relationship (`confirms` / `contradicts` / `extends` / `refines` / `resolves`), and writes a derived `Inference` record. By the time a query arrives, the cross-session conclusion is already a single retrievable artifact, not a multi-hop computation.

This is the database "materialized view" pattern applied to agent memory. No production system implements it.

## 3. Data Model

### 3.1 Five Memory Networks (aspiration) – Two in v0 (implementation)

The full data model surfaces five networks in the Pydantic type system. They share a base shape (`id`, `created_at`, `valid_at`, `valid_until`, `confidence`, `provenance[]`).

| Network | Holds | Source |
|---|---|---|
| `Fact` | Declarative atoms with bitemporal validity | Stated by user, or extracted with high confidence |
| `Experience` | Time-stamped events, episodic | Conversation turns, tool calls, sessions |
| `Observation` | Passive captures, unverified | Browser, screen, file changes |
| `Belief` | Inferred patterns with confidence + evidence | Dialectic reflector |
| `Inference` | **Materialized cross-session conclusions** | Active Reconsolidation on Write |

**v0 implementation note.** Per FIRST_PRINCIPLES commandment 2, only two of these flow through the v0 ingest → store → recall pipeline: `Experience` (the episode network) and `Fact` / `Belief` / `Inference` (the three claim networks the wedge demo requires). `Observation` is deferred until passive capture earns its keep. Storage is unified into two physical tables (`episodes`, `claims`); the network discriminator is a column. The full taxonomy is in the type system so that adding a network later is a non-event.

`Inference` is the novel network. Each inference carries:
- `relation_type ∈ {confirms, contradicts, extends, refines, resolves}`
- `parent_memory_ids: list[ULID]` (the memories combined)
- `derivation_method` (which extractor produced it)
- `confidence` and `evidence_strength`
- `valid_at`, `invalid_at` (it can become stale and be replaced)

### 3.2 Bitemporal Transition Graph

Beyond Zep's `valid_at` / `invalid_at` (which models *replacement*), Reify introduces `Transition` as a first-class memory:

```
Fact(employer="X", valid: [t1, t2])
Fact(employer="Y", valid: [t2, ∞))
Transition(type=EMPLOYER_CHANGE, from=X, to=Y, at=t2, evidence=[exp_id_1, exp_id_2])
```

Queries can now ask not only "what is true?" but "what changed?" and "when did it change?" and "what evidence drove the change?"

This resolves the staleness problem on high-relevance facts: the old fact is not deleted; it is transitioned, with an explicit evidence trail.

### 3.3 Identity Bond Graph

Every memory is scoped to an `Identity`. Identities form a probabilistic graph:

```
Anonymous(session_xyz) --[0.62]--> Device(macbook_pro_a1) --[0.94]--> User(arun)
                                                          \--[0.71]--> User(arun_work)
```

A configurable confidence threshold controls cross-identity memory inheritance at retrieval time. Bonds are formed by `IdentityResolver` — a separate subsystem that watches for shared embeddings, behavioral fingerprints, and explicit user assertions.

## 4. Write Path: Active Reconsolidation (queue-based)

The user-facing write path is short, deterministic, and never waits on an LLM. The expensive parts run behind a queue.

```
FOREGROUND  (≤ 50 ms typical, no network IO)
  INGEST → DETERMINISTIC-EXTRACT → CANONICALIZE → STRUCTURAL-DEDUPE → SQL-COMMIT → ENQUEUE → RETURN
                                                                              ↓
BACKGROUND  (consumer pool, may take seconds)
  DEQUEUE → LLM-EXTRACT-RESIDUAL → SEMANTIC-DEDUPE → RECONSOLIDATE → EMIT-INFERENCES + TRANSITIONS
```

### Foreground stages

1. **INGEST** — raw event in (chat turn, tool call, file).
2. **DETERMINISTIC-EXTRACT** — cheap pattern-based extractors (regex, simple NER, dialog markers, structured assertions like "I prefer X"). Handles ~60 % of useful extractions for zero LLM cost. Always runs.
3. **CANONICALIZE** — normalize entities ("SF" → "San Francisco"), dates, units.
4. **STRUCTURAL-DEDUPE** — hash-based dedup against existing claims; cheap.
5. **SQL-COMMIT** — atomic transaction; the new `Episode` and any deterministic `Claim`s land.
6. **ENQUEUE** — a job is appended to the reconsolidation queue. Durable (a row in a `pending_arw` table); survives crash.
7. **RETURN** — the foreground call is complete.

### Background stages

1. **LLM-EXTRACT-RESIDUAL** — the LLM is invoked only for content that the deterministic stage flagged as ambiguous. Skipped entirely if no LLM is configured.
2. **SEMANTIC-DEDUPE** — embedding-based dedup against existing claims. Skipped if no embedder is configured.
3. **RECONSOLIDATE** — for each new claim:
   * retrieve k=8 metadata-filtered candidates (entity overlap + recency window) before any vector search.
   * LLM classifies relationship to each (`none` / `confirms` / `contradicts` / `extends` / `refines` / `resolves`).
   * for each non-`none` classification, write an `Inference` record with provenance.
4. **EMIT-INFERENCES + TRANSITIONS** — if a relation is `contradicts` or `resolves`, also write a `Transition` and bind the prior claim's `valid_until`.

### Why this matters

* **Latency.** The user-facing write returns in tens of milliseconds even for content that ultimately produces inferences and transitions. The user never feels reconsolidation.
* **Robustness.** An LLM outage or rate-limit does not block writes; the queue keeps growing and drains when service returns.
* **Cost control.** The deterministic stage handles most claims for free; LLMs are a minority of writes.
* **Graceful degradation.** With no LLM and no embedder configured, the foreground stages still produce a useful Reify — ingest, store, structural recall, export. Reconsolidation simply doesn't happen, and the system is honest about it (claims carry `derivation_method="deterministic"`).

## 5. Read Path: Hybrid Retrieval

```
QUERY → INTENT-CLASS → PARALLEL FETCH → RRF FUSE → RERANK → PACK
                            │
        ┌───────────┬───────┼───────────┬───────────┐
     semantic     BM25   entity-walk  temporal   causal-walk
```

1. **INTENT-CLASS** — fast classifier decides which strategies to engage and the token budget
2. **PARALLEL FETCH** — five independent strategies run concurrently
3. **RRF FUSE** — reciprocal rank fusion (Hindsight's pattern; see RRF literature)
4. **RERANK** — cross-encoder reranker scores against the query intent
5. **PACK** — token-budget-aware packer formats for the target model (Claude / GPT / Gemini / local), with citations

Importantly, the retrieval pipeline returns `Inferences` directly when they are the strongest match — meaning the agent does not need to re-derive cross-session conclusions on the fly.

## 6. Consolidation: Sleep Phase

Periodically (configurable; default every N writes or T minutes when idle), Reify runs a consolidation sweep:

1. **Hot-buffer promotion** — memories in the probationary buffer (last 24h) are re-verified, deduplicated, importance-scored, and promoted to long-term storage. Failed candidates are summarized and demoted.
2. **Adversarial belief probing** — for each high-confidence `Belief`, generate counter-hypotheses and check stored `Experiences` and `Observations` for refuting evidence. Beliefs decay confidence without confirming evidence (Ebbinghaus-style).
3. **Cross-session synthesis pass** — sample clusters of semantically-related memories from different sessions and run reconsolidation against them, producing new `Inferences` that the write-time pass may have missed.
4. **Transition detection** — scan for facts whose validity ranges suggest unannotated transitions; emit `Transition` records.
5. **Forgetting** — apply retention policies; tombstone memories whose policy has expired (machine-unlearnable in v1; soft-deleted in v0).

## 6a. Graceful Degradation Matrix

Reify is required to function with any combination of (LLM-present, embedder-present, network-present). The matrix below is normative: the system's behavior in each cell must match what is described.

| LLM | Embedder | Network | Behavior |
|---|---|---|---|
| present | present | present | Full pipeline. ARW runs, sleep-phase synthesis runs, multimodal possible later. |
| present | present | absent | Full pipeline minus any cloud LLM provider. Local LLM (Ollama) handles ARW; embeddings work. |
| present | absent | * | Recall falls back to `tag → time → entity → BM25`. ARW runs but cannot semantic-dedupe. |
| absent | present | * | Recall has all five channels; ARW skipped (claims are deterministic only). Reconsolidation queue grows; drains when an LLM is configured. |
| absent | absent | * | Recall is `tag → time → entity → BM25` only. ARW skipped. The system is still useful: the wedge demo's step 1 and step 2 still work. |
| * | * | absent (offline) | Local-only; no cloud calls attempted. Equivalent to whichever of the rows above matches the local capabilities. |

No configuration combination is allowed to produce a *non-functional* system. Tests enforce this matrix.

## 6b. User-Value Metrics

Measured continuously by `make user-value` against a fixture workload. Releases gate on these in addition to benchmarks. See [ROADMAP.md](./ROADMAP.md) for targets.

* **Time-to-useful-recall** — median latency from query submission to retrieved memory landing in the prompt, with no LLM in the path.
* **Recall-without-LLM rate** — fraction of recalls that return useful results without invoking an LLM.
* **30-day recall accuracy** — fraction of memories successfully recalled by their natural-language probe 30 simulated days after ingest.
* **Cost per useful recall** — LLM + embedding spend per useful recall.

## 7. Surfaces

- **MCP server** (primary). Tools (v0.1): `remember`, `recall`, `reflect`, `whatchanged`, `whatdoyouknow`, `forget`, `identity_bond`. Stdio + HTTP/SSE transports. Compatible with any MCP-spec-compliant host; only Windsurf has been smoke-tested end-to-end as of v0.1, with the others left for community confirmation.
- **OpenAI-compatible HTTP proxy** — drop-in `chat/completions` that injects retrieved context.
- **REST API** (FastAPI, `reify serve-rest`) — `/health`, `POST /remember`, `POST /recall`, `POST /forget`, `GET /what-changed`, `GET /what-do-you-know`. Path versioning (`/v1/...`) is intentionally deferred until the surface stabilises.
- **Python SDK** — `from reify import Reify; r = Reify(); r.remember(...); r.recall(...)`.
- **TypeScript SDK** — for browser/Electron/Node clients.

## 8. Storage

- **SQLite** with `sqlite-vec` extension. One file at `~/.reify/store.db`. Inspectable from `sqlite3` on any platform that runs Python 3.11+ (mobile bindings are a v1.0 roadmap item; see [`ROADMAP.md`](./ROADMAP.md)).
- **Schema versioned** via migrations.
- **Bitemporal indexing** — every memory row carries `valid_at`, `valid_until`, `recorded_at`.
- **Vector index** in a single `embeddings` table keyed on `(memory_id, model)`, so multiple coexisting embedders (different model + dim combinations) can be stored side-by-side without rebuilds.
- **Property graph** modelled in SQL with `entities`, `entity_mentions`, and `entity_edges` tables; backed by a CTE-based traversal helper.
- **WAL journaling** for crash safety; nightly `VACUUM`.
- **Adapter layer** so Postgres + pgvector, DuckDB, and Turso are drop-in alternatives in v1.

## 9. Embeddings

Default: `BAAI/bge-small-en-v1.5` via `fastembed` (CPU-friendly, ONNX, 384-dim). Pluggable. Multimodal embedders (CLIP, ImageBind) on the v1 roadmap.

## 10. LLM Providers

Pluggable. v0.1 ships three concrete providers, all selected via the
`provider:model` spec string — no model name is hardcoded:

- `OpenAIProvider` — OpenAI API and any OpenAI-compatible endpoint (`OPENAI_API_KEY`, `OPENAI_BASE_URL`).
- `AnthropicProvider` — Anthropic Messages API (`ANTHROPIC_API_KEY`, `ANTHROPIC_BASE_URL`). Mirrors the OpenAI provider's contract; structured-output enforcement (`json_schema`) is intentionally not implemented in v0.1 and the provider raises rather than silently returning free-form text.
- `OllamaProvider` — any local model served by Ollama (`OLLAMA_BASE_URL`).

Per-stage configuration: cheap model for extraction, capable model for reconsolidation, fast model for intent classification — all selected via environment variables, none baked into the codebase.

## 11. Evaluation

`reify eval` exposes the harness; `.github/workflows/ci.yml` runs the bundled fixtures on every push and pull request to `main`:

- **User-value gate** — `make user-value` (always runs in CI; release-blocking).
- **LongMemEval bundled fixture** — `reify eval longmemeval-fixture` (always runs in CI).
- **LoCoMo bundled fixture** — `reify eval locomo` (always runs in CI).
- **LongMemEval (official dataset)** — `reify eval longmemeval --dataset ...` (operator-run; not in CI).
- **BEAM 1M scale** — `reify eval beam --scale 1M` (operator-run; not in CI). The 10M scale is a v1.0 roadmap item.
- **Multi-Session Synthesis** — a track derived from LongMemEval that isolates the 87% ceiling. Capability-track breakdown is shipped today; a fully native scorer is also implemented and continues to be hardened.

Failure to maintain the user-value or fixture baselines blocks merges. Numbers measured against the official datasets are not in CI yet; running them publicly is a v0.4 milestone in [`ROADMAP.md`](./ROADMAP.md).

## 12. Privacy and Control

- Local-first; nothing leaves the machine without explicit configuration.
- Per-source ACL — every memory carries the source that produced it; clients can be granted scoped access.
- PII redaction extractor in the write path is on the v1.0 roadmap; until it lands, treat memory content as untrusted-by-default and avoid feeding sensitive material into `reify remember`.
- Full export / full delete supported as first-class operations.
- E2E-encrypted multi-device sync via SQLite CRDT layer (cr-sqlite) on the v1 roadmap.

## 13. Non-Goals (v0)

- Multi-modal capture (images, audio, video) — handled later via dedicated extractors.
- Mobile native bindings — Python core ships first; Rust port + Swift/Kotlin bindings in v1.
- Agent framework — Reify is a substrate, not an agent runtime. Use Letta, LangGraph, etc. on top.
- Distributed / multi-tenant — single-user, single-machine in v0.

## 14. Module Layout

```
src/reify/
├── engine.py        Reify orchestrator: ingest → extract → commit → enqueue
├── recall.py        recall pipeline (metadata-first → lexical → graph → vector)
├── cli.py           `reify` entrypoint (Typer)
├── config.py        env + config.toml loader
├── version.py       single source of truth for `__version__`
├── core/            types, ids, errors
├── storage/         base, sqlite_store (sqlite-vec), hot_buffer, postgres (stub), _schema.sql
├── ingest/          ingest namespace (currently a stub; surface is in engine.py)
├── extract/         base, deterministic, pii_redaction (stub)
├── consolidate/     reconsolidate (ARW), sleep_scheduler, adversarial, cross_session_synthesis
├── identity/        bond_graph
├── retrieve/        entities, graph (CTE walks), rerank, ebbinghaus (decay)
├── synthesize/      pack (token-budget-aware context packer)
├── llm/             base + openai, anthropic, ollama
├── embed/           base (fastembed integrated as a runtime dep, not a separate module)
├── server/          mcp (stdio), mcp_sse, rest_api, openai_proxy, security
├── eval/            longmemeval, locomo, beam, multisession_synthesis, native_scorer, user_value
└── telemetry/       OpenTelemetry placeholder (full instrumentation is v1.0)
```

The orchestration of write and read paths lives in `engine.py` and `recall.py` rather than being split across the network sub-packages — the sub-packages contribute the *primitives* (extractors, retrievers, rerankers, etc.) that the orchestrator composes.

## 15. Quality Bar

- All public surfaces are typed (Pydantic v2 schemas, mypy strict).
- All write-path operations emit structured Python logs today. A real OpenTelemetry instrumentation layer is scaffolded under `src/reify/telemetry/` but currently a placeholder; full distributed-tracing instrumentation is on the v1.0 roadmap.
- All retrieval stages are independently testable; failure of any one stage is graceful.
- All benchmark numbers reported in [`README.md`](./README.md) are reproducible from a single command on a clean checkout. Numbers in [`README.md`](./README.md) are produced by the bundled fixtures shipped in this repo and are explicitly distinguished from the official LongMemEval / LoCoMo / BEAM datasets.

---

*See [GLOSSARY.md](./GLOSSARY.md) for abbreviations used in this document.*
