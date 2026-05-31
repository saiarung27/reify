# Reify

**A model-portable cognitive substrate for AI.**

Repository: `reify`

Reify is a memory engine that any AI model can plug into via MCP, REST, or an OpenAI-compatible proxy. It runs locally on macOS, Linux, and Windows from a single Python 3.11+ install. It is built on a small set of binding constraints (see [FIRST_PRINCIPLES.md](./FIRST_PRINCIPLES.md)) and ships when one demo works that no other production system can run.

## The Wedge

> **Day 1** (Claude Desktop): *"I just got promoted from Senior to Staff Engineer."*
>
> **Day 30** (Claude Desktop): *"What's my title?"* → *"Staff Engineer."*
>
> **Day 30** (Claude Desktop): *"Did anything change recently?"* → *"Yes — you were promoted from Senior to Staff Engineer 30 days ago."*
>
> **Day 60** (Cursor) *(roadmap item; requires the v0.4 LLM-backed reconsolidator + sleep phase)*: *"What do I usually emphasize in promotion writeups?"* → *"You frame promotions around impact-on-team."* (a derived claim materialized across sessions — not raw conversation.)

What the leaders cannot do today:

* **Mem0 / OpenMemory MCP** — cannot answer step 3 (no transitions) or step 4 (no materialized cross-session inferences).
* **Zep / Graphiti** — can almost answer step 3 (it stores `valid_at` / `valid_until`) but does not surface the change as a narrative or a queryable artifact.
* **Mastra Observational Memory** — strong recall, but synthesis is computed at read time only; "what do I usually emphasize" is a synthesis the model has to redo on every query.

Reify ships when the first three steps above run end-to-end on a real MCP host with no LLM key. That is the v0.1 acceptance bar, and it is what `make demo` verifies on every clean clone. Step 4 is a v0.4 capability tracked in [ROADMAP.md](./ROADMAP.md).

## The Thesis

> Memory is not a key-value store of facts. It is a directed graph of events with bitemporal validity, causal edges, and **materialized inferences** that the system actively maintains as new evidence arrives.

Existing memory systems leave the heavy lifting to the model at retrieval time. Top-K vector similarity, then "good luck synthesizing." This is why every leader plateaus at the same place: a hard ceiling at ~87% on multi-session synthesis on LongMemEval — admitted in their own publications.

Reify’s working hypothesis is that this ceiling is a *retrieval-time-only* artifact — not a fundamental property of long-term memory. By treating the memory layer itself as an active reasoner (Active Reconsolidation on Write, Bitemporal Transitions, sleep-phase synthesis), every write becomes a chance to consolidate, and recall returns conclusions instead of just clues. Whether this hypothesis beats the published numbers is an open question Reify is built to answer; current results are reported under [Status](#status), not asserted up front.

## How Reify relates to other memory systems

Reify borrows ideas from the public literature and existing systems and combines them in a way no single system does today. The table below describes the *architectural* difference, not a head-to-head benchmark; for current measured numbers see [Status](#status).

| System | Where it is strongest | What Reify does differently |
|---|---|---|
| **Mastra Observational Memory** | Observer + Reflector dual-agent ingest | Same dual-pass idea, decoupled from any agent framework, plus Active Reconsolidation on Write |
| **Hindsight** | Four memory networks separating evidence from inference | Adopts the split and adds **Inferences** as a materialized fifth network |
| **Zep / Graphiti** | Bitemporal `valid_at` / `invalid_at` on every edge | Adds **Transition events** as first-class memories — the change itself, not just the before / after states |
| **Honcho** | Dialectic implicit-pattern inference | MIT-licensed re-implementation, combined with bitemporal validity and ARW |
| **Mem0 / OpenMemory MCP** | Cross-vendor MCP portability | Equal portability with the additional consolidation, transition, and inference layers underneath |

## The Eight Pillars

1. **Five-network memory model**: Facts, Experiences, Observations, Beliefs, **Inferences** (novel)
2. **Active Reconsolidation on Write (ARW)** — the wedge
3. **Bitemporal Transition Graph (BTG)** — change as data
4. **Dialectic Reflection Loop (DRL)** — implicit patterns + adversarial probing
5. **Identity Bond Graph (IBG)** — probabilistic cross-device user resolution
6. **Hybrid retrieval pipeline** — parallel semantic / BM25 / graph / temporal / causal, RRF-fused, cross-encoder reranked, token-budget-aware
7. **MCP-native, multi-surface** — MCP server, OpenAI-compatible proxy, REST, SDKs
8. **Open evaluation harness in CI** — LongMemEval, LoCoMo, BEAM, plus a new Multi-Session Synthesis track

See [ARCHITECTURE.md](./ARCHITECTURE.md) for the full design.

## Status

Pre-alpha, single-maintainer. The v0.1 wedge runs end-to-end through the core engine, CLI, and automated MCP stdio tests, plus a manual smoke test against Windsurf with an OpenAI-compatible provider. The main remaining operational caveat is ARW latency during `reify reflect`.

### Current numbers (reproducible from a clean clone)

All numbers below are produced by the bundled fixture suites in this repo. Official dataset numbers are also provided for LongMemEval and LoCoMo.

| Suite | Command | Result |
|---|---|---|
| User-value gate (6 fixture cases) | `make user-value` | 6 / 6 useful recalls, **100 % recall-without-LLM**, p50 ~1 ms, p95 ~6 ms, $0 / recall |
| LongMemEval bundled fixture (4 cases) | `reify eval longmemeval-fixture` | 4 / 4, 100 % overall (single-hop + temporal sub-tracks) |
| LongMemEval official dataset (227 cases) | `reify eval longmemeval --dataset <path>` | **78.8 %** overall (knowledge-update: 87.2 %, multi-session: 57.1 %, single-session-assistant: 91.1 %, single-session-preference: 76.7 %, single-session-user: 88.6 %, temporal-reasoning: 85.7 %) |
| LoCoMo bundled fixture (6 cases) | `reify eval locomo` | 5 / 6, 83.3 % overall, target 80 % — the adversarial-negation sub-case currently fails (tracked for v0.4 adversarial reflection) |
| LoCoMo official dataset (50 conversations, 204 QAs) | `reify eval locomo --dataset <path>` | **82.9 %** overall (adversarial: 100.0 %, multi_hop: 66.7 %, open_domain: 87.9 %, single_hop: 80.5 %, temporal: 53.0 %) |
| BEAM bundled fixture (3 cases) | `reify eval beam` | **33.3 %** overall, p50 11.6 ms, p95 14.9 ms (full dataset requires generation pipeline) |
| Test suite | `make test` | 461 / 461 |

Numbers can drift between commits. CI re-runs all of the above on every push and pull request to `main`. A performance regression gate enforces that p50 recall latency stays below 5 ms and p95 below 20 ms on the bundled fixture.

## What Reify is not (yet)

To keep expectations honest:

- **Not benchmarked against BEAM full dataset or STATE-Bench.** LongMemEval (78.8%) and LoCoMo (82.9%) official dataset numbers are available with lenient retrieval proxy matching. BEAM bundled fixture tested (33.3%); full BEAM dataset requires LLM-judged evaluation with complex question types (abstention, contradiction resolution, event ordering) that are incompatible with Reify's retrieval proxy approach. STATE-Bench requires agent framework integration with their evaluation pipeline; stub provided in `src/reify/eval/statebench.py`.
- **Not production-grade.** No Postgres / pgvector adapter, no E2E-encrypted multi-device sync, no OpenTelemetry. All on the v1.0 roadmap. Note: deterministic PII redaction (emails, phone numbers, SSN-shaped strings, credit-card-shaped strings) is available via `REIFY_PII_REDACT=on`.
- **Not multi-tenant out of the box.** The HTTP surfaces refuse to bind on non-loopback hosts unless `REIFY_HTTP_TOKEN` is set; see [`SECURITY.md`](./SECURITY.md).
- **Not a peer-reviewed system.** No paper, no third-party reproduction. The architecture and design are documented in [`ARCHITECTURE.md`](./ARCHITECTURE.md) and [`FIRST_PRINCIPLES.md`](./FIRST_PRINCIPLES.md); claims that go beyond what is measured here are flagged as roadmap items.
- **Not a turnkey replacement for vector RAG.** Reify is an opinionated memory engine, not a generic vector store. If your use case is purely “embed and retrieve documents,” a plain vector DB is simpler and probably faster.

See [ROADMAP.md](./ROADMAP.md) for the full milestone plan.

## Quickstart

### 60-second smoke test

```bash
git clone https://github.com/saiarung27/reify.git
cd reify
make install        # creates .venv, installs reify + dev extras
make demo           # runs the four-step wedge end-to-end, no LLM key
make test           # full pytest suite
```

`make demo` is the headline proof: it ingests two promotions, queries the
current title bitemporally, lists every memory the engine retains, and
exports to Markdown - all against a temp SQLite store with the
deterministic extractor only. No API keys, no network.

### For developers

```bash
git clone https://github.com/saiarung27/reify.git
cd reify
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"  # editable install; required for the CLI
reify init               # creates ~/.reify with store.db and config.toml
reify serve              # start the MCP stdio server
```

`reify init` creates `~/.reify/` (overridable via `REIFY_HOME`) containing
`store.db` (SQLite + sqlite-vec for embeddings), `config.toml`, and a
`schema.sql` snapshot. It is idempotent - re-running it does not reset
existing data. To wipe state, delete `~/.reify` and rerun `reify init`.

### CLI usage (no MCP host required)

The wedge demo can be tested entirely via CLI without any MCP host:

```bash
# Step 1: Ingest promotion
reify remember "I got promoted from Senior to Staff Engineer" --identity user123 --at "2025-01-01T00:00:00Z"
reify reflect

# Step 2: Query title
reify recall "What's my title?" --identity user123 --at "2025-01-30T00:00:00Z"

# Step 3: Query changes
reify what-changed --identity user123 --days 30

# Step 4: Query knowledge
reify remember "I got promoted from Staff to Principal Engineer" --identity user123 --at "2025-04-01T00:00:00Z"
reify reflect
reify what-do-you-know "promotion" --identity user123

# Export everything
reify export /tmp/reify_export.md
```

### MCP integration

`reify serve` exposes seven tools to any MCP-compatible host: `remember`,
`recall`, `reflect`, `whatchanged`, `whatdoyouknow`, `forget`,
`identity_bond`. Working copy-paste configurations for Claude Desktop,
Cursor, and Windsurf live in
[`examples/mcp-configs/`](./examples/mcp-configs/README.md).

Quick Windsurf example (no API key required for the deterministic-only
flow):

```json
{
  "mcpServers": {
    "reify": {
      "command": "/abs/path/to/reify/.venv/bin/reify",
      "args": ["serve"],
      "env": { "REIFY_HOME": "~/.reify" }
    }
  }
}
```

To enable the LLM-backed reconsolidator (BTG + ARW), add an API key and a
provider spec to `env`:

- OpenAI: `"OPENAI_API_KEY"` + `"REIFY_LLM_RECONSOLIDATE": "openai:gpt-4o-mini"`
- Anthropic: `"ANTHROPIC_API_KEY"` + `"REIFY_LLM_RECONSOLIDATE": "anthropic:claude-3-5-sonnet-latest"`
- Fully local: `"REIFY_LLM_RECONSOLIDATE": "ollama:llama3.1"` with a running Ollama (no key needed)

Operational note: `remember` stays fast because writes are async-first,
but `reflect` takes 5-50 seconds when an LLM reconsolidator is enabled.

### Examples

- [`examples/python-quickstart/`](./examples/python-quickstart/quickstart.py) - end-to-end Python script (no LLM key)
- [`examples/node-app/`](./examples/node-app/index.js) - Node.js client against the REST API
- [`examples/mcp-configs/`](./examples/mcp-configs/README.md) - Claude Desktop / Cursor / Windsurf

### Web dashboard (optional, read-only)

A read-only web UI ships under [`dashboard/`](./dashboard/). Start it
alongside the engine:

```bash
reify serve-rest        # REST API on http://127.0.0.1:8001
reify serve-dashboard   # Vite dev server on http://127.0.0.1:3000
```

The dashboard reads through the REST API only; it cannot mutate state.
See [SECURITY.md](./SECURITY.md) before exposing any HTTP server on a
non-loopback host - the bind guard will refuse such configurations
unless `REIFY_HTTP_TOKEN` is set.

### Forgetting and retention

Reify now applies a default TTL to claim-like memories and reinforces that TTL when a claim is successfully recalled. The current v0.2 surface is:

```bash
reify forget <memory-id>
reify forget --tag employer
reify forget --before "2025-01-01T00:00:00+00:00"
```

Retention knobs:

```bash
REIFY_DEFAULT_TTL_DAYS=30
REIFY_REINFORCEMENT_FACTOR=1.5
REIFY_DECAY_SWEEP_INTERVAL_SECONDS=60
```

## Project Stance

The full constitutional layer lives in [FIRST_PRINCIPLES.md](./FIRST_PRINCIPLES.md). The summary:

- **Plain SQL is truth.** Any user can `sqlite3 ~/.reify/store.db` and read everything.
- **Async-first, LLM-optional writes.** The default write path is deterministic-extract → SQL commit → return. The user never waits for an LLM.
- **Forget by default.** Every claim has a TTL. Reinforcement extends it. Decay is the rule.
- **Recall is metadata-first.** Tag → time → entity → BM25 → embeddings. Must work with embeddings disabled.
- **Graceful degradation.** No LLM key → still works. No embedder → lexical. No internet → fully local.
- **Legible to humans.** `reify export`, `reify what-do-you-know`, `sqlite3 ~/.reify/store.db`, read-only dashboard.
- **Local-first.** Single SQLite file. Optional E2E-encrypted sync on the v1 roadmap.
- **Model-agnostic.** Pluggable LLM providers (OpenAI, Anthropic, Ollama, plus any OpenAI-compatible endpoint). Pluggable embedders.
- **Framework-agnostic.** No agent framework required. Single library + CLI is the product; MCP, REST, OpenAI-proxy are thin adapters (< 200 LOC each).
- **Open.** MIT licensed. All benchmark results reproducible from a single command.

## Maintenance & support

Reify is maintained by **one developer**. Issues are triaged on a best-effort
basis, usually within a few days. Expect the following stance:

- **Bug reports, docs fixes, and small focused PRs are welcome.** Please run
  `make demo` and `make test` first; if either fails on a clean clone, that is
  itself a high-priority bug worth filing.
- **Larger feature requests should open an issue first** so we can confirm the
  change fits [`FIRST_PRINCIPLES.md`](./FIRST_PRINCIPLES.md) before code lands.
- **No guarantee of rapid response, and no promise to support every MCP host
  or every LLM provider on day one.** Compatibility expands as users contribute
  configs and bug reports.
- **Security issues** must be reported privately via the channel in
  [`SECURITY.md`](./SECURITY.md), not in a public issue.

See [`CONTRIBUTING.md`](./CONTRIBUTING.md) for setup, coding style, and PR
expectations.

## License

MIT. See [LICENSE](./LICENSE).

---

*See [GLOSSARY.md](./GLOSSARY.md) for abbreviations used in this document.*
