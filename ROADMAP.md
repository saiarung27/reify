# Reify Roadmap

A measurable, dated plan. Each milestone has an exit criterion that is binary and testable. The order is dictated by [FIRST_PRINCIPLES.md](./FIRST_PRINCIPLES.md): the wedge demo ships first, infrastructure earns its complexity by being required for the next demo, benchmarks come *after* the wedge is proven.

## Two release gates, applied to every milestone

No milestone passes without clearing both:

1. **Functional gate.** The milestone's user-visible demo runs end-to-end on a clean install with the same `pip install -e .` command.
2. **User-value gate.** No regression in any of the four user-value metrics (measured by `make user-value`):
   * **Time-to-useful-recall** — median latency from query submission to retrieved memory landing in the prompt. Target: < 5 ms p50, < 20 ms p95, no LLM in the path.
   * **Recall-without-LLM rate** — fraction of recalls that return useful results without invoking an LLM. Target: ≥ 70 %.
   * **30-day recall accuracy** — fraction of memories successfully recalled by their natural-language probe 30 simulated days after ingest. Target: ≥ 85 %.
   * **Cost per useful recall** — LLM + embedding spend per useful recall, measured against a fixture workload. Target: ≤ $0.0005.

Benchmarks (LongMemEval, LoCoMo, BEAM) are an *additional* gate from v0.3 onward but never replace the user-value gate.

## v0.1 — The Wedge Demo (Week 1–2)

Goal: the four-step promotion narrative runs end-to-end. CLI-first, MCP as thin adapter.

What is required to clear the demo and nothing else gets implemented:

- [x] SQLite store with the minimum schema: `memories`, `transitions`, `provenance`, `embeddings`, `pending_arw`. Plain SQL inspectable from `sqlite3`.
- [x] Deterministic extractor: regex patterns for promotions, locations, employers, titles. Works with no LLM key configured.
- [x] Async write path: deterministic-extract → SQL commit → return. ARW runs behind an asyncio queue and is a no-op when no LLM is configured.
- [x] Active Reconsolidation on Write: minimal one-shot LLM classifier that checks a new claim against k=8 metadata-filtered candidates and writes a single derived `Inference` claim with `relation` set.
- [x] Bitemporal Transition Graph: writes `Transition` records when ARW observes a `CONTRADICTS` or `RESOLVES` relation; binds the prior claim's `valid_until`.
- [x] Recall: metadata-first pipeline (tag → time → entity → BM25), with embeddings as an optional fifth stage. Returns a stable, typed payload.
- [x] MCP server (stdio): `remember`, `reflect`, `recall`, `whatchanged`, `whatdoyouknow` tools. Wired to `mcp` Python SDK. (Thin adapter, < 200 LOC)
- [x] CLI: `reify init`, `reify remember`, `reify recall`, `reify what-changed`, `reify what-do-you-know`, `reify export`. All commands support `--at` flag for controllable timestamps.
- [x] `reify export` produces a single human-readable markdown file of every claim, transition, and inference. Round-trippable.
- [x] CLI-runnable wedge demo acceptance test: uses engine with injectable clock to compress 60-day story into seconds. No MCP host required.
- [x] Test suite covering types, storage, ARW happy path, CLI subprocess flows, and the wedge demo end-to-end (scripted MCP client).

**Exit criterion (functional):** the four-step promotion demo runs via CLI with `--at` timestamps, the automated acceptance suite passes, and the same flow works through the thin MCP adapter on a real host. Current manual proof point: Windsurf MCP against an OpenAI-compatible provider.

**Exit criterion (user-value):** the four user-value metrics meet their targets on a fixture conversation set.

## v0.2 — Forget by Default + Reinforcement (Week 3)

Goal: memory has a TTL by default. Reinforcement extends it. The user-perceived effect is that stale things go away on their own.

Current next-step note: the wedge is proven, the retention policy is implemented with audited decay sweeps and survivor-set validation, and the default user-value harness is runnable via `make user-value`. The remaining work is to broaden that fixture workload as retrieval gets more capable, then move into the v0.3 benchmark floor.

- [x] Per-claim TTL set at write time (default: 30 days; LLM may override during ARW).
- [x] Reinforcement: a recall that returns a claim extends its TTL by a multiplicative factor.
- [x] Decay sweep (idle scheduler): tombstones expired claims; emits structured logs so users can audit.
- [x] CLI: `reify forget <id>`, `reify forget --tag <tag>`, `reify forget --before <date>`.
- [x] MCP tool: `forget`.

**Exit criterion:** a fixture workload of 1000 claims with mixed reinforcement patterns produces the expected survivor set after a simulated 90 days. 30-day recall accuracy stays ≥ 85 %. Both claims are enforced by targeted tests plus the runnable `make user-value` fixture gate.

## v0.3 — Hybrid Retrieval + Benchmark Floor (Week 4–5)

Goal: graduate from "works on the wedge demo" to "survives a real benchmark."

Guardrails from first principles:
- Retrieval additions remain layered after metadata-first filtering. Tag/time/entity/BM25 stay the default path; graph and reranking are later stages, not replacements.
- Benchmark work may not introduce an LLM dependency into the default recall path.
- Benchmark wins do not count if they regress the user-value gate.

- [x] Entity graph extraction during write (lightweight, deterministic + LLM-residual).
- [x] Graph-walk retrieval strategy.
- [x] Cross-encoder reranker (BGE reranker base, local). Pluggable `Reranker` protocol with deterministic `FeatureReranker` (default; dep-free) and `BGEReranker` adapter (optional, `reify[rerank]`).
- [x] Token-budget-aware context packer with model profiles (Claude, GPT, Gemini, local). `reify.synthesize.pack` + `Reify.recall_packed`.
- [x] LongMemEval harness scaffold: `reify eval longmemeval-fixture` for bundled cases, plus `reify eval longmemeval --dataset ...` for external JSON/JSONL case loading and per-category breakdown.
- [x] LongMemEval official dataset run: 78.8% overall (knowledge-update: 87.2%, multi-session: 57.1%, single-session-assistant: 91.1%, single-session-preference: 76.7%, single-session-user: 88.6%, temporal-reasoning: 85.7%).
- [x] LoCoMo harness: `reify eval locomo` (bundled fixture + JSON/JSONL loader; 80% proxy target).
- [x] LoCoMo official dataset run: 82.9% overall (adversarial: 100.0%, multi_hop: 66.7%, open_domain: 87.9%, single_hop: 80.5%, temporal: 53.0%).
- [x] BEAM harness: `reify eval beam` (bundled fixture; full dataset requires LLM-judged evaluation incompatible with retrieval proxy).
- [x] BEAM bundled fixture run: 33.3% overall, p50 11.6 ms, p95 14.9 ms.
- [x] STATE-Bench adapter stub: `src/reify/eval/statebench.py` (full integration requires agent framework).
- [x] CI gates merges on benchmark baseline + user-value gate (`.github/workflows/ci.yml`: pytest + user-value + LongMemEval fixture + LoCoMo fixture across Python 3.12/3.13).

**Exit criterion (benchmarks):** Official LongMemEval (78.8%) and LoCoMo (82.9%) dataset numbers published with lenient retrieval proxy matching. BEAM bundled fixture (33.3%) tested; full BEAM dataset requires LLM-judged evaluation incompatible with Reify's retrieval proxy approach. STATE-Bench stub provided; full integration requires agent framework. User-value metrics still pass.

## v0.4 — Sleep Phase + Adversarial Reflection (Week 6)

Goal: principled consolidation (the 9.1 open challenge).

Guardrails from first principles:
- Sleep-phase and reflection work must stay off the foreground write/read path.
- The system must remain useful with no LLM configured; these stages are enhancements, not prerequisites.
- Consolidation outputs must stay inspectable in SQL/export surfaces.

- [x] Hot-buffer with promotion gate (re-verification, deduplication, importance scoring).
- [x] Cross-session synthesis pass during idle periods (the explicit attack on the 87 % multi-session ceiling).
- [x] Adversarial belief probing.
- [x] Belief confidence decay (Ebbinghaus).
- [x] Sleep-phase scheduler (idle detection + nightly run).
- [x] BEAM harness: `reify eval beam --scale 1M`. (10M scale targeted at v1.)
- [x] Multi-Session Synthesis capability breakdown: `reify eval multisession-synthesis` / `reify eval reify-native` groups the current Reify retrieval proxy into product-aligned tracks over official LongMemEval histories.
- [x] Multi-Session Synthesis native scorer: replace the proxy-backed breakdown with a genuinely Reify-native evaluation for continuity, abstention, provenance, and update truth.

**Exit criterion:** Multi-Session Synthesis track score ≥ 90 %. LongMemEval overall ≥ 92 %. Documented evidence that the ARW + sleep-phase combo outperforms read-time-only synthesis on a held-out probe.

## v0.5 — Identity Bond Graph (Week 7)

Goal: cross-session, cross-device identity resolution.

- [x] `Identity` model with anonymous → device → user hierarchy.
- [x] Bond formation from three signal classes: behavioral fingerprint, embedding similarity, explicit assertions.
- [x] Configurable confidence threshold for cross-identity inheritance (default 0.8).
- [x] CLI: `reify identity link/unlink/list`.
- [x] MCP tool: `identity_bond`.

**Exit criterion:** a fresh anonymous session is correctly linked to the user's identity at threshold 0.8 from three behavioral signals, in a fixture scenario.

## v0.6 — Surfaces (Week 8)

Goal: every developer integration path. Each adapter is < 200 LOC.

Guardrails from first principles:
- Adapters remain thin and transport-only; no adapter may contain unique memory logic that is absent from the library.
- Surface expansion does not outrank core engine quality, inspectability, or graceful degradation.

- [x] OpenAI-compatible HTTP proxy (drop-in `chat/completions` that injects retrieved context).
- [x] REST API (FastAPI).
- [x] Python SDK polish + docs.
- [x] TypeScript SDK.
- [x] MCP HTTP/SSE transport.
- [x] Read-only web dashboard: browse claims, transitions, inferences; "what do you know about me"; export.
  - **Note:** dashboard work is verified in a local browser before commit, per project rules.

**Exit criterion:** Reify works with Claude Desktop, Cursor, Cline, Zed, Windsurf, and a sample Node app from the same install.

## v1.0 — Production-grade (Week 9–12)

- [ ] Postgres + pgvector adapter
- [ ] E2E-encrypted multi-device sync (cr-sqlite or similar)
- [ ] PII redaction extractor in write path
- [ ] OpenTelemetry instrumentation across all stages
- [ ] Rust port of hot paths (storage, retrieval, ARW classifier dispatch)
- [ ] Swift + Kotlin bindings (iOS / Android)
- [ ] Multi-modal embedders (CLIP) and ingestors (screenshot)
- [ ] First public release; arXiv preprint of the architecture and evaluation

**Exit criterion:** Reify v1.0 ships with peer-reviewable benchmark results that beat 94.87% LongMemEval and 90% on the Multi-Session Synthesis track, while being framework-agnostic.

## Beyond v1

- Foundation model for memory management (the 9.9 open challenge)
- Causal discovery integration for retrieval
- Standardized evaluation harness adopted as community baseline (the 9.10 open challenge)
- Embodied / robotics memory integration

---

*See [GLOSSARY.md](./GLOSSARY.md) for abbreviations used in this document.*
