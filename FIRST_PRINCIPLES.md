# First Principles & Inversion

This document is the constitutional layer of Reify. Every other design decision has to clear it. It distinguishes between:

1. **Core invariants** that define the product floor and are intentionally hard to bend.
2. **Default strategies** that reflect how we should build v0 and near-term versions unless evidence changes them.
3. **Extension principles** that govern higher-order memory intelligence so the product can evolve without violating its own foundation.

If a design choice violates a core invariant, the choice is wrong unless we are prepared to change the constitution. If a design choice conflicts with a default strategy, the burden is to show why the new approach better serves the same product truth.

It is structured the way an engineer interrogates a problem after running it through both inversion ("how would I guarantee failure?") and first-principles decomposition ("what is physically/logically true regardless of received wisdom?").

## 1. Inversion: How would Reify fail?

The list of guaranteed-failure moves, written down so we never make them:

1. **Block writes on synchronous LLM calls.** User-facing latency goes up; users blame Reify; users churn.
2. **Couple the data model to a vendor's output shape.** When the vendor's API changes (or we want to swap it), the data model breaks.
3. **Hide data behind opaque APIs.** Users cannot inspect, edit, export. Trust evaporates. Switching cost flips: it now favors leaving us, not staying.
4. **Require an LLM call on every read.** Cost and latency scale with usage. Mobile and offline scenarios are dead on arrival.
5. **Optimize for benchmarks instead of user-perceived value.** We win the leaderboard, lose the market.
6. **Try to solve every open problem in v0.** We ship nothing. Worse, we ship something fragile.
7. **Build for AI agents and ignore the human.** The agent is a conduit. The human pays.
8. **Build a pipeline whose every stage requires LLM + embedder.** A single upstream outage takes the whole thing down.
9. **Pick a stack that doesn't extend to mobile or constrained devices.** We close off the largest market segment.
10. **Add five memory types before we've validated that two are useful.** Implementation cost balloons; the wedge stays unproven.

These are guard rails, not slogans. The architecture document is read against this list. Every section either dodges the failure mode or admits it explicitly.

## 2. First Principles: What is true regardless of received wisdom?

| Received wisdom | First-principles truth | Consequence for Reify |
|---|---|---|
| "Memory needs an LLM to extract." | An LLM is one extractor among several. Deterministic patterns (NER, regex, dialog markers, structured assertions like "I prefer X") handle the easy cases for zero marginal cost. | **Layered extraction.** Deterministic first; LLM is invoked only for residual ambiguity. |
| "Memory must persist forever." | Cost of memory grows linearly with time; value of any specific memory typically decays exponentially. Most memories should die. | **Forget by default.** Every claim has a TTL. Reinforcement extends it. Decay is the rule, not the exception. |
| "Retrieval must score everything." | The user's current task is a strong prior. Most queries are answered well by metadata-filtered recent records. | **Pre-filter aggressively.** Tag → time window → entity match → BM25 → embeddings. Embeddings are last, not first. |
| "Multi-session synthesis fails because synthesis is hard." | Half of the failure is *recall*: the right facts aren't in top-K because vector similarity missed them. | **Lift recall first, then synthesis.** Cheap structural indexes (per-entity timelines, per-topic clusters, change-logs) are the unsexy 80% of the win. |
| "The user is an AI agent." | The user is a human. The agent is a conduit. Trust requires legibility. | **Legible to humans first.** Plain SQL truth. `export` to markdown. `whatdoyouknow` command. In-place edit. Read-only dashboard. |
| "Five networks are required from day 0." | The wedge requires only the distinction between raw events and derived conclusions. Everything else is taxonomy. | **Two memory types in v0.** `Episode` and `Claim`. The five-network model is the aspiration; the implementation earns each network when it adds value. |
| "Embeddings are *the* unit of similarity." | Embeddings are one channel. Entity overlap (Jaccard), time proximity, co-session frequency, and explicit user tags are cheaper and often stronger signals. | **Hybrid by construction.** Embeddings are one input. The system must work with embeddings off. |
| "Beating 94.87% on LongMemEval is the win." | A 90% system that is 5× cheaper, 10× faster, and never loses user data beats a 95% black box. | **User-value metrics are first-class.** Time-to-useful-recall, recall-per-dollar, fraction-of-recalls-answered-without-LLM. Releases gate on these in addition to benchmarks. |
| "MCP is the primary surface." | MCP is a transport. The product is a Python library + CLI; MCP, REST, and OpenAI-proxy are thin adapters. | **Single library, thin adapters.** Each adapter is < 200 LOC. The core works without any of them. |
| "Users care about ARW, BTG, IBG." | Users care about: did it remember correctly, can I delete it, can I move my data. | **Four user-facing commands.** `remember`, `recall`, `forget`, `export`. Everything else is implementation detail. |

## 3. Core Invariants (binding)

These are the non-negotiable product-floor principles. They are about trust, operability, and user control, not about freezing the current implementation forever.

1. **Plain SQL is truth.** Pydantic is a typed view. Any user can `sqlite3 ~/.reify/store.db` and read everything.
2. **Async-first writes.** Foreground writes stay fast and durable. Expensive intelligence belongs behind a queue or other non-blocking boundary.
3. **Forget by default.** Every claim has a TTL or an equally explicit retention policy. Reinforcement extends it. Retention is earned, not assumed.
4. **Graceful degradation everywhere.** No LLM key → still useful. No embedder → lexical or structural retrieval. No internet → fully local core still works.
5. **User-value metrics gate releases.** Benchmarks matter, but only alongside time-to-useful-recall, cost, inspectability, and user trust.
6. **Single library, thin adapters.** The product is the library and CLI. MCP, REST, and OpenAI-compatible surfaces are adapters, not separate products.
7. **Legibility is a feature, not an afterthought.** `export`, `whatdoyouknow`, editability, provenance, and auditability are part of the product floor.
8. **Ship the wedge demo first.** If we cannot show Reify doing something Mem0 / OpenMemory cannot, the project has failed regardless of benchmark numbers.

## 4. Default Strategies (strong defaults, not eternal laws)

These are the current best ways to honor the invariants. They should be hard to abandon casually, but they can evolve if the replacement is still more truthful, inspectable, and useful.

1. **Two memory types in v0.** `Episode` and `Claim` (with a `kind` discriminator). The five-network taxonomy remains a future shape, not a v0 requirement.
2. **Deterministic-first extraction.** Cheap, inspectable extractors should handle the easy cases. Learned or LLM-based extraction is for residual ambiguity or higher-order synthesis.
3. **Metadata-first recall.** Tag → time → entity → BM25 → embeddings is the default retrieval path. This is a bias toward cheap, legible recall, not a ban on later semantic layers.
4. **Embeddings are optional and late.** They are one retrieval channel, not the system's definition of memory.
5. **User-facing commands stay simple.** `remember`, `recall`, `forget`, and `export` remain the core visible contract even if the internals get smarter.

## 5. Strategic Extension Principles (for higher-order memory intelligence)

These principles exist to prevent the constitution from trapping Reify at the level of a deterministic database wrapper. ARW, synthesis, and richer memory intelligence are allowed, but they have to earn their place.

1. **LLM-optional means product floor, not product ceiling.** The base product must work without an LLM. Higher-order capabilities may depend on optional learned systems if they fail gracefully and do not collapse the core.
2. **Metadata-first is a default ordering, not an absolute ceiling.** Semantic retrieval, reranking, and synthesis are allowed when they sit on top of explicit memories instead of replacing them as the source of truth.
3. **ARW is a first-class intelligence layer.** Active Reconsolidation on Write is not a gimmick. It is a strategic differentiator, but it must remain asynchronous, inspectable, and auditable.
4. **Derived intelligence must stay legible.** If Reify materializes a new claim, recommendation, transition, or summary, the user must be able to inspect the evidence and understand why it exists.
5. **Learned systems should compress ambiguity, not hide truth.** Use models to interpret, cluster, rerank, or synthesize where deterministic systems cap out. Do not let them become the only place the product's memory exists.
6. **Benchmarks are diagnostics, not constitutions.** LongMemEval, LoCoMo, BEAM, and Reify-native evals should pressure the system toward usefulness, but they do not override the product floor.

## 6. The Wedge Demo (the v0.1 acceptance bar)

The user-visible thing that proves the architecture earns its complexity. v0.1 ships when **steps 1–3** below work end-to-end. **Step 4 is the v0.4 acceptance bar** and depends on the LLM-backed reconsolidator plus sleep-phase synthesis; it is included here so the long-arc story is in one place but is *not* required to clear v0.1.

The wedge demo is built on the MCP standard, so it is **designed to work with any MCP-spec-compliant client**. As of v0.1 it has been smoke-tested end-to-end against Windsurf with an OpenAI-compatible provider; Claude Desktop, Cursor, Zed, Cline, and other MCP hosts are expected to work but have not been individually verified yet. The CLI provides a host-independent path for the automated acceptance test (`scripts/demo.sh`).

> **Day 1** (any MCP host or CLI): "I just got promoted from Senior to Staff Engineer."
>
> **Day 30** (any MCP host or CLI): "What's my title?"
> → *"Staff Engineer."*
>
> **Day 30** (any MCP host or CLI): "Did anything change recently?"
> → *"Yes — you were promoted from Senior to Staff Engineer 30 days ago."*
>
> **Day 60** *(v0.4; requires LLM-backed reconsolidator + sleep phase)* (different session, any MCP host or CLI): "What do I usually emphasize in promotion writeups?"
> → Returns a materialized `Claim` derived across sessions: *"You frame promotions around impact-on-team."* (Not the raw conversation.)

**CLI acceptance test path:**

The wedge demo can be verified entirely via the CLI without any MCP host:

```bash
# Step 1: Ingest promotion
reify remember "I got promoted from Senior to Staff Engineer." --at 2025-01-01
reify reflect

# Step 2: Query title (bitemporal recall)
reify recall "What's my title?" --at 2025-01-30

# Step 3: Query changes (BTG)
reify whatchanged --since 2025-01-01

# Step 4: Cross-session inference (ARW)
reify remember "I got promoted from Staff to Principal." --at 2025-04-01
reify reflect
reify whatdoyouknow "promotion writeups" --at 2025-05-01
```

Each command uses `--at` to inject controllable timestamps, compressing the 60-day story into seconds. `reify reflect` is the explicit CLI hook that drains the durable ARW queue between scripted steps; long-running servers run the same queue in the background. The acceptance test scripts these calls and asserts the outputs.

What this proves about the architecture:

* **Step 1** exercises ingest + deterministic extraction.
* **Step 2** exercises bitemporal validity (the new fact has superseded the old one).
* **Step 3** exercises the **Bitemporal Transition Graph**: a `Transition` was emitted, evidence was attached, and "what changed?" is now a first-class query — something OpenMemory MCP and Mem0 cannot answer.
* **Step 4** exercises **Active Reconsolidation on Write**: a derived `Claim` was materialized across two sessions and surfaces directly at retrieval time — something no production system does today.

If step 3 does not work, BTG is unproven and v0.1 is not done. Step 4 is the corresponding bar for v0.4: it proves cross-session ARW + sleep-phase synthesis on top of the v0.1 foundation.

## 7. Translation into the codebase

The current scaffold is updated to honor these commandments:

* **Type system (`src/reify/core/types.py`)** stays as-is — it documents the future five-network world. The v0.1 implementation operates on the union but only emits two specializations (`Experience` for episodes; `Fact` / `Belief` / `Inference` as the three flavors of claim that the wedge demo requires).
* **`ARCHITECTURE.md`** is updated to call out async-by-default writes, queue-based ARW, graceful degradation, and the user-value metric set.
* **`ROADMAP.md`** is reordered: v0.1 is no longer "ingest + storage + MCP + benchmarks". v0.1 is "the wedge demo runs end-to-end on a real MCP host." Benchmarks move to v0.3.
* **`README.md`** leads with the wedge demo, not the leaderboard.

## 8. What this document is not

It is not a roadmap. It is not an exhaustive design. It is the smallest set of constraints we believe — after honest inversion and first-principles audit — separate a product that ships and matters from one that does not.

If you propose a change to Reify and it violates a **core invariant**, the change is wrong unless the invariant is also wrong. If it challenges a **default strategy**, argue why the new approach better serves the same invariant. If it expands the intelligence layer, prove that it still degrades gracefully and remains inspectable.

---

*See [GLOSSARY.md](./GLOSSARY.md) for abbreviations used in this document.*
