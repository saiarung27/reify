---
name: Feature request
about: Propose a change in scope or behavior. Larger ideas should open an issue before a PR.
title: "feat: <short description>"
labels: ["enhancement", "needs-triage"]
---

<!--
Reify is opinionated. Before filing, please skim FIRST_PRINCIPLES.md and
ROADMAP.md so the request is framed against the existing constraints.
-->

## The problem

What user-visible problem does this solve? Concrete scenario, not abstract.

## Proposed change

What would Reify do differently after this lands? CLI surface, MCP tool,
config flag, library API — be specific where possible.

## How does this fit the first principles?

Pick the invariants in [`FIRST_PRINCIPLES.md`](../FIRST_PRINCIPLES.md) that
this change preserves, and any it bends. Examples:

- Does it keep the no-LLM-required write path?
- Does it keep recall metadata-first?
- Does it stay legible from `sqlite3`?
- Does it grow the dependency footprint? If so, what is the justification?

## User-value impact

Which of the four user-value metrics (time-to-useful-recall, recall-without-LLM
rate, 30-day recall accuracy, cost per useful recall) does this move, and in
what direction?

## Alternatives considered

Other shapes you thought about and why they are worse.

## Additional context

Links to prior art, papers, related issues.
