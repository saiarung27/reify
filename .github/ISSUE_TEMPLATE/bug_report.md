---
name: Bug report
about: Something Reify does wrong, or documentation that does not match behavior.
title: "bug: <short description>"
labels: ["bug", "needs-triage"]
---

<!--
Thanks for filing a bug. Reify is solo-maintained, so structured reports get
triaged faster than free-form ones. Run `make demo` and `make test` first and
include their result below.
-->

## What happened

A clear description of the unexpected behavior.

## What you expected

A clear description of what should have happened instead. If the behavior
contradicts a doc, please link the section.

## Reproduction

Minimal CLI commands or short Python script. Please trim to the smallest
example that still triggers the bug.

```bash
# example
reify init
reify remember "..."
reify recall "..."
```

## Environment

- Reify version: <output of `reify version`>
- Python version: <output of `python --version`>
- OS: <e.g. macOS 14.5, Ubuntu 24.04>
- LLM provider configured (if any): <openai / ollama / none>

## Status of the smoke tests

- [ ] `make demo` passes on a clean clone
- [ ] `make test` passes on a clean clone
- [ ] `make user-value` passes on a clean clone

If any of these fail, please paste the last ~30 lines of output below.

## Additional context

Logs, screenshots, anything else that helps. Please redact API keys.
