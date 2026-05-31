# Contributing to Reify

Thanks for your interest. Reify is a pre-alpha project. We welcome bug reports,
small fixes, benchmark contributions, and design feedback. Please read this
document and [`FIRST_PRINCIPLES.md`](./FIRST_PRINCIPLES.md) before opening a
substantial pull request.

## Getting set up

```bash
git clone https://github.com/saiarung27/reify.git
cd reify
make install     # creates .venv and installs package + dev extras
make test        # full pytest suite
make demo        # end-to-end wedge demo (no LLM key required)
```

Supported Python versions: **3.11, 3.12, 3.13**.

## Repository layout

- `src/reify/` — the library and CLI.
- `tests/` — pytest suite. Async tests use `asyncio_mode = "auto"`.
- `scripts/` — operator scripts (`demo.sh`, `fetch_longmemeval.py`, ...).
- `dashboard/` — optional read-only web UI.
- `sdk/python/`, `sdk/ts/` — thin client bindings.
- `examples/` — copy-paste integrations.
- `docs (root *.md)` — `ARCHITECTURE.md`, `ROADMAP.md`, `FIRST_PRINCIPLES.md`,
  `GLOSSARY.md`. Read these before proposing architectural change.

## Branching and commits

- Default branch is `main`. CI must be green on `main`.
- Open a feature branch (`feat/...`, `fix/...`, `docs/...`) and PR into `main`.
- Commit messages follow Conventional Commits (`feat:`, `fix:`, `chore:`,
  `test:`, `docs:`, `refactor:`, `perf:`).

## What we expect from a PR

1. **Honor the constitution.** If your change conflicts with a core invariant
   in `FIRST_PRINCIPLES.md`, the PR description must call that out explicitly.
2. **Tests.** New behavior gets tests. Bug fixes get a regression test that
   would have failed before the fix.
3. **Green CI.** `make test`, `make lint`, and `make demo` must pass locally
   before review.
4. **Small surface area.** Prefer minimal, focused diffs. Large refactors
   require a design note in the PR description.
5. **No new heavyweight dependencies** without justification. Reify is
   designed to run on a laptop with no API keys.

## Filing a bug

Please include:

- Reify version (`reify version`).
- Python version and OS.
- Minimal reproduction (CLI commands or a short script).
- Observed vs expected behavior.
- Whether the wedge demo (`make demo`) still passes.

## Filing a feature request

Frame the request against `FIRST_PRINCIPLES.md`:

- Which invariant is preserved?
- Which default strategy does it challenge, and why?
- What user-value metric does it move?

## Security issues

Do **not** open a public issue for a security vulnerability. See
[`SECURITY.md`](./SECURITY.md).

## Code style

- Python: ruff for formatting/lint, mypy in strict mode.
  - `ruff check src tests` and `mypy src/reify` must pass.
  - Line length 100. Async-first. Type hints on every public symbol.
- Markdown: 100-column soft wrap. Code blocks have a language tag.
- Tests: one fact per test. Test names describe the behavior, not the
  implementation.

## License

By contributing, you agree that your contributions are licensed under the
[MIT License](./LICENSE).
