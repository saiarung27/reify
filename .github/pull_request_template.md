<!--
Thanks for the PR. Reify is solo-maintained, so a clear PR description
shortens the review cycle a lot.
-->

## Summary

One paragraph: what does this change, and why?

## Type of change

- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] Feature (non-breaking change that adds capability)
- [ ] Docs only
- [ ] Refactor / internal cleanup (no behavior change)
- [ ] Breaking change (please link the issue that pre-discussed it)

## First-principles check

If this change interacts with any of the invariants in
[`FIRST_PRINCIPLES.md`](../FIRST_PRINCIPLES.md), call them out here.
Otherwise write `n/a`.

## Verification

- [ ] `make test` passes locally
- [ ] `make demo` passes locally
- [ ] `make user-value` passes locally (if you touched recall, ARW, or storage)
- [ ] New behavior has tests; bug fixes include a regression test that would
      have failed before the fix
- [ ] Docs updated if user-visible behavior changed

## Notes for the reviewer

Anything that is not obvious from the diff: design trade-offs, follow-ups
you intentionally deferred, performance implications, etc.
