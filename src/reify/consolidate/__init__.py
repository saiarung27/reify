"""Consolidation subsystem.

Three modules, deferred to their respective milestones:

* :mod:`reify.consolidate.reconsolidate` — Active Reconsolidation on Write
  (ARW). The novel core. Runs synchronously after every commit-eligible
  memory and produces :class:`reify.core.types.Inference` records. v0.1.
* :mod:`reify.consolidate.transitions` — detection and emission of
  :class:`reify.core.types.Transition` records during write and during
  sleep-phase sweeps. v0.2.
* :mod:`reify.consolidate.sleep` — periodic sleep-phase consolidator:
  hot-buffer promotion, adversarial belief probing, cross-session synthesis
  pass, transition detection, forgetting. v0.4.
"""

from reify.consolidate.reconsolidate import ARWWorker, Relation

__all__ = ["ARWWorker", "Relation"]
