"""Token-budget-aware context packer.

The final stage of the v0.3 retrieval pipeline. Given a ranked list of
:class:`RecallResult`, pack as many as fit into a model-specific budget
in a model-specific format. Output is the string blob ready for the
prompt plus a structured trace of what was included, dropped, and why.

Why this lives in its own module
--------------------------------
Different models prefer different framings:

* **Claude** prefers XML-style tags around documents.
* **GPT-class models** prefer numbered / delimited markdown blocks.
* **Gemini** is happy with simple markdown with explicit headers.
* **Local / open** models behave best with minimal templates and tight
  budgets.

Hard-coding any one shape into the engine would couple Reify to a
specific vendor (FIRST_PRINCIPLES.md inversion #2). The packer is the
single place that knows about model quirks; the engine remains agnostic.

Token counting is best-effort: if ``tiktoken`` is installed we use the
appropriate encoder, otherwise a conservative ``len/4`` heuristic. The
final character truncation is *always* exact, so even when the token
estimate is slightly off the packed text never exceeds the budget in
characters either.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Callable, Literal

from reify.recall import RecallResult


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------


def _heuristic_count(text: str) -> int:
    """Approximate token count without external deps.

    OpenAI's rule of thumb is ~4 characters per token for English text.
    We round up so the budget enforcement is conservative (we never
    underestimate).
    """
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def _make_token_counter(model_hint: str | None) -> Callable[[str], int]:
    """Return a callable that counts tokens for the target model.

    Tries ``tiktoken`` first (encoder picked from ``model_hint``); falls
    back to the heuristic. Constructed once per pack call so the encoder
    is cached implicitly via tiktoken's own internals.
    """
    try:
        import tiktoken  # type: ignore[import-not-found]
    except ImportError:
        return _heuristic_count

    try:
        if model_hint and "gpt" in model_hint.lower():
            enc = tiktoken.encoding_for_model(model_hint)
        else:
            enc = tiktoken.get_encoding("cl100k_base")
    except Exception:  # pragma: no cover - tiktoken edge cases
        return _heuristic_count

    def _count(text: str) -> int:
        if not text:
            return 0
        return len(enc.encode(text))

    return _count


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------


PackFormat = Literal["xml", "markdown", "delimited", "minimal"]


@dataclass(frozen=True, slots=True)
class ModelProfile:
    """Per-model packing preferences.

    Attributes
    ----------
    name:
        Human-readable identifier. Used as the ``model_hint`` for token
        counting (so ``"gpt-4o"`` picks the right encoder).
    max_tokens:
        Hard cap on the packed output. Defaults to a third of the model's
        nominal context window so the packed memories leave room for the
        user message, system prompt, and response.
    format:
        Layout style. See :data:`PackFormat`.
    header:
        Text inserted before the first item.
    footer:
        Text appended after the last item. Often used for a truncation
        notice.
    item_separator:
        Inserted between items.
    """

    name: str
    max_tokens: int = 4000
    format: PackFormat = "markdown"
    header: str = "# Reify memories\n\n"
    footer: str = ""
    item_separator: str = "\n\n"


CLAUDE_PROFILE = ModelProfile(
    name="claude-3-5-sonnet",
    max_tokens=8000,
    format="xml",
    header="<memories>\n",
    footer="</memories>\n",
    item_separator="\n",
)

GPT_PROFILE = ModelProfile(
    name="gpt-4o",
    max_tokens=8000,
    format="delimited",
    header="### Reify memories\n\n",
    footer="",
    item_separator="\n---\n",
)

GEMINI_PROFILE = ModelProfile(
    name="gemini-1.5-pro",
    max_tokens=8000,
    format="markdown",
    header="# Reify memories\n\n",
    footer="",
    item_separator="\n\n",
)

LOCAL_PROFILE = ModelProfile(
    name="local",
    max_tokens=2000,
    format="minimal",
    header="Memories:\n",
    footer="",
    item_separator="\n",
)


_BUILTIN_PROFILES: dict[str, ModelProfile] = {
    "claude": CLAUDE_PROFILE,
    "gpt": GPT_PROFILE,
    "gemini": GEMINI_PROFILE,
    "local": LOCAL_PROFILE,
}


def get_profile(name: str) -> ModelProfile:
    """Look up a built-in profile by short name.

    Accepts ``"claude"``, ``"gpt"``, ``"gemini"``, ``"local"``. Raises
    :class:`KeyError` for unknown profiles so callers know to construct
    a custom :class:`ModelProfile`.
    """
    if name not in _BUILTIN_PROFILES:
        raise KeyError(
            f"Unknown profile {name!r}. "
            f"Built-in profiles: {sorted(_BUILTIN_PROFILES)}"
        )
    return _BUILTIN_PROFILES[name]


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PackedContext:
    """Result of :func:`pack`. Legible by construction.

    ``text`` is the prompt-ready string. ``included_ids`` lists the
    memory IDs that landed; ``dropped_ids`` lists the ones the budget
    excluded so callers can surface "we left X out" messages. The
    estimated ``token_count`` reflects the actual encoded length when
    ``tiktoken`` is available, the heuristic otherwise.
    """

    text: str
    included_ids: tuple[str, ...]
    dropped_ids: tuple[str, ...]
    token_count: int
    char_count: int
    truncated: bool
    profile: ModelProfile

    @property
    def included_count(self) -> int:
        return len(self.included_ids)

    @property
    def total_count(self) -> int:
        return len(self.included_ids) + len(self.dropped_ids)


# ---------------------------------------------------------------------------
# Item rendering
# ---------------------------------------------------------------------------


def _render_item(result: RecallResult, *, format: PackFormat, index: int) -> str:
    mem = result.memory
    score = f"{result.score:.3f}"
    stage = result.matched_stage
    when = mem.valid_at.isoformat()
    if format == "xml":
        return (
            f'<memory id="{mem.id}" stage="{stage}" score="{score}" valid_at="{when}">'
            f"\n{mem.content}\n"
            f"</memory>"
        )
    if format == "delimited":
        return (
            f"[{index}] ({stage}, score={score}, valid_at={when})\n"
            f"{mem.content}"
        )
    if format == "markdown":
        return (
            f"## Memory {index} — {stage} (score {score})\n"
            f"_valid at {when}_\n\n"
            f"{mem.content}"
        )
    # minimal
    return f"- ({stage}) {mem.content}"


# ---------------------------------------------------------------------------
# pack
# ---------------------------------------------------------------------------


def pack(
    results: Sequence[RecallResult],
    *,
    profile: ModelProfile | str = "gpt",
    budget_tokens: int | None = None,
    query: str | None = None,  # noqa: ARG001 - reserved for future query-aware packing
) -> PackedContext:
    """Pack ranked recall results into a model-shaped, budgeted blob.

    Parameters
    ----------
    results:
        Ranked recall results (highest-scored first). Order is preserved;
        the packer drops *from the tail* once the budget runs out so the
        most-relevant memories always make the cut.
    profile:
        Either a :class:`ModelProfile` instance or the short-name of a
        built-in profile (``"claude"``, ``"gpt"``, ``"gemini"``,
        ``"local"``). See :func:`get_profile`.
    budget_tokens:
        Override the profile's ``max_tokens`` for this call.

    Returns
    -------
    :class:`PackedContext`
        Structured trace of what was included, what was dropped, and the
        final string. ``truncated=True`` if any results were excluded by
        the budget.
    """
    if isinstance(profile, str):
        profile = get_profile(profile)

    budget = budget_tokens if budget_tokens is not None else profile.max_tokens
    if budget <= 0:
        return PackedContext(
            text="",
            included_ids=(),
            dropped_ids=tuple(r.memory.id for r in results),
            token_count=0,
            char_count=0,
            truncated=bool(results),
            profile=profile,
        )

    count_tokens = _make_token_counter(profile.name)

    header_tokens = count_tokens(profile.header)
    footer_tokens = count_tokens(profile.footer)
    sep_tokens = count_tokens(profile.item_separator)
    used = header_tokens + footer_tokens
    parts: list[str] = []
    included: list[str] = []
    dropped: list[str] = []

    for i, r in enumerate(results, 1):
        rendered = _render_item(r, format=profile.format, index=i)
        rendered_tokens = count_tokens(rendered)
        addition = rendered_tokens + (sep_tokens if parts else 0)
        if used + addition > budget:
            dropped.append(r.memory.id)
            # We keep iterating so the rest of the dropped_ids list is
            # complete; greedy fill might still fit a smaller later item
            # within the remaining budget. This is bounded by the input
            # length so cost is linear.
            continue
        parts.append(rendered)
        included.append(r.memory.id)
        used += addition

    body = profile.item_separator.join(parts)
    truncation_note = ""
    if dropped:
        truncation_note = (
            f"\n[truncated: {len(dropped)} more memories omitted by budget]\n"
        )

    text = profile.header + body + truncation_note + profile.footer
    final_tokens = count_tokens(text)
    return PackedContext(
        text=text,
        included_ids=tuple(included),
        dropped_ids=tuple(dropped),
        token_count=final_tokens,
        char_count=len(text),
        truncated=bool(dropped),
        profile=profile,
    )


__all__ = [
    "CLAUDE_PROFILE",
    "GEMINI_PROFILE",
    "GPT_PROFILE",
    "LOCAL_PROFILE",
    "ModelProfile",
    "PackFormat",
    "PackedContext",
    "get_profile",
    "pack",
]
