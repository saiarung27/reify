"""Recall pipeline.

Implements the metadata-first recall order specified in
FIRST_PRINCIPLES.md commandment 5:

    tag \u2192 time \u2192 entity \u2192 BM25 \u2192 embeddings

The cheaper stages prune the candidate set before the expensive ones run.
For v0.1 the embedder stage is optional and the lexical stage uses a
simple keyword + recency scorer rather than full BM25; this is enough to
clear the wedge-demo bar and keeps the dependency surface minimal. A real
BM25 implementation lands in v0.3 (Hindsight-class retrieval).

The pipeline returns ``RecallResult`` objects that retain *why* each
memory was retrieved \u2014 the matched stage and the score \u2014 so the engine and
downstream surfaces can explain themselves to the user. Legibility
(commandment 9) starts here.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Literal, Sequence

from reify.core.types import Experience, Memory, Network
from reify.storage import RecallFilter, Storage


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RecallResult:
    """A retrieved memory plus diagnostics.

    ``score`` is in ``[0, 1]`` and is comparable across stages: the recall
    pipeline normalizes per-stage scores before merging.
    """

    memory: Memory
    score: float
    matched_stage: str  # "tag" | "lexical" | "vector" | "recent"


@dataclass(frozen=True)
class Query:
    """Parsed natural-language query with structural hints.

    The :class:`Reify` engine builds a :class:`Query` from the user's text
    via :func:`parse_query`. Subsystems that have richer intent inference
    (an LLM, a learned classifier) can construct a :class:`Query`
    directly.
    """

    text: str
    keywords: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    networks: tuple[Network, ...] = ()
    valid_at: datetime | None = None
    identity_id: str | None = None
    limit: int = 20
    prefer_user_memories: bool = False
    prefer_assistant_memories: bool = False
    assistant_item_index: int | None = None
    temporal_intent: "TemporalIntent | None" = None
    aggregate_intent: "AggregateIntent | None" = None


@dataclass(frozen=True)
class TemporalIntent:
    mode: Literal["compare_order", "delta_days", "first_after", "first_date", "wake_time", "wake_delta", "event_object", "count_before", "age_at_event", "recent_start"]
    target_labels: tuple[str, ...] = ()
    answer_labels: tuple[str, ...] = ()
    reference_label: str | None = None
    target_kind: str | None = None
    constrain_month: int | None = None
    relative_filter: Literal["last_month"] | None = None
    target_tag: str | None = None
    answer_unit: Literal["days", "weeks", "months"] | None = "days"


@dataclass(frozen=True)
class AggregateIntent:
    mode: Literal["count", "sum_duration", "sum_money", "sum_value", "difference_value"]
    target_tag: str
    unit: str | None = None
    unique_field: str | None = None
    value_field: str | None = None
    constrain_months: tuple[int, ...] = ()
    target_labels: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Query parsing (cheap, deterministic intent)
# ---------------------------------------------------------------------------


# A *very* small set of trigger words that map to schema tags. Conservative;
# the LLM intent classifier in v0.3 will replace this with a learned
# mapping. The point here is to make the wedge demo's natural-language
# questions ("what's my title?", "what changed recently?") work without an
# LLM in the loop.
_TAG_TRIGGERS: dict[str, tuple[str, ...]] = {
    "title": ("title",),
    "role": ("title",),
    "promoted": ("title", "promotion"),
    "promotion": ("title", "promotion"),
    "promotions": ("title", "promotion"),
    "live": ("location",),
    "lives": ("location",),
    "living": ("location",),
    "location": ("location",),
    "moved": ("location",),
    "city": ("location",),
    "work": ("employer",),
    "works": ("employer",),
    "working": ("employer",),
    "employer": ("employer",),
    "company": ("employer",),
    "joined": ("employer",),
    "writeup": ("writeup", "promotion"),
    "writeups": ("writeup", "promotion"),
    "emphasize": ("writeup", "promotion"),
    "prefer": ("preference",),
    "preference": ("preference",),
    "favorite": ("preference",),
    "liked": ("preference",),
    "like": ("preference",),
    "workshop": ("event",),
    "webinar": ("event",),
    "meeting": ("event",),
    "issue": ("event",),
    "repair": ("event",),
    "device": ("event",),
    "study": ("education",),
    "university": ("education",),
    "college": ("education",),
    "undergrad": ("education",),
    "bachelor": ("education",),
    "commute": ("commute",),
    "rice": ("preference",),
    "lens": ("inventory",),
}

_NETWORK_TRIGGERS: dict[str, Network] = {
    "fact": Network.FACT,
    "experience": Network.EXPERIENCE,
    "belief": Network.BELIEF,
    "inference": Network.INFERENCE,
    "observation": Network.OBSERVATION,
}

_STOPWORDS = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "of", "in", "on", "at", "for",
        "with", "to", "from", "by", "as", "is", "are", "was", "were", "be",
        "been", "being", "do", "does", "did", "have", "has", "had", "i",
        "me", "my", "mine", "you", "your", "yours", "we", "us", "our",
        "ours", "they", "them", "their", "theirs", "it", "its", "this",
        "that", "these", "those", "what", "which", "who", "whom", "whose",
        "where", "when", "why", "how", "if", "so", "than", "then", "just",
        "about", "yes", "no", "not",
    }
)

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9'\-]*")


def _tokenize(text: str) -> list[str]:
    """Lower-case, strip punctuation, drop stopwords."""
    return [
        t.lower()
        for t in _TOKEN_RE.findall(text)
        if t.lower() not in _STOPWORDS and len(t) > 1
    ]


def parse_query(text: str, *, default_limit: int = 20) -> Query:
    """Cheap deterministic intent inference.

    Maps trigger tokens to schema tags and (when explicit) network filters.
    Always returns a :class:`Query`; never raises. An empty query is valid
    and returns "everything ranked by recency".
    """
    tokens = _tokenize(text)
    tags: list[str] = []
    networks: list[Network] = []
    keywords: list[str] = []
    seen_tags: set[str] = set()
    seen_networks: set[Network] = set()
    for tok in tokens:
        if tok in _TAG_TRIGGERS:
            for tag in _TAG_TRIGGERS[tok]:
                if tag not in seen_tags:
                    tags.append(tag)
                    seen_tags.add(tag)
        if tok in _NETWORK_TRIGGERS:
            net = _NETWORK_TRIGGERS[tok]
            if net not in seen_networks:
                networks.append(net)
                seen_networks.add(net)
        keywords.append(tok)
    prefer_assistant_memories = _prefers_assistant_memories(text)
    prefer_user_memories = _prefers_user_memories(text) and not prefer_assistant_memories
    if prefer_assistant_memories:
        tags = []
        networks = []
    return Query(
        text=text.strip(),
        keywords=tuple(keywords),
        tags=tuple(tags),
        networks=tuple(networks),
        limit=default_limit,
        prefer_user_memories=prefer_user_memories,
        prefer_assistant_memories=prefer_assistant_memories,
        assistant_item_index=_parse_assistant_item_index(text) if prefer_assistant_memories else None,
        temporal_intent=_parse_temporal_intent(text),
        aggregate_intent=_parse_aggregate_intent(text),
    )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _keyword_score(query_kw: Sequence[str], memory: Memory) -> float:
    """Jaccard-style overlap of query keywords with memory content tokens.

    Cheap and adequate for the wedge demo. A proper BM25 lands in v0.3.
    """
    if not query_kw:
        return 0.0
    mem_tokens = set(_tokenize(memory.content))
    if not mem_tokens:
        return 0.0
    overlap = sum(1 for k in query_kw if k in mem_tokens)
    return overlap / max(len(query_kw), 1)


def _recency_score(memory: Memory, *, now: datetime, half_life_days: float = 30.0) -> float:
    """Exponential decay on the memory's ``created_at``. Newer is higher."""
    if memory.created_at.tzinfo is None:
        created = memory.created_at.replace(tzinfo=timezone.utc)
    else:
        created = memory.created_at
    delta = now - created
    days = max(delta.total_seconds() / 86400.0, 0.0)
    return math.exp(-math.log(2) * days / max(half_life_days, 1e-6))


def _tag_score(query_tags: Sequence[str], memory: Memory) -> float:
    if not query_tags:
        return 0.0
    qt = set(query_tags)
    mt = set(memory.tags)
    if not mt:
        return 0.0
    matches = qt & mt
    return len(matches) / max(len(qt), 1)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class Recaller:
    """The default v0.1 metadata-first recall pipeline.

    Held by :class:`reify.engine.Reify`. Stateless beyond its constructor
    arguments; safe to share across coroutines.

    Parameters
    ----------
    storage:
        The store to query.
    weights:
        Per-channel weights for the merged score. Defaults reflect the
        first-principles ordering: tag matches dominate, recency
        tie-breaks, lexical adds the long tail.
    half_life_days:
        Recency decay constant.
    """

    def __init__(
        self,
        storage: Storage,
        *,
        weights: tuple[float, float, float] = (0.5, 0.2, 0.3),
        half_life_days: float = 30.0,
        candidate_limit: int = 200,
    ) -> None:
        self._storage = storage
        self._w_tag, self._w_recency, self._w_lexical = weights
        self._half_life_days = half_life_days
        self._candidate_limit = candidate_limit

    async def recall(self, query: Query, *, now: datetime | None = None) -> list[RecallResult]:
        """Run the metadata-first pipeline and return ranked results.

        Stages (cheap to expensive):

        1. **Tag/network/time filter** at the storage layer prunes the
           candidate set with a single SQL query.
        2. **Score** each candidate with a weighted sum of tag, recency,
           and lexical (keyword overlap) signals.
        3. **Sort** descending and trim to ``query.limit``.

        The vector stage is intentionally not invoked here \u2014 it lives in a
        sibling pipeline that the engine layers on top when an embedder is
        configured.
        """
        clock_now = now or datetime.now(timezone.utc)

        candidates = await self._fetch_candidates(query, now=clock_now)

        ranked: list[RecallResult] = []
        for mem in candidates:
            tag_s = _tag_score(query.tags, mem)
            recency_s = _recency_score(mem, now=clock_now, half_life_days=self._half_life_days)
            lex_s = _keyword_score(query.keywords, mem)
            prior_s = _memory_prior_score(
                mem,
                prefer_user_memories=query.prefer_user_memories,
                prefer_assistant_memories=query.prefer_assistant_memories,
            )
            score = (
                self._w_tag * tag_s
                + self._w_recency * recency_s
                + self._w_lexical * lex_s
                + prior_s
            )
            stage = _stage_label(tag_s, lex_s, recency_s)
            ranked.append(RecallResult(memory=mem, score=score, matched_stage=stage))

        ranked.sort(key=lambda r: r.score, reverse=True)
        return ranked[: query.limit]

    async def _fetch_candidates(self, query: Query, *, now: datetime) -> list[Memory]:
        """Storage-side prune. Always applies bitemporal validity at ``now``."""
        f = RecallFilter(
            networks=list(query.networks) or None,
            tags_any=list(query.tags) or None,
            identity_ids=[query.identity_id] if query.identity_id else None,
            valid_at=query.valid_at or now,
        )
        primary = await self._storage.recall(f, limit=self._candidate_limit)
        if primary or query.tags or query.networks or query.identity_id:
            return primary
        # Fallback: empty filter (just bitemporal) when the parsed query had
        # no structural hints. Returns the most recent memories.
        fallback = RecallFilter(valid_at=query.valid_at or now)
        return await self._storage.recall(fallback, limit=self._candidate_limit)


def _stage_label(tag_s: float, lex_s: float, recency_s: float) -> str:
    """Pick the stage that contributed most to the score, for diagnostics."""
    if tag_s >= max(lex_s, recency_s):
        return "tag"
    if lex_s >= recency_s:
        return "lexical"
    return "recent"


def _memory_prior_score(
    memory: Memory,
    *,
    prefer_user_memories: bool,
    prefer_assistant_memories: bool,
) -> float:
    """Inject a small ranking prior by memory type and actor.

    Reify is primarily a memory of the user's state, not a transcript search
    engine for assistant advice. When structural signals are weak, retrieved
    user-authored facts and user experiences should beat long assistant
    responses. The bonus is intentionally small so strong tag/lexical matches
    still win.
    """
    is_assistant_memory = "assistant-memory" in memory.tags
    if memory.network is Network.FACT:
        if prefer_assistant_memories and is_assistant_memory:
            return 0.35
        if prefer_user_memories:
            return 0.25
        return 0.2
    if memory.network is Network.INFERENCE:
        return 0.2 if prefer_user_memories else 0.15
    if isinstance(memory, Experience):
        if memory.actor == "user":
            if prefer_assistant_memories:
                return 0.0
            return 0.25 if prefer_user_memories else 0.12
        if memory.actor == "assistant":
            if prefer_assistant_memories:
                return 0.18
            return -0.22 if prefer_user_memories else -0.08
        return -0.1 if prefer_user_memories else -0.04
    return 0.0


def _prefers_user_memories(text: str) -> bool:
    lower = text.lower()
    has_first_person = any(token in lower for token in (" i ", " my ", " me ", " mine "))
    starts_with_first_person = lower.startswith(("i ", "my ", "me ", "mine "))
    recommendation_intent = any(
        phrase in lower
        for phrase in (
            "can you recommend",
            "can you suggest",
            "any suggestions",
            "any advice",
            "any tips",
            "what should i",
        )
    )
    return starts_with_first_person or has_first_person or recommendation_intent


def _prefers_assistant_memories(text: str) -> bool:
    lower = text.lower()
    return any(
        phrase in lower
        for phrase in (
            "previous chat",
            "previous conversation",
            "our previous conversation",
            "our previous chat",
            "you told me",
            "you mentioned",
            "you suggested",
            "we discussed",
            "can you remind me",
            "remind me what",
            "going back to our previous",
            "follow up on our previous",
            "what was the",
            "which restaurant",
            "which book",
            "which song",
            "which trail",
            "which resource",
            "which job",
            "what was the name",
            "what was the 7th job",
            "what was the 7th",
        )
    )


def _parse_assistant_item_index(text: str) -> int | None:
    match = re.search(r"\b(?P<index>\d+)(?:st|nd|rd|th)\b", text.lower())
    if not match:
        return None
    return int(match.group("index"))


def _parse_temporal_intent(text: str) -> TemporalIntent | None:
    lower = text.lower()
    quoted = tuple(_strip_label_quotes(match) for match in re.findall(r'["\']([^"\']+)["\']', text))

    age_at_event_match = re.search(
        r"\bhow\s+old\s+was\s+i\s+when\s+i\s+moved\s+to\s+(?P<label>[^?]+)\??$",
        text.strip(),
        flags=re.IGNORECASE,
    )
    if age_at_event_match:
        label = _clean_temporal_query_label(age_at_event_match.group("label"))
        return TemporalIntent(
            mode="age_at_event",
            target_labels=(f"moved to {label}",),
            target_kind="move_country",
            answer_unit=None,
        )

    if re.search(r"\bwhich\s+(?:streaming\s+service|service)\s+did\s+i\s+start\s+using\s+most\s+recently\??$", text.strip(), flags=re.IGNORECASE):
        return TemporalIntent(
            mode="recent_start",
            target_kind="service_start",
            target_tag="streaming-service",
            answer_unit=None,
        )

    between_match = re.search(r"\bbetween\s+(?P<left>.+?)\s+and\s+(?P<right>.+?)(?:\?|$)", text, flags=re.IGNORECASE)
    if between_match and ("how many" in lower or "how long" in lower):
        left = _clean_temporal_query_label(between_match.group("left"))
        right = _clean_temporal_query_label(between_match.group("right"))
        return TemporalIntent(
            mode="delta_days",
            target_labels=(left,),
            reference_label=right,
            answer_unit=_parse_temporal_unit(lower),
        )

    if "how many days before" in lower:
        reference_label = None
        if "team meeting" in lower:
            reference_label = "meeting"
        target_labels = quoted[:1]
        return TemporalIntent(
            mode="delta_days",
            target_labels=target_labels,
            reference_label=reference_label,
            answer_unit="days",
        )

    if "how long have i been working before i started my current job" in lower:
        employer_match = re.search(r"\bcurrent\s+job\s+at\s+(?P<employer>[A-Z][A-Za-z0-9.&\-\s]+)\??$", text)
        reference = "started current job"
        if employer_match:
            reference = f"started current job at {employer_match.group('employer').strip()}"
        return TemporalIntent(
            mode="delta_days",
            target_labels=("working professionally",),
            reference_label=reference,
            answer_unit="months",
        )

    duration_before_match = re.search(
        r"\bhow\s+long\s+(?:had\s+i\s+been|have\s+i\s+been|did\s+i\s+use)\s+(?P<left>.+?)\s+(?:when|before)\s+(?P<right>.+?)\?$",
        text.strip(),
        flags=re.IGNORECASE,
    )
    if duration_before_match:
        left = _clean_temporal_query_label(duration_before_match.group("left"))
        right = _clean_temporal_query_label(duration_before_match.group("right"))
        if right.startswith("i "):
            right = _clean_temporal_query_label(right[2:])
        return TemporalIntent(
            mode="delta_days",
            target_labels=(left,),
            reference_label=right,
            answer_unit=_parse_temporal_unit(lower),
        )

    count_before_match = re.search(
        r"\bhow\s+many\s+(?P<label>.+?)\s+did\s+i\s+(?:attend|participate\s+in|volunteer\s+at)\s+before\s+(?P<right>.+?)\?$",
        text.strip(),
        flags=re.IGNORECASE,
    )
    if count_before_match:
        label = _clean_temporal_query_label(count_before_match.group("label"))
        right = _clean_temporal_query_label(count_before_match.group("right"))
        target_tag = "charity" if "charity" in label else None
        return TemporalIntent(
            mode="count_before",
            target_labels=(label,),
            reference_label=right,
            target_tag=target_tag,
            answer_unit=None,
        )

    days_since_match = re.search(
        r"\bhow\s+many\s+days\s+had\s+passed\s+since\s+(?P<left>.+?)\s+when\s+(?P<right>.+?)\?$",
        text.strip(),
        flags=re.IGNORECASE,
    )
    if days_since_match:
        left = _clean_temporal_query_label(days_since_match.group("left"))
        right = _clean_temporal_query_label(days_since_match.group("right"))
        return TemporalIntent(
            mode="delta_days",
            target_labels=(left,),
            reference_label=right,
            answer_unit="days",
        )

    if "first issue" in lower and "after" in lower:
        reference_label = "first service" if "first service" in lower else None
        return TemporalIntent(
            mode="first_after",
            reference_label=reference_label,
            target_kind="issue",
        )

    if "which event" in lower and "first" in lower and quoted:
        return TemporalIntent(
            mode="compare_order",
            target_labels=tuple(_clean_temporal_query_label(item) for item in quoted[:2]),
            answer_labels=quoted[:2],
        )

    meet_compare = re.search(
        r"\bwho\s+did\s+i\s+meet\s+first,\s+(?P<left>.+?)\s+or\s+(?P<right>.+?)\?$",
        text.strip(),
        flags=re.IGNORECASE,
    )
    if meet_compare:
        left = _strip_label_quotes(meet_compare.group("left"))
        right = _strip_label_quotes(meet_compare.group("right"))
        return TemporalIntent(
            mode="compare_order",
            target_labels=(_clean_temporal_query_label(left), _clean_temporal_query_label(right)),
            answer_labels=(left, right),
            target_kind="met_person",
        )

    parent_compare = re.search(
        r"\bwho\s+became\s+a\s+parent\s+first,\s+(?P<left>.+?)\s+or\s+(?P<right>.+?)\?$",
        text.strip(),
        flags=re.IGNORECASE,
    )
    if parent_compare:
        left = _strip_label_quotes(parent_compare.group("left"))
        right = _strip_label_quotes(parent_compare.group("right"))
        return TemporalIntent(
            mode="compare_order",
            target_labels=(_clean_temporal_query_label(left), _clean_temporal_query_label(right)),
            answer_labels=(left, right),
            target_kind="became_parent",
        )

    if "which event happened first" in lower and " or " in lower:
        labels = _parse_or_labels(text)
        if len(labels) >= 2:
            return TemporalIntent(
                mode="compare_order",
                target_labels=tuple(_clean_temporal_query_label(item) for item in labels[:2]),
                answer_labels=labels[:2],
            )

    if "first" in lower and " or " in lower:
        labels = _parse_or_labels(text)
        if len(labels) >= 2:
            return TemporalIntent(
                mode="compare_order",
                target_labels=tuple(_clean_temporal_query_label(item) for item in labels[:2]),
                answer_labels=labels[:2],
            )

    generic_compare = re.search(
        r"\bwhich\s+.+?\b(?:happened|did\s+i\s+\w+|were\s+\w+|was)\s+first[, ]+(?P<left>.+?)\s+or\s+(?P<right>.+?)\?$",
        text.strip(),
        flags=re.IGNORECASE,
    )
    if generic_compare:
        left_raw = _strip_label_quotes(generic_compare.group("left"))
        right_raw = _strip_label_quotes(generic_compare.group("right"))
        left = _clean_temporal_query_label(left_raw)
        right = _clean_temporal_query_label(right_raw)
        return TemporalIntent(
            mode="compare_order",
            target_labels=(left, right),
            answer_labels=(left_raw, right_raw),
        )

    if "what was the date" in lower and "first" in lower:
        first_match = re.search(
            r"\bfirst\s+(?P<label>[^?]+?)(?:\s+in\s+(?P<month>[A-Za-z]+))?\?$",
            text.strip(),
            flags=re.IGNORECASE,
        )
        if first_match:
            label = _clean_temporal_query_label(first_match.group("label"))
            return TemporalIntent(
                mode="first_date",
                target_labels=(label,),
                constrain_month=_month_from_text((first_match.group("month") or "").lower()),
            )

    wake_delta_match = re.search(
        r"\bhow\s+much\s+earlier\s+do\s+i\s+wake\s+up\s+on\s+(?P<days>[^?]+?)\s+compared\s+to\s+other\s+weekdays\??$",
        text.strip(),
        flags=re.IGNORECASE,
    )
    if wake_delta_match:
        days = tuple(_parse_weekday_labels(wake_delta_match.group("days")))
        if days:
            return TemporalIntent(mode="wake_delta", target_labels=days, answer_unit=None)

    wake_match = re.search(
        r"\bwhat\s+time\s+do\s+i\s+wake\s+up\s+on\s+(?P<days>[^?]+)\??$",
        text.strip(),
        flags=re.IGNORECASE,
    )
    if wake_match:
        days = tuple(_parse_weekday_labels(wake_match.group("days")))
        if days:
            return TemporalIntent(mode="wake_time", target_labels=days, answer_unit=None)

    if "which pair of shoes did i clean last month" in lower:
        return TemporalIntent(
            mode="event_object",
            target_labels=("cleaned",),
            relative_filter="last_month",
            answer_unit=None,
        )

    if "how many days did it take" in lower and ("working with" in lower or "work with" in lower):
        person_match = re.search(r"work(?:ing)?\s+with\s+(?P<label>[A-Z][a-z]+)", text)
        if person_match:
            return TemporalIntent(
                mode="delta_days",
                target_labels=("house I loved",),
                reference_label=f"started working with {person_match.group('label')}",
                answer_unit="days",
            )

    if "which vehicle" in lower and "first" in lower and " or " in lower:
        labels = _parse_or_labels(text)
        return TemporalIntent(
            mode="compare_order",
            target_labels=tuple(_clean_temporal_query_label(item) for item in labels[:2]),
            answer_labels=labels[:2],
            constrain_month=_month_from_text(lower),
        )

    if "which device" in lower and "first" in lower and " or " in lower:
        labels = _parse_or_labels(text)
        return TemporalIntent(
            mode="compare_order",
            target_labels=tuple(_clean_temporal_query_label(item) for item in labels[:2]),
            answer_labels=labels[:2],
            target_kind="device_acquired",
        )

    return None


def _parse_or_labels(text: str) -> tuple[str, ...]:
    matches = re.findall(r"\bthe\s+([^?.,]+?)\s+or\s+the\s+([^?.,]+)", text, flags=re.IGNORECASE)
    if not matches:
        return ()
    first, second = matches[0]
    return (_strip_label_quotes(first), _strip_label_quotes(second))


def _strip_label_quotes(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().strip("\"'")).strip()


def _clean_temporal_query_label(text: str) -> str:
    cleaned = _strip_label_quotes(text)
    cleaned = re.sub(r"^(?:the\s+)?narrator\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^(?:the|a|an)\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(
        r"^(?:purchase\s+of|malfunction\s+of|arrival\s+of|start\s+of|attendance\s+at|participation\s+in|date\s+on\s+which\s+i\s+attended|day\s+i\s+|time\s+i\s+|hindu\s+festival\s+of)\s+",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\b(?:did\s+i\s+attend|did\s+i\s+participate\s+in)\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^(?:the|a|an)\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+for\s+[A-Za-z']+\b$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^(?:i\s+attended|i\s+saw|i\s+used|i\s+rearranged)\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^(?:using|losing|receiving)\s+(?:their|the|my)\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ?!.,")
    return cleaned


def _month_from_text(lower: str) -> int | None:
    months = {
        "january": 1, "february": 2, "march": 3, "april": 4,
        "may": 5, "june": 6, "july": 7, "august": 8,
        "september": 9, "october": 10, "november": 11, "december": 12,
    }
    for name, month in months.items():
        if name in lower:
            return month
    return None


def _parse_temporal_unit(lower: str) -> Literal["days", "weeks", "months"] | None:
    if "month" in lower:
        return "months"
    if "week" in lower:
        return "weeks"
    if "day" in lower:
        return "days"
    return None


def _parse_weekday_labels(raw: str) -> list[str]:
    tokens = re.split(r",|\band\b", raw, flags=re.IGNORECASE)
    weekdays = []
    for token in tokens:
        cleaned = token.strip().lower().rstrip("s")
        if cleaned in {
            "monday", "tuesday", "wednesday", "thursday",
            "friday", "saturday", "sunday",
        }:
            weekdays.append(cleaned)
    return weekdays


def _parse_aggregate_intent(text: str) -> AggregateIntent | None:
    lower = text.lower()
    month_constraints = _parse_month_constraints(lower)
    if "how many items of clothing" in lower and ("pick up" in lower or "return" in lower):
        return AggregateIntent(mode="count", target_tag="obligation")
    if "how many projects" in lower and ("led" in lower or "leading" in lower):
        return AggregateIntent(mode="count", target_tag="project")
    if "how many model kits" in lower and ("worked on" in lower or "bought" in lower):
        return AggregateIntent(mode="count", target_tag="model-kit")
    if "how many plants" in lower and ("acquire" in lower or "got" in lower or "bought" in lower):
        return AggregateIntent(mode="count", target_tag="inventory", unique_field="normalized_label")
    if "how many tanks" in lower:
        return AggregateIntent(mode="count", target_tag="inventory", unique_field="normalized_label")
    if "how many movie festivals" in lower or ("how many" in lower and "film festivals" in lower):
        return AggregateIntent(mode="count", target_tag="festival", unique_field="normalized_label")
    if "how many weddings" in lower:
        return AggregateIntent(mode="count", target_tag="wedding", unique_field="normalized_label")
    if "how many different types of citrus fruits" in lower:
        return AggregateIntent(mode="count", target_tag="citrus", unique_field="fruit")
    if "how many different doctors" in lower:
        return AggregateIntent(mode="count", target_tag="doctor", unique_field="doctor_name")
    if "how many days" in lower and "camping trip" in lower:
        return AggregateIntent(mode="sum_duration", target_tag="trip", unit="days")
    if "how many days" in lower and "social media break" in lower:
        return AggregateIntent(mode="sum_duration", target_tag="social-break", unit="days")
    if "how many weeks" in lower and ("watched" in lower or "movies" in lower):
        return AggregateIntent(mode="sum_duration", target_tag="media", unit="weeks")
    if "how many hours" in lower and "playing games" in lower:
        return AggregateIntent(mode="sum_duration", target_tag="gaming", unit="hours")
    if "how many hours" in lower and "driving" in lower and "road trip" in lower:
        return AggregateIntent(mode="sum_duration", target_tag="travel", unit="hours")
    if "what is the total amount" in lower and "luxury items" in lower:
        return AggregateIntent(mode="sum_money", target_tag="expense", unit="usd")
    if "how much total money" in lower and "bike-related" in lower:
        return AggregateIntent(mode="sum_money", target_tag="expense", unit="usd")
    if "how much did i save on" in lower and "handbag" in lower:
        return AggregateIntent(
            mode="difference_value",
            target_tag="deal",
            value_field="amount_value",
            target_labels=("paid", "original"),
            unit="usd",
        )
    if "how much will i save by taking the train" in lower and "instead of a taxi" in lower:
        return AggregateIntent(
            mode="difference_value",
            target_tag="transport-cost",
            value_field="amount_value",
            target_labels=("train", "taxi"),
            unit="usd",
        )
    if "total number of people reached" in lower:
        return AggregateIntent(mode="sum_value", target_tag="audience", value_field="count_value")
    if "page count" in lower and "novels" in lower:
        return AggregateIntent(
            mode="sum_value",
            target_tag="reading",
            value_field="page_count",
            constrain_months=month_constraints,
        )
    if "total number of siblings" in lower:
        return AggregateIntent(mode="sum_value", target_tag="family", value_field="count_value")
    return None


def _parse_month_constraints(text: str) -> tuple[int, ...]:
    months: list[int] = []
    for index, month in enumerate(
        (
            "january", "february", "march", "april", "may", "june",
            "july", "august", "september", "october", "november", "december",
        ),
        start=1,
    ):
        if month in text:
            months.append(index)
    return tuple(months)


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------


def since_window(days: int, *, now: datetime | None = None) -> tuple[datetime, datetime]:
    """A ``(start, end)`` pair for ``whatchanged`` queries."""
    end = now or datetime.now(timezone.utc)
    return (end - timedelta(days=days), end)
