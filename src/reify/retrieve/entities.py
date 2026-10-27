"""Entity extraction and canonicalization for the v0.3 hybrid retrieval graph.

This module produces the typed records that populate the ``entities``,
``entity_mentions``, and ``entity_edges`` tables. It is intentionally
LLM-free and pure-Python: graph-walk retrieval must work on the no-LLM
floor (FIRST_PRINCIPLES.md commandment 4).

Design constraints
------------------
* **Canonicalization is conservative.** Two surface forms collapse to one
  entity only when the normalized slug matches exactly. This avoids the
  classic NER trap of merging unrelated people/places named similarly.
* **Confidence-weighted.** Mentions derived from a :class:`Fact`'s explicit
  ``subject`` / ``obj`` carry confidence 1.0. Mentions captured from
  free text via the capitalization heuristic carry 0.6. Downstream
  graph-walk weighting uses these.
* **Co-occurrence is the cheap edge.** Any two entities mentioned in the
  same memory get a ``co_occurs`` edge. Facts also produce a typed
  ``predicate`` edge from subject to object using the predicate as the
  relation label, so the graph carries semantically-meaningful structure
  alongside bag-of-mentions co-occurrence.
* **Returns plain dataclasses, not SQL.** This module knows nothing about
  storage; the engine adapter wires the records to the storage layer.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable, Sequence

from reify.core.types import Fact, Memory


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Entity:
    """A canonical node in the entity graph.

    ``id`` is the canonical ``kind:slug`` form (e.g. ``person:arun``,
    ``role:staff_engineer``). ``label`` preserves the most-readable surface
    form encountered for display.
    """

    id: str
    label: str
    kind: str


@dataclass(frozen=True, slots=True)
class EntityMention:
    """A single occurrence of an entity inside one memory's content."""

    memory_id: str
    entity_id: str
    surface: str
    span_start: int = 0
    span_end: int = 0
    confidence: float = 1.0
    role: str = "mention"  # 'subject' | 'object' | 'mention'


@dataclass(frozen=True, slots=True)
class EntityEdge:
    """A weighted edge between two entities."""

    from_entity: str
    to_entity: str
    relation: str
    weight: float = 1.0
    last_seen_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(frozen=True, slots=True)
class EntityGraphDelta:
    """The triplet a single memory contributes to the entity graph."""

    entities: tuple[Entity, ...] = ()
    mentions: tuple[EntityMention, ...] = ()
    edges: tuple[EntityEdge, ...] = ()


# ---------------------------------------------------------------------------
# Canonicalization
# ---------------------------------------------------------------------------


_SLUG_RE = re.compile(r"[^a-z0-9]+")
_ARTICLES = frozenset({"the", "a", "an", "of", "for", "and", "or", "to", "at", "in", "on"})

# Lexicons used to classify free-text matches into a coarse kind. Order
# matters: the first hit wins. Each pattern is checked against the
# lowercased label so we can be liberal here.
_ROLE_TAILS: tuple[str, ...] = (
    "engineer", "developer", "designer", "manager", "director", "founder",
    "researcher", "scientist", "analyst", "consultant", "architect", "lead",
    "officer", "president", "cto", "ceo", "cfo", "coo", "vp", "svp", "evp",
    "intern", "advisor", "partner",
)

_LOCATION_HINTS: tuple[str, ...] = (
    "city", "town", "county", "country", "state", "province", "district",
    "village", "valley", "river", "lake",
)

_ORG_TAILS: tuple[str, ...] = (
    "inc", "inc.", "corp", "corp.", "ltd", "llc", "labs", "studios",
    "university", "institute", "foundation", "group", "team",
)


_POSSESSIVE_RE = re.compile(r"['\u2019]s\b", re.IGNORECASE)


def slugify(text: str) -> str:
    """Lower-case, ASCII, hyphen-separated. ``"Staff Engineer "`` -> ``staff_engineer``.

    Possessives are stripped (``"Arun's"`` -> ``"arun"``) so the same
    surface form does not produce two different graph nodes depending on
    grammatical context.

    Non-ASCII characters are folded to their nearest ASCII equivalent
    (NFKD + drop combining marks). Articles and trivial connector words
    are *not* dropped here; downstream callers that want a tighter match
    can post-process. Empty input returns ``""``.
    """
    nfkd = unicodedata.normalize("NFKD", text)
    ascii_only = "".join(ch for ch in nfkd if not unicodedata.combining(ch))
    # Strip possessives before slugifying so "Arun's" collapses to "arun".
    stripped = _POSSESSIVE_RE.sub("", ascii_only)
    lower = stripped.lower().strip()
    slug = _SLUG_RE.sub("_", lower).strip("_")
    return slug


def classify_kind(label: str, *, hint: str | None = None) -> str:
    """Heuristically classify a label into one of the canonical kinds.

    ``hint`` is preferred when present (e.g. a Fact's predicate carries
    semantic information). Returns one of ``person``, ``org``, ``location``,
    ``role``, ``topic`` (the catch-all).
    """
    if hint:
        h = hint.lower()
        if h in {"person", "org", "location", "role", "topic"}:
            return h
        if "title" in h or "role" in h or "promot" in h:
            return "role"
        if "lives_in" in h or "located" in h or "based" in h:
            return "location"
        if "works_at" in h or "employer" in h or "company" in h:
            return "org"

    low = label.lower()
    tokens = low.split()
    last = tokens[-1] if tokens else ""

    if last in _ROLE_TAILS:
        return "role"
    if any(tok in _ORG_TAILS for tok in tokens):
        return "org"
    if any(tok in _LOCATION_HINTS for tok in tokens):
        return "location"

    # All-caps acronym tokens (CPI, FOMC, NVTS, DM) are topics, not people:
    # the old person default turned every ticker and initialism into a
    # bogus persona candidate.
    if len(tokens) == 1 and len(label) >= 2 and label.isupper() and label.isalpha():
        return "topic"

    # Multi-word title-cased proper noun without role/org/location markers:
    # default to person if length 1-3 tokens, else topic.
    if 1 <= len(tokens) <= 3:
        return "person"
    return "topic"


# Higher number = stronger kind classification. When the same surface is
# classified differently by two passes (e.g. "Acme" + hint="works_at" -> org;
# same "Acme" + hint=None -> person), we keep the stronger kind so the entity
# does not fragment into multiple nodes.
_KIND_PRIORITY: dict[str, int] = {
    "topic": 1,
    "person": 2,
    "role": 3,
    "location": 4,
    "org": 5,
}


def canonicalize(label: str, *, hint: str | None = None) -> Entity | None:
    """Return the canonical :class:`Entity` for a surface form, or ``None``.

    The canonical id is the **slug only** (e.g. ``"acme"``, ``"san_francisco"``).
    ``kind`` is metadata, upgradeable when later passes produce a stronger
    classification. This avoids the classic fragmentation trap where the
    same surface form generates two graph nodes (``person:acme`` from a
    free-text scan and ``org:acme`` from a Fact's ``works_at`` predicate).

    ``None`` is returned for empty or stop-word-only inputs so callers can
    skip the mention without a special case.
    """
    label = label.strip()
    if not label:
        return None
    slug = slugify(label)
    if not slug or slug in _ARTICLES:
        return None
    kind = classify_kind(label, hint=hint)
    return Entity(id=slug, label=label, kind=kind)


# ---------------------------------------------------------------------------
# Surface extraction
# ---------------------------------------------------------------------------


# Capitalized noun phrases anywhere in the text. We capture sequences of
# 1-4 capitalized tokens, optionally separated by ``of``/``the``/``and``
# in lowercase (so "Bank of America" survives as one phrase). This is
# the long tail; Facts already give us their subject/obj for free.
_CAPS_PHRASE = re.compile(
    r"""
    \b
    (?P<phrase>
        (?:[A-Z][\w\-']+)               # first capitalized token
        (?:\s+(?:of|the|and|de|la)\s+   # optional connector (lowercase)
            [A-Z][\w\-']+)*?
        (?:\s+[A-Z][\w\-']+){0,3}        # up to 3 more capitalized tokens
    )
    \b
    """,
    re.VERBOSE,
)

# Common false-positive heads we want to drop even if capitalized.
_BLOCK_TOKENS = frozenset({
    "I", "I'm", "I'll", "I've", "I'd", "Im",
    "Mr", "Mrs", "Ms", "Dr",
    "Yes", "No", "OK", "Ok", "Okay",
    "Hey", "Hello", "Hi",
    "Today", "Yesterday", "Tomorrow",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
})

# Single-token capitalized phrases that are common English words, not names.
# Applied ONLY to single-token phrases from the free-text scan (multi-word
# proper names such as "Maya Chen" are never affected, and Fact subjects
# bypass this filter entirely -- they carry explicit semantics). Without
# this, sentence-initial verbs and pronouns ("Holds index funds...",
# "Checks her phone...", "It sends...") become bogus "person" entities and
# downstream persona generation invents people for them. Ambiguous
# name-words (Will, May, Mark, Bill, Frank, Grace, Hope, Summer) are
# deliberately NOT listed: a real single first name must survive the scan.
_COMMON_WORD_SINGLES = frozenset({
    # Pronouns / question words / determiners / quantifiers.
    "it", "he", "she", "they", "we", "you", "nobody", "everybody",
    "everyone", "someone", "anyone", "somebody", "anybody", "himself",
    "herself", "itself", "themselves", "ourselves", "yourself",
    "whoever", "whatever", "whichever", "why", "what", "when", "where",
    "which", "who", "whom", "whose", "how", "all", "any", "some", "none",
    "more", "most", "other", "others", "another", "each", "every", "few",
    "many", "much", "own", "same",
    # Auxiliaries / modals. "will" IS listed despite also being a name:
    # bare sentence-initial "Will" is overwhelmingly the verb ("Will test
    # any claim..."), while a real person is introduced by full name
    # ("Will Smith"), which this single-token filter never touches. May
    # stays excluded: it is month-blocked already and a common name.
    "will", "would", "can", "could", "shall", "should", "might", "must", "do",
    "does", "did", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had",
    # Adverbs.
    "very", "just", "also", "still", "already", "even", "ever", "never",
    "always", "often", "sometimes", "usually", "really", "actually",
    "finally", "recently", "currently", "previously", "especially",
    "particularly", "mainly", "mostly", "largely", "highly", "fully",
    "entirely", "completely", "exactly", "nearly", "almost", "quite",
    "rather", "pretty", "fairly", "somewhat", "slightly", "extremely",
    # Common verbs / adjectives / nouns observed as sentence-start junk.
    "want", "wants", "need", "needs", "use", "uses", "hold", "holds",
    "check", "checks", "watch", "watches", "hate", "hates", "trust",
    "trusts", "follow", "follows", "like", "likes", "know", "knows",
    # Added 2026-10-27 by live validation evidence: the Telegram rerun
    # graph still produced "Dislikes" / "Compares" / "Invests" person
    # nodes from sentence-initial verbs the original list missed.
    "dislike", "dislikes", "compare", "compares", "invest", "invests",
    "think", "thinks", "get", "gets", "make", "makes", "take", "takes",
    "see", "sees", "come", "comes", "go", "goes", "say", "says", "ask",
    "asks", "try", "tries", "help", "helps", "show", "shows", "give",
    "gives", "put", "puts", "keep", "keeps", "let", "lets", "feel",
    "feels", "mean", "means", "become", "becomes", "leave", "leaves",
    "bring", "brings", "happen", "happens", "write", "writes", "sit",
    "sits", "stand", "stands", "lose", "loses", "pay", "pays", "meet",
    "meets", "include", "includes", "continue", "continues", "set",
    "sets", "learn", "learns", "change", "changes", "lead", "leads",
    "understand", "understands", "read", "reads", "allow", "allows",
    "add", "adds", "spend", "spends", "grow", "grows", "open", "opens",
    "walk", "walks", "win", "wins", "offer", "offers", "remember",
    "remembers", "love", "loves", "consider", "considers", "appear",
    "appears", "buy", "buys", "wait", "waits", "serve", "serves",
    "die", "dies", "send", "sends", "expect", "expects", "build",
    "builds", "stay", "stays", "fall", "falls", "cut", "cuts", "reach",
    "reaches", "kill", "kills", "remain", "remains", "suggest",
    "suggests", "raise", "raises", "pass", "passes", "sell", "sells",
    "require", "requires", "report", "reports", "decide", "decides",
    "pull", "pulls", "afraid", "skeptical", "privacy", "free", "paid",
    "launch", "pick", "levels", "people", "questions",
})


def _extract_capitalized_phrases(text: str) -> list[tuple[str, int, int]]:
    """Return ``(surface, span_start, span_end)`` for capitalized phrases.

    Skips phrases at sentence start when they are a single common stop
    word (``"I"``, day-names, etc.) to keep precision high.
    """
    out: list[tuple[str, int, int]] = []
    for match in _CAPS_PHRASE.finditer(text):
        phrase = match.group("phrase").strip()
        if not phrase:
            continue
        first_token = phrase.split()[0]
        if first_token in _BLOCK_TOKENS and " " not in phrase:
            continue
        if " " not in phrase and phrase.lower() in _COMMON_WORD_SINGLES:
            continue
        out.append((phrase, match.start("phrase"), match.end("phrase")))
    return out


# ---------------------------------------------------------------------------
# Per-memory extraction
# ---------------------------------------------------------------------------


def extract_for_memory(memory: Memory) -> EntityGraphDelta:
    """Build the entity graph delta for a single memory.

    Strategy:

    1. If the memory is a :class:`Fact`, the subject and object are
       canonicalized with the predicate as a kind hint. A typed predicate
       edge is emitted from subject to object.
    2. The memory's ``content`` is scanned for capitalized phrases via
       :func:`_extract_capitalized_phrases` and each unique entity becomes
       a low-confidence ``mention``.
    3. Every pair of entities found on the same memory gets a
       symmetric ``co_occurs`` edge so the walk has bidirectional
       traversal even if the predicate-typed edges are sparse.
    """
    entities: dict[str, Entity] = {}
    # One mention per (memory, entity) pair. Higher-priority roles
    # (``subject`` > ``object`` > ``mention``) win when the same entity
    # is captured by multiple passes (e.g. a Fact whose content text
    # also matches the capitalized-phrase scan). Higher-priority roles
    # carry higher confidence, so this is just keeping the strongest
    # signal we have for each entity.
    mention_by_entity: dict[str, EntityMention] = {}
    role_priority = {"subject": 3, "object": 2, "mention": 1}
    edges: list[EntityEdge] = []
    now = memory.created_at

    def add_mention(label: str, *, hint: str | None, role: str, confidence: float, span: tuple[int, int] = (0, 0)) -> str | None:
        ent = canonicalize(label, hint=hint)
        if ent is None:
            return None
        existing_ent = entities.get(ent.id)
        if existing_ent is None:
            entities[ent.id] = ent
        elif _KIND_PRIORITY.get(ent.kind, 0) > _KIND_PRIORITY.get(existing_ent.kind, 0):
            # Upgrade the kind when this pass produces a stronger one.
            entities[ent.id] = ent
        candidate = EntityMention(
            memory_id=memory.id,
            entity_id=ent.id,
            surface=label,
            span_start=span[0],
            span_end=span[1] if span[1] >= span[0] else span[0],
            confidence=confidence,
            role=role,
        )
        existing = mention_by_entity.get(ent.id)
        if existing is None or role_priority.get(role, 0) > role_priority.get(existing.role, 0):
            mention_by_entity[ent.id] = candidate
        return ent.id

    # 1. Fact subject/object -------------------------------------------------
    # The predicate describes how the subject relates to the object, which
    # carries strong semantic information about the *object's* kind
    # (``works_at`` -> object is an org, ``lives_in`` -> location). The
    # subject is typically a person doing the action, so we let the
    # lexicon classify the subject without the predicate hint.
    if isinstance(memory, Fact):
        subj_id = add_mention(
            memory.subject, hint=None, role="subject", confidence=1.0
        )
        obj_id = add_mention(
            memory.obj, hint=memory.predicate, role="object", confidence=1.0
        )
        if subj_id and obj_id and subj_id != obj_id:
            edges.append(
                EntityEdge(
                    from_entity=subj_id,
                    to_entity=obj_id,
                    relation=memory.predicate,
                    weight=1.0,
                    last_seen_at=now,
                )
            )

    # 2. Free-text capitalized phrases ---------------------------------------
    for phrase, start, end in _extract_capitalized_phrases(memory.content):
        add_mention(
            phrase, hint=None, role="mention", confidence=0.6, span=(start, end)
        )

    # 3. Co-occurrence edges -------------------------------------------------
    ids = list(entities.keys())
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            if a == b:
                continue
            edges.append(
                EntityEdge(
                    from_entity=a, to_entity=b, relation="co_occurs",
                    weight=0.5, last_seen_at=now,
                )
            )
            edges.append(
                EntityEdge(
                    from_entity=b, to_entity=a, relation="co_occurs",
                    weight=0.5, last_seen_at=now,
                )
            )

    return EntityGraphDelta(
        entities=tuple(entities.values()),
        mentions=tuple(mention_by_entity.values()),
        edges=tuple(edges),
    )


def extract_for_query(text: str) -> tuple[Entity, ...]:
    """Extract candidate seed entities from a user query string.

    Used by graph-walk retrieval to pick the entities to start the walk
    from. Returns canonicalized entities; duplicates are collapsed.
    """
    out: dict[str, Entity] = {}
    for phrase, _, _ in _extract_capitalized_phrases(text):
        ent = canonicalize(phrase)
        if ent is not None and ent.id not in out:
            out[ent.id] = ent
    return tuple(out.values())


def merge_deltas(deltas: Iterable[EntityGraphDelta]) -> EntityGraphDelta:
    """Merge a sequence of per-memory deltas into one batch-shaped delta.

    Entities are deduplicated by id (last writer wins on label). Mentions
    and edges are concatenated; downstream storage UPSERT handles edge
    weight aggregation.
    """
    ents: dict[str, Entity] = {}
    mentions: list[EntityMention] = []
    edges: list[EntityEdge] = []
    for d in deltas:
        for e in d.entities:
            existing = ents.get(e.id)
            if existing is None:
                ents[e.id] = e
            elif _KIND_PRIORITY.get(e.kind, 0) > _KIND_PRIORITY.get(existing.kind, 0):
                ents[e.id] = e
        mentions.extend(d.mentions)
        edges.extend(d.edges)
    return EntityGraphDelta(
        entities=tuple(ents.values()),
        mentions=tuple(mentions),
        edges=tuple(edges),
    )


__all__ = [
    "Entity",
    "EntityMention",
    "EntityEdge",
    "EntityGraphDelta",
    "canonicalize",
    "classify_kind",
    "extract_for_memory",
    "extract_for_query",
    "merge_deltas",
    "slugify",
]
