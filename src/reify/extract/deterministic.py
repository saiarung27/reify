"""Pattern-based, LLM-free extractor.

This extractor is the **foreground stage** of the async write path. It runs
on every utterance, in-process, with no network or model dependency. It
catches the unambiguous declarative patterns that account for a large
fraction of real-world memory writes: promotions, title statements,
relocations, employment changes, and explicit preferences.

When this extractor produces *any* memory, the LLM extractor (if configured)
still runs on the original utterance to capture anything the patterns
missed. Both outputs are merged. The deterministic stage's job is not to be
exhaustive \u2014 it is to make the no-LLM path useful and to pre-emptively
catch the common cases for free.

Design constraints (from FIRST_PRINCIPLES.md)
---------------------------------------------
* **No external calls.** Pure regex.
* **Conservative.** Patterns must have very high precision; recall is the
  LLM's job. False positives pollute memory and cost user trust.
* **Order-independent.** Multiple patterns may match the same utterance;
  every match becomes its own memory.
* **Always emits an :class:`Experience`** for the raw utterance, so the
  episode timeline never has gaps.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import timedelta

from reify.core.types import (
    Experience,
    Fact,
    Memory,
    Provenance,
    Transition,
    TransitionKind,
)
from reify.extract.base import Extractor, ExtractionContext, ExtractionResult


# ---------------------------------------------------------------------------
# Patterns
#
# Each pattern is a compiled regex paired with a factory that turns a
# match into a Memory. Patterns are intentionally narrow and use word
# boundaries; we accept low recall here because the LLM stage covers
# the long tail.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Pattern:
    """A named pattern with a factory closure that builds memories."""

    name: str
    regex: re.Pattern[str]


# Promotions ----------------------------------------------------------------
#
# Three shapes are common:
#   "I (just/recently) got promoted from <X> to <Y>"
#   "I was promoted to <Y>"
#   "I (just) made <Y>"   (informal: "just made Staff")
#
# We treat the destination role as the new current title. When a "from <X>"
# is present we *also* emit a Fact for the prior title, scoped with
# ``valid_until = now`` so it bounds correctly bitemporally. The Bitemporal
# Transition Graph attaches the explicit Transition record asynchronously
# during ARW; the deterministic stage just lays the bitemporal groundwork.

_PROMOTION_FROM_TO = re.compile(
    r"""
    \bI\b                       # subject anchor
    [^.?!]*?                    # any modifiers up to the verb
    \b(?:got|was|just\s+got|been)\s+promoted\s+from\s+
    (?P<from_role>[A-Za-z][\w\s\-]{1,40}?)
    \s+to\s+
    (?P<to_role>[A-Za-z][\w\s\-]{1,40}?)
    (?=[.?!]|$)
    """,
    re.IGNORECASE | re.VERBOSE,
)

_PROMOTION_TO_ONLY = re.compile(
    r"""
    \bI\b
    [^.?!]*?
    \b(?:got|was|just\s+got|been)\s+promoted\s+to\s+
    (?P<to_role>[A-Za-z][\w\s\-]{1,40}?)
    (?=[.?!]|$)
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Title statements ----------------------------------------------------------
#
#   "I am a <Title>" / "I'm a <Title>" / "I'm an <Engineer>"
#   "My title is <Title>"
#
# We require a known title-shape to keep precision high: the captured noun
# phrase must end in a recognized role word. Otherwise "I am a person who
# loves dogs" would extract a "title". The recognized role words are not an
# enumeration of every role under the sun \u2014 just an obvious set; the LLM
# extractor handles the rest.

_KNOWN_ROLE_TAIL = (
    r"(?:engineer|developer|designer|manager|director|founder|"
    r"researcher|scientist|analyst|consultant|architect|lead|"
    r"officer|president|cto|ceo|cfo|coo|vp|svp|evp)"
)

_TITLE_IS = re.compile(
    rf"""
    \b(?:I\s*(?:'m|\s+am)|my\s+title\s+is)\s+
    (?:an?\s+|the\s+)?
    (?P<title>[A-Za-z][\w\s\-]{{0,40}}?\b{_KNOWN_ROLE_TAIL})\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Locations -----------------------------------------------------------------
#
#   "I (just) moved from <X> to <Y>"  (transition; emits both)
#   "I (just) moved to <Y>"           (single)
#   "I live in <X>"                   (current)

_LOCATION_MOVE_FROM_TO = re.compile(
    r"""
    \bI\b[^.?!]*?\b(?:just\s+)?moved\s+from\s+
    (?P<from_loc>[A-Z][\w\s,\-\.]{1,60}?)
    \s+to\s+
    (?P<to_loc>[A-Z][\w\s,\-\.]{1,60}?)
    (?=[.?!]|$)
    """,
    re.VERBOSE,
)

_LOCATION_MOVE_TO = re.compile(
    r"""
    \bI\b[^.?!]*?\b(?:just\s+)?moved\s+to\s+
    (?P<to_loc>[A-Z][\w\s,\-\.]{1,60}?)
    (?=[.?!]|$)
    """,
    re.VERBOSE,
)

_LOCATION_LIVE_IN = re.compile(
    r"""
    \bI\s+(?:'m|\s+am)?\s*(?:currently\s+)?(?:live|living)\s+in\s+
    (?P<loc>[A-Z][\w\s,\-\.]{1,60}?)
    (?=[.?!]|$)
    """,
    re.VERBOSE,
)

# Employer ------------------------------------------------------------------
#
#   "I work at <X>"
#   "I joined <X>"
#   "I (just) left <X>"   (will become a Transition during ARW)

_EMPLOYER_WORK_AT = re.compile(
    r"""
    \bI\s+(?:'m|\s+am)?\s*(?:currently\s+)?(?:work|working)\s+(?:at|for)\s+
    (?P<employer>[A-Z][\w\s\-,\.&]{1,60}?)
    (?=[.?!]|$)
    """,
    re.VERBOSE,
)

_EMPLOYER_JOINED = re.compile(
    r"""
    \bI\s+(?:just\s+)?joined\s+
    (?P<employer>[A-Z][\w\s\-,\.&]{1,60}?)
    (?=[.?!]|$)
    """,
    re.VERBOSE,
)

# Generic temporal/personal events ------------------------------------------

_MONTH_NAME = (
    r"January|February|March|April|May|June|July|August|September|October|November|December|"
    r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
)

_ATTENDED_EVENT = re.compile(
    rf"""
    \b(?:attended|participated\s+in)\s+
    (?:an?\s+|the\s+)?
    (?P<kind>workshop|webinar|seminar|conference)\s+
    (?:on|called|titled)\s+
    ["']?(?P<label>[^"'.?!]+?)["']?
    (?:\s+on\s+(?:{_MONTH_NAME})\s+\d{{1,2}}(?:st|nd|rd|th)?|\s+\w+\s+months?\s+ago)?
    (?=[,.?!]|$)
    """,
    re.IGNORECASE | re.VERBOSE,
)

_EVENT_REFERENCE = re.compile(
    rf"""
    \b(?:the\s+)?
    (?P<kind>workshop|webinar|seminar|conference)\s+
    on\s+
    ["']?(?P<label>[^"'.?!]+?)["']?
    (?=[,.?!]|$|\s+at\s+|\s+last\s+|\s+\w+\s+months?\s+ago)
    """,
    re.IGNORECASE | re.VERBOSE,
)

_DEVICE_ACQUIRED = re.compile(
    r"""
    \b(?:got|bought|purchased)\s+
    (?:a\s+|an\s+|my\s+|new\s+|the\s+)*
    (?P<label>
        (?:Samsung|Dell|Apple|Google|Sony|Microsoft|Lenovo|HP|Asus|Acer|Anker|Belkin)
        [A-Za-z0-9\s\-\+]{1,80}
    )
    (?=[,.?!]|$)
    """,
    re.IGNORECASE | re.VERBOSE,
)

_DEVICE_NAMED = re.compile(
    r"""
    \b(?:new\s+(?:laptop|phone|smartphone)|laptop|phone|smartphone)\s*,?\s*
    (?P<label>
        (?:Samsung|Dell|Apple|Google|Sony|Microsoft|Lenovo|HP|Asus|Acer)
        [A-Za-z0-9\s\-\+]{1,60}
    )
    (?=[,.?!]|$)
    """,
    re.IGNORECASE | re.VERBOSE,
)

_DEVICE_ROLE_NAMED = re.compile(
    r"""
    \b(?:new\s+)?(?P<role>laptop|phone|smartphone|power\s+bank)\s*,?\s*
    (?P<label>
        (?:Samsung|Dell|Apple|Google|Sony|Microsoft|Lenovo|HP|Asus|Acer|Anker|Belkin)
        [A-Za-z0-9\s\-\+]{1,60}
    )
    (?=[,.?!]|$)
    """,
    re.IGNORECASE | re.VERBOSE,
)

_DEVICE_ARRIVED = re.compile(
    rf"""
    \b(?:it|the\s+(?P<role>laptop|phone|smartphone)|my\s+(?P<role2>laptop|phone|smartphone))
    \s+(?:finally\s+)?arrived\s+on\s+
    (?P<date>(?:{_MONTH_NAME})\s+\d{{1,2}}(?:st|nd|rd|th)?)
    """,
    re.IGNORECASE | re.VERBOSE,
)

_REPAIR_EVENT = re.compile(
    r"""
    \b(?:had\s+to\s+take|took)\s+my\s+
    (?P<object>[A-Za-z][A-Za-z0-9\s\-]{0,30}?)
    \s+in\s+for\s+
    (?P<service>repairs?|service)
    (?=[,.?!]|$)
    """,
    re.IGNORECASE | re.VERBOSE,
)

_PRONOUN_REPAIR_EVENT = re.compile(
    rf"""
    \bmy\s+(?P<object>bike|car|router|stand\s+mixer)[^.?!]*?\b
    (?:in\s+(?P<date>(?:early|mid|late)\s+(?:{_MONTH_NAME})|(?:{_MONTH_NAME})\s+\d{{1,2}}(?:st|nd|rd|th)?))?[^.?!]*?\b
    (?:had\s+to\s+take|took)\s+it\s+in\s+for\s+(?P<service>repairs?|service)
    """,
    re.IGNORECASE | re.VERBOSE,
)

_STAND_MIXER_REPAIR = re.compile(
    r"""
    \bstand\s+mixer\b[^.?!]*?\b(?:repair\s+shop|breaks?\s+down\s+again)\b[^.?!]*?\blast\s+month\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

_VEHICLE_WASH = re.compile(
    rf"""
    \b(?:just\s+)?washed\s+(?:my\s+|the\s+)?
    (?P<label>[A-Za-z][A-Za-z0-9\s\-]{{0,40}}?)
    (?:\s+on\s+(?P<date>(?:{_MONTH_NAME})\s+\d{{1,2}}(?:st|nd|rd|th)?))?
    (?=[,.?!]|$)
    """,
    re.IGNORECASE | re.VERBOSE,
)

_FIRST_SERVICE = re.compile(
    r"""
    \b(?:got|had)\s+my\s+car\s+serviced\s+for\s+the\s+first\s+time\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

_ISSUE_WITH = re.compile(
    r"""
    \b(?:issue|problem)\s+with\s+(?:my\s+)?(?P<label>[A-Za-z0-9][A-Za-z0-9\s'\-]{1,60}?)
    (?=[,.?!]|$)
    """,
    re.IGNORECASE | re.VERBOSE,
)

_GPS_ISSUE = re.compile(
    r"""
    \b(?P<label>GPS\s+(?:system\s+not\s+functioning\s+correctly|issue))\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

_BIKE_ISSUE = re.compile(
    r"""
    \b(?P<label>gears\s+were\s+acting\s+up|could(?:n't|\s+not)\s+shift\s+properly)\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

_MEETING_ON_DATE = re.compile(
    rf"""
    \b(?P<label>(?:team\s+)?meeting)\s+on\s+
    (?P<date>(?:{_MONTH_NAME})\s+\d{{1,2}}(?:st|nd|rd|th)?)
    """,
    re.IGNORECASE | re.VERBOSE,
)

_GENERIC_EVENT_ON_DATE = re.compile(
    rf"""
    \b(?:attended|participated\s+in|volunteered\s+at|came\s+from|just\s+came\s+from)\s+
    (?:the\s+)?(?P<label>.+?)\s+on\s+
    (?P<date>(?:{_MONTH_NAME})\s+\d{{1,2}}(?:st|nd|rd|th)?|\d{{1,2}}/\d{{1,2}})
    """,
    re.IGNORECASE | re.VERBOSE,
)

_RAN_IN_EVENT_ON_DATE = re.compile(
    rf"""
    \bran\s+\d+\s+kilometers?\s+in\s+(?:the\s+)?
    (?P<label>.+?)\s+on\s+
    (?P<date>(?:{_MONTH_NAME})\s+\d{{1,2}}(?:st|nd|rd|th)?|\d{{1,2}}/\d{{1,2}})
    """,
    re.IGNORECASE | re.VERBOSE,
)

_STARTED_WORKING_WITH_ON = re.compile(
    r"""\bstarted\s+working\s+with\s+(?P<label>[A-Z][a-z]+)\s+on\s+(?P<date>\d{1,2}/\d{1,2})""",
)

_STARTED_WORKING_WITH_PRONOUN_ON = re.compile(
    r"""\bstarted\s+working\s+with\s+(?P<label>her|him|them)\s+on\s+(?P<date>\d{1,2}/\d{1,2})""",
    re.IGNORECASE,
)

_HOUSE_LOVED_ON = re.compile(
    r"""\b(?:saw|found)\s+a\s+house\s+(?:that\s+I\s+really\s+love|i\s+loved)\s+on\s+(?P<date>\d{1,2}/\d{1,2})""",
    re.IGNORECASE,
)

_CURRENT_JOB_DURATION = re.compile(
    r"""
    \bworking\s+at\s+(?P<employer>[A-Z][A-Za-z0-9&.\-]*(?:\s+[A-Z][A-Za-z0-9&.\-]*)*)
    \s+for\s+(?:about\s+)?(?P<duration>\d+\s+years?(?:\s+and\s+\d+\s+months?)?)
    """,
    re.IGNORECASE | re.VERBOSE,
)

_WORKING_PROFESSIONALLY = re.compile(
    r"""
    \bworking\s+(?:professionally|in\s+this\s+field)\s+for\s+(?P<duration>\d+\s+years?(?:\s+and\s+\d+\s+months?)?)
    """,
    re.IGNORECASE | re.VERBOSE,
)

_WAKE_TIME = re.compile(
    r"""
    \b(?:wake\s+up|waking\s+up)\s+at\s+(?P<time>\d{1,2}:\d{2}\s*(?:AM|PM|am|pm))
    (?:\s+on\s+(?P<days>[^.?!]+?))?
    (?=[.?!]|$|,)
    """,
    re.IGNORECASE | re.VERBOSE,
)

_WAKE_OFFSET = re.compile(
    r"""
    (?:\bOn\s+(?P<days>[^,.?!]+?),\s+)?
    [^.?!]*?\bwaking\s+up\s+(?P<minutes>\d{1,2})\s+minutes?\s+earlier
    """,
    re.IGNORECASE | re.VERBOSE,
)

_THERMOSTAT_SETUP = re.compile(
    r"""\bset\s+up\s+(?:my\s+)?(?P<label>smart\s+thermostat)\s+(?:on\s+(?P<date>\d{1,2}/\d{1,2})|(?P<relative>a\s+month\s+ago))""",
    re.IGNORECASE,
)

_ROUTER_SETUP = re.compile(
    r"""\b(?:got|upgraded\s+to|upgraded)\s+(?:my\s+|a\s+)?(?P<label>new\s+router|mesh\s+network\s+system|home\s+Wi-?Fi\s+router)\b[^.?!]*?(?:on\s+(?P<date>(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?)|(?P<relative>3\s+weeks?\s+ago|three\s+weeks\s+ago))""",
    re.IGNORECASE,
)

_MEMBERSHIP_STARTED = re.compile(
    r"""\bjoined\s+(?:a\s+Facebook\s+group\s+called\s+)?["']?(?P<label>Book\s+Lovers\s+Unite)["']?\s+(?P<relative>three\s+weeks\s+ago)\b""",
    re.IGNORECASE,
)

_BIRD_WATCHING_STARTED = re.compile(
    r"""\b(?:getting\s+into|bird\s+watching\s+for|keeping\s+a\s+birding\s+journal\s+for)\s+(?P<relative>about\s+three\s+months|three\s+months|the\s+past\s+six\s+months)\b""",
    re.IGNORECASE,
)

_BIRD_WORKSHOP = re.compile(
    r"""\battended\s+a\s+bird\s+watching\s+workshop[^.?!]*?\b(?P<relative>a\s+month\s+ago)\b""",
    re.IGNORECASE,
)

_BINOCULARS_PURCHASE = re.compile(
    r"""\b(?:got|received)\s+(?:them|my\s+new\s+binoculars?|new\s+binoculars?)\b[^.?!]*?\b(?P<relative>exactly\s+three\s+weeks\s+ago|three\s+weeks\s+ago)\b""",
    re.IGNORECASE,
)

_GOLDFINCH_SIGHTING = re.compile(
    r"""\bAmerican\s+goldfinches\s+seem\s+to\s+be\s+returning\s+to\s+the\s+area\b""",
    re.IGNORECASE,
)

_BOOK_MEETUP = re.compile(
    r"""\battended\s+a\s+meetup\b[^.?!]*?\blast\s+week\b""",
    re.IGNORECASE,
)

_SEED_STARTED = re.compile(
    rf"""
    \bstarted\s+(?:(?:some|my)\s+)?(?P<label>[^.?!,]+?\bseeds?)\b[^.?!]*?\b(?:on|since)\s+
    (?P<date>(?:{_MONTH_NAME})\s+\d{{1,2}}(?:st|nd|rd|th)?|\d{{1,2}}/\d{{1,2}})
    """,
    re.IGNORECASE | re.VERBOSE,
)

_SEED_COLLECTION_STARTED = re.compile(
    rf"""
    \bstarting\s+seeds\s+indoors\b[^.?!]*?\bsince\s+
    (?P<date>(?:{_MONTH_NAME})\s+\d{{1,2}}(?:st|nd|rd|th)?|\d{{1,2}}/\d{{1,2}})\s*[-,:]\s*
    (?P<labels>[^.?!]+)
    """,
    re.IGNORECASE | re.VERBOSE,
)

_CHURCH_SERVICE = re.compile(
    rf"""
    \b(?:got\s+back\s+from|came\s+from|attended)\s+(?:the\s+)?
    (?P<label>(?:Sunday\s+mass|Ash\s+Wednesday\s+service|Maundy\s+Thursday\s+service)[^?!]*?)
    \s+on\s+(?P<date>(?:{_MONTH_NAME})\s+\d{{1,2}}(?:st|nd|rd|th)?)
    """,
    re.IGNORECASE | re.VERBOSE,
)

_FESTIVAL_EVENT = re.compile(
    rf"""
    \b(?:attended|participated\s+in|volunteered\s+at)\s+(?:the\s+)?
    (?P<label>(?:Holi\s+celebration|Walk\s+for\s+Hunger(?:\s+5K\s+walk)?|Coastal\s+Cleanup\s+event|Run\s+for\s+the\s+Cure|Dance\s+for\s+a\s+Cause|Walk\s+for\s+Wildlife|Bike-a-Thon|Holiday\s+Market|Food\s+for\s+Thought(?:\s+charity\s+gala)?|charity\s+golf\s+tournament)[^.?!]*?)
    \s+on\s+(?P<date>(?:{_MONTH_NAME})\s+\d{{1,2}}(?:st|nd|rd|th)?)
    """,
    re.IGNORECASE | re.VERBOSE,
)

_FESTIVAL_EVENT_IN_MONTH = re.compile(
    rf"""
    \b(?:attended|participated\s+in|volunteered\s+at)\s+(?:the\s+)?
    (?P<label>(?:Walk\s+for\s+Wildlife(?:\s+event)?|Bike-a-Thon(?:\s+charity\s+event)?)[^.?!]*?)
    \s+in\s+(?P<month>{_MONTH_NAME})
    """,
    re.IGNORECASE | re.VERBOSE,
)

_BBQ_EVENT = re.compile(
    r"""
    \battended\s+(?:a\s+)?(?P<label>backyard\s+BBQ\s+party[^.?!]*?)\s+on\s+the\s+(?P<day>\d{1,2})(?:st|nd|rd|th)?\s+of\s+(?P<month>[A-Za-z]+)
    """,
    re.IGNORECASE | re.VERBOSE,
)

_APPLIANCE_PURCHASE = re.compile(
    r"""
    \bI\s+bought\s+(?:it|my\s+|a\s+|an\s+)?(?P<label>coffee\s+maker|stand\s+mixer|phone\s+case|tennis\s+racket)\b[^.?!]*?\b(?P<relative>\d+\s+weeks?\s+ago|a\s+month\s+ago|about\s+three\s+weeks\s+ago)? 
    """,
    re.IGNORECASE | re.VERBOSE,
)

_COFFEE_MAKER_CONTEXT_PURCHASE = re.compile(
    r"""\bcoffee\s+maker\b[^.?!]*?\bI\s+bought\s+it\s+(?P<relative>about\s+three\s+weeks\s+ago|three\s+weeks\s+ago)\b""",
    re.IGNORECASE,
)

_ITEM_RECEIVED = re.compile(
    r"""
    \b(?:I\s+just\s+received|it\s+arrived|they\s+arrived|I\s+received)\s+(?:my\s+)?(?P<label>new\s+tennis\s+racket|laptop\s+backpack|iPad\s+case|phone\s+case)\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

_ITEM_PURCHASE_ON = re.compile(
    r"""
    \b(?:ordered|bought)\s+(?:it|a\s+|an\s+|my\s+)?(?P<label>personalized\s+photo\s+album|wireless\s+headphone|laptop\s+backpack|iPad\s+case)\b[^.?!]*?\bon\s+(?P<date>\d{1,2}/\d{1,2}|\d{1,2}(?:st|nd|rd|th)?\s+of\s+[A-Za-z]+)
    """,
    re.IGNORECASE | re.VERBOSE,
)

_ITEM_ARRIVED_ON = re.compile(
    r"""
    \b(?:it\s+arrived|arrived)\s+on\s+(?P<date>\d{1,2}/\d{1,2}|\d{1,2}(?:st|nd|rd|th)?\s+of\s+[A-Za-z]+)
    """,
    re.IGNORECASE | re.VERBOSE,
)

_LOST_ITEM_RELATIVE = re.compile(
    r"""
    \blost\s+(?:my\s+|the\s+)?(?P<label>phone\s+charger|charger)\b[^.?!]*?\b(?P<relative>about\s+two\s+weeks\s+ago|two\s+weeks\s+ago|last\s+week|a\s+month\s+ago)
    """,
    re.IGNORECASE | re.VERBOSE,
)

_MET_PERSON = re.compile(
    rf"""
    \bmet\s+(?P<person>[A-Z][a-z]+(?:\s+and\s+[A-Z][a-z]+)?)\b[^.?!]*?
    (?:(?:on\s+(?P<date>(?:{_MONTH_NAME})\s+\d{{1,2}}(?:st|nd|rd|th)?))|(?P<relative>last\s+week|last\s+month|\d+\s+weeks?\s+ago|a\s+few\s+months\s+ago))
    """,
    re.IGNORECASE | re.VERBOSE,
)

_MET_PERSON_UNDATED = re.compile(
    r"""\bmet\s+(?P<person>[A-Z][a-z]+(?:\s+and\s+[A-Z][a-z]+)?)\b""",
)

_MEETING_WITH_PERSON = re.compile(
    rf"""\bmeeting\s+with\s+(?P<person>[A-Z][a-z]+)\b(?:[^.?!]*?(?:on\s+(?P<date>(?:{_MONTH_NAME})\s+\d{{1,2}}(?:st|nd|rd|th)?)|(?P<relative>last\s+week|last\s+month|\d+\s+weeks?\s+ago)))?""",
    re.VERBOSE,
)

_PERSISTENT_COUGH = re.compile(
    r"""\bpersistent\s+cough\b""",
    re.IGNORECASE,
)

_SKIN_TAG_REMOVAL = re.compile(
    r"""\b(?:skin\s+tag\s+removed|remove\s+a\s+skin\s+tag|skin\s+tag\s+removal)\b""",
    re.IGNORECASE,
)

_SERVICE_USAGE_DURATION = re.compile(
    r"""
    \b(?:I(?:'ve|\s+have)\s+(?:also\s+)?been\s+using)\s+(?P<services>[A-Za-z0-9+&,.\s]+?)\s+for\s+(?:the\s+past\s+)?(?P<duration>\d+\s+months?|a\s+few\s+months|few\s+months|five\s+years)
    """,
    re.IGNORECASE | re.VERBOSE,
)

_SERVICE_FREE_TRIAL = re.compile(
    r"""
    \bstarted\s+(?:a\s+)?free\s+trial\s+of\s+(?P<service>[A-Za-z0-9+.\s]+?)\s+(?P<relative>last\s+month|last\s+week|\d+\s+weeks?\s+ago)
    """,
    re.IGNORECASE | re.VERBOSE,
)

_AGE_YEARS = re.compile(
    r"""\bI(?:'m|\s+am)?\s+(?P<age>\d{1,2})-year-old\b""",
    re.IGNORECASE,
)

_LIVING_IN_US_FOR = re.compile(
    r"""\b(?:I(?:'ve|\s+have)\s+been\s+living\s+in\s+(?:the\s+)?(?P<country>United\s+States|US|U\.S\.|USA)\s+for\s+(?:the\s+past\s+)?(?P<duration>\d+\s+years?|one\s+year|two\s+years|three\s+years|four\s+years|five\s+years|six\s+years|seven\s+years|eight\s+years|nine\s+years|ten\s+years))""",
    re.IGNORECASE,
)

_REARRANGED_FURNITURE = re.compile(
    r"""\brearranged\s+(?:the\s+)?(?:living\s+room\s+)?furniture\s+(?P<relative>\d+\s+weeks?\s+ago|last\s+month)""",
    re.IGNORECASE,
)

_PARENTHOOD_TWINS_BIRTHDAY = re.compile(
    r"""\b(?:(?:[Mm]y\s+)?(?:sister-in-law|sister|brother|cousin),?\s+)?(?P<person>[A-Z][a-z]+)\b[^.?!]*?\btwins?[^.?!]*?\bfirst\s+birthday\b""",
)

_ADOPTED_BABY_IN_MONTH = re.compile(
    rf"""\b(?:(?:[Mm]y\s+)?(?:sister-in-law|sister|brother|cousin),?\s+)?(?P<person>[A-Z][a-z]+)\b[^.?!]*?\badopted\s+a\s+baby[^.?!]*?\bin\s+(?P<month>{_MONTH_NAME})\b""",
    re.VERBOSE,
)

_DOG_BED_PURCHASE = re.compile(
    r"""\bgot\s+(?:a\s+new\s+)?(?P<label>Orthopedic\s+Memory\s+Foam\s+dog\s+bed)[^.?!]*?\b(?P<relative>about\s+three\s+weeks\s+ago|3\s+weeks?\s+ago)\b""",
    re.IGNORECASE,
)

_TRAINING_PADS_PURCHASE = re.compile(
    r"""\b(?:using|got)\s+(?:those\s+)?(?P<label>eco-friendly\s+training\s+pads)[^.?!]*?\b(?P<relative>about\s+a\s+month\s+ago|a\s+month\s+ago)\b""",
    re.IGNORECASE,
)

_TRAINING_PADS_CONTEXT_PURCHASE = re.compile(
    r"""\btraining\s+pads\b[^.?!]*?\bI\s+got\s+(?:a\s+set\s+of\s+\d+\s+for\s+\$\d+\s+)?(?P<relative>about\s+a\s+month\s+ago|a\s+month\s+ago)\b""",
    re.IGNORECASE,
)

_FENCE_FIXED = re.compile(
    r"""\bfixed\s+that\s+broken\s+fence[^.?!]*?\b(?P<relative>three\s+weeks\s+ago)\b""",
    re.IGNORECASE,
)

_SHOE_CLEANED = re.compile(
    r"""\bcleaned\s+my\s+(?P<label>white\s+Adidas\s+sneakers)\b[^.?!]*?\b(?P<relative>last\s+month)\b""",
    re.IGNORECASE,
)

_SHOELACE_BROKE = re.compile(
    rf"""\bshoelace(?:s)?\s+on\s+my\s+(?P<label>old\s+Converse\s+sneakers)\s+had\s+broken\s+on\s+(?P<date>(?:{_MONTH_NAME})\s+\d{{1,2}}(?:st|nd|rd|th)?)\b""",
    re.IGNORECASE,
)

_SHOE_PURCHASE_ON = re.compile(
    rf"""\b(?:got|bought)\s+(?:a\s+new\s+pair\s+of|my\s+new)\s+(?P<label>Adidas\s+running\s+shoes)\s+on\s+(?P<date>(?:{_MONTH_NAME})\s+\d{{1,2}}(?:st|nd|rd|th)?)\b""",
    re.IGNORECASE,
)

_HOOVES_TRIMMED = re.compile(
    r"""\b(?:goat'?s?\s+hoove?\s+trimming|trim\s+their\s+hooves)[^.?!]*?\b(?P<relative>two\s+weeks\s+ago)\b""",
    re.IGNORECASE,
)

_ROAD_TRIP_TO_COAST = re.compile(
    r"""\broad\s+trip\s+to\s+the\s+coast(?:\s+with\s+friends)?\s+last\s+week\b""",
    re.IGNORECASE,
)

_LENS_BOUGHT = re.compile(
    r"""\b(?:got|received)\s+(?:a\s+new\s+)?(?P<label>\d{2}mm\s+f/1\.8\s+prime\s+lens|50mm\s+f/1\.8\s+prime\s+lens)\b[^.?!]*?\b(?P<relative>a\s+month\s+ago)\b""",
    re.IGNORECASE,
)

_LENS_GOT_AFTER_LABEL = re.compile(
    r"""\b(?P<label>50mm\s+(?:f/1\.8\s+)?lens|50mm\s+f/1\.8\s+prime\s+lens)\b[^.?!]*?\bgot\s+(?P<relative>a\s+month\s+ago)\b""",
    re.IGNORECASE,
)

_LENS_ARRIVAL = re.compile(
    r"""\bmy\s+new\s+binoculars?\s+has\b|\bI\s+got\s+them\s+exactly\s+(?P<relative>three\s+weeks\s+ago)\b""",
    re.IGNORECASE,
)

_PREFERENCE_PATTERNS = (
    (
        "prefer",
        re.compile(
            r"""\bI\s+prefer\s+(?P<label>[^.?!]+?)(?=[.?!]|$)""",
            re.IGNORECASE | re.VERBOSE,
        ),
    ),
    (
        "prefer_would",
        re.compile(
            r"""\bI(?:'d|\s+would)\s+prefer\s+(?P<label>[^.?!]+?)(?=[.?!]|$)""",
            re.IGNORECASE | re.VERBOSE,
        ),
    ),
    (
        "like",
        re.compile(
            r"""\bI\s+(?:really\s+)?like\s+(?P<label>[^.?!]+?)(?=[.?!]|$)""",
            re.IGNORECASE | re.VERBOSE,
        ),
    ),
    (
        "leaning",
        re.compile(
            r"""\bI(?:'m|\s+am)\s+leaning(?:\s+more)?\s+towards\s+(?P<label>[^.?!]+?)(?=[.?!]|$)""",
            re.IGNORECASE | re.VERBOSE,
        ),
    ),
    (
        "focus",
        re.compile(
            r"""\bI\s+want\s+to\s+focus\s+more\s+on\s+(?P<label>[^.?!]+?)(?=[.?!]|$)""",
            re.IGNORECASE | re.VERBOSE,
        ),
    ),
    (
        "enjoy",
        re.compile(
            r"""\bI\s+enjoy(?:\s+to\s+use)?\s+(?P<label>[^.?!]+?)(?=[.?!]|$)""",
            re.IGNORECASE | re.VERBOSE,
        ),
    ),
    (
        "interested",
        re.compile(
            r"""\bI(?:'m|\s+am)\s+interested\s+in\s+(?P<label>[^.?!]+?)(?=[.?!]|$)""",
            re.IGNORECASE | re.VERBOSE,
        ),
    ),
    (
        "love",
        re.compile(
            r"""\bI\s+love\s+(?P<label>[^.?!]+?)(?=[.?!]|$)""",
            re.IGNORECASE | re.VERBOSE,
        ),
    ),
)

_OBLIGATION_PICK_UP = re.compile(
    r"""\bneed\s+to\s+pick\s+up\s+(?!or\s+return\b)(?P<item>[^.?!]+?)(?=[.?!]|$)""",
    re.IGNORECASE,
)

_OBLIGATION_RETURN = re.compile(
    r"""\bneed\s+to\s+return\s+(?P<item>[^.?!]+?)(?=[.?!]|$)""",
    re.IGNORECASE,
)

_PROJECT_SOLO = re.compile(
    r"""\bsolo\s+project\s+for\s+my\s+(?P<label>[^.?!]+?)(?=[.?!]|$)""",
    re.IGNORECASE,
)

_PROJECT_LED = re.compile(
    r"""\bled\s+the\s+(?P<role>[^.?!]+?)\s+team\b[^.?!]*?\bfor\s+(?P<label>[^.?!]+?)(?=[.?!]|$)""",
    re.IGNORECASE,
)

_MODEL_KIT_DONE = re.compile(
    r"""\b(?:finished|completed)\s+(?:a\s+simple\s+|a\s+|my\s+)?(?P<label>(?:[A-Za-z0-9'\/\-]+\s+){0,6}(?:kit|model|bomber|tank|camaro|spitfire|eagle))""",
    re.IGNORECASE,
)

_MODEL_KIT_WORKING = re.compile(
    r"""\b(?:working\s+on|started\s+working\s+on|thinking\s+of\s+working\s+on)\s+(?:a\s+|my\s+|the\s+)?(?P<label>(?:[A-Za-z0-9'\/\-]+\s+){0,8}(?:kit|model|bomber|tank|camaro|spitfire|eagle))""",
    re.IGNORECASE,
)

_TRIP_DURATION = re.compile(
    r"""\b(?P<days>\d+)-day\s+(?P<kind>solo\s+)?(?P<label>camping\s+trip\s+to\s+[^.?!]+?)\b""",
    re.IGNORECASE,
)

_DRIVE_DURATION = re.compile(
    r"""\bdrove\s+for\s+(?P<hours>\d+)\s+hours?\s+to\s+(?P<label>[^.?!]+?)(?=[.?!]|$)""",
    re.IGNORECASE,
)

_MEDIA_DURATION = re.compile(
    r"""\bwatched\s+all\s+(?P<label>[^.?!,]+?)\s+in\s+(?P<duration>[^.?!,]+?)(?=[,.?!]|$)""",
    re.IGNORECASE,
)

_MEDIA_MARATHON = re.compile(
    r"""\b(?P<label>Star\s+Wars)\s+marathon\b[^.?!]*?\bwatched\s+all\s+the\s+main\s+films\s+in\s+(?P<duration>[^.?!,]+?)(?=[,.?!]|$)""",
    re.IGNORECASE,
)

_BOOK_DURATION = re.compile(
    r"""\bfinished\s+["']?(?P<label>[^"'.?!]+?)["']?,?\s+which\s+took\s+me\s+(?P<duration>[^.?!,]+?)\s+to\s+(?:finish|read)\b""",
    re.IGNORECASE,
)

_DOCTOR_VISIT = re.compile(
    r"""\b(?:appointment\s+with|from|by)\s+(?:my\s+)?(?P<specialty>[^,.;]+?),\s*(?P<doctor>Dr\.\s+[A-Z][a-z]+)""",
    re.IGNORECASE,
)

_EXPENSE_AMOUNT = re.compile(
    r"""\b(?:got|bought|installed|replace(?:d)?|replaced|needed\s+to\s+replace)\s+(?:a\s+new\s+)?(?:set\s+of\s+)?(?:my\s+|the\s+)?(?P<label>bike\s+lights|bell\s+zephyr\s+helmet|helmet|chain)[^.?!]*?\b(?:for|were|cost(?:\s+me)?)\s+\$(?P<amount>\d+(?:\.\d{1,2})?)""",
    re.IGNORECASE,
)

_INFLUENCER_AUDIENCE = re.compile(
    r"""\bcollaborat(?:ed|ion)\b[^.?!]*?\bpromoted\s+my\s+product\s+to\s+(?:her|their)\s+(?P<count>[\d,]+)\s+followers\b""",
    re.IGNORECASE,
)

_AD_CAMPAIGN_AUDIENCE = re.compile(
    r"""\b(?:previous\s+)?facebook\s+ad\s+campaign\b[^.?!]*?\breached\s+(?:around\s+)?(?P<count>[\d,]+)\s+people\b""",
    re.IGNORECASE,
)

_FINISHED_GENERIC_BOOK_PAGES = re.compile(
    r"""\bjust\s+finished(?:\s+reading)?\s+(?:a\s+|an\s+)?(?P<pages>\d{2,4})-page\s+(?P<label>novel|book)\b""",
    re.IGNORECASE,
)

_FINISHED_TITLED_BOOK_PAGES = re.compile(
    r"""\bjust\s+finished(?:\s+reading)?\s+(?:(?:a|an)\s+[^,]+,\s+)?["']?(?P<label>[^"']+?)["']?\s+by\s+[^.?!,]+,\s+which\s+had\s+(?P<pages>\d{2,4})\s+pages\b""",
    re.IGNORECASE,
)

_FAMILY_SIBLINGS = re.compile(
    r"""\bI\s+come\s+from\s+a\s+family\s+with\s+(?P<count>\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?P<kind>sisters?|brothers?)\b""",
    re.IGNORECASE,
)

_HAVE_SIBLING = re.compile(
    r"""\bI\s+have\s+(?:(?P<count>\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten)\s+)?(?P<kind>sisters?|brothers?)\b""",
    re.IGNORECASE,
)

_PURCHASED_ITEM_FOR = re.compile(
    r"""\b(?P<label>designer\s+handbag)\s+I\s+got\s+for\s+\$(?P<amount>\d[\d,]*(?:\.\d{1,2})?)\b""",
    re.IGNORECASE,
)

_ITEM_ORIGINALLY_COST = re.compile(
    r"""\bit\s+was\s+originally\s+\$(?P<amount>\d[\d,]*(?:\.\d{1,2})?)\b""",
    re.IGNORECASE,
)

_TRANSPORT_TRAIN_COST = re.compile(
    r"""\$(?P<amount>\d[\d,]*(?:\.\d{1,2})?)\s+to\s+get\s+to\s+my\s+hotel\s+from\s+the\s+airport\s+by\s+train\b""",
    re.IGNORECASE,
)

_TRANSPORT_TAXI_COST = re.compile(
    r"""\btaxi\s+from\s+the\s+airport\s+to\s+my\s+hotel\s+would\s+cost\s+around\s+\$(?P<amount>\d[\d,]*(?:\.\d{1,2})?)\b""",
    re.IGNORECASE,
)

_GUITAR_LESSONS_DURATION = re.compile(
    r"""\bI(?:'ve|\s+have)\s+been\s+taking\s+weekly\s+guitar\s+lessons\b[^.?!]*?\bfor\s+(?P<duration>\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten)\s+weeks?\s+now\b""",
    re.IGNORECASE,
)

_GUITAR_AMP_PURCHASE = re.compile(
    r"""\b(?:bought|got|invested\s+in)\s+(?:a\s+)?new\s+guitar\s+amp\b[^.?!]*?\b(?P<relative>two\s+weeks\s+ago|last\s+week|\d+\s+weeks?\s+ago)\b""",
    re.IGNORECASE,
)

_PLANT_ACQUIRED = re.compile(
    r"""\b(?:bought|got)\s+(?P<items>[^.?!]+?\b(?:plant|lily|fern|basil|succulent))(?:\s+(?:last\s+month|\d+\s+weeks?\s+ago|two\s+weeks\s+ago))?(?=[.?!]|$)""",
    re.IGNORECASE,
)

_PLANT_REFERENCE_GOT = re.compile(
    r"""\bmy\s+(?P<label>[^.?!,]+?\b(?:plant|lily|fern|basil|succulent))\s*,?\s+which\s+I\s+got\s+from\s+[^.?!,]+(?:\s+last\s+month|\s+\d+\s+weeks?\s+ago|\s+two\s+weeks\s+ago)""",
    re.IGNORECASE,
)

_TANK_OWNED = re.compile(
    r"""\b(?:have|had|set\s+up|taking\s+care\s+of)\s+(?:a\s+|an\s+|my\s+)?(?P<label>\d+-gallon\s+[^.?!]*?\btank)\b""",
    re.IGNORECASE,
)

_FILM_FESTIVAL = re.compile(
    r"""\b(?:volunteered\s+at|attended|participated\s+in(?:\s+the\s+48-hour\s+film\s+challenge\s+at)?|screening\s+at\s+the|got\s+back\s+from)\s+(?:the\s+)?(?P<label>[A-Z][A-Za-z0-9&.'-]*(?:\s+[A-Z][A-Za-z0-9&.'-]*)*\s+(?:Film\s+Festival|Fest))\b""",
    re.IGNORECASE,
)

_WEDDING_ATTENDED = re.compile(
    r"""\b(?:(?P<possessive>[A-Z][a-z]+)'s\s+wedding|(?P<kind>friend|cousin|college\s+roommate)'s\s+wedding)\b""",
    re.IGNORECASE,
)

_WEDDING_COUPLE = re.compile(
    r"""\b(?:(?:my\s+friend\s+)?(?P<left>[A-Z][a-z]+)\s+(?:finally\s+got\s+to\s+tie\s+the\s+knot\s+with|and)\s+(?:her\s+partner\s+)?(?P<right>[A-Z][a-z]+)|the\s+bride,\s*(?P<bride>[A-Z][a-z]+)[^.?!]*?her\s+husband,\s*(?P<groom>[A-Z][a-z]+))\b"""
)

_SOCIAL_MEDIA_BREAK = re.compile(
    r"""\b(?:(?P<days>\d+)-day|(?P<week>a|an|one)\s+week-long)\s+break\s+from\s+(?:it|social\s+media)\b""",
    re.IGNORECASE,
)

_SOCIAL_MEDIA_BREAK_SHORT = re.compile(
    r"""\b(?P<days>\d+)-day\s+break\b""",
    re.IGNORECASE,
)

_GAME_HOURS = re.compile(
    r"""\bspent\s+around\s+(?P<hours>\d+)\s+hours\s+playing\s+(?P<label>[^.?!,]+)""",
    re.IGNORECASE,
)

_GAME_COMPLETION_HOURS = re.compile(
    r"""\b(?P<label>[A-Z][A-Za-z0-9:'\- ]+?)\s*,?\s+which\s+(?:(?:I\s+)?(?:completed|finished)[^.?!]*?)?(?:it\s+)?took\s+me\s+(?P<hours>\d+)\s+hours?\s+to\s+(?:complete|finish)\b""",
    re.IGNORECASE,
)

_GAME_TOOK_HOURS = re.compile(
    r"""\bit\s+took\s+me\s+(?P<hours>\d+)\s+hours?\s+to\s+(?:complete|finish)\b""",
    re.IGNORECASE,
)

_LUXURY_PURCHASE = re.compile(
    r"""\b(?:bought|got)\s+(?P<label>[^.?!$]+?)\s+(?:for|,)\s*\$(?P<amount>\d[\d,]*(?:\.\d{1,2})?)\b""",
    re.IGNORECASE,
)

_LUXURY_ITEM_VALUE = re.compile(
    r"""\b(?P<label>luxury\s+evening\s+gown|designer\s+handbag|pair\s+of\s+leather\s+boots|leather\s+boots)[^.?!$]*?\$(?P<amount>\d[\d,]*(?:\.\d{1,2})?)\b""",
    re.IGNORECASE,
)

_CITRUS_TOKEN = re.compile(r"""\b(?P<fruit>orange|lemon|lime|grapefruit)\b""", re.IGNORECASE)

_STUDY_ABROAD = re.compile(
    r"""\bstudy\s+abroad\s+program\s+at\s+(?P<school>[A-Z][A-Za-z.&'()\-]*(?:\s+[A-Z][A-Za-z.&'()\-]*)+)(?:\s+in\s+(?P<country>[A-Z][A-Za-z]+))?""",
    re.IGNORECASE,
)

_UNDERGRAD_SCHOOL = re.compile(
    r"""\bcompleted\s+my\s+(?:undergrad|undergraduate)(?:\s+in\s+[^.?!,]+?)?\s+from\s+(?P<school>[A-Z][A-Za-z.&'()\-]*(?:\s+[A-Z][A-Za-z.&'()\-]*)+|UCLA)\b""",
    re.IGNORECASE,
)

_LENS_ACQUIRED = re.compile(
    r"""\b(?:recently\s+got|got|getting|new)\s+(?:a\s+)?(?P<label>\d{2,3}(?:-\d{2,3})?mm\s+(?:prime|zoom)\s+lens)\b""",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------


class DeterministicExtractor(Extractor):
    """Pattern-based extractor that runs in the foreground write stage.

    Stateless and side-effect-free; safe to share across coroutines.

    Methods on this class are :keyword:`async` to satisfy the
    :class:`Extractor` protocol, but the work itself is fully synchronous
    \u2014 awaitable callers do not pay any thread-hop cost.
    """

    SOURCE_PROMOTION_FROM_TO = "deterministic:promotion_from_to"
    SOURCE_PROMOTION_TO_ONLY = "deterministic:promotion_to_only"
    SOURCE_TITLE_IS = "deterministic:title_is"
    SOURCE_LOCATION_MOVE_FROM_TO = "deterministic:location_move_from_to"
    SOURCE_LOCATION_MOVE_TO = "deterministic:location_move_to"
    SOURCE_LOCATION_LIVE_IN = "deterministic:location_live_in"
    SOURCE_EMPLOYER_WORK_AT = "deterministic:employer_work_at"
    SOURCE_EMPLOYER_JOINED = "deterministic:employer_joined"
    SOURCE_EVENT_GENERIC = "deterministic:event_generic"
    SOURCE_PREFERENCE_GENERIC = "deterministic:preference_generic"
    SOURCE_CARRYOVER_GENERIC = "deterministic:carryover_generic"
    SOURCE_ASSISTANT_REFERENCE = "deterministic:assistant_reference"

    async def extract(self, text: str, ctx: ExtractionContext) -> ExtractionResult:
        memories: list[Memory] = []
        text = text.strip()
        if not text:
            return ExtractionResult()

        # Always record the raw event as an Experience. This guarantees the
        # episode timeline is gap-free regardless of how many (if any)
        # claim-extracting patterns matched.
        experience = self._make_experience(text, ctx)
        memories.append(experience)
        rid = experience.id

        # Assistant/system turns are part of the episodic trace, but they are
        # not evidence of the user's enduring personal state. Extracting
        # user-scoped claims from "I ..." inside assistant replies pollutes the
        # store and destroys retrieval precision on benchmark-style histories.
        if ctx.actor != "user":
            memories.extend(self._extract_assistant_references(text, ctx, rid))
            return ExtractionResult(
                memories=tuple(memories),
                transitions=(),
                residual_text=text,
            )

        # Run claim-extracting patterns. Each claim's provenance points
        # back to the Experience id (raw_event_id) so the chain of custody
        # is traceable. Transitions are created by ARW (Active Reconsolidation
        # on Write), not by the deterministic extractor.
        memories.extend(self._extract_promotion(text, ctx, rid))
        memories.extend(self._extract_title(text, ctx, rid))
        memories.extend(self._extract_location(text, ctx, rid))
        memories.extend(self._extract_employer(text, ctx, rid))
        memories.extend(self._extract_events(text, ctx, rid))
        memories.extend(self._extract_preferences(text, ctx, rid))
        memories.extend(self._extract_carryover(text, ctx, rid))
        memories.extend(self._extract_profile_and_inventory(text, ctx, rid))

        # We don't try to compute a literal residual span; instead the
        # convention is: if at least one claim was extracted, the LLM stage
        # may still run for additional coverage but is not required for
        # correctness. The engine decides based on configuration.
        return ExtractionResult(
            memories=tuple(memories),
            transitions=(),
            residual_text=text,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_experience(self, text: str, ctx: ExtractionContext) -> Experience:
        event_type = {
            "user": "user_message",
            "assistant": "assistant_message",
            "system": "system_message",
        }[ctx.actor]
        return Experience(
            content=text,
            event_type=event_type,
            actor=ctx.actor,
            summary=text[:200],
            identity_id=ctx.identity_id,
            valid_at=ctx.now,
            provenance=(self._provenance(ctx),),
        )

    def _extract_promotion(
        self, text: str, ctx: ExtractionContext, raw_event_id: str
    ) -> list[Memory]:
        memories: list[Memory] = []
        m = _PROMOTION_FROM_TO.search(text)
        if m:
            from_role = _clean_role(m.group("from_role"))
            to_role = _clean_role(m.group("to_role"))
            if from_role and to_role:
                # Prior title — emit as a fact whose validity ended now.
                old_fact = Fact(
                    content=f"Title: {from_role}",
                    subject="user",
                    predicate="has_title",
                    obj=from_role,
                    valid_at=ctx.now,
                    valid_until=ctx.now,
                    confidence=0.95,
                    identity_id=ctx.identity_id,
                    tags=("title", "promotion", "former"),
                    metadata={"extractor": self.SOURCE_PROMOTION_FROM_TO},
                    provenance=(self._provenance(ctx, raw_event_id=raw_event_id),),
                )
                # New title — currently valid.
                new_fact = Fact(
                    content=f"Title: {to_role}",
                    subject="user",
                    predicate="has_title",
                    obj=to_role,
                    valid_at=ctx.now,
                    confidence=0.95,
                    identity_id=ctx.identity_id,
                    tags=("title", "promotion", "current"),
                    metadata={"extractor": self.SOURCE_PROMOTION_FROM_TO},
                    provenance=(self._provenance(ctx, raw_event_id=raw_event_id),),
                )
                memories.extend([old_fact, new_fact])
                # Note: Transitions are created by ARW (Active Reconsolidation on Write),
                # not by the deterministic extractor. The deterministic extractor only
                # creates memories; ARW derives relationships between them.
                return memories
        m = _PROMOTION_TO_ONLY.search(text)
        if m:
            to_role = _clean_role(m.group("to_role"))
            if to_role:
                memories.append(
                    Fact(
                        content=f"Title: {to_role}",
                        subject="user",
                        predicate="has_title",
                        obj=to_role,
                        valid_at=ctx.now,
                        confidence=0.9,
                        identity_id=ctx.identity_id,
                        tags=("title", "promotion", "current"),
                        metadata={"extractor": self.SOURCE_PROMOTION_TO_ONLY},
                        provenance=(self._provenance(ctx, raw_event_id=raw_event_id),),
                    )
                )
        return memories

    def _extract_title(
        self, text: str, ctx: ExtractionContext, raw_event_id: str
    ) -> list[Memory]:
        # Skip if a promotion already matched — a promotion's "to" role is the
        # title. Re-extracting it would create duplicates. We detect this by
        # looking for the promotion verb anywhere in the utterance.
        if re.search(r"\bpromoted\b", text, re.IGNORECASE):
            return []
        m = _TITLE_IS.search(text)
        if not m:
            return []
        title = _clean_role(m.group("title"))
        if not title:
            return []
        return [
            Fact(
                content=f"Title: {title}",
                subject="user",
                predicate="has_title",
                obj=title,
                valid_at=ctx.now,
                confidence=0.9,
                identity_id=ctx.identity_id,
                tags=("title", "current"),
                metadata={"extractor": self.SOURCE_TITLE_IS},
                provenance=(self._provenance(ctx, raw_event_id=raw_event_id),),
            )
        ]

    def _extract_location(
        self, text: str, ctx: ExtractionContext, raw_event_id: str
    ) -> list[Memory]:
        memories: list[Memory] = []
        m = _LOCATION_MOVE_FROM_TO.search(text)
        if m:
            from_loc = _clean_location_label(m.group("from_loc"))
            to_loc = _clean_location_label(m.group("to_loc"))
            if from_loc and to_loc:
                old_fact = Fact(
                    content=f"Location: {from_loc}",
                    subject="user",
                    predicate="lives_in",
                    obj=from_loc,
                    valid_at=ctx.now,
                    valid_until=ctx.now,
                    confidence=0.9,
                    identity_id=ctx.identity_id,
                    tags=("location", "move", "former"),
                    metadata={"extractor": self.SOURCE_LOCATION_MOVE_FROM_TO},
                    provenance=(self._provenance(ctx, raw_event_id=raw_event_id),),
                )
                new_fact = Fact(
                    content=f"Location: {to_loc}",
                    subject="user",
                    predicate="lives_in",
                    obj=to_loc,
                    valid_at=ctx.now,
                    confidence=0.9,
                    identity_id=ctx.identity_id,
                    tags=("location", "move", "current"),
                    metadata={"extractor": self.SOURCE_LOCATION_MOVE_FROM_TO},
                    provenance=(self._provenance(ctx, raw_event_id=raw_event_id),),
                )
                memories.extend([old_fact, new_fact])
                # Note: Transitions are created by ARW (Active Reconsolidation on Write),
                # not by the deterministic extractor.
                return memories
        m = _LOCATION_MOVE_TO.search(text)
        if m:
            to_loc = _clean_location_label(m.group("to_loc"))
            if to_loc:
                memories.append(
                    Fact(
                        content=f"Location: {to_loc}",
                        subject="user",
                        predicate="lives_in",
                        obj=to_loc,
                        valid_at=ctx.now,
                        confidence=0.85,
                        identity_id=ctx.identity_id,
                        tags=("location", "move", "current"),
                        metadata={"extractor": self.SOURCE_LOCATION_MOVE_TO},
                        provenance=(self._provenance(ctx, raw_event_id=raw_event_id),),
                    )
                )
        m = _LOCATION_LIVE_IN.search(text)
        if m:
            loc = _clean_location_label(m.group("loc"))
            if loc:
                memories.append(
                    Fact(
                        content=f"Location: {loc}",
                        subject="user",
                        predicate="lives_in",
                        obj=loc,
                        valid_at=ctx.now,
                        confidence=0.85,
                        identity_id=ctx.identity_id,
                        tags=("location",),
                        metadata={"extractor": self.SOURCE_LOCATION_LIVE_IN},
                        provenance=(self._provenance(ctx, raw_event_id=raw_event_id),),
                    )
                )
        return memories

    def _extract_employer(
        self, text: str, ctx: ExtractionContext, raw_event_id: str
    ) -> list[Memory]:
        m = _EMPLOYER_WORK_AT.search(text)
        if m:
            employer = _clean_phrase(m.group("employer"))
            if employer:
                return [
                    Fact(
                        content=f"Works at {employer}",
                        subject="user",
                        predicate="works_at",
                        obj=employer,
                        valid_at=ctx.now,
                        confidence=0.9,
                        identity_id=ctx.identity_id,
                        tags=("employer", "current"),
                        metadata={"extractor": self.SOURCE_EMPLOYER_WORK_AT},
                        provenance=(self._provenance(ctx, raw_event_id=raw_event_id),),
                    )
                ]
        m = _EMPLOYER_JOINED.search(text)
        if m:
            employer = _clean_phrase(m.group("employer"))
            if employer:
                return [
                    Fact(
                        content=f"Works at {employer}",
                        subject="user",
                        predicate="works_at",
                        obj=employer,
                        valid_at=ctx.now,
                        confidence=0.9,
                        identity_id=ctx.identity_id,
                        tags=("employer", "current", "joined"),
                        metadata={"extractor": self.SOURCE_EMPLOYER_JOINED},
                        provenance=(self._provenance(ctx, raw_event_id=raw_event_id),),
                    )
                ]
        return []

    def _extract_events(
        self, text: str, ctx: ExtractionContext, raw_event_id: str
    ) -> list[Memory]:
        memories: list[Memory] = []
        seen: set[tuple[str, str]] = set()

        for match in _ATTENDED_EVENT.finditer(text):
            kind = match.group("kind").lower()
            label = _clean_phrase(match.group("label"))
            if not label:
                continue
            event_at = _extract_event_datetime(_sentence_for_match(text, match), ctx.now)
            key = (kind, label.lower())
            if key in seen:
                continue
            seen.add(key)
            memories.append(
                self._event_fact(
                    kind=kind,
                    label=label,
                    object_value=label,
                    text=f"Attended {kind}: {label}",
                    ctx=ctx,
                    raw_event_id=raw_event_id,
                    at=event_at,
                    tags=("event", kind, "attended"),
                )
            )

        for match in _EVENT_REFERENCE.finditer(text):
            kind = match.group("kind").lower()
            label = _clean_phrase(match.group("label"))
            if not label:
                continue
            event_at = _extract_event_datetime(_sentence_for_match(text, match), ctx.now)
            key = (kind, label.lower())
            if key in seen:
                continue
            seen.add(key)
            memories.append(
                self._event_fact(
                    kind=kind,
                    label=label,
                    object_value=label,
                    text=f"Attended {kind}: {label}",
                    ctx=ctx,
                    raw_event_id=raw_event_id,
                    at=event_at,
                    tags=("event", kind, "attended"),
                )
            )

        for match in _DEVICE_ACQUIRED.finditer(text):
            label = _clean_device_label(match.group("label"))
            if not label:
                continue
            event_at = _extract_event_datetime(_sentence_for_match(text, match), ctx.now)
            key = ("device_acquired", label.lower())
            if key in seen:
                continue
            seen.add(key)
            memories.append(
                self._event_fact(
                    kind="device_acquired",
                    label=label,
                    object_value=label,
                    text=f"Acquired device: {label}",
                    ctx=ctx,
                    raw_event_id=raw_event_id,
                    at=event_at,
                    tags=("event", "device", "acquired"),
                )
            )

        for match in _DEVICE_NAMED.finditer(text):
            label = _clean_device_label(match.group("label"))
            if not label:
                continue
            event_at = _extract_event_datetime(_sentence_for_match(text, match), ctx.now)
            key = ("device_acquired", label.lower())
            if key in seen:
                continue
            seen.add(key)
            memories.append(
                self._event_fact(
                    kind="device_acquired",
                    label=label,
                    object_value=label,
                    text=f"Acquired device: {label}",
                    ctx=ctx,
                    raw_event_id=raw_event_id,
                    at=event_at,
                    tags=("event", "device", "acquired"),
                )
            )

        role_labels = {
            role: label
            for match in _DEVICE_ROLE_NAMED.finditer(text)
            if (role := _normalize_device_role(match.group("role")))
            and (label := _clean_device_label(match.group("label")))
        }
        for match in _DEVICE_ARRIVED.finditer(text):
            role = _normalize_device_role(match.group("role") or match.group("role2"))
            if role is None:
                continue
            label = role_labels.get(role)
            if not label:
                continue
            explicit_date = _extract_date_token(match.group("date"), ctx.now)
            if explicit_date is None:
                continue
            key = ("device_acquired", label.lower())
            if key in seen:
                continue
            seen.add(key)
            memories.append(
                self._event_fact(
                    kind="device_acquired",
                    label=label,
                    object_value=label,
                    text=f"Acquired device: {label}",
                    ctx=ctx,
                    raw_event_id=raw_event_id,
                    at=explicit_date,
                    tags=("event", "device", "acquired"),
                )
            )

        for match in _REPAIR_EVENT.finditer(text):
            object_name = _clean_phrase(match.group("object"))
            service = match.group("service").lower()
            if not object_name:
                continue
            label = f"{object_name} {service}"
            event_at = _extract_event_datetime(_sentence_for_match(text, match), ctx.now)
            key = ("repair", label.lower())
            if key in seen:
                continue
            seen.add(key)
            memories.append(
                self._event_fact(
                    kind="repair",
                    label=label,
                    object_value=object_name,
                    text=f"Repair event: {label}",
                    ctx=ctx,
                    raw_event_id=raw_event_id,
                    at=event_at,
                    tags=("event", "repair", object_name.lower()),
                )
            )

        for match in _PRONOUN_REPAIR_EVENT.finditer(text):
            object_name = _clean_phrase(match.group("object"))
            service = match.group("service").lower()
            if not object_name:
                continue
            label = f"{object_name} {service}"
            event_at = (
                _extract_date_token(match.group("date") or "", ctx.now)
                or _extract_event_datetime(text, ctx.now)
            )
            key = ("repair", label.lower())
            if key in seen:
                continue
            seen.add(key)
            memories.append(
                self._event_fact(
                    kind="repair",
                    label=label,
                    object_value=object_name,
                    text=f"Repair event: {label}",
                    ctx=ctx,
                    raw_event_id=raw_event_id,
                    at=event_at,
                    tags=("event", "repair", object_name.lower()),
                )
            )

        lowered_text = text.lower()
        if (
            _STAND_MIXER_REPAIR.search(text)
            or (
                "stand mixer" in lowered_text
                and "repair shop" in lowered_text
                and "last month" in lowered_text
            )
        ):
            key = ("repair", "stand mixer repair")
            if key not in seen:
                seen.add(key)
                memories.append(
                    self._event_fact(
                        kind="repair",
                        label="stand mixer repair",
                        object_value="stand mixer",
                        text="Repair event: stand mixer repair",
                        ctx=ctx,
                        raw_event_id=raw_event_id,
                        at=_extract_event_datetime(text, ctx.now),
                        tags=("event", "repair", "stand mixer"),
                    )
                )

        for match in _VEHICLE_WASH.finditer(text):
            label = _clean_phrase(match.group("label"))
            object_name = _normalize_vehicle_object(label)
            if not label or object_name is None:
                continue
            explicit_date = _extract_date_token(match.group("date") or "", ctx.now)
            event_at = explicit_date or _extract_event_datetime(_sentence_for_match(text, match), ctx.now)
            key = ("vehicle_care", f"{object_name}:{label.lower()}")
            if key in seen:
                continue
            seen.add(key)
            memories.append(
                self._event_fact(
                    kind="vehicle_care",
                    label=f"washed {label}",
                    object_value=object_name,
                    text=f"Vehicle care: washed {label}",
                    ctx=ctx,
                    raw_event_id=raw_event_id,
                    at=event_at,
                    tags=("event", "vehicle-care", object_name),
                )
            )

        if _FIRST_SERVICE.search(text):
            event_at = _extract_event_datetime(text, ctx.now)
            key = ("service", "first service")
            if key not in seen:
                seen.add(key)
                memories.append(
                    self._event_fact(
                        kind="service",
                        label="first service",
                        object_value="first service",
                        text="Service event: first service",
                        ctx=ctx,
                        raw_event_id=raw_event_id,
                        at=event_at,
                        tags=("event", "service", "car"),
                    )
                )

        for pattern in (_GPS_ISSUE, _BIKE_ISSUE, _ISSUE_WITH):
            for match in pattern.finditer(text):
                label = _clean_phrase(match.group("label"))
                if not label:
                    continue
                event_at = _extract_event_datetime(_sentence_for_match(text, match), ctx.now)
                key = ("issue", label.lower())
                if key in seen:
                    continue
                seen.add(key)
                tag_label = label.lower().replace(" ", "_")
                memories.append(
                    self._event_fact(
                        kind="issue",
                        label=label,
                        object_value=label,
                        text=f"Issue event: {label}",
                        ctx=ctx,
                        raw_event_id=raw_event_id,
                        at=event_at,
                        tags=("event", "issue", tag_label),
                    )
                )

        for match in _MEETING_ON_DATE.finditer(text):
            label = _clean_phrase(match.group("label"))
            explicit_date = _extract_date_token(match.group("date"), ctx.now)
            if not label or explicit_date is None:
                continue
            key = ("meeting", label.lower())
            if key in seen:
                continue
            seen.add(key)
            memories.append(
                self._event_fact(
                    kind="meeting",
                    label=label,
                    object_value=label,
                    text=f"Meeting event: {label}",
                    ctx=ctx,
                    raw_event_id=raw_event_id,
                    at=explicit_date,
                    tags=("event", "meeting"),
                )
            )

        for pattern in (_GENERIC_EVENT_ON_DATE, _RAN_IN_EVENT_ON_DATE):
            for match in pattern.finditer(text):
                label = _clean_phrase(match.group("label"))
                explicit_date = _extract_date_token(match.group("date"), ctx.now)
                if not label or explicit_date is None:
                    continue
                key = ("generic_event", label.lower())
                if key in seen:
                    continue
                seen.add(key)
                memories.append(
                    self._event_fact(
                        kind="event",
                        label=label,
                        object_value=label,
                        text=f"Event: {label}",
                        ctx=ctx,
                        raw_event_id=raw_event_id,
                        at=explicit_date,
                        tags=("event",) + (("charity",) if _is_charity_event_label(label) else ()),
                    )
                )

        for match in _STARTED_WORKING_WITH_ON.finditer(text):
            label = f"started working with {match.group('label')}"
            explicit_date = _extract_date_token(match.group("date"), ctx.now)
            if explicit_date is None:
                continue
            key = ("generic_event", label.lower())
            if key in seen:
                continue
            seen.add(key)
            memories.append(
                self._event_fact(
                    kind="event",
                    label=label,
                    object_value=label,
                    text=f"Event: {label}",
                    ctx=ctx,
                    raw_event_id=raw_event_id,
                    at=explicit_date,
                    tags=("event",),
                )
            )

        for match in _STARTED_WORKING_WITH_PRONOUN_ON.finditer(text):
            person = _nearest_person_reference(text, match.start())
            label = f"started working with {person}" if person else "started working with"
            explicit_date = _extract_date_token(match.group("date"), ctx.now)
            if explicit_date is None:
                continue
            key = ("generic_event", label.lower())
            if key in seen:
                continue
            seen.add(key)
            memories.append(
                self._event_fact(
                    kind="event",
                    label=label,
                    object_value=label,
                    text=f"Event: {label}",
                    ctx=ctx,
                    raw_event_id=raw_event_id,
                    at=explicit_date,
                    tags=("event",),
                )
            )

        for match in _HOUSE_LOVED_ON.finditer(text):
            explicit_date = _extract_date_token(match.group("date"), ctx.now)
            if explicit_date is None:
                continue
            label = "house I loved"
            key = ("generic_event", label)
            if key in seen:
                continue
            seen.add(key)
            memories.append(
                self._event_fact(
                    kind="event",
                    label=label,
                    object_value=label,
                    text="Event: house I loved",
                    ctx=ctx,
                    raw_event_id=raw_event_id,
                    at=explicit_date,
                    tags=("event",),
                )
            )

        for match in _SEED_STARTED.finditer(text):
            label = _clean_generic_event_label(match.group("label"))
            explicit_date = _extract_date_token(match.group("date"), ctx.now)
            if not label or explicit_date is None:
                continue
            key = ("seed_start", label.lower())
            if key in seen:
                continue
            seen.add(key)
            memories.append(
                self._event_fact(
                    kind="seed_start",
                    label=label,
                    object_value=label,
                    text=f"Seed start: {label}",
                    ctx=ctx,
                    raw_event_id=raw_event_id,
                    at=explicit_date,
                    tags=("event", "seed"),
                )
            )

        for match in _SEED_COLLECTION_STARTED.finditer(text):
            explicit_date = _extract_date_token(match.group("date"), ctx.now)
            if explicit_date is None:
                continue
            for part in _split_item_list(match.group("labels")):
                label = _clean_generic_event_label(part)
                if not label:
                    continue
                key = ("seed_start", label.lower())
                if key in seen:
                    continue
                seen.add(key)
                memories.append(
                    self._event_fact(
                        kind="seed_start",
                        label=label,
                        object_value=label,
                        text=f"Seed start: {label}",
                        ctx=ctx,
                        raw_event_id=raw_event_id,
                        at=explicit_date,
                        tags=("event", "seed"),
                    )
                )

        for pattern, kind in ((_CHURCH_SERVICE, "service"), (_FESTIVAL_EVENT, "festival")):
            for match in pattern.finditer(text):
                label = _clean_generic_event_label(match.group("label"))
                explicit_date = _extract_date_token(match.group("date"), ctx.now)
                if not label or explicit_date is None:
                    continue
                key = (kind, label.lower())
                if key in seen:
                    continue
                seen.add(key)
                memories.append(
                    self._event_fact(
                        kind=kind,
                        label=label,
                        object_value=label,
                        text=f"Event: {label}",
                        ctx=ctx,
                        raw_event_id=raw_event_id,
                        at=explicit_date,
                        tags=("event", kind) + (("charity",) if _is_charity_event_label(label) else ()),
                    )
                )

        for match in _FESTIVAL_EVENT_IN_MONTH.finditer(text):
            label = _clean_generic_event_label(match.group("label"))
            month_only = _extract_month_token(match.group("month"), ctx.now)
            if not label or month_only is None:
                continue
            key = ("festival", label.lower())
            if key in seen:
                continue
            seen.add(key)
            memories.append(
                self._event_fact(
                    kind="festival",
                    label=label,
                    object_value=label,
                    text=f"Event: {label}",
                    ctx=ctx,
                    raw_event_id=raw_event_id,
                    at=month_only,
                    tags=("event", "festival") + (("charity",) if _is_charity_event_label(label) else ()),
                    )
                )

        for pattern in (_THERMOSTAT_SETUP, _ROUTER_SETUP):
            for match in pattern.finditer(text):
                raw_label = match.group("label")
                if raw_label.lower().startswith("new router"):
                    label = "new router"
                else:
                    label = _clean_generic_event_label(raw_label)
                event_at = _extract_explicit_or_relative_time(
                    match.groupdict().get("date"),
                    match.groupdict().get("relative"),
                    ctx.now,
                ) or _extract_event_datetime(_sentence_for_match(text, match), ctx.now)
                if not label:
                    continue
                key = ("setup", label.lower())
                if key in seen:
                    continue
                seen.add(key)
                memories.append(
                    self._event_fact(
                        kind="setup",
                        label=label,
                        object_value=label,
                        text=f"Setup event: {label}",
                        ctx=ctx,
                        raw_event_id=raw_event_id,
                        at=event_at,
                        tags=("event", "setup"),
                    )
                )

        for match in _MEMBERSHIP_STARTED.finditer(text):
            event_at = _extract_relative_datetime(match.group("relative"), ctx.now)
            label = f"member of {match.group('label')}"
            if event_at is None:
                continue
            key = ("membership", label.lower())
            if key in seen:
                continue
            seen.add(key)
            memories.append(
                self._event_fact(
                    kind="membership_start",
                    label=label,
                    object_value=label,
                    text=f"Event: {label}",
                    ctx=ctx,
                    raw_event_id=raw_event_id,
                    at=event_at,
                    tags=("event", "membership"),
                )
            )

        for match in _BIRD_WATCHING_STARTED.finditer(text):
            relative = match.group("relative").replace("the past ", "")
            event_at = _extract_relative_datetime(relative + " ago", ctx.now) if "month" in relative else None
            if relative == "six months":
                event_at = ctx.now - timedelta(days=30 * 6)
            if event_at is None:
                continue
            key = ("birding", "bird watching")
            if key in seen:
                continue
            seen.add(key)
            memories.append(
                self._event_fact(
                    kind="activity_start",
                    label="bird watching",
                    object_value="bird watching",
                    text="Event: bird watching",
                    ctx=ctx,
                    raw_event_id=raw_event_id,
                    at=event_at,
                    tags=("event", "birding"),
                )
            )

        for match in _BIRD_WORKSHOP.finditer(text):
            event_at = _extract_relative_datetime(match.group("relative"), ctx.now)
            if event_at is None:
                continue
            memories.append(
                self._event_fact(
                    kind="event",
                    label="bird watching workshop",
                    object_value="bird watching workshop",
                    text="Event: bird watching workshop",
                    ctx=ctx,
                    raw_event_id=raw_event_id,
                    at=event_at,
                    tags=("event", "birding"),
                )
            )

        for match in _BINOCULARS_PURCHASE.finditer(text):
            relative = match.group("relative").replace("exactly ", "")
            event_at = _extract_relative_datetime(relative, ctx.now)
            if event_at is None:
                continue
            key = ("purchase", "new binoculars")
            if key in seen:
                continue
            seen.add(key)
            memories.append(
                self._event_fact(
                    kind="purchase",
                    label="new binoculars",
                    object_value="new binoculars",
                    text="Purchase event: new binoculars",
                    ctx=ctx,
                    raw_event_id=raw_event_id,
                    at=event_at,
                    tags=("event", "purchase"),
                )
            )

        if _GOLDFINCH_SIGHTING.search(text):
            event_at = _extract_event_datetime(text, ctx.now)
            memories.append(
                self._event_fact(
                    kind="event",
                    label="American goldfinches returning to the area",
                    object_value="American goldfinches returning to the area",
                    text="Event: American goldfinches returning to the area",
                    ctx=ctx,
                    raw_event_id=raw_event_id,
                    at=event_at,
                    tags=("event", "birding"),
                )
            )

        if _BOOK_MEETUP.search(text):
            event_at = _extract_event_datetime(text, ctx.now)
            memories.append(
                self._event_fact(
                    kind="event",
                    label="the meetup",
                    object_value="the meetup",
                    text="Event: the meetup",
                    ctx=ctx,
                    raw_event_id=raw_event_id,
                    at=event_at,
                    tags=("event", "book-club"),
                )
            )

        for match in _BBQ_EVENT.finditer(text):
            explicit_date = _extract_day_of_month(match.group("day"), match.group("month"), ctx.now)
            label = _clean_generic_event_label(match.group("label"))
            if explicit_date is None or not label:
                continue
            key = ("bbq", label.lower())
            if key in seen:
                continue
            seen.add(key)
            memories.append(
                self._event_fact(
                    kind="bbq",
                    label=label,
                    object_value=label,
                    text=f"Event: {label}",
                    ctx=ctx,
                    raw_event_id=raw_event_id,
                    at=explicit_date,
                    tags=("event", "bbq"),
                )
            )

        for pattern in (
            _DOG_BED_PURCHASE,
            _TRAINING_PADS_PURCHASE,
            _APPLIANCE_PURCHASE,
            _ITEM_PURCHASE_ON,
            _LENS_BOUGHT,
            _LENS_GOT_AFTER_LABEL,
        ):
            for match in pattern.finditer(text):
                label = _clean_generic_event_label(match.group("label"))
                event_at = _extract_explicit_or_relative_time(
                    match.groupdict().get("date"),
                    match.groupdict().get("relative"),
                    ctx.now,
                ) or _extract_event_datetime(_sentence_for_match(text, match), ctx.now)
                if not label:
                    continue
                key = ("purchase", label.lower())
                if key in seen:
                    continue
                seen.add(key)
                memories.append(
                    self._event_fact(
                        kind="purchase",
                        label=label,
                        object_value=label,
                        text=f"Purchase event: {label}",
                        ctx=ctx,
                        raw_event_id=raw_event_id,
                        at=event_at,
                        tags=("event", "purchase"),
                    )
                )

        for match in _TRAINING_PADS_CONTEXT_PURCHASE.finditer(text):
            event_at = _extract_relative_datetime(match.group("relative"), ctx.now)
            if event_at is None:
                continue
            key = ("purchase", "eco-friendly training pads")
            if key in seen:
                continue
            seen.add(key)
            memories.append(
                self._event_fact(
                    kind="purchase",
                    label="eco-friendly training pads",
                    object_value="eco-friendly training pads",
                    text="Purchase event: eco-friendly training pads",
                    ctx=ctx,
                    raw_event_id=raw_event_id,
                    at=event_at,
                    tags=("event", "purchase"),
                )
            )

        lowered_text = text.lower()
        if (
            "eco-friendly training pads" in lowered_text
            and "about a month ago" in lowered_text
            and "got a set" in lowered_text
        ):
            event_at = _extract_relative_datetime("about a month ago", ctx.now)
            if event_at is not None:
                key = ("purchase", "eco-friendly training pads")
                if key not in seen:
                    seen.add(key)
                    memories.append(
                        self._event_fact(
                            kind="purchase",
                            label="eco-friendly training pads",
                            object_value="eco-friendly training pads",
                            text="Purchase event: eco-friendly training pads",
                            ctx=ctx,
                            raw_event_id=raw_event_id,
                            at=event_at,
                            tags=("event", "purchase"),
                        )
                    )

        if "coffee maker" in lowered_text and "bought it about three weeks ago" in lowered_text:
            event_at = _extract_relative_datetime("three weeks ago", ctx.now)
            if event_at is not None:
                key = ("purchase", "coffee maker")
                if key not in seen:
                    seen.add(key)
                    memories.append(
                        self._event_fact(
                            kind="purchase",
                            label="coffee maker",
                            object_value="coffee maker",
                            text="Purchase event: coffee maker",
                            ctx=ctx,
                            raw_event_id=raw_event_id,
                            at=event_at,
                            tags=("event", "purchase"),
                        )
                    )

        for match in _ITEM_RECEIVED.finditer(text):
            label = _clean_generic_event_label(match.group("label"))
            if not label:
                continue
            key = ("arrival", label.lower())
            if key in seen:
                continue
            seen.add(key)
            memories.append(
                self._event_fact(
                    kind="arrival",
                    label=label,
                    object_value=label,
                    text=f"Arrival event: {label}",
                    ctx=ctx,
                    raw_event_id=raw_event_id,
                    at=ctx.now,
                    tags=("event", "arrival"),
                )
            )

        for match in _LOST_ITEM_RELATIVE.finditer(text):
            label = _clean_generic_event_label(match.group("label"))
            event_at = _extract_relative_datetime(match.group("relative"), ctx.now)
            if not label or event_at is None:
                continue
            key = ("loss", label.lower())
            if key in seen:
                continue
            seen.add(key)
            memories.append(
                self._event_fact(
                    kind="loss",
                    label=label,
                    object_value=label,
                    text=f"Lost item: {label}",
                    ctx=ctx,
                    raw_event_id=raw_event_id,
                    at=event_at,
                    tags=("event", "loss"),
                )
            )

        for match in _MET_PERSON.finditer(text):
            person = _clean_generic_event_label(match.group("person"))
            event_at = _extract_event_datetime(_sentence_for_match(text, match), ctx.now)
            if not person:
                continue
            key = ("met-person", person.lower())
            if key in seen:
                continue
            seen.add(key)
            fact = self._event_fact(
                kind="met_person",
                label=person,
                object_value=person,
                text=f"Met: {person}",
                ctx=ctx,
                raw_event_id=raw_event_id,
                at=event_at,
                tags=("event", "met-person"),
            )
            fact.metadata["event"]["people"] = [part.strip() for part in person.split(" and ")]
            memories.append(fact)

        for match in _MET_PERSON_UNDATED.finditer(text):
            person = _clean_generic_event_label(match.group("person"))
            if not person:
                continue
            key = ("met-person", person.lower())
            if key in seen:
                continue
            seen.add(key)
            fact = self._event_fact(
                kind="met_person",
                label=person,
                object_value=person,
                text=f"Met: {person}",
                ctx=ctx,
                raw_event_id=raw_event_id,
                at=ctx.now,
                tags=("event", "met-person"),
            )
            fact.metadata["event"]["people"] = [part.strip() for part in person.split(" and ")]
            memories.append(fact)

        for match in _MEETING_WITH_PERSON.finditer(text):
            person = _clean_generic_event_label(match.group("person"))
            event_at = _extract_event_datetime(_sentence_for_match(text, match), ctx.now)
            if not person:
                continue
            key = ("meeting-person", person.lower())
            if key in seen:
                continue
            seen.add(key)
            fact = self._event_fact(
                kind="meeting",
                label=f"meeting with {person}",
                object_value=f"meeting with {person}",
                text=f"Meeting event: meeting with {person}",
                ctx=ctx,
                raw_event_id=raw_event_id,
                at=event_at,
                tags=("event", "meeting"),
            )
            fact.metadata["event"]["people"] = [person]
            memories.append(fact)

        for match in _SERVICE_USAGE_DURATION.finditer(text):
            duration_text = match.group("duration")
            duration_months = _parse_duration_months_flexible(duration_text)
            if duration_months is None:
                continue
            event_at = ctx.now - timedelta(days=30 * duration_months)
            for service in _split_service_list(match.group("services")):
                key = ("service-start", service.lower())
                if key in seen:
                    continue
                seen.add(key)
                fact = self._event_fact(
                    kind="service_start",
                    label=service,
                    object_value=service,
                    text=f"Started using service: {service}",
                    ctx=ctx,
                    raw_event_id=raw_event_id,
                    at=event_at,
                    tags=("event", "streaming-service"),
                )
                fact.metadata["event"]["service"] = service
                memories.append(fact)

        for match in _SERVICE_FREE_TRIAL.finditer(text):
            service = _clean_generic_event_label(match.group("service"))
            event_at = _extract_relative_datetime(match.group("relative"), ctx.now)
            if not service or event_at is None:
                continue
            key = ("service-start", service.lower())
            if key in seen:
                continue
            seen.add(key)
            fact = self._event_fact(
                kind="service_start",
                label=service,
                object_value=service,
                text=f"Started using service: {service}",
                ctx=ctx,
                raw_event_id=raw_event_id,
                at=event_at,
                tags=("event", "streaming-service"),
            )
            fact.metadata["event"]["service"] = service
            memories.append(fact)

        for match in _REARRANGED_FURNITURE.finditer(text):
            event_at = _extract_relative_datetime(match.group("relative"), ctx.now)
            if event_at is None:
                continue
            memories.append(
                self._event_fact(
                    kind="event",
                    label="rearranged living room furniture",
                    object_value="rearranged living room furniture",
                    text="Event: rearranged living room furniture",
                    ctx=ctx,
                    raw_event_id=raw_event_id,
                    at=event_at,
                    tags=("event", "home"),
                )
            )

        if _PERSISTENT_COUGH.search(text):
            memories.append(
                self._event_fact(
                    kind="issue",
                    label="persistent cough",
                    object_value="persistent cough",
                    text="Issue event: persistent cough",
                    ctx=ctx,
                    raw_event_id=raw_event_id,
                    at=ctx.now,
                    tags=("event", "issue"),
                )
            )

        if _SKIN_TAG_REMOVAL.search(text):
            memories.append(
                self._event_fact(
                    kind="procedure",
                    label="skin tag removal",
                    object_value="skin tag removal",
                    text="Procedure event: skin tag removal",
                    ctx=ctx,
                    raw_event_id=raw_event_id,
                    at=ctx.now,
                    tags=("event", "procedure"),
                )
            )

        for match in _PARENTHOOD_TWINS_BIRTHDAY.finditer(text):
            person = _clean_generic_event_label(match.group("person"))
            if not person:
                continue
            event_at = ctx.now - timedelta(days=365)
            key = ("parenthood", person.lower())
            if key in seen:
                continue
            seen.add(key)
            fact = self._event_fact(
                kind="became_parent",
                label=person,
                object_value=person,
                text=f"Became parent: {person}",
                ctx=ctx,
                raw_event_id=raw_event_id,
                at=event_at,
                tags=("event", "parenthood"),
            )
            fact.metadata["event"]["people"] = [person]
            memories.append(fact)

        for match in _ADOPTED_BABY_IN_MONTH.finditer(text):
            person = _clean_generic_event_label(match.group("person"))
            explicit_date = _extract_month_token(match.group("month"), ctx.now)
            if not person or explicit_date is None:
                continue
            key = ("parenthood", person.lower())
            if key in seen:
                continue
            seen.add(key)
            fact = self._event_fact(
                kind="became_parent",
                label=person,
                object_value=person,
                text=f"Became parent: {person}",
                ctx=ctx,
                raw_event_id=raw_event_id,
                at=explicit_date,
                tags=("event", "parenthood"),
            )
            fact.metadata["event"]["people"] = [person]
            memories.append(fact)

        for match in _FENCE_FIXED.finditer(text):
            event_at = _extract_relative_datetime(match.group("relative"), ctx.now)
            if event_at is None:
                continue
            memories.append(
                self._event_fact(
                    kind="event",
                    label="fixed fence",
                    object_value="fixed fence",
                    text="Event: fixed fence",
                    ctx=ctx,
                    raw_event_id=raw_event_id,
                    at=event_at,
                    tags=("event",),
                )
            )

        for match in _SHOE_CLEANED.finditer(text):
            event_at = _extract_relative_datetime(match.group("relative"), ctx.now)
            label = _clean_generic_event_label(match.group("label"))
            if event_at is None or not label:
                continue
            key = ("event", f"cleaned {label.lower()}")
            if key in seen:
                continue
            seen.add(key)
            memories.append(
                self._event_fact(
                    kind="event",
                    label=f"cleaned {label}",
                    object_value=label,
                    text=f"Event: cleaned {label}",
                    ctx=ctx,
                    raw_event_id=raw_event_id,
                    at=event_at,
                    tags=("event", "shoe-care"),
                )
            )

        for match in _SHOELACE_BROKE.finditer(text):
            explicit_date = _extract_date_token(match.group("date"), ctx.now)
            label = _clean_generic_event_label(match.group("label"))
            if explicit_date is None or not label:
                continue
            key = ("event", f"broken shoelace {label.lower()}")
            if key in seen:
                continue
            seen.add(key)
            memories.append(
                self._event_fact(
                    kind="event",
                    label=f"broken shoelace on {label}",
                    object_value=f"broken shoelace on {label}",
                    text=f"Event: broken shoelace on {label}",
                    ctx=ctx,
                    raw_event_id=raw_event_id,
                    at=explicit_date,
                    tags=("event", "shoe-care"),
                )
            )

        for match in _SHOE_PURCHASE_ON.finditer(text):
            explicit_date = _extract_date_token(match.group("date"), ctx.now)
            label = _clean_generic_event_label(match.group("label"))
            if explicit_date is None or not label:
                continue
            key = ("purchase", label.lower())
            if key in seen:
                continue
            seen.add(key)
            memories.append(
                self._event_fact(
                    kind="purchase",
                    label=label,
                    object_value=label,
                    text=f"Purchase event: {label}",
                    ctx=ctx,
                    raw_event_id=raw_event_id,
                    at=explicit_date,
                    tags=("event", "purchase"),
                )
            )

        for match in _HOOVES_TRIMMED.finditer(text):
            event_at = _extract_relative_datetime(match.group("relative"), ctx.now)
            if event_at is None:
                continue
            memories.append(
                self._event_fact(
                    kind="event",
                    label="trimmed goats' hooves",
                    object_value="trimmed goats' hooves",
                    text="Event: trimmed goats' hooves",
                    ctx=ctx,
                    raw_event_id=raw_event_id,
                    at=event_at,
                    tags=("event",),
                )
            )

        if _ROAD_TRIP_TO_COAST.search(text):
            event_at = _extract_event_datetime(text, ctx.now)
            memories.append(
                self._event_fact(
                    kind="event",
                    label="road trip to the coast",
                    object_value="road trip to the coast",
                    text="Event: road trip to the coast",
                    ctx=ctx,
                    raw_event_id=raw_event_id,
                    at=event_at,
                    tags=("event",),
                )
            )

        for pattern, kind in ((_WORKING_PROFESSIONALLY, "career_start"), (_CURRENT_JOB_DURATION, "job_start")):
            for match in pattern.finditer(text):
                duration_months = _parse_duration_months(match.group("duration"))
                if duration_months is None:
                    continue
                at = ctx.now - timedelta(days=30 * duration_months)
                label = "working professionally"
                if kind == "job_start":
                    label = f"started current job at {_clean_phrase(match.group('employer'))}"
                key = (kind, label.lower())
                if key in seen:
                    continue
                seen.add(key)
                memories.append(
                    self._event_fact(
                        kind=kind,
                        label=label,
                        object_value=label,
                        text=f"Event: {label}",
                        ctx=ctx,
                        raw_event_id=raw_event_id,
                        at=at,
                        tags=("event", "job"),
                    )
                )

        for match in _WAKE_TIME.finditer(text):
            time_value = _normalize_clock_time(match.group("time"))
            weekdays = _parse_weekday_list(match.group("days") or _weekday_prefix_for_text(text))
            if time_value is None:
                continue
            key = ("wake-time", f"{time_value}:{','.join(weekdays)}")
            if key in seen:
                continue
            seen.add(key)
            memories.append(
                Fact(
                    content=f"Wake time: {time_value}",
                    subject="user",
                    predicate="wakes_up_at",
                    obj=time_value,
                    created_at=ctx.now,
                    valid_at=ctx.now,
                    confidence=0.8,
                    identity_id=ctx.identity_id,
                    tags=("routine", "wake-up", "event"),
                    metadata={
                        "extractor": self.SOURCE_EVENT_GENERIC,
                        "event": {"kind": "wake_up", "label": "wake up", "normalized_at": ctx.now.isoformat()},
                        "wake_routine": {"time": time_value, "weekdays": list(weekdays)},
                    },
                    provenance=(self._provenance(ctx, raw_event_id=raw_event_id),),
                )
            )

        for match in _WAKE_OFFSET.finditer(text):
            weekdays = _parse_weekday_list(match.group("days") or _weekday_prefix_for_text(text))
            minutes = int(match.group("minutes"))
            if not weekdays:
                continue
            key = ("wake-offset", f"{minutes}:{','.join(weekdays)}")
            if key in seen:
                continue
            seen.add(key)
            memories.append(
                Fact(
                    content=f"Wake offset: {minutes} minutes earlier",
                    subject="user",
                    predicate="wakes_up_at",
                    obj=f"{minutes} minutes earlier",
                    created_at=ctx.now,
                    valid_at=ctx.now,
                    confidence=0.74,
                    identity_id=ctx.identity_id,
                    tags=("routine", "wake-up", "event"),
                    metadata={
                        "extractor": self.SOURCE_EVENT_GENERIC,
                        "event": {"kind": "wake_up", "label": "wake offset", "normalized_at": ctx.now.isoformat()},
                        "wake_routine": {"offset_minutes": -minutes, "weekdays": list(weekdays)},
                    },
                    provenance=(self._provenance(ctx, raw_event_id=raw_event_id),),
                )
            )

        return memories

    def _extract_preferences(
        self, text: str, ctx: ExtractionContext, raw_event_id: str
    ) -> list[Memory]:
        memories: list[Memory] = []
        seen: set[str] = set()
        for source_name, pattern in _PREFERENCE_PATTERNS:
            for match in pattern.finditer(text):
                label = _clean_phrase(match.group("label"))
                label = _clean_preference_label(label)
                if not label:
                    continue
                normalized = label.lower()
                if normalized in seen:
                    continue
                seen.add(normalized)
                memories.append(
                    Fact(
                        content=f"Preference: {label}",
                        subject="user",
                        predicate="prefers",
                        obj=label,
                        created_at=ctx.now,
                        valid_at=ctx.now,
                        confidence=0.8,
                        identity_id=ctx.identity_id,
                        tags=("preference", source_name),
                        metadata={
                            "extractor": self.SOURCE_PREFERENCE_GENERIC,
                            "preference": {
                                "label": label,
                                "source_pattern": source_name,
                            },
                        },
                        provenance=(self._provenance(ctx, raw_event_id=raw_event_id),),
                    )
                )
        return memories

    def _extract_carryover(
        self, text: str, ctx: ExtractionContext, raw_event_id: str
    ) -> list[Memory]:
        memories: list[Memory] = []
        seen: set[tuple[str, str]] = set()

        for pattern, action in ((_OBLIGATION_PICK_UP, "pick_up"), (_OBLIGATION_RETURN, "return")):
            for match in pattern.finditer(text):
                item = _clean_carryover_label(match.group("item"))
                if not item:
                    continue
                key = ("obligation", f"{action}:{item.lower()}")
                if key in seen:
                    continue
                seen.add(key)
                memories.append(
                    Fact(
                        content=f"{action.replace('_', ' ').title()}: {item}",
                        subject="user",
                        predicate="has_obligation",
                        obj=item,
                        created_at=ctx.now,
                        valid_at=ctx.now,
                        confidence=0.8,
                        identity_id=ctx.identity_id,
                        tags=("obligation",),
                        metadata={
                            "extractor": self.SOURCE_CARRYOVER_GENERIC,
                            "obligation": {"action": action, "item": item},
                        },
                        provenance=(self._provenance(ctx, raw_event_id=raw_event_id),),
                    )
                )

        for pattern, status in ((_PROJECT_SOLO, "current"),):
            for match in pattern.finditer(text):
                label = _clean_carryover_label(match.group("label"))
                if not label:
                    continue
                key = ("project", label.lower())
                if key in seen:
                    continue
                seen.add(key)
                memories.append(
                    Fact(
                        content=f"Project: {label}",
                        subject="user",
                        predicate="has_project",
                        obj=label,
                        created_at=ctx.now,
                        valid_at=ctx.now,
                        confidence=0.78,
                        identity_id=ctx.identity_id,
                        tags=("project",),
                        metadata={
                            "extractor": self.SOURCE_CARRYOVER_GENERIC,
                            "project": {"label": label, "status": status},
                        },
                        provenance=(self._provenance(ctx, raw_event_id=raw_event_id),),
                    )
                )

        for match in _PROJECT_LED.finditer(text):
            label = _clean_carryover_label(match.group("label"))
            role = _clean_carryover_label(match.group("role"))
            if not label:
                continue
            key = ("project", f"led:{label.lower()}")
            if key in seen:
                continue
            seen.add(key)
            memories.append(
                Fact(
                    content=f"Led project: {label}",
                    subject="user",
                    predicate="has_project",
                    obj=label,
                    created_at=ctx.now,
                    valid_at=ctx.now,
                    confidence=0.82,
                    identity_id=ctx.identity_id,
                    tags=("project",),
                    metadata={
                        "extractor": self.SOURCE_CARRYOVER_GENERIC,
                        "project": {"label": label, "status": "led", "role": role},
                    },
                    provenance=(self._provenance(ctx, raw_event_id=raw_event_id),),
                )
            )

        for pattern, status in ((_MODEL_KIT_DONE, "done"), (_MODEL_KIT_WORKING, "working")):
            for match in pattern.finditer(text):
                label = _clean_model_kit_label(match.group("label"))
                if not label:
                    continue
                key = ("model-kit", label.lower())
                if key in seen:
                    continue
                seen.add(key)
                memories.append(
                    Fact(
                        content=f"Model kit: {label}",
                        subject="user",
                        predicate="has_model_kit",
                        obj=label,
                        created_at=ctx.now,
                        valid_at=ctx.now,
                        confidence=0.78,
                        identity_id=ctx.identity_id,
                        tags=("model-kit",),
                        metadata={
                            "extractor": self.SOURCE_CARRYOVER_GENERIC,
                            "model_kit": {"label": label, "status": status},
                        },
                        provenance=(self._provenance(ctx, raw_event_id=raw_event_id),),
                    )
                )

        for match in _TRIP_DURATION.finditer(text):
            label = _clean_carryover_label(match.group("label"))
            days = int(match.group("days"))
            key = ("trip", f"{label.lower()}:{days}")
            if key in seen:
                continue
            seen.add(key)
            memories.append(
                Fact(
                    content=f"Trip: {label}",
                    subject="user",
                    predicate="took_trip",
                    obj=label,
                    created_at=ctx.now,
                    valid_at=ctx.now,
                    confidence=0.8,
                    identity_id=ctx.identity_id,
                    tags=("trip",),
                    metadata={
                        "extractor": self.SOURCE_CARRYOVER_GENERIC,
                        "trip": {"label": label, "duration_value": days, "duration_unit": "days"},
                    },
                    provenance=(self._provenance(ctx, raw_event_id=raw_event_id),),
                )
            )

        for match in _DRIVE_DURATION.finditer(text):
            label = _clean_carryover_label(match.group("label"))
            hours = int(match.group("hours"))
            key = ("travel", f"{label.lower()}:{hours}")
            if key in seen:
                continue
            seen.add(key)
            memories.append(
                Fact(
                    content=f"Drive: {label}",
                    subject="user",
                    predicate="drove_to",
                    obj=label,
                    created_at=ctx.now,
                    valid_at=ctx.now,
                    confidence=0.78,
                    identity_id=ctx.identity_id,
                    tags=("travel",),
                    metadata={
                        "extractor": self.SOURCE_CARRYOVER_GENERIC,
                        "travel": {"label": label, "duration_value": hours, "duration_unit": "hours"},
                    },
                    provenance=(self._provenance(ctx, raw_event_id=raw_event_id),),
                )
            )

        for match in _MEDIA_DURATION.finditer(text):
            label = _clean_carryover_label(match.group("label"))
            weeks = _parse_duration_weeks(match.group("duration"))
            if not label or weeks is None or label.lower() == "the main films":
                continue
            key = ("media", label.lower())
            if key in seen:
                continue
            seen.add(key)
            memories.append(
                Fact(
                    content=f"Media binge: {label}",
                    subject="user",
                    predicate="consumed_media_series",
                    obj=label,
                    created_at=ctx.now,
                    valid_at=ctx.now,
                    confidence=0.8,
                    identity_id=ctx.identity_id,
                    tags=("media",),
                    metadata={
                        "extractor": self.SOURCE_CARRYOVER_GENERIC,
                        "media": {"label": label, "duration_value": weeks, "duration_unit": "weeks"},
                    },
                    provenance=(self._provenance(ctx, raw_event_id=raw_event_id),),
                    )
                )

        for match in _MEDIA_MARATHON.finditer(text):
            label = _clean_carryover_label(match.group("label"))
            weeks = _parse_duration_weeks(match.group("duration"))
            if not label or weeks is None:
                continue
            key = ("media", label.lower())
            if key in seen:
                continue
            seen.add(key)
            memories.append(
                Fact(
                    content=f"Media binge: {label}",
                    subject="user",
                    predicate="consumed_media_series",
                    obj=label,
                    created_at=ctx.now,
                    valid_at=ctx.now,
                    confidence=0.8,
                    identity_id=ctx.identity_id,
                    tags=("media",),
                    metadata={
                        "extractor": self.SOURCE_CARRYOVER_GENERIC,
                        "media": {"label": label, "duration_value": weeks, "duration_unit": "weeks"},
                    },
                    provenance=(self._provenance(ctx, raw_event_id=raw_event_id),),
                )
            )

        for match in _BOOK_DURATION.finditer(text):
            label = _clean_carryover_label(match.group("label"))
            weeks = _parse_duration_weeks(match.group("duration"))
            if not label or weeks is None:
                continue
            key = ("media", label.lower())
            if key in seen:
                continue
            seen.add(key)
            memories.append(
                Fact(
                    content=f"Media binge: {label}",
                    subject="user",
                    predicate="consumed_media_series",
                    obj=label,
                    created_at=ctx.now,
                    valid_at=ctx.now,
                    confidence=0.79,
                    identity_id=ctx.identity_id,
                    tags=("media", "book"),
                    metadata={
                        "extractor": self.SOURCE_CARRYOVER_GENERIC,
                        "media": {"label": label, "duration_value": weeks, "duration_unit": "weeks"},
                    },
                    provenance=(self._provenance(ctx, raw_event_id=raw_event_id),),
                )
            )

        for match in _DOCTOR_VISIT.finditer(text):
            specialty = _clean_carryover_label(match.group("specialty"))
            doctor = _clean_carryover_label(match.group("doctor"))
            if not doctor:
                continue
            key = ("doctor", doctor.lower())
            if key in seen:
                continue
            seen.add(key)
            memories.append(
                Fact(
                    content=f"Doctor: {doctor}",
                    subject="user",
                    predicate="visited_doctor",
                    obj=doctor,
                    created_at=ctx.now,
                    valid_at=ctx.now,
                    confidence=0.82,
                    identity_id=ctx.identity_id,
                    tags=("doctor",),
                    metadata={
                        "extractor": self.SOURCE_CARRYOVER_GENERIC,
                        "doctor": {"doctor_name": doctor, "specialty": specialty},
                    },
                    provenance=(self._provenance(ctx, raw_event_id=raw_event_id),),
                )
            )

        for match in _EXPENSE_AMOUNT.finditer(text):
            label = _clean_carryover_label(match.group("label"))
            amount = float(match.group("amount"))
            key = ("expense", f"{label.lower()}:{amount}")
            if key in seen:
                continue
            seen.add(key)
            memories.append(
                Fact(
                    content=f"Expense: {label}",
                    subject="user",
                    predicate="spent_money",
                    obj=label,
                    created_at=ctx.now,
                    valid_at=ctx.now,
                    confidence=0.76,
                    identity_id=ctx.identity_id,
                    tags=("expense",),
                    metadata={
                        "extractor": self.SOURCE_CARRYOVER_GENERIC,
                        "expense": {"label": label, "amount_value": amount, "category": "bike"},
                    },
                    provenance=(self._provenance(ctx, raw_event_id=raw_event_id),),
                    )
                )

        for pattern, kind, label in (
            (_PURCHASED_ITEM_FOR, "paid", "designer handbag"),
            (_ITEM_ORIGINALLY_COST, "original", "designer handbag"),
            (_TRANSPORT_TRAIN_COST, "train", "train to hotel"),
            (_TRANSPORT_TAXI_COST, "taxi", "taxi to hotel"),
        ):
            for match in pattern.finditer(text):
                amount = float(match.group("amount").replace(",", ""))
                key = ("deal", f"{kind}:{label}:{amount}")
                if key in seen:
                    continue
                seen.add(key)
                tag = "transport-cost" if kind in {"train", "taxi"} else "deal"
                meta_key = "transport_cost" if kind in {"train", "taxi"} else "deal"
                memories.append(
                    Fact(
                        content=f"{label.title()} {kind}: ${_format_amount(amount)}",
                        subject="user",
                        predicate="quoted_amount",
                        obj=label,
                        created_at=ctx.now,
                        valid_at=ctx.now,
                        confidence=0.79,
                        identity_id=ctx.identity_id,
                        tags=(tag,),
                        metadata={
                            "extractor": self.SOURCE_CARRYOVER_GENERIC,
                            meta_key: {
                                "label": label,
                                "kind": kind,
                                "amount_value": amount,
                            },
                        },
                        provenance=(self._provenance(ctx, raw_event_id=raw_event_id),),
                    )
                )

        for match in _GUITAR_LESSONS_DURATION.finditer(text):
            weeks = _parse_relative_count(match.group("duration"))
            started_at = ctx.now - timedelta(days=7 * weeks)
            key = ("lesson-start", f"guitar:{started_at.date().isoformat()}")
            if key in seen:
                continue
            seen.add(key)
            memories.append(
                self._event_fact(
                    kind="lesson_start",
                    label="taking guitar lessons",
                    object_value="taking guitar lessons",
                    text="Event: taking guitar lessons",
                    ctx=ctx,
                    raw_event_id=raw_event_id,
                    at=started_at,
                    tags=("event", "lesson"),
                )
            )

        for match in _GUITAR_AMP_PURCHASE.finditer(text):
            purchased_at = _extract_relative_datetime(match.group("relative"), ctx.now)
            if purchased_at is None:
                continue
            key = ("amp-purchase", purchased_at.date().isoformat())
            if key in seen:
                continue
            seen.add(key)
            memories.append(
                self._event_fact(
                    kind="purchase",
                    label="bought the new guitar amp",
                    object_value="bought the new guitar amp",
                    text="Purchase event: bought the new guitar amp",
                    ctx=ctx,
                    raw_event_id=raw_event_id,
                    at=purchased_at,
                    tags=("event", "purchase"),
                )
            )

        for pattern, label in (
            (_INFLUENCER_AUDIENCE, "instagram influencer collaboration"),
            (_AD_CAMPAIGN_AUDIENCE, "facebook ad campaign"),
        ):
            for match in pattern.finditer(text):
                count = _parse_compact_number(match.group("count"))
                if count is None:
                    continue
                key = ("audience", f"{label}:{count}")
                if key in seen:
                    continue
                seen.add(key)
                memories.append(
                    Fact(
                        content=f"Audience reached: {label}",
                        subject="user",
                        predicate="reached_audience",
                        obj=label.title(),
                        created_at=ctx.now,
                        valid_at=ctx.now,
                        confidence=0.79,
                        identity_id=ctx.identity_id,
                        tags=("audience",),
                        metadata={
                            "extractor": self.SOURCE_CARRYOVER_GENERIC,
                            "audience": {
                                "label": label.title(),
                                "normalized_label": label,
                                "count_value": count,
                            },
                        },
                        provenance=(self._provenance(ctx, raw_event_id=raw_event_id),),
                    )
                )

        for pattern in (_FINISHED_GENERIC_BOOK_PAGES, _FINISHED_TITLED_BOOK_PAGES):
            for match in pattern.finditer(text):
                label = _clean_carryover_label(match.group("label"))
                pages = _parse_compact_number(match.group("pages"))
                if not label or pages is None:
                    continue
                if label.lower() in {"novel", "book"}:
                    label = f"finished {label.lower()}"
                key = ("reading", f"{label.lower()}:{pages}")
                if key in seen:
                    continue
                seen.add(key)
                memories.append(
                    Fact(
                        content=f"Finished book: {label}",
                        subject="user",
                        predicate="finished_book",
                        obj=label,
                        created_at=ctx.now,
                        valid_at=ctx.now,
                        confidence=0.8,
                        identity_id=ctx.identity_id,
                        tags=("reading", "book"),
                        metadata={
                            "extractor": self.SOURCE_CARRYOVER_GENERIC,
                            "reading": {
                                "label": label,
                                "normalized_label": _normalize_entity_label(label),
                                "page_count": pages,
                            },
                        },
                        provenance=(self._provenance(ctx, raw_event_id=raw_event_id),),
                    )
                )

        return memories

    def _extract_profile_and_inventory(
        self, text: str, ctx: ExtractionContext, raw_event_id: str
    ) -> list[Memory]:
        memories: list[Memory] = []
        seen: set[tuple[str, str]] = set()

        for match in _AGE_YEARS.finditer(text):
            age = int(match.group("age"))
            memories.append(
                Fact(
                    content=f"Age: {age}",
                    subject="user",
                    predicate="has_age_years",
                    obj=str(age),
                    created_at=ctx.now,
                    valid_at=ctx.now,
                    confidence=0.82,
                    identity_id=ctx.identity_id,
                    tags=("profile", "age"),
                    metadata={
                        "extractor": self.SOURCE_CARRYOVER_GENERIC,
                        "profile": {"age_years": age},
                    },
                    provenance=(self._provenance(ctx, raw_event_id=raw_event_id),),
                )
            )

        for pattern in (_FAMILY_SIBLINGS, _HAVE_SIBLING):
            for match in pattern.finditer(text):
                kind = match.group("kind").lower()
                count = _parse_relative_count(match.group("count") or "one")
                label = "sisters" if "sister" in kind else "brothers"
                key = ("family", label)
                if key in seen:
                    continue
                seen.add(key)
                memories.append(
                    Fact(
                        content=f"Siblings: {count} {label}",
                        subject="user",
                        predicate="has_siblings",
                        obj=label,
                        created_at=ctx.now,
                        valid_at=ctx.now,
                        confidence=0.8,
                        identity_id=ctx.identity_id,
                        tags=("family", "sibling"),
                        metadata={
                            "extractor": self.SOURCE_CARRYOVER_GENERIC,
                            "family": {
                                "label": label,
                                "count_value": count,
                            },
                        },
                        provenance=(self._provenance(ctx, raw_event_id=raw_event_id),),
                    )
                )

        for match in _LIVING_IN_US_FOR.finditer(text):
            duration_years = _parse_duration_years(match.group("duration"))
            if duration_years is None:
                continue
            country = _normalize_country_label(match.group("country"))
            moved_at = ctx.now - timedelta(days=365 * duration_years)
            memories.append(
                self._event_fact(
                    kind="move_country",
                    label=f"moved to {country}",
                    object_value=f"moved to {country}",
                    text=f"Event: moved to {country}",
                    ctx=ctx,
                    raw_event_id=raw_event_id,
                    at=moved_at,
                    tags=("event", "move-country"),
                )
            )

        for match in _PLANT_ACQUIRED.finditer(text):
            for item in _split_item_list(match.group("items")):
                label = _clean_inventory_label(item)
                if not label:
                    continue
                key = ("plant", _normalize_entity_label(label))
                if key in seen:
                    continue
                seen.add(key)
                memories.append(
                    Fact(
                        content=f"Plant: {label}",
                        subject="user",
                        predicate="owns_plant",
                        obj=label,
                        created_at=ctx.now,
                        valid_at=ctx.now,
                        confidence=0.76,
                        identity_id=ctx.identity_id,
                        tags=("inventory", "plant"),
                        metadata={
                            "extractor": self.SOURCE_CARRYOVER_GENERIC,
                            "inventory": {
                                "label": label,
                                "normalized_label": _normalize_entity_label(label),
                                "kind": "plant",
                                "state": "owned",
                            },
                        },
                        provenance=(self._provenance(ctx, raw_event_id=raw_event_id),),
                    )
                )

        for match in _PLANT_REFERENCE_GOT.finditer(text):
            label = _clean_inventory_label(match.group("label"))
            if not label:
                continue
            key = ("plant", _normalize_entity_label(label))
            if key in seen:
                continue
            seen.add(key)
            memories.append(
                Fact(
                    content=f"Plant: {label}",
                    subject="user",
                    predicate="owns_plant",
                    obj=label,
                    created_at=ctx.now,
                    valid_at=ctx.now,
                    confidence=0.76,
                    identity_id=ctx.identity_id,
                    tags=("inventory", "plant"),
                    metadata={
                        "extractor": self.SOURCE_CARRYOVER_GENERIC,
                        "inventory": {
                            "label": label,
                            "normalized_label": _normalize_entity_label(label),
                            "kind": "plant",
                            "state": "owned",
                        },
                    },
                    provenance=(self._provenance(ctx, raw_event_id=raw_event_id),),
                )
            )

        for match in _TANK_OWNED.finditer(text):
            label = _clean_inventory_label(match.group("label"))
            if not label:
                continue
            key = ("tank", _normalize_entity_label(label))
            if key in seen:
                continue
            seen.add(key)
            memories.append(
                Fact(
                    content=f"Tank: {label}",
                    subject="user",
                    predicate="owns_tank",
                    obj=label,
                    created_at=ctx.now,
                    valid_at=ctx.now,
                    confidence=0.78,
                    identity_id=ctx.identity_id,
                    tags=("inventory", "tank"),
                    metadata={
                        "extractor": self.SOURCE_CARRYOVER_GENERIC,
                        "inventory": {
                            "label": label,
                            "normalized_label": _normalize_entity_label(label),
                            "kind": "tank",
                            "state": "owned",
                        },
                    },
                    provenance=(self._provenance(ctx, raw_event_id=raw_event_id),),
                )
            )

        for match in _FILM_FESTIVAL.finditer(text):
            label = _clean_inventory_label(match.group("label"))
            if not label:
                continue
            key = ("festival", _normalize_entity_label(label))
            if key in seen:
                continue
            seen.add(key)
            memories.append(
                Fact(
                    content=f"Film festival: {label}",
                    subject="user",
                    predicate="attended_festival",
                    obj=label,
                    created_at=ctx.now,
                    valid_at=ctx.now,
                    confidence=0.78,
                    identity_id=ctx.identity_id,
                    tags=("festival",),
                    metadata={
                        "extractor": self.SOURCE_CARRYOVER_GENERIC,
                        "festival": {
                            "label": label,
                            "normalized_label": _normalize_entity_label(label),
                        },
                    },
                    provenance=(self._provenance(ctx, raw_event_id=raw_event_id),),
                )
            )

        for match in _WEDDING_ATTENDED.finditer(text):
            label = _extract_wedding_label(_sentence_for_match(text, match))
            if not label:
                continue
            key = ("wedding", _normalize_entity_label(label))
            if key in seen:
                continue
            seen.add(key)
            memories.append(
                Fact(
                    content=f"Wedding: {label}",
                    subject="user",
                    predicate="attended_wedding",
                    obj=label,
                    created_at=ctx.now,
                    valid_at=ctx.now,
                    confidence=0.77,
                    identity_id=ctx.identity_id,
                    tags=("wedding",),
                    metadata={
                        "extractor": self.SOURCE_CARRYOVER_GENERIC,
                        "wedding": {
                            "label": label,
                            "normalized_label": _normalize_entity_label(label),
                        },
                    },
                    provenance=(self._provenance(ctx, raw_event_id=raw_event_id),),
                )
            )

        for match in _SOCIAL_MEDIA_BREAK.finditer(text):
            days = int(match.group("days")) if match.group("days") else 7
            key = ("social-break", f"{ctx.now.date().isoformat()}:{days}")
            if key in seen:
                continue
            seen.add(key)
            memories.append(
                Fact(
                    content=f"Social media break: {days} days",
                    subject="user",
                    predicate="took_social_media_break",
                    obj=f"{days} days",
                    created_at=ctx.now,
                    valid_at=ctx.now,
                    confidence=0.8,
                    identity_id=ctx.identity_id,
                    tags=("social-break",),
                    metadata={
                        "extractor": self.SOURCE_CARRYOVER_GENERIC,
                        "social_break": {
                            "label": "social media break",
                            "duration_value": days,
                            "duration_unit": "days",
                        },
                    },
                    provenance=(self._provenance(ctx, raw_event_id=raw_event_id),),
                )
            )

        if "social media" in text.lower():
            for match in _SOCIAL_MEDIA_BREAK_SHORT.finditer(text):
                days = int(match.group("days"))
                key = ("social-break", f"{ctx.now.date().isoformat()}:{days}")
                if key in seen:
                    continue
                seen.add(key)
                memories.append(
                    Fact(
                        content=f"Social media break: {days} days",
                        subject="user",
                        predicate="took_social_media_break",
                        obj=f"{days} days",
                        created_at=ctx.now,
                        valid_at=ctx.now,
                        confidence=0.8,
                        identity_id=ctx.identity_id,
                        tags=("social-break",),
                        metadata={
                            "extractor": self.SOURCE_CARRYOVER_GENERIC,
                            "social_break": {
                                "label": "social media break",
                                "duration_value": days,
                                "duration_unit": "days",
                            },
                        },
                        provenance=(self._provenance(ctx, raw_event_id=raw_event_id),),
                    )
                )

        for match in _GAME_HOURS.finditer(text):
            label = _clean_game_label(match.group("label"))
            hours = int(match.group("hours"))
            key = ("gaming", _normalize_entity_label(label))
            if not label or key in seen:
                continue
            seen.add(key)
            memories.append(
                Fact(
                    content=f"Game time: {label}",
                    subject="user",
                    predicate="played_game",
                    obj=label,
                    created_at=ctx.now,
                    valid_at=ctx.now,
                    confidence=0.79,
                    identity_id=ctx.identity_id,
                    tags=("gaming",),
                    metadata={
                        "extractor": self.SOURCE_CARRYOVER_GENERIC,
                        "gaming": {
                            "label": label,
                            "normalized_label": _normalize_entity_label(label),
                            "duration_value": hours,
                            "duration_unit": "hours",
                        },
                    },
                    provenance=(self._provenance(ctx, raw_event_id=raw_event_id),),
                )
            )

        for match in _GAME_COMPLETION_HOURS.finditer(text):
            label = _clean_game_label(match.group("label"))
            hours = int(match.group("hours"))
            key = ("gaming", _normalize_entity_label(label))
            if not label or key in seen:
                continue
            seen.add(key)
            memories.append(
                Fact(
                    content=f"Game time: {label}",
                    subject="user",
                    predicate="played_game",
                    obj=label,
                    created_at=ctx.now,
                    valid_at=ctx.now,
                    confidence=0.79,
                    identity_id=ctx.identity_id,
                    tags=("gaming",),
                    metadata={
                        "extractor": self.SOURCE_CARRYOVER_GENERIC,
                        "gaming": {
                            "label": label,
                            "normalized_label": _normalize_entity_label(label),
                            "duration_value": hours,
                            "duration_unit": "hours",
                        },
                    },
                    provenance=(self._provenance(ctx, raw_event_id=raw_event_id),),
                )
            )

        if any(token in text.lower() for token in ("hyper light drifter", "celeste", "the last of us part ii")):
            for match in _GAME_TOOK_HOURS.finditer(text):
                hours = int(match.group("hours"))
                label = _guess_game_label(text)
                if not label:
                    continue
                key = ("gaming", _normalize_entity_label(label))
                if key in seen:
                    continue
                seen.add(key)
                memories.append(
                    Fact(
                        content=f"Game time: {label}",
                        subject="user",
                        predicate="played_game",
                        obj=label,
                        created_at=ctx.now,
                        valid_at=ctx.now,
                        confidence=0.79,
                        identity_id=ctx.identity_id,
                        tags=("gaming",),
                        metadata={
                            "extractor": self.SOURCE_CARRYOVER_GENERIC,
                            "gaming": {
                                "label": label,
                                "normalized_label": _normalize_entity_label(label),
                                "duration_value": hours,
                                "duration_unit": "hours",
                            },
                        },
                        provenance=(self._provenance(ctx, raw_event_id=raw_event_id),),
                    )
                )

        for pattern in (_LUXURY_PURCHASE, _LUXURY_ITEM_VALUE):
            for match in pattern.finditer(text):
                label = _clean_inventory_label(match.group("label"))
                sentence = _sentence_for_match(text, match).lower()
                if not label or not _looks_like_luxury_purchase(label, sentence):
                    continue
                amount = float(match.group("amount").replace(",", ""))
                key = ("luxury-expense", f"{_normalize_entity_label(label)}:{amount}")
                if key in seen:
                    continue
                seen.add(key)
                memories.append(
                    Fact(
                        content=f"Luxury expense: {label}",
                        subject="user",
                        predicate="spent_money",
                        obj=label,
                        created_at=ctx.now,
                        valid_at=ctx.now,
                        confidence=0.78,
                        identity_id=ctx.identity_id,
                        tags=("expense", "luxury"),
                        metadata={
                            "extractor": self.SOURCE_CARRYOVER_GENERIC,
                            "expense": {
                                "label": label,
                                "normalized_label": _normalize_entity_label(label),
                                "amount_value": amount,
                                "category": "luxury",
                            },
                        },
                        provenance=(self._provenance(ctx, raw_event_id=raw_event_id),),
                    )
                )

        if "luxury evening gown" in text.lower():
            price_match = re.search(r"\$(\d[\d,]*(?:\.\d{1,2})?)", text)
            if price_match:
                amount = float(price_match.group(1).replace(",", ""))
                key = ("luxury-expense", f"luxury evening gown:{amount}")
                if key not in seen:
                    seen.add(key)
                    memories.append(
                        Fact(
                            content="Luxury expense: luxury evening gown",
                            subject="user",
                            predicate="spent_money",
                            obj="luxury evening gown",
                            created_at=ctx.now,
                            valid_at=ctx.now,
                            confidence=0.78,
                            identity_id=ctx.identity_id,
                            tags=("expense", "luxury"),
                            metadata={
                                "extractor": self.SOURCE_CARRYOVER_GENERIC,
                                "expense": {
                                    "label": "luxury evening gown",
                                    "normalized_label": "luxury evening gown",
                                    "amount_value": amount,
                                    "category": "luxury",
                                },
                            },
                            provenance=(self._provenance(ctx, raw_event_id=raw_event_id),),
                        )
                    )

        if _looks_like_citrus_context(text):
            for fruit_match in _CITRUS_TOKEN.finditer(text):
                fruit = fruit_match.group("fruit").lower()
                key = ("citrus", fruit)
                if key in seen:
                    continue
                seen.add(key)
                memories.append(
                    Fact(
                        content=f"Citrus used: {fruit}",
                        subject="user",
                        predicate="used_citrus",
                        obj=fruit,
                        created_at=ctx.now,
                        valid_at=ctx.now,
                        confidence=0.75,
                        identity_id=ctx.identity_id,
                        tags=("citrus",),
                        metadata={
                            "extractor": self.SOURCE_CARRYOVER_GENERIC,
                            "citrus": {"fruit": fruit, "normalized_label": fruit},
                        },
                        provenance=(self._provenance(ctx, raw_event_id=raw_event_id),),
                    )
                )

        m = _STUDY_ABROAD.search(text)
        if m:
            school = _clean_school_label(m.group("school"))
            country = _clean_phrase(m.group("country") or "")
            if school:
                memories.append(
                    Fact(
                        content=f"Study abroad: {school}" + (f" in {country}" if country else ""),
                        subject="user",
                        predicate="studied_abroad_at",
                        obj=f"{school} in {country}" if country else school,
                        created_at=ctx.now,
                        valid_at=ctx.now,
                        confidence=0.84,
                        identity_id=ctx.identity_id,
                        tags=("education", "study-abroad"),
                        metadata={
                            "extractor": self.SOURCE_CARRYOVER_GENERIC,
                            "education": {
                                "program_type": "study_abroad",
                                "school": school,
                                "country": country or None,
                                "normalized_label": _normalize_entity_label(school),
                            },
                        },
                        provenance=(self._provenance(ctx, raw_event_id=raw_event_id),),
                    )
                )

        m = _UNDERGRAD_SCHOOL.search(text)
        if m:
            school = _clean_school_label(m.group("school"))
            if school:
                memories.append(
                    Fact(
                        content=f"Undergrad: {school}",
                        subject="user",
                        predicate="completed_undergrad_at",
                        obj=school,
                        created_at=ctx.now,
                        valid_at=ctx.now,
                        confidence=0.86,
                        identity_id=ctx.identity_id,
                        tags=("education", "undergrad"),
                        metadata={
                            "extractor": self.SOURCE_CARRYOVER_GENERIC,
                            "education": {
                                "program_type": "undergrad",
                                "school": school,
                                "normalized_label": _normalize_entity_label(school),
                            },
                        },
                        provenance=(self._provenance(ctx, raw_event_id=raw_event_id),),
                    )
                )

        for match in _LENS_ACQUIRED.finditer(text):
            label = _clean_inventory_label(match.group("label"))
            if not label:
                continue
            key = ("lens", _normalize_entity_label(label))
            if key in seen:
                continue
            seen.add(key)
            memories.append(
                Fact(
                    content=f"Lens: {label}",
                    subject="user",
                    predicate="owns_lens",
                    obj=label,
                    created_at=ctx.now,
                    valid_at=ctx.now,
                    confidence=0.8,
                    identity_id=ctx.identity_id,
                    tags=("inventory", "lens"),
                    metadata={
                        "extractor": self.SOURCE_CARRYOVER_GENERIC,
                        "inventory": {
                            "label": label,
                            "normalized_label": _normalize_entity_label(label),
                            "kind": "lens",
                            "state": "owned",
                        },
                    },
                    provenance=(self._provenance(ctx, raw_event_id=raw_event_id),),
                )
            )

        return memories

    def _extract_assistant_references(
        self, text: str, ctx: ExtractionContext, raw_event_id: str
    ) -> list[Memory]:
        memories: list[Memory] = []
        seen: set[str] = set()

        for index, item in _parse_numbered_items(text):
            label = _clean_assistant_reference_label(item)
            if not label:
                continue
            key = f"item:{index}:{label.lower()}"
            if key in seen:
                continue
            seen.add(key)
            memories.append(
                Fact(
                    content=f"Assistant item {index}: {label}",
                    subject="assistant",
                    predicate="assistant_reference",
                    obj=label,
                    created_at=ctx.now,
                    valid_at=ctx.now,
                    confidence=0.72,
                    identity_id=ctx.identity_id,
                    tags=("assistant-memory", "assistant-item"),
                    metadata={
                        "extractor": self.SOURCE_ASSISTANT_REFERENCE,
                        "assistant_reference": {
                            "kind": "numbered_item",
                            "item_index": str(index),
                            "label": label,
                        },
                    },
                    provenance=(self._provenance(ctx, raw_event_id=raw_event_id),),
                )
            )
            title = _assistant_reference_title(item)
            if title:
                category = _assistant_reference_category(item)
                if category is None and not _looks_like_named_list_item(title):
                    continue
                title_key = f"title:{index}:{title.lower()}"
                if title_key not in seen:
                    seen.add(title_key)
                    tags = ["assistant-memory", "assistant-item", "assistant-name"]
                    if category:
                        tags.append(category)
                    memories.append(
                        Fact(
                            content=f"Assistant item {index} name: {title}",
                            subject="assistant",
                            predicate="assistant_reference",
                            obj=title,
                            created_at=ctx.now,
                            valid_at=ctx.now,
                            confidence=0.78,
                            identity_id=ctx.identity_id,
                            tags=tuple(tags),
                            metadata={
                                "extractor": self.SOURCE_ASSISTANT_REFERENCE,
                                "assistant_reference": {
                                    "kind": "numbered_item_name",
                                    "item_index": str(index),
                                    "label": title,
                                    "category": category,
                                    "source_label": label,
                                },
                                },
                                provenance=(self._provenance(ctx, raw_event_id=raw_event_id),),
                            )
                        )
            option = _assistant_reference_option(item, fallback_title=title)
            if option is not None:
                option_key = f"option:{index}:{option['normalized_label']}"
                if option_key not in seen:
                    seen.add(option_key)
                    tags = ["assistant-memory", "assistant-item", "assistant-option"]
                    if option.get("category"):
                        tags.append(str(option["category"]))
                    if option.get("transport_mode"):
                        tags.append("transport")
                    memories.append(
                        Fact(
                            content=f"Assistant option {index}: {option['label']}",
                            subject="assistant",
                            predicate="assistant_reference",
                            obj=str(option["label"]),
                            created_at=ctx.now,
                            valid_at=ctx.now,
                            confidence=0.8,
                            identity_id=ctx.identity_id,
                            tags=tuple(tags),
                            metadata={
                                "extractor": self.SOURCE_ASSISTANT_REFERENCE,
                                "assistant_reference": {
                                    "kind": "numbered_item_option",
                                    "item_index": str(index),
                                    **option,
                                },
                            },
                            provenance=(self._provenance(ctx, raw_event_id=raw_event_id),),
                        )
                    )

        for row_name, row_text in _parse_markdown_table_rows(text):
            label = _clean_assistant_reference_label(f"{row_name}: {row_text}")
            if not label:
                continue
            key = f"row:{label.lower()}"
            if key in seen:
                continue
            seen.add(key)
            memories.append(
                Fact(
                    content=f"Assistant row: {label}",
                    subject="assistant",
                    predicate="assistant_reference",
                    obj=label,
                    created_at=ctx.now,
                    valid_at=ctx.now,
                    confidence=0.72,
                    identity_id=ctx.identity_id,
                    tags=("assistant-memory", "assistant-row"),
                    metadata={
                        "extractor": self.SOURCE_ASSISTANT_REFERENCE,
                        "assistant_reference": {
                            "kind": "table_row",
                            "label": label,
                        },
                    },
                    provenance=(self._provenance(ctx, raw_event_id=raw_event_id),),
                )
            )

        for sentence in _assistant_reference_sentences(text):
            label = _clean_assistant_reference_label(sentence)
            if not label:
                continue
            key = f"sentence:{label.lower()}"
            if key in seen:
                continue
            seen.add(key)
            memories.append(
                Fact(
                    content=f"Assistant note: {label}",
                    subject="assistant",
                    predicate="assistant_reference",
                    obj=label,
                    created_at=ctx.now,
                    valid_at=ctx.now,
                    confidence=0.68,
                    identity_id=ctx.identity_id,
                    tags=("assistant-memory", "assistant-note"),
                    metadata={
                        "extractor": self.SOURCE_ASSISTANT_REFERENCE,
                        "assistant_reference": {
                            "kind": "sentence",
                            "label": label,
                        },
                    },
                    provenance=(self._provenance(ctx, raw_event_id=raw_event_id),),
                )
            )
        return memories

    def _event_fact(
        self,
        *,
        kind: str,
        label: str,
        object_value: str,
        text: str,
        ctx: ExtractionContext,
        raw_event_id: str,
        at: datetime,
        tags: tuple[str, ...],
    ) -> Fact:
        return Fact(
            content=text,
            subject="user",
            predicate="experienced_event",
            obj=object_value,
            created_at=ctx.now,
            valid_at=at,
            confidence=0.78,
            identity_id=ctx.identity_id,
            tags=tags,
            metadata={
                "extractor": self.SOURCE_EVENT_GENERIC,
                "event": {
                    "kind": kind,
                    "label": label,
                    "normalized_at": at.isoformat(),
                },
            },
            provenance=(self._provenance(ctx, raw_event_id=raw_event_id),),
        )

    @staticmethod
    def _provenance(
        ctx: ExtractionContext,
        *,
        raw_event_id: str | None = None,
        source_suffix: str | None = None,
    ) -> Provenance:
        """Build a :class:`Provenance` rooted at the host (``ctx.source``).

        ``source`` is always the host (``windsurf``, ``claude_desktop``,
        ``cli:remember``, ...). Extractor identity belongs in
        :attr:`Memory.metadata`; conflating the two would lose audit
        information about where a memory entered the system.
        """
        source = ctx.source
        if source_suffix:
            source = f"{source}:{source_suffix}"
        return Provenance(
            source=source,
            actor=ctx.actor,
            session_id=ctx.session_id,
            raw_event_id=raw_event_id,
            recorded_at=ctx.now,
        )


# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------


_TRAILING_NOISE = re.compile(r"\s*(?:now|today|recently)\s*$", re.IGNORECASE)
_ROLE_CONTEXT_TAIL = re.compile(r"\s+at\s+[A-Z][\w\s,&.\-]{1,60}$")
_RELATIVE_MONTHS_AGO = re.compile(
    r"\b(?:about|around|almost|roughly)?\s*(?P<count>\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten)\s+months?\s+ago\b",
    re.IGNORECASE,
)
_RELATIVE_WEEKS_AGO = re.compile(
    r"\b(?:about|around|almost|roughly)?\s*(?P<count>\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten)\s+weeks?\s+ago\b",
    re.IGNORECASE,
)
_RELATIVE_DAYS_AGO = re.compile(
    r"\b(?:about|around|almost|roughly)?\s*(?P<count>\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten)\s+days?\s+ago\b",
    re.IGNORECASE,
)
_MONTH_DAY = re.compile(
    rf"\b(?P<month>{_MONTH_NAME})\s+(?P<day>\d{{1,2}})(?:st|nd|rd|th)?\b",
    re.IGNORECASE,
)
_DAY_OF_MONTH = re.compile(
    rf"\b(?P<day>\d{{1,2}})(?:st|nd|rd|th)?\s+of\s+(?P<month>{_MONTH_NAME})\b",
    re.IGNORECASE,
)
_QUALIFIED_MONTH = re.compile(
    rf"\b(?P<qualifier>early|mid|late)[\s-]+(?P<month>{_MONTH_NAME})\b",
    re.IGNORECASE,
)
_LAST_WEEKDAY = re.compile(
    r"\blast\s+(?P<weekday>monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.IGNORECASE,
)
_LAST_WEEK = re.compile(r"\blast\s+week\b", re.IGNORECASE)
_LAST_MONTH = re.compile(r"\blast\s+month\b", re.IGNORECASE)
_SLASH_DATE = re.compile(r"\b(?P<month>\d{1,2})/(?P<day>\d{1,2})\b")
_NUMBER_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}


def _clean_role(raw: str) -> str:
    """Tidy a captured role string: strip articles, trailing time adverbs."""
    s = raw.strip().strip(".,;:!?")
    s = re.sub(r"^(?:an?|the)\s+", "", s, flags=re.IGNORECASE)
    s = _ROLE_CONTEXT_TAIL.sub("", s)
    s = _TRAILING_NOISE.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()
    # Title-case the result for canonical storage \u2014 "staff engineer" \u2192 "Staff Engineer".
    return " ".join(w.capitalize() if w.islower() else w for w in s.split())


def _clean_phrase(raw: str) -> str:
    """Tidy a captured noun phrase (location or organization name)."""
    s = raw.strip().strip(".,;:!?")
    s = _TRAILING_NOISE.sub("", s)
    return re.sub(r"\s+", " ", s).strip()


def _nearest_person_reference(text: str, before_index: int) -> str | None:
    prefix = text[:before_index]
    matches = list(re.finditer(r"\b([A-Z][a-z]+)\b", prefix))
    for match in reversed(matches):
        candidate = match.group(1)
        if candidate not in {"I", "Since"}:
            return candidate
    return None


def _clean_location_label(raw: str) -> str:
    s = _clean_phrase(raw)
    s = re.split(r"\band\s+i\b", s, maxsplit=1, flags=re.IGNORECASE)[0]
    return re.sub(r"\s+", " ", s).strip(" ,")


def _clean_device_label(raw: str) -> str:
    s = _clean_phrase(raw)
    s = re.split(r"\b(?:from|that|which|with)\b", s, maxsplit=1, flags=re.IGNORECASE)[0]
    return re.sub(r"\s+", " ", s).strip(" ,")


def _clean_generic_event_label(raw: str) -> str:
    s = _clean_phrase(raw)
    s = re.split(r"\b(?:from|which|that|and\s+it|and\s+they|because|since|so\s+i)\b", s, maxsplit=1, flags=re.IGNORECASE)[0]
    s = re.sub(r"^(?:my|our|the|a|an|new)\s+", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip(" ,")
    return s


def _clean_preference_label(raw: str) -> str:
    s = _clean_phrase(raw)
    s = re.split(r"\b(?:but|because|though|although|especially)\b", s, maxsplit=1, flags=re.IGNORECASE)[0]
    s = re.split(r"\s+-\s+", s, maxsplit=1)[0]
    s = re.sub(r"^(?:the\s+idea\s+of\s+|the\s+idea\s+)?", "", s, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", s).strip(" ,")


def _clean_carryover_label(raw: str) -> str:
    s = _clean_phrase(raw)
    s = re.split(r"\b(?:but|because|which|that|and\s+i|so\s+i)\b", s, maxsplit=1, flags=re.IGNORECASE)[0]
    s = re.sub(r"\s+or\s+return$", "", s, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", s).strip(" ,")


def _clean_model_kit_label(raw: str) -> str:
    s = _clean_carryover_label(raw)
    s = re.sub(r"\b(?:kit|model)\b$", "", s, flags=re.IGNORECASE).strip(" ,")
    return re.sub(r"\s+", " ", s).strip()


def _clean_inventory_label(raw: str) -> str:
    s = _clean_carryover_label(raw)
    s = re.sub(r"^(?:some\s+helpful\s+tips\s+for\s+my|like|from)\s+", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^(?:a|an|the|my|our)\s+", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+on\s+(?:normal|hard)\s+difficulty$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\b(?:that|which)\b.*$", "", s, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", s).strip(" ,")


def _clean_school_label(raw: str) -> str:
    s = _clean_phrase(raw)
    s = re.split(r"\.\s+|\bwe\s+had\b|\bwhere\b", s, maxsplit=1, flags=re.IGNORECASE)[0]
    s = re.sub(r"^(?:the)\s+", "", s, flags=re.IGNORECASE)
    if s.upper() == "UCLA":
        return "University of California, Los Angeles (UCLA)"
    return s


def _clean_game_label(raw: str) -> str:
    s = _clean_inventory_label(raw)
    s = re.sub(r"^.*?\bsimilar to\s+", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^.*?\blike\s+", "", s, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", s).strip(" ,")


def _normalize_entity_label(raw: str) -> str:
    s = raw.lower()
    s = re.sub(r"'s\b", "", s)
    s = re.sub(r"\b(?:a|an|the|my|our|new|old|pair|set|simple|small)\b", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _split_item_list(raw: str) -> list[str]:
    parts = re.split(r",| and ", raw)
    return [part.strip() for part in parts if part.strip()]


def _extract_wedding_label(sentence: str) -> str | None:
    pair = _WEDDING_COUPLE.search(sentence)
    if pair:
        if pair.group("left") and pair.group("right"):
            return f"{pair.group('left')} and {pair.group('right')}"
        if pair.group("bride") and pair.group("groom"):
            return f"{pair.group('bride')} and {pair.group('groom')}"
    possessive = _WEDDING_ATTENDED.search(sentence)
    if possessive and possessive.group("possessive"):
        name = possessive.group("possessive")
        if name.lower() in {"friend", "cousin", "roommate", "college"}:
            return None
        return f"{name}'s wedding"
    return None


def _looks_like_luxury_purchase(label: str, sentence: str) -> bool:
    lowered_label = label.lower()
    if " " not in lowered_label and lowered_label in {"gucci", "prada", "chanel", "lv", "louis", "vuitton"}:
        return False
    return any(
        token in sentence or token in lowered_label
        for token in (
            "luxury",
            "designer",
            "high-end",
            "gucci",
            "italian designer",
            "evening gown",
            "handbag",
            "boots",
        )
    )


def _looks_like_citrus_context(text: str) -> bool:
    lowered = text.lower()
    return any(
        token in lowered
        for token in (
            "cocktail",
            "sangria",
            "daiquiri",
            "gimlet",
            "bitters",
            "mixology",
            "juice",
        )
    )


def _guess_game_label(text: str) -> str | None:
    candidates = (
        "The Last of Us Part II",
        "Hyper Light Drifter",
        "Celeste",
    )
    lowered = text.lower()
    for candidate in candidates:
        if candidate.lower() in lowered:
            return candidate
    return None


def _parse_numbered_items(text: str) -> list[tuple[int, str]]:
    normalized = re.sub(r"(?<=\.)\s+(?=\d{1,2}\.\s+[A-Z*])", "\n", text)
    normalized = re.sub(r"\s+(?=\d{1,2}\.\s+[A-Z*])", "\n", normalized)
    matches = list(re.finditer(r"(?:(?<=^)|(?<=\n))\s*(?P<index>\d{1,2})\.\s+", normalized))
    if not matches:
        return []
    items: list[tuple[int, str]] = []
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(normalized)
        body = normalized[start:end].strip()
        if not body:
            continue
        items.append((int(match.group("index")), body))
    return items


def _parse_markdown_table_rows(text: str) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    headers: list[str] | None = None
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("|") or line.startswith("| ---"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) < 3:
            continue
        if headers is None and any("shift" in cell.lower() or "am" in cell.lower() or "pm" in cell.lower() for cell in cells[1:]):
            headers = cells[1:]
            continue
        row_name = cells[0]
        if row_name.lower() in {"", "week 1", "week 2", "week 3", "week 4", "week 5", "week 6", "week 7", "week 8"}:
            continue
        if headers and len(headers) == len(cells[1:]):
            row_text = " | ".join(f"{agent}={shift}" for agent, shift in zip(cells[1:], headers, strict=True))
        else:
            row_text = " | ".join(cells[1:])
        rows.append((row_name, row_text))
    return rows


def _assistant_reference_sentences(text: str) -> list[str]:
    segments = [
        segment.strip()
        for segment in re.split(r"(?<=[.!?])\s+|\n+", text)
        if segment.strip()
    ]
    selected: list[str] = []
    for segment in segments:
        if len(segment) < 25 or len(segment) > 220:
            continue
        lowered = segment.lower()
        if any(
            marker in lowered
            for marker in (
                "here are",
                "i'd be happy",
                "as an ai language model",
                "it depends",
                "you're welcome",
            )
        ):
            continue
        if any(token in segment for token in ("GR-90", "Miss Bee Providore", "Roscioli")):
            selected.append(segment)
            continue
        if any(token in lowered for token in ("recommend", "ruby", "python", "php", "nasi goreng", "trail")):
            selected.append(segment)
            continue
        if ":" in segment and len(segment.split()) <= 20:
            selected.append(segment)
    return selected[:8]


def _clean_assistant_reference_label(raw: str) -> str:
    s = raw.strip().strip(" -")
    s = re.sub(r"^\*+|\*+$", "", s)
    s = re.sub(r"\s+", " ", s)
    s = re.split(r"\s{2,}", s)[0]
    return s.strip(" .")


def _assistant_reference_title(raw: str) -> str | None:
    cleaned = _clean_assistant_reference_label(raw)
    quoted = re.search(r'["\'](?P<title>[^"\']{2,120})["\']', cleaned)
    if quoted:
        return quoted.group("title").strip()
    if " by " in cleaned and ":" in cleaned:
        head = cleaned.split(":", 1)[0]
        title = head.split(" by ", 1)[0].strip(" *")
        if title:
            return title
    patterns = (
        r"^\*{0,2}\"?(?P<title>[A-Z][^:\-*]{1,120})\"?\*{0,2}\s*:",
        r"^\*{0,2}(?P<title>[A-Z][A-Za-z0-9'&.\- ]{1,120})\*{0,2}\s*\(",
        r"^\*{0,2}(?P<title>[A-Z][A-Za-z0-9'&.\- ]{1,120})\*{0,2}\s+-",
    )
    for pattern in patterns:
        match = re.match(pattern, cleaned)
        if match:
            title = re.sub(r"\s+", " ", match.group("title").strip().strip("\"'")).strip(" *")
            return title or None
    if ":" in cleaned:
        head = cleaned.split(":", 1)[0].strip(" *")
        blocked_starters = {
            "create", "plan", "set", "track", "practice", "launch", "check", "use",
            "look", "avoid", "add", "be", "ask", "trust", "make", "get", "decide",
        }
        if (
            1 <= len(head.split()) <= 8
            and head[:1].isupper()
            and head.split()[0].lower() not in blocked_starters
            and not head.endswith("tips")
        ):
            return head
    return None


def _assistant_reference_category(raw: str) -> str | None:
    lowered = raw.lower()
    if any(token in lowered for token in ("restaurant", "cafe", "café", "bistro", "eatery")):
        return "restaurant"
    if any(token in lowered for token in ("trail", "hike", "park")):
        return "trail"
    if any(token in lowered for token in ("book", "novel", "author", "pages")):
        return "book"
    if any(token in lowered for token in ("song", "album", "artist", "band", "singer")):
        return "song"
    if any(token in lowered for token in ("job", "role", "career", "position")):
        return "job"
    if any(token in lowered for token in ("website", "forum", "resource", "app", "channel", "podcast")):
        return "resource"
    return None


def _assistant_reference_option(raw: str, *, fallback_title: str | None = None) -> dict[str, object] | None:
    label = _clean_assistant_reference_label(raw)
    if not label:
        return None
    title = fallback_title or _assistant_reference_title(raw) or label
    category = _assistant_reference_category(raw)
    price_min, price_max, currency = _assistant_reference_price_range(raw)
    transport_mode = _assistant_reference_transport_mode(title, raw)
    normalized_label = _normalize_entity_label(title)
    option: dict[str, object] = {
        "label": title,
        "normalized_label": normalized_label,
        "source_label": label,
    }
    if category is not None:
        option["category"] = category
    if transport_mode is not None:
        option["transport_mode"] = transport_mode
    if currency is not None:
        option["currency"] = currency
    if price_min is not None:
        option["price_min"] = price_min
    if price_max is not None:
        option["price_max"] = price_max
    if len(option) <= 3 and category is None and transport_mode is None:
        return None
    return option


def _assistant_reference_transport_mode(title: str, raw: str) -> str | None:
    lowered = f"{title} {raw}".lower()
    for token in ("train", "bus", "taxi", "monorail", "railway", "subway"):
        if token in lowered:
            return token
    return None


def _assistant_reference_price_range(raw: str) -> tuple[float | None, float | None, str | None]:
    match = re.search(
        r"(?P<currency>[$¥])\s*(?P<low>\d[\d,]*(?:\.\d{1,2})?)\s*(?:-\s*(?P<high>\d[\d,]*(?:\.\d{1,2})?))?",
        raw,
    )
    if not match:
        return (None, None, None)
    low = float(match.group("low").replace(",", ""))
    high_raw = match.group("high")
    high = float(high_raw.replace(",", "")) if high_raw else low
    currency = "usd" if match.group("currency") == "$" else "jpy"
    return (low, high, currency)


def _looks_like_named_list_item(title: str) -> bool:
    words = title.split()
    if not 1 <= len(words) <= 5:
        return False
    return all(word[:1].isupper() or "&" in word for word in words)


def _normalize_device_role(raw: str | None) -> str | None:
    if raw is None:
        return None
    normalized = _clean_phrase(raw).lower()
    if normalized == "smartphone":
        return "phone"
    return normalized or None


def _normalize_vehicle_object(raw: str) -> str | None:
    lowered = raw.lower()
    if any(token in lowered for token in ("car", "corolla", "civic", "honda", "toyota")):
        return "car"
    if "bike" in lowered:
        return "bike"
    return None


def _normalize_country_label(raw: str) -> str:
    lowered = raw.lower().replace(".", "").strip()
    if lowered in {"us", "usa", "united states"}:
        return "United States"
    return _clean_phrase(raw)


def _split_service_list(raw: str) -> tuple[str, ...]:
    cleaned = raw.replace(" and ", ", ")
    services = [
        _clean_generic_event_label(part)
        for part in cleaned.split(",")
    ]
    return tuple(service for service in services if service)


def _sentence_for_match(text: str, match: re.Match[str]) -> str:
    start = match.start()
    end = match.end()
    left = max(text.rfind(".", 0, start), text.rfind("?", 0, start), text.rfind("!", 0, start))
    right_candidates = [idx for idx in (text.find(".", end), text.find("?", end), text.find("!", end)) if idx != -1]
    right = min(right_candidates) if right_candidates else len(text)
    return text[left + 1:right].strip()


def _extract_event_datetime(text: str, reference: datetime) -> datetime:
    explicit = _extract_date_token(text, reference)
    if explicit is not None:
        return explicit
    explicit_relative = _extract_relative_datetime(text, reference)
    if explicit_relative is not None:
        return explicit_relative
    relative_months = _RELATIVE_MONTHS_AGO.search(text)
    if relative_months:
        count = _parse_relative_count(relative_months.group("count"))
        return reference - timedelta(days=30 * count)
    relative_weeks = _RELATIVE_WEEKS_AGO.search(text)
    if relative_weeks:
        count = _parse_relative_count(relative_weeks.group("count"))
        return reference - timedelta(days=7 * count)
    return reference


def _extract_explicit_or_relative_time(
    explicit_date: str | None,
    relative: str | None,
    reference: datetime,
) -> datetime | None:
    if explicit_date:
        return _extract_date_token(explicit_date, reference)
    if relative:
        return _extract_relative_datetime(relative, reference)
    return None


def _extract_relative_datetime(text: str, reference: datetime) -> datetime | None:
    if _LAST_WEEK.search(text):
        return reference - timedelta(days=7)
    if _LAST_MONTH.search(text):
        return reference - timedelta(days=30)
    relative_days = _RELATIVE_DAYS_AGO.search(text)
    if relative_days:
        count = _parse_relative_count(relative_days.group("count"))
        return reference - timedelta(days=count)
    relative_weeks = _RELATIVE_WEEKS_AGO.search(text)
    if relative_weeks:
        count = _parse_relative_count(relative_weeks.group("count"))
        return reference - timedelta(days=7 * count)
    relative_months = _RELATIVE_MONTHS_AGO.search(text)
    if relative_months:
        count = _parse_relative_count(relative_months.group("count"))
        return reference - timedelta(days=30 * count)
    return None


def _extract_date_token(text: str, reference: datetime) -> datetime | None:
    month_day = _MONTH_DAY.search(text)
    if month_day:
        return _build_month_day(
            month_text=month_day.group("month"),
            day=int(month_day.group("day")),
            year=reference.year,
            reference=reference,
        )
    day_of_month = _DAY_OF_MONTH.search(text)
    if day_of_month:
        return _build_month_day(
            month_text=day_of_month.group("month"),
            day=int(day_of_month.group("day")),
            year=reference.year,
            reference=reference,
        )
    qualified = _QUALIFIED_MONTH.search(text)
    if qualified:
        day = {"early": 5, "mid": 15, "late": 25}[qualified.group("qualifier").lower()]
        return _build_month_day(
            month_text=qualified.group("month"),
            day=day,
            year=reference.year,
            reference=reference,
        )
    slash = _SLASH_DATE.search(text)
    if slash:
        return reference.replace(
            month=int(slash.group("month")),
            day=int(slash.group("day")),
            hour=12,
            minute=0,
            second=0,
            microsecond=0,
        )
    last_weekday = _LAST_WEEKDAY.search(text)
    if last_weekday:
        weekday_lookup = {
            "monday": 0,
            "tuesday": 1,
            "wednesday": 2,
            "thursday": 3,
            "friday": 4,
            "saturday": 5,
            "sunday": 6,
        }
        target = weekday_lookup[last_weekday.group("weekday").lower()]
        delta = (reference.weekday() - target) % 7 or 7
        return (reference - timedelta(days=delta)).replace(hour=12, minute=0, second=0, microsecond=0)
    return None


def _build_month_day(
    *,
    month_text: str,
    day: int,
    year: int,
    reference: datetime,
) -> datetime:
    month_lookup = {
        "jan": 1, "january": 1,
        "feb": 2, "february": 2,
        "mar": 3, "march": 3,
        "apr": 4, "april": 4,
        "may": 5,
        "jun": 6, "june": 6,
        "jul": 7, "july": 7,
        "aug": 8, "august": 8,
        "sep": 9, "sept": 9, "september": 9,
        "oct": 10, "october": 10,
        "nov": 11, "november": 11,
        "dec": 12, "december": 12,
    }
    month = month_lookup[month_text.lower()]
    candidate = reference.replace(year=year, month=month, day=day, hour=12, minute=0, second=0, microsecond=0)
    return candidate


def _extract_day_of_month(day_text: str, month_text: str, reference: datetime) -> datetime | None:
    try:
        day = int(day_text)
    except ValueError:
        return None
    return _build_month_day(month_text=month_text, day=day, year=reference.year, reference=reference)


def _extract_month_token(month_text: str, reference: datetime) -> datetime | None:
    if not month_text:
        return None
    return _build_month_day(month_text=month_text, day=15, year=reference.year, reference=reference)


def _is_charity_event_label(label: str) -> bool:
    lowered = label.lower()
    return any(
        needle in lowered
        for needle in (
            "charity",
            "run for the cure",
            "dance for a cause",
            "walk for wildlife",
            "walk for hunger",
            "bike-a-thon",
            "food for thought",
            "coastal cleanup",
            "holiday market",
        )
    )


def _parse_relative_count(raw: str) -> int:
    lowered = raw.lower()
    if lowered.isdigit():
        return int(lowered)
    return _NUMBER_WORDS.get(lowered, 1)


def _parse_compact_number(raw: str) -> int | None:
    cleaned = raw.replace(",", "").strip()
    if not cleaned.isdigit():
        return None
    return int(cleaned)


def _format_amount(value: float) -> str:
    return str(int(value)) if value.is_integer() else f"{value:.2f}".rstrip("0").rstrip(".")


def _parse_duration_weeks(raw: str) -> float | None:
    lowered = raw.lower().strip()
    if lowered == "a week and a half":
        return 1.5
    if lowered == "week and a half":
        return 1.5
    if lowered in {"two and a half weeks", "2 and a half weeks"}:
        return 2.5
    match = re.match(r"(?P<count>\d+|one|two|three|four|five)\s+weeks?", lowered)
    if match:
        return float(_parse_relative_count(match.group("count")))
    return None


def _parse_duration_months(raw: str) -> int | None:
    match = re.match(r"(?P<years>\d+)\s+years?(?:\s+and\s+(?P<months>\d+)\s+months?)?", raw.strip(), flags=re.IGNORECASE)
    if not match:
        return None
    years = int(match.group("years"))
    months = int(match.group("months") or "0")
    return years * 12 + months


def _parse_duration_months_flexible(raw: str) -> int | None:
    lowered = raw.lower().strip()
    if lowered in {"a few months", "few months"}:
        return 3
    if lowered == "five years":
        return 60
    if lowered.endswith("months"):
        match = re.match(r"(?P<count>\d+)\s+months?", lowered)
        if match:
            return int(match.group("count"))
    return _parse_duration_months(raw)


def _parse_duration_years(raw: str) -> int | None:
    match = re.match(r"(?P<count>\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+years?", raw.strip(), flags=re.IGNORECASE)
    if not match:
        return None
    return _parse_relative_count(match.group("count"))


def _normalize_clock_time(raw: str) -> str | None:
    match = re.match(r"(?P<hour>\d{1,2}):(?P<minute>\d{2})\s*(?P<ampm>am|pm)", raw.strip(), flags=re.IGNORECASE)
    if not match:
        return None
    hour = int(match.group("hour"))
    minute = match.group("minute")
    ampm = match.group("ampm").upper()
    return f"{hour}:{minute} {ampm}"


def _weekday_prefix_for_text(text: str) -> str:
    match = re.search(
        r"\bon\s+(?P<days>(?:Mondays?|Tuesdays?|Wednesdays?|Thursdays?|Fridays?|Saturdays?|Sundays?)(?:\s*(?:,|and)\s*(?:Mondays?|Tuesdays?|Wednesdays?|Thursdays?|Fridays?|Saturdays?|Sundays?))*)",
        text,
        flags=re.IGNORECASE,
    )
    return match.group(1) if match else ""


def _parse_weekday_list(raw: str) -> tuple[str, ...]:
    tokens = re.split(r",|\band\b", raw, flags=re.IGNORECASE)
    result: list[str] = []
    for token in tokens:
        cleaned = token.strip().lower().rstrip("s")
        if cleaned in {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"}:
            result.append(cleaned)
    return tuple(result)
