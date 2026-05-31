"""Context synthesis: token-budget-aware packing.

Given a ranked list of memories and a target model with a fixed token
budget, produce the final string blob (or structured tool-result payload)
that will land in the prompt. Awareness of model-specific quirks (Claude's
preference for XML, GPT's preference for delimited blocks) lives here.

v0.3.
"""

from reify.synthesize.pack import (
    CLAUDE_PROFILE,
    GEMINI_PROFILE,
    GPT_PROFILE,
    LOCAL_PROFILE,
    ModelProfile,
    PackFormat,
    PackedContext,
    get_profile,
    pack,
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
