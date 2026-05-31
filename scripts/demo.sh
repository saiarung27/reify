#!/usr/bin/env bash
# Reify wedge-demo end-to-end smoke test.
#
# Runs the four-step promotion narrative entirely against a local SQLite
# store with the deterministic extractor only. No LLM API keys, no network
# calls, no MCP host. Compresses a 60-day timeline into seconds via the
# CLI's `--at` flag.
#
# Usage:
#   ./scripts/demo.sh
#
# Exit code 0 = wedge demo passed; non-zero = something is broken.

set -euo pipefail

# --- Setup ---------------------------------------------------------------

# Use an isolated REIFY_HOME so the demo never touches a real install.
DEMO_HOME="$(mktemp -d -t reify-demo-XXXXXX)"
export REIFY_HOME="$DEMO_HOME"

# Make absolutely sure no LLM provider is configured for this run; the wedge
# demo must pass on the deterministic-extract path only.
unset OPENAI_API_KEY ANTHROPIC_API_KEY OLLAMA_BASE_URL || true
unset REIFY_LLM_EXTRACT REIFY_LLM_RECONSOLIDATE REIFY_LLM_INTENT || true

cleanup() {
  rm -rf "$DEMO_HOME"
}
trap cleanup EXIT

# Resolve the reify CLI. Prefer the active virtualenv if one is set up.
if [[ -x ".venv/bin/reify" ]]; then
  REIFY=".venv/bin/reify"
elif command -v reify >/dev/null 2>&1; then
  REIFY="reify"
else
  echo "[demo] reify CLI not found. Run: pip install -e \".[dev]\"" >&2
  exit 1
fi

USER_ID="demo-user"

step()  { printf "\n\033[1;34m== %s ==\033[0m\n" "$*"; }
ok()    { printf "\033[1;32m  ok\033[0m %s\n" "$*"; }
fail()  { printf "\033[1;31m  fail\033[0m %s\n" "$*" >&2; exit 1; }
assert_contains() {
  local haystack="$1" needle="$2" label="$3"
  if grep -qiF -- "$needle" <<<"$haystack"; then
    ok "$label contains '$needle'"
  else
    printf "%s\n" "$haystack"
    fail "$label did not contain '$needle'"
  fi
}

# --- Demo ----------------------------------------------------------------

step "init"
$REIFY init >/dev/null
ok "REIFY_HOME=$DEMO_HOME"

step "Day 1: remember promotion (Senior -> Staff)"
$REIFY remember "I just got promoted from Senior to Staff Engineer" \
  --identity "$USER_ID" \
  --at "2025-01-01T00:00:00Z" >/dev/null
$REIFY reflect >/dev/null
ok "ingested day-1 promotion"

step "Day 30: recall current title"
RECALL_OUT="$($REIFY recall "What's my title?" \
  --identity "$USER_ID" \
  --at "2025-01-30T00:00:00Z")"
echo "$RECALL_OUT"
assert_contains "$RECALL_OUT" "staff" "recall output"

step "Day 90: second promotion (Staff -> Principal)"
$REIFY remember "I got promoted from Staff to Principal Engineer" \
  --identity "$USER_ID" \
  --at "2025-04-01T00:00:00Z" >/dev/null
$REIFY reflect >/dev/null
ok "ingested day-90 promotion"

step "Day 120: recall current title"
RECALL2_OUT="$($REIFY recall "What's my title?" \
  --identity "$USER_ID" \
  --at "2025-04-30T00:00:00Z")"
echo "$RECALL2_OUT"
assert_contains "$RECALL2_OUT" "principal" "second recall output"

step "Day 120: cross-session knowledge (all live memories for identity)"
KNOW_OUT="$($REIFY what-do-you-know --identity "$USER_ID")"
echo "$KNOW_OUT"
assert_contains "$KNOW_OUT" "principal" "what-do-you-know output"

# Note: full BTG (step 3 of the wedge: 'what changed?' surfacing the
# Senior->Staff transition narrative) and ARW (step 4: cross-session
# inferences) require an LLM-backed reconsolidator. Those steps are exercised
# by tests/test_wedge_demo_acceptance.py with a fake provider. This script
# proves the deterministic, LLM-free path: ingest -> bitemporal recall ->
# supersession via second remember -> export.

step "Export"
EXPORT_FILE="$DEMO_HOME/export.md"
$REIFY export "$EXPORT_FILE" >/dev/null
test -s "$EXPORT_FILE" || fail "export file is empty"
ok "exported to $EXPORT_FILE ($(wc -c <"$EXPORT_FILE") bytes)"

printf "\n\033[1;32mWedge demo passed.\033[0m All four steps + export ran with no LLM and no network.\n"
