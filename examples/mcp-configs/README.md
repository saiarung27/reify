# MCP host configurations

Reify ships an MCP stdio server (`reify serve`) that exposes seven tools to
any MCP-compatible host: `remember`, `recall`, `reflect`, `whatchanged`,
`whatdoyouknow`, `forget`, `identity_bond`.

The snippets below are working, copy-paste configurations for the three
most common hosts. Each example assumes:

- Reify is installed in a virtualenv at `~/projects/reify/.venv` (adjust
  the absolute path if yours differs), and
- `reify init` has been run at least once so `~/.reify/store.db` exists.

If you installed Reify globally (`pipx install reify`), replace
`/abs/path/to/reify/.venv/bin/reify` with the output of `which reify`.

---

## Claude Desktop

File: `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS)
or `%APPDATA%\Claude\claude_desktop_config.json` (Windows).

```json
{
  "mcpServers": {
    "reify": {
      "command": "/abs/path/to/reify/.venv/bin/reify",
      "args": ["serve"],
      "env": {
        "REIFY_HOME": "~/.reify"
      }
    }
  }
}
```

Restart Claude Desktop. The seven Reify tools appear in the tools list.

## Cursor

File: `~/.cursor/mcp.json` (or via Settings -> MCP in the UI).

```json
{
  "mcpServers": {
    "reify": {
      "command": "/abs/path/to/reify/.venv/bin/reify",
      "args": ["serve"]
    }
  }
}
```

## Windsurf

File: `~/.codeium/windsurf/mcp_config.json`.

```json
{
  "mcpServers": {
    "reify": {
      "command": "/abs/path/to/reify/.venv/bin/reify",
      "args": ["serve"],
      "env": {
        "REIFY_HOME": "/Users/<you>/.reify"
      }
    }
  }
}
```

---

## Optional: enable an LLM-backed reconsolidator

By default Reify uses a deterministic extractor only. To turn on the LLM
path (BTG and ARW improvements), add the relevant env var:

```json
"env": {
  "OPENAI_API_KEY": "sk-...",
  "REIFY_LLM_RECONSOLIDATE": "openai:gpt-4o-mini"
}
```

Or for a fully local stack via Ollama:

```json
"env": {
  "OLLAMA_BASE_URL": "http://localhost:11434",
  "REIFY_LLM_RECONSOLIDATE": "ollama:llama3.1"
}
```

No API keys are required for the wedge demo or for any deterministic-only
flow. See `SECURITY.md` for hardening guidance before exposing the HTTP
servers (`reify serve-rest`, `serve-proxy`, `serve-mcp-sse`) on a
non-loopback interface.
