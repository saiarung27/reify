# Security Policy

Reify is pre-alpha software. Treat any deployment as experimental.

## Reporting a vulnerability

**Do not open a public issue.** If you believe you have found a security
vulnerability in Reify, please report it privately:

- Open a GitHub Security Advisory:
  <https://github.com/saiarung27/reify/security/advisories/new>
- Or email the maintainers (address listed in the GitHub repository profile).

Please include:

- A description of the vulnerability and the impact.
- Steps to reproduce, ideally with a minimal proof of concept.
- The affected version(s) (`reify version`).
- Any suggested mitigation, if you have one.

We will acknowledge your report within **3 business days** and aim to confirm
or refute the issue within **10 business days**. Coordinated disclosure
timelines are negotiated case-by-case.

## Scope

In scope:

- The Reify Python library and CLI (`src/reify/`).
- The MCP stdio server, REST API, OpenAI-compatible proxy, and MCP SSE
  server shipped under `src/reify/server/`.
- The official SDKs under `sdk/`.

Out of scope:

- Vulnerabilities in third-party dependencies (please report upstream).
- Issues that require physical or local-account access to a machine
  already running Reify.
- Findings against demo or example code clearly marked as such.

## Hardening guidance for operators

The defaults assume a single-user, local-machine deployment.

- **Default bind:** All HTTP servers (`reify serve-rest`, `serve-proxy`,
  `serve-mcp-sse`, `serve-dashboard`) bind to `127.0.0.1` by default.
- **Non-loopback bind guard:** If you pass `--host 0.0.0.0` (or any other
  non-loopback address), Reify **refuses to start** unless `REIFY_HTTP_TOKEN`
  is set in the environment. When the token is set, every non-health
  request must present a matching `Authorization: Bearer <token>` header,
  enforced by the `BearerTokenMiddleware` in `src/reify/server/security.py`.
  The guard is implemented by `assert_safe_bind` and is invoked by the CLI
  before `uvicorn.run()`.
- **Defense in depth:** the bind guard is a backstop, not a substitute for
  a properly configured reverse proxy with TLS. For anything beyond a
  single-user laptop, terminate TLS upstream and treat the bearer token as
  a shared secret.
- **Database:** `~/.reify/store.db` contains everything Reify remembers in
  plain text. Restrict file-system permissions accordingly.
- **PII redaction:** Reify includes a deterministic regex-based PII redactor
  that can detect and redact emails, phone numbers, SSN-shaped strings, and
  credit-card-shaped strings before storage. Enable with `REIFY_PII_REDACT=on`.
  This is a first-pass defense; contextual PII detection (e.g., "my mother's
  maiden name") is not covered by the current implementation.
- **API keys:** Reify reads provider credentials from the environment only
  when the matching provider is selected via the `provider:model` spec:
  - `OPENAI_API_KEY` for `openai:...`
  - `ANTHROPIC_API_KEY` for `anthropic:...`
  - Ollama uses `OLLAMA_BASE_URL` and is local by default; no key required.

  Never check any of these into source control. `.env` is gitignored.
- **Logs:** Reify uses standard Python logging. Do not pipe logs to a public
  destination if your memories contain PII.

If you are deploying Reify in a multi-user setting, treat it as untrusted
until v1.0 (multi-tenant auth, audit logging, and CORS hardening are on the
roadmap, see `ROADMAP.md`).

## Disclosure

After a fix is released, the advisory will be published with credit to the
reporter (unless you request otherwise) and a CVE will be requested if the
issue warrants one.
