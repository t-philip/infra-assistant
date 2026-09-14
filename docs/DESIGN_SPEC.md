# infra-assistant — Design Specification

**Status:** alpha — feature-complete for its scope; not yet run against a live
Telegram bot or a live LLM provider.
**Last reconciled to code:** 2026-09-01.

This document explains how infra-assistant is built and why, and is kept honest
against the code. Where the build does not match the intent, §0.1 says so.

It is the bot half of a two-part system: a companion MCP server
(`homelab-mcp` is the reference one) supplies the read-only tools; this repo is
**only** the assistant.

---

## 0. Purpose

Replace a heavyweight agent framework whose only wanted function was "a
conversational bot over the infrastructure" with the smallest thing that does
that well:

- long-poll Telegram, restricted to an owner allowlist;
- one LLM call per model turn, behind an **ordered provider fallback chain**;
- a system prompt assembled from plain files;
- tools taken from one or more **MCP servers** — never defined locally;
- a bounded **model → tools → model** loop.

Non-goals, enforced: no self-writing skills, no plugins, no autonomous code
execution, no multi-channel, no built-in scheduler. If the model wants an action
taken, it returns the command for a human to run.

### 0.1 Deviations from intent (as of 2026-09-14, after a full code-review pass)

| Area | Intended | As built | Status |
|---|---|---|---|
| Telegram | `python-telegram-bot`, long-poll, owner allowlist | PTB v21+, `concurrent_updates(True)` + a **per-chat `asyncio.Lock`** (one chat's slow turn no longer freezes others), `filters.Chat` allowlist **plus** an in-handler id check; `/start` `/help` `/reset`; an `add_error_handler` that logs with a traceback and DMs the owner | **done** |
| Provider chain | ordered failover + peak-window skip, no difficulty routing | `ProviderChain` in `llm.py`; window skip via `[[llm.chain]].skip_between` UTC ranges; **failover now catches every per-entry failure** — a `ProviderError`, a `ConfigError` from *constructing* the provider, or any unforeseen exception (only `CancelledError` re-raises), so one misconfigured earlier entry can't disable the rest | **done** |
| Provider back ends | `litellm` default, hand-rolled `--no-litellm` mode | `LiteLLMProvider` (lazy import; exception text bounded to a class name + 300 chars before it is logged) + `HTTPProvider` (httpx → OpenAI-compatible `/chat/completions`, Ollama via `/v1`, trailing `/v1` in `base_url` tolerated; response **streamed** with an 8 MiB running cap so an oversized body can't be buffered) | **done** |
| System prompt | assembled from `persona.md` / `user.md` / `context.md`, tools injected from MCP | `prompt.py`; `persona.md` required, other two optional; fixed boundary trailer always appended; `prompt_dir` resolved so a not-found error names the absolute path | **done** |
| Memory | per-chat rolling history (last N), optional per-chat transcript log | `memory.py`; in-process `deque`, `history_turns` cap; `log_dir` transcripts are `chmod 0o600` on create and warn once past 5 MiB (no rotation — document a logrotate path) | **done** |
| MCP client | one or more servers, tools flattened, dispatch routed back | `mcp_client.py`; `mcp` 2.x high-level `Client`; stdio + streamable-http (+ bearer) + sse; tools namespaced `<server>__<tool>`; **`asyncio.wait_for` on every `call_tool`** (`[llm].tool_call_timeout_seconds`, default 30); a raw exception from `call_tool` is logged and returned as a generic string, never echoed; the one private-SDK import is `try`/`except`-guarded | **done** |
| Tool-call loop | cap on calls/turn + wall-clock cap; never a half-formed reply | `agent.py`; `max_tool_calls_per_turn` then tools withdrawn + one nudge; **`turn_wall_time_seconds` is now enforced with `asyncio.timeout` around each model call** (the between-calls check alone could not bound a hung `await`) → `TurnTimeout` → clean bot message; a cumulative `[llm].max_tool_output_chars` budget trims later tool results so a chatty turn can't overflow a small local fallback model | **done** |
| Explicitly absent | no exec/shell/file-write/self-mod/scheduler | `scripts/check_no_exec.py` — a **tripwire** (its own docstring says so): regex-matched imports/calls for `subprocess`, `asyncio.create_subprocess_*`, `os.system`/`popen`/`exec*`/`spawn*`/`fork`, `multiprocessing`, `runpy`, `importlib`, `ctypes`/`cffi`, `eval`/`exec`/`compile`, and file-writes (`.open("w"/"a")`, `open(p, "w")`, `.write_text/_bytes`) outside `memory.py`. Wired into CI | **done** |
| Startup failure | fail loud, not a silent crash loop | `cli._best_effort_startup_alert` — on a non-zero `run` exit it DMs the first owner one line before exiting (best-effort, silent on any problem); the real error is still in the log. Deployment-level supervision (systemd `StartLimit*`, watchdog) is left to the operator | **done** (minimal) |
| `check` | validate + probe | resolves secrets, assembles the prompt, connects each MCP server; **exits non-zero** if a secret is missing, a server is unreachable, every provider is in a peak window now, or a connected server exposes 0 tools. `--probe-llm` additionally makes one 1-token call per eligible provider and fails on any error | **done** |
| Packaging | `pipx install infra-assistant` from a release | builds locally with hatchling; not published; `httpx`/`litellm`/`mcp` now upper-bounded | pre-release |
| Live run | bot proven end to end against a real Telegram bot + real provider | every automated test uses fakes; a real stdio handshake against `homelab-mcp` is verified; `--probe-llm` against a real endpoint is not | **unverified** |
| Public-repo privacy tooling | `check_privacy.sh` + hook + workflow + denylist | present, byte-identical to the other public repos; active now that this is a standalone repo | **done** |
| Parallel tool calls | — | the tool calls the model asks for in one turn run **concurrently** (`asyncio.gather`), each still bounded by its own `tool_call_timeout`; results are appended in the model's original order and the caps/budgets applied in that order | **done** |
| Per-turn usage/cost logging | — | `LLMResponse` carries optional `provider_name`/`prompt_tokens`/`completion_tokens`/`total_tokens`/`cost_usd` (`None`, not `0`, when a provider doesn't report usage or the model is unpriced); `Agent.answer()` sums these across every model call in a turn and logs one `"turn complete: latency_ms=… model_calls=… tool_calls=… provider=… […]"` line, plain `key=value` so a log-scraping exporter needs no JSON dependency. Logging only — no exporter or dashboard ships in this repo; an operator wires their own from the log line | **done** (emit side only) |

---

## 1. Architecture

```
Telegram ──long-poll──► telegram_bot.on_message
                              │  owner allowlist (filter + explicit check)
                              ▼
                        Agent.answer(text, history)
                              │  system prompt + history + user msg
                              ▼
                     ┌── loop (≤ max_tool_calls, ≤ wall_time) ──┐
                     │  ProviderChain.complete(messages, tools)  │
                     │      try each [[llm.chain]] entry:        │
                     │        skip if inside a peak window       │
                     │        else call provider; on error →next │
                     │  response.wants_tools?                    │
                     │    yes → MCPPool.call(name, args) ─┐      │
                     │          append tool result ◄──────┘      │
                     │    no  → return text                      │
                     └──────────────────────────────────────────┘
                              │
                    memory.record(chat, q, a)   (only on success)
                    reply_chunked(message, text)  (≤ 4096 chars each)
```

Module map (`src/infra_assistant/`):

| Module | Responsibility |
|---|---|
| `config.py` | Parse + validate the TOML. `TimeWindow` (UTC, may wrap midnight). Resolve every `*_env` field against the environment; fail closed listing all gaps. |
| `prompt.py` | Read `persona.md` (+ optional `user.md` / `context.md`); append the fixed boundary trailer. |
| `memory.py` | `ChatMemory` — per-chat `deque` capped at `history_turns` exchanges; optional append-only transcript, `chmod 0o600`, size-warn. |
| `llm.py` | Provider-neutral types (`LLMResponse`, `ToolCall`), the `Completer`/`Provider` protocols, `ProviderChain` (failover — of any exception — + window skip), OpenAI-shape parsing helpers. |
| `providers/litellm_provider.py` | Default back end: one `litellm.acompletion`. Lazy import; exception text bounded before it is logged. |
| `providers/http_provider.py` | `--no-litellm` back end: a direct `httpx` POST to `/chat/completions`, response **streamed** with an 8 MiB running cap; error bodies logged, never echoed. |
| `mcp_client.py` | `MCPPool` — open N MCP `Client`s, flatten + namespace their tools, route calls back (each under `asyncio.wait_for`), render results (16 k-char cap). Injectable `transport_factory` for tests. |
| `agent.py` | `Agent.answer` — the bounded tool-calling loop; `_complete` wraps each model call in `asyncio.timeout(remaining budget)`; cumulative tool-output budget. |
| `telegram_bot.py` | PTB wiring (`concurrent_updates` + per-chat lock), allowlist, command handlers, `add_error_handler`, UTF-16-aware 4096-unit reply splitting. |
| `runner.py` | `build_runtime` (shared by `run`/`check`; closes the pool if a later step fails) and `serve` (drives the long-poll loop, cleans up). |
| `cli.py` | `infra-assistant run` / `infra-assistant check` (`--probe-llm`), `--no-litellm`, `--prompt-dir`, `--log-level`; best-effort owner alert on a `run` startup failure. |

---

## 2. The "no autonomy" guarantee

infra-assistant offers the model **no local capability**. It has no shell, no
code execution, no file-write tool, no self-modification, no scheduler. Every
tool it can call is one an external MCP server chose to expose.

Backed — as a **tripwire, not a proof** (the script's own docstring says so) —
by `scripts/check_no_exec.py` in CI: it regex-matches the package source for
imports/dotted calls of `subprocess`, `asyncio.create_subprocess_*`,
`os.system` / `popen` / `exec*` / `spawn*` / `fork`, `multiprocessing`,
`runpy`, `importlib`, `ctypes` / `cffi`, and `eval` / `exec` / `compile` as
calls, plus a file opened for writing (`.open("w"/"a")`, `open(p, "w")`,
`.write_text` / `.write_bytes`) anywhere but `memory.py`. Deliberate
obfuscation would slip past it; an accidental regression will not. The real
argument is this section and code review.

The read-only-ness of the *tools* is a separate guarantee that lives in the MCP
server — `homelab-mcp` enforces it four independent ways. This bot does not
re-check it; it simply never offers a write path of its own.

---

## 3. Configuration & secrets

TOML. Secrets are never stored in the file: a field ending in `_env` holds the
*name* of an environment variable. `Config.resolve_secrets()` reads the values
at startup and raises `ConfigError` listing **every** missing variable at once
(`telegram.bot_token_env`, each `[[llm.chain]].api_key_env`, each
`[[mcp.servers]].token_env`). A half-configured bot that silently drops a
provider or a tool server is worse than one that refuses to start.

`[[mcp.servers]].name` is constrained to `[a-z0-9][a-z0-9_-]*` so the
`<name>__<tool>` prefix keeps every tool name inside `^[A-Za-z0-9_-]+$`.

---

## 4. The provider fallback chain (capability 15)

`[[llm.chain]]` entries are tried in order. The first that returns a response
wins. **Any** per-entry failure moves the chain on: a `ProviderError`, a
`ConfigError` raised while *constructing* the provider (missing `base_url`,
missing key), or any other exception (`asyncio.CancelledError` alone
re-raises). Only when every entry has failed does it raise
`AllProvidersFailed`, carrying the class name of the last underlying error
(never its message — that can contain response text). Provider construction is
inside the per-entry `try`, so one misconfigured earlier entry can never
disable the entries after it.

`skip_between = ["01:00-04:00Z"]` marks a daily UTC window in which that entry
is skipped and the chain moves on — the replacement for a "refuse during peak
pricing" behaviour: route to the next provider (typically a local model)
instead of declining. A window may wrap midnight. Only UTC is understood in v1
(every provider quotes its peak window in UTC; a tz-aware parser is a
maintenance cost with no caller). If *every* entry is inside a window,
`AllProvidersFailed("every provider is inside a peak window right now")`.

No difficulty-based routing: it needs a classifier to maintain and buys little.

Two back ends, one interface:

- **`LiteLLMProvider`** (default). One `litellm.acompletion`; the model string
  carries the provider prefix (`deepseek/…`, `ollama/…`). `litellm` is an
  optional dependency, imported lazily, with a clear `ConfigError` if
  `use_litellm` is true and it is not installed.
- **`HTTPProvider`** (`use_litellm = false` / `--no-litellm`). A direct `httpx`
  POST to an OpenAI-compatible `/chat/completions` endpoint. `ollama` → `/v1`
  shim, no auth; `deepseek` → `https://api.deepseek.com/v1`, key required;
  anything else needs an explicit `base_url` (a trailing `/v1` on the
  `ollama` / `deepseek` `base_url` is tolerated). The response is **streamed**
  and aborted if `Content-Length` or the running byte count passes 8 MiB, so an
  oversized body is never fully buffered. A ≥400 response logs its (truncated)
  body and raises `ProviderError("… returned HTTP <code>")` — the body never
  reaches the raised message.

---

## 5. The tool-calling loop

`Agent.answer(user_message, history)`:

1. Build `messages` = system prompt + retained history + the user message.
2. Loop:
   - **wall-clock** — `_complete()` checks the remaining budget and then runs
     the model call inside `async with asyncio.timeout(remaining)`. The
     between-calls check alone could not bound a hung `await`; the timeout can.
     Either way an over-budget turn raises `TurnTimeout` and the bot renders a
     clean "couldn't finish in time" message, never a partial one.
   - `ProviderChain.complete(messages, tools)` — `tools` is the flattened MCP
     tool list until `max_tool_calls_per_turn` calls have run, then `None`.
   - text response → return it (a placeholder if the model returned empty).
   - tool-call response → append the assistant turn, run the calls **concurrently**
     (`asyncio.gather`) via `MCPPool.call` (each under
     `[llm].tool_call_timeout_seconds`; a tool error, a timeout, or an unexpected
     exception all come back as an `ERROR: …` string, never raised), append the
     results **in the model's original order**, loop. Once the running total of
     tool output passes `[llm].max_tool_output_chars` (applied in that order),
     later results are trimmed to a stub so the context can't overflow a small
     fallback model.
   - a tool-call response *after* tools were withdrawn → one plain-language
     nudge with no tools, then return whatever text comes back.

`MCPPool.call` never raises: an unknown tool, a tool error, a timeout, or a raw
exception from `call_tool` (logged, returned as a generic string — no upstream
detail is echoed) all become an `ERROR: …` string the model can explain.

---

## 6. Memory

`ChatMemory` keeps, per chat id, the last `history_turns` user/assistant
exchanges in an in-process `deque`. Intermediate tool-call transcripts are not
retained across turns — only the final question and answer — which keeps the
replayed context compact and provider-neutral. `history_turns = 0` disables
retention. A failed turn (provider outage, timeout) is **not** recorded.

If `[memory].log_dir` is set, every turn is also appended to
`<log_dir>/chat-<id>.md` — created `chmod 0o600` (it holds the operator's
questions and infra detail), and logged once with a warning past 5 MiB. It is
**not** rotated; point `log_dir` at a logrotate-managed path if you enable it.
That file is for the operator; it is never read back.

---

## 7. Transports (MCP)

Per `[[mcp.servers]]`:

- **`stdio`** (default, and what the reference deployment uses): the bot spawns
  the server as a subprocess. No port, no auth. Verified with a real
  `homelab-mcp` subprocess handshake.
- **`streamable-http`**: `url` required; `token_env` → `Authorization: Bearer`
  via a custom `httpx` client built by a private SDK helper — the import is
  `try`/`except ImportError`-guarded and a bearer-over-HTTP server on an SDK
  build that lacks it fails closed with a clear message. Bind the server to
  LAN/VPN only.
- **`sse`**: the legacy HTTP transport, via the SDK's `sse_client`. The
  bearer-header path here is wired but not exercised against a real authed SSE
  server (see §10).

Tools from all servers are flattened into one set, each renamed
`<server>__<tool>`; the prefix is stripped on dispatch so a call always routes
back to the right server.

---

## 8. Testing

`uv run pytest` — 112 tests, all green. Coverage:

- `config` — full parse, every rejection path, `TimeWindow` (simple / wrapping /
  garbage), the new `[llm]` knobs parse + reject, `resolve_secrets` reports all
  gaps at once, the shipped `infra-assistant.example.toml` parses and names its
  secrets.
- `prompt` — persona-only, all three files, plain vs. own-heading optional
  sections, missing dir / missing persona / empty persona, the shipped
  `prompts/` dir is valid.
- `memory` — rolling cap, `history_turns = 0`, per-chat reset isolation,
  transcript append, provider-message rendering.
- `llm_chain` — first wins, failover, **construction `ConfigError` fails over
  (not chain-killing)**, **an unforeseen exception fails over**, all-fail →
  `AllProvidersFailed` with the last error *class*, peak-window skip,
  all-skipped, provider instance cached, argument/response parsing incl.
  malformed JSON, OpenAI-envelope wrapping.
- `providers` — endpoint resolution table incl. **trailing-`/v1` tolerance**,
  `base_url` / key requirements, `respx`-mocked happy path (auth header
  asserted), HTTP 500 with the body **not** in the error, **oversized rejected
  by `Content-Length` and while streaming**, bad response shape;
  `LiteLLMProvider` with a monkeypatched `acompletion` (params forwarded, errors
  wrapped); `build_provider` back-end selection.
- `mcp_client` — an in-process `mcp.MCPServer` with real tools: namespacing,
  routing, unknown tool and tool error as strings, **a raw `call_tool`
  exception → generic string (no leak)**, **a hung tool → timeout string**,
  fail-closed on a connection error, `_transport` shapes + the missing
  command/url and missing-auth-helper guards, result rendering + truncation.
- `agent` — plain answer, history replay, one tool call then answer, the
  tool-call cap forcing a final answer (cap never exceeded), wall-clock
  `TurnTimeout` via both the budget check **and a genuinely hanging model call**,
  **the cumulative tool-output budget trims later results**, **multiple tool calls in one turn run concurrently and in order**, `AllProvidersFailed`
  propagation, empty-reply placeholder.
- `telegram_bot` — non-allowlisted chat ignored (no reply, nothing recorded),
  happy path replies + records + typing action, `AllProvidersFailed` and
  `TurnTimeout` as clean messages that are **not** recorded, oversized reply
  chunked, **the splitter counts UTF-16 units** (emoji), `/reset`,
  `on_error` DMs the owner (and is quiet with no owner).
- `cli` — `--version`, subcommand required, missing config, `check` reporting
  providers + peak-window state + no-MCP, **`check` exits non-zero when every
  provider is in a peak window**, **`--probe-llm` reports a provider failure**,
  missing-secret exit code, `--no-litellm` flips the config.

**All test data is fictitious:** chat ids `111` / `222`, hosts `node-a` /
`node-b`, addresses in 192.0.2.0/24 and 198.51.100.0/24, free text
`"Test Data …"`, every absolute date before 2002 (relative-to-`now` fixtures
stay relative). No real infrastructure detail appears anywhere in this repo.

CI: `ruff check` + `ruff format --check` + `mypy --strict` + `pytest` +
`scripts/check_no_exec.py`. `scripts/check_privacy.sh` + `.githooks/pre-push` +
`.github/workflows/check-privacy.yml` are present and byte-identical to the
other public repos; they become active when infra-assistant is split into its
own git repo (see §0.1).

---

## 9. Acceptance criteria

| # | Criterion | Verified? |
|---|---|---|
| A1 | A message from a chat id not in `owner_allow_from` produces no reply and changes no state | **Yes** (test_telegram_bot) — checked by a PTB filter *and* an in-handler id check |
| A2 | Every model call goes through the fallback chain; **any** per-entry failure — including a misconfigured earlier entry — moves to the next | **Yes** (test_llm_chain: construction `ConfigError` and an unforeseen exception both fail over) |
| A3 | A `skip_between` window removes that entry for the duration and the chain continues | **Yes** (test_llm_chain, test_cli shows it in `check`) |
| A4 | The tool-call loop cannot exceed `max_tool_calls_per_turn`; it still returns an answer | **Yes** (test_agent — cap 2, 3 requests, 2 invocations, final text) |
| A5 | A turn over `turn_wall_time_seconds` ends with a clean message, never a truncated one — **even if a model call hangs** | **Yes** (test_agent: the budget check *and* a genuinely hanging `complete()` both raise `TurnTimeout`; test_telegram_bot renders it) |
| A6 | A reply over Telegram's limit is split into valid chunks, counting **UTF-16 units** | **Yes** (test_telegram_bot: ASCII sizes + a 2500-emoji case) |
| A7 | A missing secret / bot token / empty chain prevents startup, naming the gap; a `run` startup failure DMs the owner | **Yes** (test_config, test_cli; `_best_effort_startup_alert` best-effort) |
| A8 | The bot exposes no shell / exec / file-write / scheduler of its own | **Tripwire** — `scripts/check_no_exec.py` in CI catches accidental regressions; not proof against deliberate obfuscation (§2) |
| A9 | The MCP client completes a real stdio handshake with a real server | **Yes** — manual smoke against `homelab-mcp`; in-process `mcp.MCPServer` round-trip in tests |
| A10 | A hung MCP tool call cannot hang the turn | **Yes** — `asyncio.wait_for` per call (test_mcp_client: a sleeping tool → timeout string) |
| A11 | An unexpected `call_tool` exception surfaces to the model without leaking upstream detail | **Yes** (test_mcp_client) |
| A12 | `check` exits non-zero for a config that would leave the bot unable to answer | **Yes** — missing secret, unreachable server, all-providers-in-peak-window, 0-tool server; `--probe-llm` adds a live per-provider call |
| A13 | `pipx install infra-assistant` yields a working `infra-assistant` command | **No** — not published |
| A14 | End-to-end against a live Telegram bot and a live LLM provider | **No** — every automated test uses fakes |

---

## 10. Open decisions

1. **Streaming replies.** Currently one message (chunked) after the full turn.
   Telegram "typing…" covers the wait; token streaming would add reconnection
   and edit-rate handling for little gain. Revisit if turns get long.
2. **`sse` auth.** `streamable-http` gets a bearer header; `sse_client` is wired
   for `headers` too but untested against a real authed SSE server.
3. **Difficulty routing.** Deliberately absent (§4). Reconsider only with
   evidence that the fallback model is being used for turns it can't handle.
4. **Per-user personas.** One `prompts/` dir for the whole bot. Multi-operator
   deployments would want per-chat overrides; out of scope for v1.

*Resolved 2026-09-01:* `python-telegram-bot` + `litellm` confirmed; provider
chain = DeepSeek → Ollama with peak-window routing (generic in this repo, real
values in the private layer); stdio the default transport; no write tools / no
autonomous code (tripwire lint + this document). A full code-review pass hardened
the reliability surface — the wall-clock cap is now enforced with
`asyncio.timeout`, every MCP tool call has its own timeout, the failover net
catches every per-entry failure (construction included), the HTTP response is
size-capped while streaming, tool calls in one turn run concurrently (order and
budgets preserved), `check` gained real exit codes and an optional live provider
probe, PTB got an error handler + concurrent updates, and startup failure DMs
the owner.
