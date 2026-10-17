# infra-assistant

A small Telegram assistant for **infrastructure Q&A**: one LLM, a system prompt
assembled from plain files, **read-only** tools from one or more
[MCP](https://modelcontextprotocol.io) servers, and an ordered provider fallback
chain. The deliberate anti-framework — no skills, no plugins, no autonomy, no
scheduler.

> **Status: v1.0.0.** 146 tests against fakes; no automated test exercises a
> real Telegram bot or LLM provider end to end. See `docs/DESIGN_SPEC.md`.

Pairs naturally with [`homelab-mcp`](https://github.com/t-philip/homelab-mcp) as the tool server, but it
will drive any MCP server.

## What it does

1. Long-polls Telegram. A message from a chat id not in `owner_allow_from` is
   ignored.
2. Builds the request: system prompt + the last _N_ exchanges for that chat +
   the new message.
3. Offers the model the MCP tool list and runs a **model → tools → model** loop,
   bounded by a per-turn tool-call cap and a wall-clock cap.
4. Sends the answer — in 4096-character chunks if needed. Never a half-formed
   reply.
5. On a provider error or timeout, falls through the provider chain before
   giving up.

## What it will not do

No `execute_code`, no shell, no file-write tool, no self-modification, no
scheduler. If the model wants an action taken, it returns the command for you to
run. The tools themselves are read-only — that guarantee lives in the MCP server
(`homelab-mcp` enforces it four ways).

## Install

```sh
pipx install "infra-assistant[litellm]"     # or: uv tool install "infra-assistant[litellm]"
```

Drop `[litellm]` and set `use_litellm = false` to run with only `httpx` on the
provider side.

## Configure

```sh
cp infra-assistant.example.toml infra-assistant.toml
$EDITOR infra-assistant.toml
mkdir prompts && $EDITOR prompts/persona.md   # + optional user.md / context.md
```

Secrets are never stored in the file. A field ending in `_env` names an
environment variable, read at startup:

```sh
export TELEGRAM_BOT_TOKEN=...
export DEEPSEEK_API_KEY=...
infra-assistant check -c infra-assistant.toml              # static validation, then exit
infra-assistant check -c infra-assistant.toml --probe-llm  # also make one live call per provider
```

`check` resolves every secret, assembles the system prompt, connects to each MCP
server to count its tools, and reports which providers are eligible right now
(peak windows). It exits non-zero if a secret is missing, an MCP server can't be
reached, every provider is currently inside a peak window, or a connected server
exposes no tools. `--probe-llm` additionally makes one 1-token call to each
eligible provider and fails on any that error.

## Run

```sh
infra-assistant run -c infra-assistant.toml
```

`--no-litellm` forces the direct-`httpx` provider path for this run.

## The system prompt

Assembled at startup from `prompts/`:

| File | Required | Purpose |
|---|---|---|
| `persona.md` | yes | who the assistant is, its tone, its boundaries |
| `user.md` | no | facts about the operator and their preferences |
| `context.md` | no | a short description of the environment |

Tool descriptions are **not** in these files — they come from the MCP servers at
request time, so there is nothing to keep in sync. A fixed boundary clause is
always appended so a thin `persona.md` can't remove the read-only framing.

## The provider chain

```toml
[[llm.chain]]
provider = "deepseek"
model = "deepseek/deepseek-chat"
api_key_env = "DEEPSEEK_API_KEY"
# skip_between = ["01:00-04:00Z"]     # skip this entry inside a UTC window

[[llm.chain]]
provider = "ollama"
model = "ollama/llama3.1:8b"
base_url = "http://ollama.example.invalid:11434"
```

Each entry is tried in order. `skip_between` skips an entry while the clock is
inside a UTC window and moves to the next — route to a local model during peak
pricing instead of refusing to answer. No difficulty-based routing.

## Develop

```sh
uv sync --all-extras
uv run pytest
uv run ruff check .
uv run mypy
uv run python scripts/check_no_exec.py
git config core.hooksPath .githooks    # once per clone — privacy pre-push hook
```

See `docs/PRIVACY.md` before pushing anything public.

## License

GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later). See
[`LICENSE`](LICENSE).
