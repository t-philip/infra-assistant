# Publishing rule — what may and may not appear in this repo

`infra-assistant` is a bot that talks to real infrastructure through MCP
servers, and its system prompt is assembled from files that *describe an
environment*. Its README, `DESIGN_SPEC.md`, example config, the `prompts/`
templates and the tests all describe how that works. That is exactly how a
specific environment's host names, IP addresses, endpoint URLs, chat ids or a
verbatim persona line can end up in a public repository.

## The rule

> **A comment, an example, a prompt template, or a fixture may describe
> behaviour and reasoning. It may not describe a specific installation or a
> specific operator.**

The test to apply to any line you are about to write:

> *Could this have been written by someone who has never seen the author's
> network or met the operator?*

## Specifics

1. **Code comments and docstrings** — general behaviour only. Never a real host
   name, internal DNS name, IP address, endpoint URL, SSH alias, or Telegram
   chat id.
2. **Example config (`infra-assistant.example.toml`)** — every `base_url`,
   `url` and `command` uses a `.invalid` / `.example` hostname or a plain
   binary name; `owner_allow_from` is the placeholder `[123456789]`; every
   `*_env` names a variable, never a value.
3. **Prompt templates (`prompts/*.md`)** — `persona.md`, `user.md` and
   `context.md` here are **generic templates**. No real operator facts, no real
   host inventory, no real endpoint. The real prompt files live in the private
   deployment layer. `context.md` explicitly tells the model to use whatever
   the MCP servers report, not anything in the file.
4. **Test fixtures — the fiction window.** Any absolute date in test data uses
   **a year before 2002**, and nothing else. Relative fixtures (built from
   `now - timedelta(...)`) stay relative — they carry no calendar fact.
   Chat ids in tests are `111` / `222`; host names are `node-a` / `node-b`;
   addresses are in `192.0.2.0/24` and `198.51.100.0/24`; free text is
   `"Test Data ..."`.
5. **`docs/DESIGN_SPEC.md`** — describes the design of this bot, not the
   author's deployment. No real host inventory, no real endpoint, no dated
   observation, no operator details.
6. **Network examples** — RFC 5737 documentation ranges only
   (`192.0.2.0/24`, `198.51.100.0/24`, `203.0.113.0/24`) plus loopback.

## Where the specific configuration goes instead

Into a private deployment layer, kept outside this repository. The real MCP
server config, real endpoint URLs, the owner's real chat id, and the real
`persona.md` / `user.md` / `context.md` live there. This repo is the generic
engine; the deployment is private.

## Checking

Two halves, and **both** are required.

**Structural** — `scripts/check_privacy.sh`, run automatically by the `pre-push`
hook in the tracked `.githooks/` directory (`.git/hooks` is never cloned).
Activate it once per clone:

```sh
git config core.hooksPath .githooks
```

It checks the *diff*, not the tree, for internal host names, real IPs, emails,
dated prose, absolute dates at or after 2002, non-placeholder free-text fields,
quoted log-note-shaped phrases, and the patterns in `.privacy-denylist` (SSH
aliases, internal DNS suffixes, real endpoint URLs, real-looking chat ids).
`.github/workflows/check-privacy.yml` runs the same script server-side, since
the hook is bypassable with `git push --no-verify`.

**Semantic** — a read of the diff by someone, or something, with no memory of
writing it, answering the one question above.

**A pass from the script does not mean the diff is safe to publish.** It means
the cheap half found nothing.
