"""Assemble the system prompt from plain files at startup.

Three files in the configured prompt directory (this is the port target for an
agent framework's ``SOUL.md`` / ``USER.md`` / ``TOOLS.md`` / ``AGENTS.md``):

* ``persona.md``  — who the assistant is, its tone, its boundaries. **Required.**
* ``user.md``     — facts about the operator and their preferences. Optional.
* ``context.md``  — a short description of the environment. Optional.

Tool descriptions are **not** part of this file — they are injected from the MCP
servers at request time, so there is no hand-maintained tool prose to drift.

A fixed trailer is always appended after the persona text. It restates the
read-only boundary in the assistant's own voice so that a thin or careless
``persona.md`` cannot remove the guarantee — the tools are read-only by
construction regardless, this just keeps the model from *promising* actions.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .errors import ConfigError

logger = logging.getLogger("infra_assistant.prompt")

PERSONA_FILE = "persona.md"
USER_FILE = "user.md"
CONTEXT_FILE = "context.md"

_BOUNDARY_TRAILER = """\
## Operating boundaries (fixed)

- Every tool available to you is **read-only**. You can look things up; you
  cannot change anything.
- If a change is needed — restart a service, edit a file, deploy — do **not**
  claim you have done it. Give the exact command or steps for the operator to
  run themselves.
- You have no shell, no code execution, no file access, and no scheduler.
- If a question is outside infrastructure and operations, say so briefly rather
  than guessing.
- When a tool call fails or returns nothing, say what you tried and what came
  back. Do not invent a plausible-looking answer.\
"""


def assemble_system_prompt(prompt_dir: str | Path) -> str:
    """Read the prompt files under *prompt_dir* and return the system prompt.

    Raises :class:`ConfigError` if the directory or ``persona.md`` is missing —
    the bot fails closed rather than running with an empty persona.
    """
    directory = Path(prompt_dir).expanduser().resolve()
    if not directory.is_dir():
        raise ConfigError(f"prompt directory not found: {directory}")

    persona_path = directory / PERSONA_FILE
    if not persona_path.is_file():
        raise ConfigError(f"required prompt file not found: {persona_path}")
    persona = persona_path.read_text(encoding="utf-8").strip()
    if not persona:
        raise ConfigError(f"{persona_path} is empty")

    sections: list[str] = [persona, _BOUNDARY_TRAILER]

    user_text = _optional_section(directory / USER_FILE, "About the operator")
    if user_text:
        sections.append(user_text)

    context_text = _optional_section(directory / CONTEXT_FILE, "About this environment")
    if context_text:
        sections.append(context_text)

    return "\n\n".join(sections) + "\n"


def _optional_section(path: Path, heading: str) -> str:
    if not path.is_file():
        logger.info("optional prompt file %s not present — skipping", path.name)
        return ""
    body = path.read_text(encoding="utf-8").strip()
    if not body:
        return ""
    # If the file already opens with a markdown heading, trust its own structure.
    if body.lstrip().startswith("#"):
        return body
    return f"## {heading}\n\n{body}"
