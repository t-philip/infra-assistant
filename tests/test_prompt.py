from __future__ import annotations

from pathlib import Path

import pytest

from infra_assistant.errors import ConfigError
from infra_assistant.prompt import assemble_system_prompt


def _write(dir_: Path, name: str, body: str) -> None:
    (dir_ / name).write_text(body, encoding="utf-8")


def test_persona_only(tmp_path: Path) -> None:
    _write(tmp_path, "persona.md", "# Persona\n\nYou are Test Data assistant.")
    prompt = assemble_system_prompt(tmp_path)
    assert "Test Data assistant" in prompt
    assert "Operating boundaries (fixed)" in prompt  # trailer always present
    assert prompt.endswith("\n")


def test_all_three_files(tmp_path: Path) -> None:
    _write(tmp_path, "persona.md", "You are terse.")
    _write(tmp_path, "user.md", "Operator prefers UTC.")
    _write(tmp_path, "context.md", "# Environment\n\nOne node named node-a.")
    prompt = assemble_system_prompt(tmp_path)
    assert "You are terse." in prompt
    # A plain-text optional file gets a synthesised heading...
    assert "## About the operator" in prompt
    assert "Operator prefers UTC." in prompt
    # ...but one that starts with its own heading is trusted as-is.
    assert "## About this environment" not in prompt
    assert "# Environment" in prompt


def test_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="prompt directory not found"):
        assemble_system_prompt(tmp_path / "absent")


def test_missing_persona(tmp_path: Path) -> None:
    _write(tmp_path, "user.md", "something")
    with pytest.raises(ConfigError, match=r"persona\.md"):
        assemble_system_prompt(tmp_path)


def test_empty_persona(tmp_path: Path) -> None:
    _write(tmp_path, "persona.md", "   \n")
    with pytest.raises(ConfigError, match="is empty"):
        assemble_system_prompt(tmp_path)


def test_shipped_prompt_dir_is_valid() -> None:
    prompts = Path(__file__).resolve().parent.parent / "prompts"
    out = assemble_system_prompt(prompts)
    assert "infra-assistant" in out
    assert "read-only" in out.lower()
