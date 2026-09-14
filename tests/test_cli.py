from __future__ import annotations

import logging
from pathlib import Path

import pytest

from infra_assistant.cli import _configure_logging, _load, main

_CONFIG = """
[telegram]
bot_token_env = "BOT_TOKEN"
owner_allow_from = [111]

[llm]
use_litellm = true

[[llm.chain]]
provider = "deepseek"
model = "deepseek/deepseek-chat"
api_key_env = "DEEPSEEK_API_KEY"
skip_between = ["00:00-23:59Z"]

[[llm.chain]]
provider = "ollama"
model = "ollama/llama3.1:8b"
base_url = "http://ollama.example.invalid:11434"
"""


def _prompt_dir(base: Path) -> Path:
    d = base / "prompts"
    d.mkdir()
    (d / "persona.md").write_text("You are Test Data assistant.", encoding="utf-8")
    return d


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "infra-assistant" in capsys.readouterr().out


def test_subcommand_required() -> None:
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2


def test_check_missing_config_returns_error(tmp_path: Path) -> None:
    rc = main(["check", "-c", str(tmp_path / "absent.toml")])
    assert rc == 2


def test_check_reports_providers_and_no_mcp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = tmp_path / "c.toml"
    cfg.write_text(_CONFIG, encoding="utf-8")
    prompts = _prompt_dir(tmp_path)
    monkeypatch.setenv("BOT_TOKEN", "t")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")

    rc = main(["check", "-c", str(cfg), "--prompt-dir", str(prompts)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "providers   : 2 in the chain" in out
    assert "SKIPPED now (peak window)" in out  # the 00:00-23:59 window always hits
    assert "mcp servers : none configured" in out


def test_check_missing_secret(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = tmp_path / "c.toml"
    cfg.write_text(_CONFIG, encoding="utf-8")
    monkeypatch.delenv("BOT_TOKEN", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    rc = main(["check", "-c", str(cfg), "--prompt-dir", str(_prompt_dir(tmp_path))])
    assert rc == 2


def test_no_litellm_flag_flips_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = tmp_path / "c.toml"
    cfg.write_text(_CONFIG, encoding="utf-8")
    monkeypatch.setenv("BOT_TOKEN", "t")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    loaded = _load(str(cfg), no_litellm=True)
    assert loaded.llm.use_litellm is False


_ALL_SKIPPED = _CONFIG.replace(
    'model = "ollama/llama3.1:8b"\nbase_url',
    'model = "ollama/llama3.1:8b"\nskip_between = ["00:00-23:59Z"]\nbase_url',
)


def test_check_fails_when_every_provider_is_in_a_peak_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = tmp_path / "c.toml"
    cfg.write_text(_ALL_SKIPPED, encoding="utf-8")
    monkeypatch.setenv("BOT_TOKEN", "t")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    rc = main(["check", "-c", str(cfg), "--prompt-dir", str(_prompt_dir(tmp_path))])
    out = capsys.readouterr().out
    assert rc == 1
    assert "every provider is inside a peak window" in out


def test_check_probe_llm_reports_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = tmp_path / "c.toml"
    # neither entry skipped, so both get probed
    cfg.write_text(_CONFIG.replace('skip_between = ["00:00-23:59Z"]\n', ""), encoding="utf-8")
    monkeypatch.setenv("BOT_TOKEN", "t")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")

    from infra_assistant.errors import ProviderError

    class Boom:
        async def complete(self, *_: object) -> object:
            raise ProviderError("unreachable")

        async def aclose(self) -> None:
            return None

    monkeypatch.setattr("infra_assistant.cli.build_provider", lambda *a, **k: Boom())
    rc = main(["check", "-c", str(cfg), "--prompt-dir", str(_prompt_dir(tmp_path)), "--probe-llm"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "FAIL" in out and "ProviderError" in out


@pytest.mark.parametrize("level", ["DEBUG", "INFO"])
def test_configure_logging_muzzles_credential_bearing_loggers(level: str) -> None:
    # httpx logs the full request URL at INFO and the Telegram bot token rides
    # in that URL's path — a --log-level of DEBUG must not expose it.
    for name in ("httpx", "httpcore", "telegram.request"):
        logging.getLogger(name).setLevel(logging.NOTSET)
    _configure_logging(level)
    for name in ("httpx", "httpcore", "telegram.request"):
        assert logging.getLogger(name).level == logging.WARNING
