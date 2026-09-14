from __future__ import annotations

from pathlib import Path

from infra_assistant.memory import ChatMemory, Message, as_provider_messages


def test_history_rolls_at_the_cap() -> None:
    mem = ChatMemory(history_turns=2)
    for i in range(4):
        mem.record(111, f"q{i}", f"a{i}")
    hist = mem.history(111)
    assert [m.content for m in hist] == ["q2", "a2", "q3", "a3"]
    assert all(isinstance(m, Message) for m in hist)


def test_history_turns_zero_keeps_nothing() -> None:
    mem = ChatMemory(history_turns=0)
    mem.record(111, "q", "a")
    assert mem.history(111) == []


def test_reset_clears_one_chat_only() -> None:
    mem = ChatMemory(history_turns=3)
    mem.record(111, "q", "a")
    mem.record(222, "q", "a")
    mem.reset(111)
    assert mem.history(111) == []
    assert mem.history(222) != []


def test_transcript_is_appended(tmp_path: Path) -> None:
    mem = ChatMemory(history_turns=1, log_dir=tmp_path)
    mem.record(111, "is node-a up?", "yes, active since Test Data")
    mem.record(111, "and node-b?", "inactive")
    text = (tmp_path / "chat-111.md").read_text(encoding="utf-8")
    assert "is node-a up?" in text
    assert "and node-b?" in text
    assert text.count("**operator:**") == 2


def test_as_provider_messages_shape() -> None:
    msgs = as_provider_messages([Message("user", "hi"), Message("assistant", "hello")])
    assert msgs == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
