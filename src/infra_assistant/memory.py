"""Per-chat rolling conversation history.

In-process only, bounded to the last ``history_turns`` user/assistant exchanges
per chat. Tool-call transcripts within a turn are *not* retained across turns —
only the final user message and the assistant's answer — which keeps the history
compact and provider-neutral.

If ``[memory].log_dir`` is set, each turn is also appended to a per-chat
markdown transcript there. That file is a convenience for the operator; it is
never read back.
"""

from __future__ import annotations

import contextlib
import logging
from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger("infra_assistant.memory")

#: Warn (once) when a per-chat transcript passes this size — it is not rotated.
_TRANSCRIPT_WARN_BYTES = 5 * 1024 * 1024


@dataclass(frozen=True)
class Message:
    role: str  # "user" | "assistant"
    content: str


class ChatMemory:
    def __init__(self, history_turns: int, *, log_dir: str | Path | None = None) -> None:
        self._history_turns = max(0, history_turns)
        # N turns == N user + N assistant messages.
        self._maxlen = self._history_turns * 2
        self._chats: dict[int, deque[Message]] = defaultdict(self._new_deque)
        self._log_dir = Path(log_dir).expanduser() if log_dir else None
        self._warned_large: set[int] = set()
        if self._log_dir is not None:
            self._log_dir.mkdir(parents=True, exist_ok=True)

    def _new_deque(self) -> deque[Message]:
        # maxlen=0 would discard everything on append; use None to mean "keep none"
        # by simply never appending (handled in record()).
        return deque(maxlen=self._maxlen or None)

    def history(self, chat_id: int) -> list[Message]:
        """The retained messages for *chat_id*, oldest first."""
        if self._history_turns == 0:
            return []
        return list(self._chats.get(chat_id, ()))

    def record(self, chat_id: int, user_message: str, assistant_message: str) -> None:
        """Append one completed exchange and, if configured, log it."""
        if self._history_turns > 0:
            buffer = self._chats[chat_id]
            buffer.append(Message("user", user_message))
            buffer.append(Message("assistant", assistant_message))
        self._append_transcript(chat_id, user_message, assistant_message)

    def reset(self, chat_id: int) -> None:
        self._chats.pop(chat_id, None)

    def _append_transcript(self, chat_id: int, user_message: str, assistant_message: str) -> None:
        if self._log_dir is None:
            return
        stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        path = self._log_dir / f"chat-{chat_id}.md"
        block = (
            f"\n### {stamp}\n\n**operator:** {user_message}\n\n**assistant:** {assistant_message}\n"
        )
        try:
            existed = path.exists()
            with path.open("a", encoding="utf-8") as handle:
                handle.write(block)
            if not existed:
                # transcripts hold the operator's questions and infra detail —
                # keep them owner-only, not umask-default world-readable.
                with contextlib.suppress(OSError, NotImplementedError):
                    path.chmod(0o600)
            if chat_id not in self._warned_large and path.stat().st_size > _TRANSCRIPT_WARN_BYTES:
                self._warned_large.add(chat_id)
                logger.warning(
                    "transcript %s is over %d MiB and is not rotated — prune it or "
                    "point [memory].log_dir at a logrotate-managed path",
                    path.name,
                    _TRANSCRIPT_WARN_BYTES // (1024 * 1024),
                )
        except OSError:
            logger.warning("could not append transcript for chat %s to %s", chat_id, path)


def as_provider_messages(history: Iterable[Message]) -> list[dict[str, str]]:
    """Render retained history as ``[{"role": ..., "content": ...}, ...]``."""
    return [{"role": m.role, "content": m.content} for m in history]
