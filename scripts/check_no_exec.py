#!/usr/bin/env python3
"""CI tripwire for the "no autonomy" guarantee (see docs/DESIGN_SPEC.md §2).

infra-assistant offers the model *no* local capability: no shell, no code
execution, no arbitrary file write, no self-modification. Every tool it can call
comes from an external MCP server.

This is a **tripwire, not a proof** — a substring/regex scan that catches the
obvious ways the guarantee would be broken (importing `subprocess`, calling
`eval`, spawning a process, writing a file outside the one sanctioned place).
It does not defeat someone deliberately obfuscating; that is what code review
and `docs/DESIGN_SPEC.md` §2 are for. Its job is to make an *accidental*
regression fail the build.

Exit 0 on success, 1 on any violation.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parent.parent / "src" / "infra_assistant"

# Regex -> why it is banned. Patterns match *usage* (an import, or a dotted
# call), not the bare word, so prose like "closes the stdio subprocess" in a
# comment does not trip. Deliberately broad on process creation, dynamic import,
# and low-level FFI.
BANNED: dict[str, str] = {
    r"\b(import\s+subprocess|from\s+subprocess\s+import|subprocess\.[A-Za-z_])": "no spawning processes",
    r"\bcreate_subprocess_(exec|shell)\b": "no spawning processes (asyncio)",
    r"\bos\.system\(": "no shell",
    r"\bos\.popen\(": "no shell",
    r"\bos\.(execl|execle|execlp|execv|execve|execvp|execvpe)\b": "no exec* family",
    r"\bos\.(spawnl|spawnle|spawnlp|spawnv|spawnve|spawnvp|posix_spawn)\b": "no spawn* family",
    r"\bos\.fork\(": "no fork",
    r"\bpty\.(spawn|fork|openpty)\b": "no pseudo-terminals",
    r"\bptyprocess\b": "no pseudo-terminals",
    r"\b(import\s+multiprocessing|from\s+multiprocessing\s+import|multiprocessing\.[A-Za-z_])": "no process pools",
    r"\b(import\s+runpy|runpy\.[A-Za-z_])": "no running modules as scripts",
    r"\b__import__\(": "no dynamic import",
    r"\b(import\s+importlib|from\s+importlib\s+import|importlib\.[A-Za-z_])": "no dynamic import",
    r"\b(import\s+ctypes|from\s+ctypes\s+import|ctypes\.[A-Za-z_])": "no FFI",
    r"\b(import\s+cffi|from\s+cffi\s+import)": "no FFI",
}

# `eval(` / `exec(` / `compile(` as whole-word calls, so "execute", "evaluate"
# and `re.compile(` do not trip.
CALL_RE = re.compile(r"(?<![A-Za-z_.])(eval|exec|compile)\s*\(")

# A file opened for writing is allowed only in memory.py (the append-only
# per-chat transcript). Catches `x.open("w"/"a")`, builtin `open(p, "w"/"a")`,
# and pathlib `.write_text` / `.write_bytes`.
WRITE_OPEN_RE = re.compile(
    r"""\.open\(\s*[^)]*["'][rwax+bt]*[wax]|"""  # p.open("a"), p.open(mode="w")
    r"""(?<![A-Za-z_.])open\([^)]*,\s*["'][rwax+bt]*[wax]|"""  # open(p, "w")
    r"""\.write_(text|bytes)\("""
)
WRITE_OPEN_ALLOWED = {"memory.py"}


def main() -> int:
    violations: list[str] = []
    banned = [(re.compile(pat), why) for pat, why in BANNED.items()]
    for path in sorted(PKG.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(PKG.parent.parent)
        for pattern, why in banned:
            for match in pattern.finditer(text):
                line = text[: match.start()].count("\n") + 1
                violations.append(f"{rel}:{line}: {match.group(0)!r} — {why}")
        for match in CALL_RE.finditer(text):
            line = text[: match.start()].count("\n") + 1
            violations.append(f"{rel}:{line}: calls {match.group(1)}() — no dynamic evaluation")
        if WRITE_OPEN_RE.search(text) and path.name not in WRITE_OPEN_ALLOWED:
            violations.append(
                f"{rel}: opens a file for writing outside {sorted(WRITE_OPEN_ALLOWED)}"
            )

    if violations:
        print("FAIL: no-autonomy guarantee")
        for item in violations:
            print(f"  - {item}")
        return 1

    print("ok: no shell / exec / dynamic-import / stray file-write in the package")
    return 0


if __name__ == "__main__":
    sys.exit(main())
