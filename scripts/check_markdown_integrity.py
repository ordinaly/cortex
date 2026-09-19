#!/usr/bin/env python3
"""Repository-wide Markdown integrity checks."""
from __future__ import annotations

from pathlib import Path
import re
import sys
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_FILES = sorted(
    path for path in ROOT.rglob("*.md") if ".git" not in path.parts
)

LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
FENCE_RE = re.compile(r"^\s*\x60\x60\x60")
DISPLAY_MATH_RE = re.compile(r"^\s*\$\$\s*$")
LEGACY_DISPLAY_RE = re.compile(r"^\s*\\[\[\]]\s*$")
LITERAL_BRACKET_RE = re.compile(r"^\s*[\[\]]\s*$")
TABLE_LINE_RE = re.compile(r"^\s*\|.*\|\s*$")


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def unescaped_pipe_count(line: str) -> int:
    count = 0
    escaped = False
    for char in line:
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == "|":
            count += 1
    return count


def resolve_link(source: Path, target: str) -> Path | None:
    target = target.strip()
    if not target or target.startswith("#"):
        return None
    if re.match(r"^(?:https?://|mailto:)", target):
        return None
    path_part = unquote(target.split("#", 1)[0])
    if not path_part:
        return None
    return (source.parent / path_part).resolve()


def check_file(path: Path) -> list[str]:
    errors: list[str] = []
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    name = rel(path)

    for index, char in enumerate(text):
        code = ord(char)
        if code < 32 and char != "\n":
            line = text.count("\n", 0, index) + 1
            errors.append(
                f"{name}:{line}: control character code {code}; possible broken escape sequence"
            )

    in_fence = False
    fence_start = 0
    math_open = False
    math_start = 0
    table_rows: list[tuple[int, int]] = []

    def flush_table() -> None:
        nonlocal table_rows
        if len(table_rows) >= 2:
            counts = {count for _, count in table_rows}
            if len(counts) != 1:
                detail = ", ".join(
                    f"L{line}:{count} pipes" for line, count in table_rows
                )
                errors.append(
                    f"{name}: inconsistent Markdown table columns ({detail}); "
                    "escape literal pipe characters or avoid raw pipes in formulas"
                )
        table_rows = []

    for lineno, line in enumerate(lines, 1):
        if FENCE_RE.match(line):
            flush_table()
            if in_fence:
                in_fence = False
            else:
                in_fence = True
                fence_start = lineno
            continue

        if in_fence:
            continue

        if SINGLE_DOLLAR_RE.match(line):
            errors.append(
                f"{name}:{lineno}: standalone single-dollar line; "
                "use $ for a display-math delimiter"
            )

        if LEGACY_DISPLAY_RE.match(line):
            errors.append(
                f"{name}:{lineno}: legacy display-math delimiter; use $ for GitHub Markdown"
            )

        if LITERAL_BRACKET_RE.match(line):
            errors.append(
                f"{name}:{lineno}: standalone bracket line; likely malformed display math"
            )

        if DISPLAY_MATH_RE.match(line):
            if math_open:
                math_open = False
            else:
                math_open = True
                math_start = lineno

        if TABLE_LINE_RE.match(line):
            table_rows.append((lineno, unescaped_pipe_count(line)))
        else:
            flush_table()

    flush_table()

    if in_fence:
        errors.append(f"{name}:{fence_start}: unclosed fenced code block")
    if math_open:
        errors.append(f"{name}:{math_start}: unclosed $$ display-math block")

    for match in LINK_RE.finditer(text):
        target = match.group(1)
        resolved = resolve_link(path, target)
        if resolved is not None and not resolved.exists():
            line = text.count("\n", 0, match.start()) + 1
            errors.append(f"{name}:{line}: broken relative link {target!r}")

    return errors


def main() -> int:
    errors: list[str] = []
    for path in MARKDOWN_FILES:
        errors.extend(check_file(path))

    if errors:
        print("Markdown integrity check failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print(f"Markdown integrity check passed for {len(MARKDOWN_FILES)} files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
