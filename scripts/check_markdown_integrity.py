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
MATH_SETEXT_COLLISION_RE = re.compile(r"^\s*(?:=+|-+)\s*$")
SINGLE_DOLLAR_RE = re.compile(r"^\s*\$\s*$")
LEGACY_DISPLAY_RE = re.compile(r"^\s*\\[\[\]]\s*$")
LITERAL_BRACKET_RE = re.compile(r"^\s*[\[\]]\s*$")
TABLE_LINE_RE = re.compile(r"^\s*\|.*\|\s*$")
HEADING_RE = re.compile(r"^\s*#{1,6}\s+(.+?)\s*$")
INLINE_CODE_RE = re.compile(r"\`[^\`]*\`")
HEADING_LATEX_RE = re.compile(
    r"(?:\$|\\[A-Za-z]+|\\[()\[\]]|_\{|\^\{)"
)
LEGACY_INLINE_MATH_RE = re.compile(r"\\[()\[\]]")
INLINE_MATH_SPAN_RE = re.compile(r"(?<!\\)\$[^$\n]+?(?<!\\)\$")
TEX_COMMAND_RE = re.compile(r"\\[A-Za-z]+")
FORBIDDEN_KATEX_MACRO_RE = re.compile(r"\\operatorname\\b")


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


def unescaped_dollar_count(line: str) -> int:
    count = 0
    escaped = False
    for char in line:
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == "$":
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
                f"{name}:{line}: control character code {code}; "
                "possible broken escape sequence"
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
                    f"L{line}:{count} pipes"
                    for line, count in table_rows
                )
                errors.append(
                    f"{name}: inconsistent Markdown table columns ({detail}); "
                    "escape literal pipe characters or avoid raw pipes in formulas"
                )
        table_rows = []

    for lineno, line in enumerate(lines, 1):
        if FORBIDDEN_KATEX_MACRO_RE.search(line):
            errors.append(
                f"{name}:{lineno}: disallowed GitHub math macro \\operatorname; "
                "use a GitHub-supported form such as \\mathrm{...}"
            )

        if FENCE_RE.match(line):
            flush_table()
            in_fence = not in_fence
            if in_fence:
                fence_start = lineno
            continue

        if in_fence:
            continue

        heading_match = HEADING_RE.match(line)
        if heading_match:
            heading = INLINE_CODE_RE.sub("", heading_match.group(1))
            if HEADING_LATEX_RE.search(heading):
                errors.append(
                    f"{name}:{lineno}: raw LaTeX/math syntax in Markdown heading; "
                    "use plain Unicode/text in headings and put equations in the body"
                )

        if SINGLE_DOLLAR_RE.match(line):
            errors.append(
                f"{name}:{lineno}: standalone single-dollar line; "
                "use $$ for a display-math delimiter"
            )

        if LEGACY_DISPLAY_RE.match(line):
            errors.append(
                f"{name}:{lineno}: legacy display-math delimiter; "
                "use $$ for GitHub Markdown"
            )

        if LITERAL_BRACKET_RE.match(line):
            errors.append(
                f"{name}:{lineno}: standalone bracket line; "
                "likely malformed display math"
            )

        if LEGACY_INLINE_MATH_RE.search(line):
            errors.append(
                f"{name}:{lineno}: legacy LaTeX delimiter; "
                "use GitHub-compatible $...$ or $$ blocks"
            )

        if DISPLAY_MATH_RE.match(line):
            math_open = not math_open
            if math_open:
                math_start = lineno
        elif math_open:
            if MATH_SETEXT_COLLISION_RE.match(line):
                errors.append(
                    f"{name}:{lineno}: Setext/heading collision inside $ "
                    "display math; use a fenced ```math block"
                )
        else:
            if unescaped_dollar_count(line) % 2 != 0:
                errors.append(
                    f"{name}:{lineno}: unbalanced inline $ delimiter"
                )

            outside_math = INLINE_CODE_RE.sub("", line)
            outside_math = INLINE_MATH_SPAN_RE.sub("", outside_math)
            if TEX_COMMAND_RE.search(outside_math):
                errors.append(
                    f"{name}:{lineno}: raw TeX command outside a math span; "
                    "wrap math in $...$ or use plain Unicode/text"
                )

        if TABLE_LINE_RE.match(line):
            table_rows.append((lineno, unescaped_pipe_count(line)))
        else:
            flush_table()

    flush_table()

    if in_fence:
        errors.append(
            f"{name}:{fence_start}: unclosed fenced code block"
        )
    if math_open:
        errors.append(
            f"{name}:{math_start}: unclosed $$ display-math block"
        )

    for match in LINK_RE.finditer(text):
        target = match.group(1)
        resolved = resolve_link(path, target)
        if resolved is not None and not resolved.exists():
            line = text.count("\n", 0, match.start()) + 1
            errors.append(
                f"{name}:{line}: broken relative link {target!r}"
            )

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

    print(
        f"Markdown integrity check passed for "
        f"{len(MARKDOWN_FILES)} files."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
