#!/usr/bin/env python3
"""Convert a Wikidata JSON dump slice to Cortex Knowledge Packet JSONL."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cortex.knowledge import statement_to_json
from cortex.wikidata import iter_dump_statements


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--entity",
        action="append",
        default=[],
        help="Wikidata entity ID to retain, e.g. Q142; repeatable",
    )
    parser.add_argument("--include-deprecated", action="store_true")
    parser.add_argument("--max-statements", type=int, default=None)
    args = parser.parse_args()

    wanted = set(args.entity) or None
    count = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for statement in iter_dump_statements(
            args.input,
            wanted_ids=wanted,
            include_deprecated=args.include_deprecated,
        ):
            handle.write(
                json.dumps(
                    statement_to_json(statement),
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
            count += 1
            if args.max_statements is not None and count >= args.max_statements:
                break

    print(
        json.dumps(
            {
                "input": str(args.input),
                "output": str(args.output),
                "statements": count,
                "entities_filter": sorted(wanted) if wanted else None,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
