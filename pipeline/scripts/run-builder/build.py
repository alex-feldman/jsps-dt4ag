#!/usr/bin/env python3
"""Embed the pipeline's config schema in ``run-builder.html``.

Usage, from the repository root::

    python3 pipeline/scripts/run-builder/build.py           # rewrite the embedded schema
    python3 pipeline/scripts/run-builder/build.py --check   # exit 1 if it is stale

The page takes every key, section, default, allowed value and help text from
``dt4ag_config.config_schema()`` and copies none of them by hand. It carries
that schema as JSON inside ``<script type="application/json" id="config-schema">``
so it also works opened directly as a file. Run this after changing
``CONFIG_SCHEMA`` or the key tables in ``pipeline/configs/README.md``;
``tests/test_run_builder.py`` fails while the embedded copy is stale.
Standard library only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TOOL_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = TOOL_DIR.parents[1]
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from dt4ag_config import config_schema  # noqa: E402

PAGE = TOOL_DIR / "run-builder.html"
MARKER = '<script type="application/json" id="config-schema">'


def schema_json() -> str:
    # "</" is escaped so no help text can close the script element early.
    return json.dumps(config_schema(), indent=1).replace("</", "<\\/")


def embed(text: str) -> str:
    start = text.find(MARKER)
    if start < 0:
        raise SystemExit(f"{PAGE} has no {MARKER} element")
    start += len(MARKER)
    end = text.find("</script>", start)
    return text[:start] + "\n" + schema_json() + "\n" + text[end:]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="build.py", description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="report a stale page; write nothing")
    args = parser.parse_args(argv)
    current = PAGE.read_text(encoding="utf-8")
    updated = embed(current)
    if args.check:
        if updated != current:
            print(f"{PAGE} is stale: run python3 {Path(__file__).relative_to(PIPELINE_DIR.parent)}")
            return 1
        print(f"{PAGE} is up to date")
        return 0
    if updated != current:
        PAGE.write_text(updated, encoding="utf-8")
        print(f"updated {PAGE}")
    else:
        print(f"{PAGE} already up to date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
