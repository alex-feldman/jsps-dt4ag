#!/usr/bin/env python3
"""Create a no-overwrite copy of a JSPS-DT4AG run config.

Copies ``configs/example.ini`` (or another INI given with ``--source``) to a new
file, optionally changing existing keys with ``--set SECTION.KEY=VALUE``. The
source is never edited and an existing file is never overwritten. A key the
source lacks is added under its section (created if absent) only when the
loader knows it (``CONFIG_SCHEMA`` in ``dt4ag_config.py``); other keys are
refused. The copy gets a syntax check only, not ``load_config``: the dry run
(``python pipeline/dt4ag_config.py <copy>``, then ``run_pipeline.py --dry-run``)
is the full check.
"""

from __future__ import annotations

import argparse
import configparser
import itertools
import re
import shutil
import sys
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = PIPELINE_DIR / "configs"


def output_names(source: Path, requested: str | None, prefix: str | None):
    """Yield exact, prefix-based, or default copy names."""
    if requested:
        yield requested
        return
    base = prefix or f"{source.stem}-copy"
    yield f"{base}.ini"
    for number in itertools.count(2):
        yield f"{base}-{number}.ini"


def validate_name(name: str) -> None:
    if Path(name).name != name or "/" in name or "\\" in name:
        raise ValueError("--name must be a filename, not a path")
    if Path(name).suffix.lower() != ".ini":
        raise ValueError("--name must end in .ini")


def validate_prefix(prefix: str) -> None:
    if (
        not prefix
        or prefix in {".", ".."}
        or Path(prefix).name != prefix
        or "/" in prefix
        or "\\" in prefix
    ):
        raise ValueError("--name-prefix must be a filename stem, not a path")
    if prefix.lower().endswith(".ini"):
        raise ValueError("--name-prefix must not include a file extension")


def parse_overrides(raw_values: list[str]) -> dict[tuple[str, str], str]:
    """Parse repeated SECTION.KEY=VALUE assignments."""
    overrides = {}
    for raw in raw_values:
        left, separator, value = raw.partition("=")
        if not separator or "." not in left:
            raise ValueError(f"--set must use SECTION.KEY=VALUE: {raw}")
        section, key = left.rsplit(".", 1)
        section, key = section.strip(), key.strip()
        if not section or not key or any(c in section + key for c in "[]\r\n"):
            raise ValueError(f"invalid --set target: {raw}")
        # configparser keeps section names case-sensitive and lowercases keys.
        target = (section, key.casefold())
        if target in overrides:
            raise ValueError(f"duplicate --set target: [{section}] {key}")
        if "\r" in value or "\n" in value:
            raise ValueError(f"--set value must be one line: {raw}")
        if re.search(r"(^|\s)[#;]", value):
            raise ValueError(
                "--set value must not contain ' #' or ' ;' (or start with # or ;): "
                f"the loader reads it as an inline comment and drops the rest: {raw}"
            )
        overrides[target] = value
    return overrides


def known_keys() -> set[tuple[str, str]]:
    """Return every (section, key) the loader reads, from its own registry."""
    from dt4ag_config import CONFIG_SCHEMA

    return {(item.section, item.key.casefold()) for item in CONFIG_SCHEMA}


def add_missing(lines: list[str], missing: dict[tuple[str, str], str]) -> list[str]:
    """Insert keys the source lacks, under their section (created if absent)."""
    newline = "\r\n" if any(line.endswith("\r\n") for line in lines) else "\n"
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += newline
    for (section, key), value in sorted(missing.items()):
        header = re.compile(r"^\s*\[" + re.escape(section) + r"\]\s*(?:[#;].*)?$")
        any_header = re.compile(r"^\s*\[[^\]]+\]")
        start = next((i for i, l in enumerate(lines) if header.match(l.rstrip("\r\n"))), None)
        if start is None:
            if lines and lines[-1].strip():
                lines.append(newline)
            lines.extend([f"[{section}]{newline}", f"{key} = {value}{newline}"])
            continue
        end = start
        for i in range(start + 1, len(lines)):
            if any_header.match(lines[i]):
                break
            if lines[i].strip():
                end = i
        lines.insert(end + 1, f"{key} = {value}{newline}")
    return lines


def apply_overrides(text: str, overrides: dict[tuple[str, str], str]) -> str:
    """Change existing INI values while retaining other lines and comments."""
    if not overrides:
        return text
    lines = text.splitlines(keepends=True)
    section = ""
    found = set()
    for index, line in enumerate(lines):
        section_match = re.match(r"^\s*\[([^\]]+)\]\s*(?:[#;].*)?(?:\r?\n)?$", line)
        if section_match:
            section = section_match.group(1).strip()
            continue
        if not section:
            continue
        key_match = re.match(r"^([ \t]*)([^=\r\n]+?)([ \t]*=[ \t]*)([^\r\n]*?)(\r?\n)?$", line)
        if not key_match:
            continue
        key = key_match.group(2).strip().casefold()
        target = (section, key)
        if target not in overrides:
            continue
        if target in found:
            raise ValueError(f"duplicate INI key: [{section}] {key}")
        newline = key_match.group(5) or ""
        lines[index] = (
            key_match.group(1)
            + key_match.group(2)
            + key_match.group(3)
            + overrides[target]
            + newline
        )
        found.add(target)
    missing = set(overrides) - found
    if missing:
        unknown = missing - known_keys()
        if unknown:
            labels = ", ".join(f"[{s}] {k}" for s, k in sorted(unknown))
            raise ValueError(
                f"--set key is not in the source and is not a key the loader "
                f"knows: {labels}"
            )
        lines = add_missing(lines, {t: overrides[t] for t in missing})
    result = "".join(lines)
    # Parse the way dt4ag_config.py does, so a value the loader would read
    # differently (or reject) fails here instead of at the first dry run.
    check = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    try:
        check.read_string(result)
    except configparser.Error as exc:
        raise ValueError(f"--set produced an unreadable INI: {exc}")
    return result


def copy_new(
    source: Path,
    destination: Path,
    overrides: dict[tuple[str, str], str],
) -> None:
    """Write a no-overwrite copy, applying only the requested key overrides."""
    created = False
    try:
        if overrides:
            content = apply_overrides(source.read_text(encoding="utf-8"), overrides)
            with destination.open("x", encoding="utf-8", newline="") as destination_file:
                created = True
                destination_file.write(content)
        else:
            with source.open("rb") as source_file:
                with destination.open("xb") as destination_file:
                    created = True
                    shutil.copyfileobj(source_file, destination_file)
        shutil.copystat(source, destination)
    except Exception:
        if created:
            destination.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Copy a source INI into a working directory without editing the source "
            "or overwriting an existing config."
        )
    )
    parser.add_argument(
        "--source",
        default=str(CONFIG_DIR / "example.ini"),
        help="existing source INI to copy; defaults to pipeline/configs/example.ini",
    )
    parser.add_argument(
        "--destination-dir",
        help="directory for the copy; defaults to pipeline/configs/",
    )
    names = parser.add_mutually_exclusive_group()
    names.add_argument("--name", help="exact destination filename; never overwrites")
    names.add_argument(
        "--name-prefix",
        help="filename stem; adds a numeric suffix until an unused INI name is found",
    )
    parser.add_argument(
        "--set", dest="sets", action="append", default=[], metavar="SECTION.KEY=VALUE",
        help="override an existing key in the copied INI; may be repeated",
    )
    args = parser.parse_args(argv)

    try:
        source = Path(args.source).expanduser().resolve(strict=True)
        if not source.is_file():
            raise ValueError(f"source is not a regular file: {source}")
        if source.suffix.lower() != ".ini":
            raise ValueError(f"source must be an .ini file: {source}")
        if args.name:
            validate_name(args.name)
        if args.name_prefix:
            validate_prefix(args.name_prefix)
        overrides = parse_overrides(args.sets)
        if args.destination_dir:
            destination_dir = Path(args.destination_dir).expanduser().resolve()
            destination_dir.mkdir(parents=True, exist_ok=True)
        else:
            destination_dir = CONFIG_DIR
            if not destination_dir.is_dir():
                raise ValueError(f"working config directory is missing: {destination_dir}")

        for name in output_names(source, args.name, args.name_prefix):
            destination = destination_dir / name
            if destination.resolve() == source:
                raise ValueError("source and destination are the same file")
            try:
                copy_new(source, destination, overrides)
                print(destination.resolve())
                return 0
            except FileExistsError:
                if args.name:
                    raise ValueError(f"destination already exists: {destination}")
                continue
        raise ValueError("could not choose an unused destination filename")
    except (OSError, ValueError) as exc:
        print(f"CONFIG COPY ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
