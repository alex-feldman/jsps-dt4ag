"""The exported config schema (``dt4ag_config.CONFIG_SCHEMA``) agrees with the loader.

``CONFIG_SCHEMA`` exists so tools that write configs (``scripts/run-builder/``)
take keys, defaults and allowed values from the pipeline instead of copying
them. These tests fail when the table and ``load_config`` disagree:

- the set of keys the loader actually reads (recorded by wrapping its parser)
  equals the schema's keys plus the retired ones;
- a config with every key written at its schema default loads to exactly the
  same values as a config that omits every optional key;
- every listed choice loads, and an unlisted one is refused;
- every key has help text in ``configs/README.md`` and every key the README
  documents is in the schema.

Stdlib ``unittest`` only, hermetic: each test builds its data tree in a
temporary directory.
"""

from __future__ import annotations

import configparser
import dataclasses
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

PIPELINE_DIR = Path(__file__).resolve().parent.parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import dt4ag_config  # noqa: E402
from dt4ag_config import (  # noqa: E402
    CONFIG_SCHEMA,
    REQUIRED_SECTIONS,
    RETIRED_KEYS,
    ConfigError,
    config_schema,
    load_config,
    readme_key_help,
)


def _write(path: Path, sections: dict) -> Path:
    lines = []
    for name, values in sections.items():
        lines.append(f"[{name}]")
        lines += [f"{key} = {value}" for key, value in values.items()]
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


class _Tree:
    """A data root with one canonical capture, deleted afterwards."""

    def __init__(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.data_root = self.root / "data"
        (self.data_root / "datasets" / "scene-01" / "images").mkdir(parents=True)

    def required(self) -> dict:
        sections = {name: {} for name in REQUIRED_SECTIONS}
        sections["paths"]["data_root"] = str(self.data_root)
        sections["dataset"]["images_subpath"] = "scene-01/images"
        return sections

    def full(self) -> dict:
        """Every schema key written explicitly at its schema default."""
        sections = self.required()
        for key in CONFIG_SCHEMA:
            sections.setdefault(key.section, {})
            if key.required:
                continue
            sections[key.section][key.key] = key.default if key.default is not None else ""
        return sections

    def cleanup(self):
        self._tmp.cleanup()


def _comparable(cfg) -> dict:
    values = dataclasses.asdict(cfg)
    values.pop("source")
    return values


class TestSchemaMatchesLoader(unittest.TestCase):
    def setUp(self):
        self.tree = _Tree()
        self.addCleanup(self.tree.cleanup)

    def test_loader_reads_exactly_the_schema_keys(self):
        read = set()

        class Tracking(configparser.ConfigParser):
            def has_option(self, section, option):
                read.add((section, option))
                return super().has_option(section, option)

            def get(self, section, option, *args, **kwargs):
                read.add((section, option))
                return super().get(section, option, *args, **kwargs)

        fake = types.SimpleNamespace(ConfigParser=Tracking, Error=configparser.Error)
        path = _write(self.tree.root / "full.ini", self.tree.full())
        with mock.patch.object(dt4ag_config, "configparser", fake):
            load_config(path)
        expected = {(k.section, k.key) for k in CONFIG_SCHEMA} | set(RETIRED_KEYS)
        self.assertEqual(
            read, expected,
            "dt4ag_config.CONFIG_SCHEMA is out of step with load_config: "
            f"read but not in the schema {sorted(read - expected)}, "
            f"in the schema but never read {sorted(expected - read)}")

    def test_schema_defaults_are_the_loader_defaults(self):
        minimal = load_config(_write(self.tree.root / "minimal.ini", self.tree.required()))
        explicit = load_config(_write(self.tree.root / "full.ini", self.tree.full()))
        a, b = _comparable(minimal), _comparable(explicit)
        differing = sorted(name for name in a if a[name] != b[name])
        self.assertEqual(differing, [], "writing every key at its CONFIG_SCHEMA default "
                         f"changed these loaded values: {differing}")

    def test_every_choice_loads_and_others_are_refused(self):
        for key in CONFIG_SCHEMA:
            if key.kind != "choice":
                continue
            for value in key.choices:
                sections = self.tree.full()
                sections[key.section][key.key] = value
                # scene_type = video needs a video path; give one to every case.
                sections["nerfstudio"]["video_path"] = "clip.mp4"
                with self.subTest(key=key.key, value=value):
                    load_config(_write(self.tree.root / "choice.ini", sections))
            sections = self.tree.full()
            sections[key.section][key.key] = "not-a-listed-value"
            with self.subTest(key=key.key, value="unlisted"):
                with self.assertRaises(ConfigError) as caught:
                    load_config(_write(self.tree.root / "choice.ini", sections))
                self.assertIn(key.key, str(caught.exception))

    def test_bool_and_number_kinds_reject_text(self):
        for key in CONFIG_SCHEMA:
            if key.kind not in ("bool", "int", "float"):
                continue
            sections = self.tree.full()
            sections[key.section][key.key] = "abc"
            with self.subTest(key=key.key):
                with self.assertRaises(ConfigError):
                    load_config(_write(self.tree.root / "kind.ini", sections))

    def test_required_keys_are_required(self):
        for key in CONFIG_SCHEMA:
            if not key.required:
                continue
            sections = self.tree.required()
            del sections[key.section][key.key]
            with self.subTest(key=key.key):
                with self.assertRaises(ConfigError) as caught:
                    load_config(_write(self.tree.root / "missing.ini", sections))
                self.assertIn(key.key, str(caught.exception))

    def test_sections_flagged_required_are_required(self):
        exported = config_schema()["sections"]
        for section in exported:
            sections = self.tree.required()
            sections.pop(section["name"], None)
            with self.subTest(section=section["name"]):
                if section["required"]:
                    with self.assertRaises(ConfigError):
                        load_config(_write(self.tree.root / "section.ini", sections))
                else:
                    load_config(_write(self.tree.root / "section.ini", sections))


class TestSchemaHelp(unittest.TestCase):
    def test_every_key_has_readme_help(self):
        help_text = readme_key_help()
        missing = [f"[{k.section}] {k.key}" for k in CONFIG_SCHEMA
                   if not help_text.get((k.section, k.key))]
        self.assertEqual(missing, [], "add a row for each of these to the key tables in "
                         "pipeline/configs/README.md")

    def test_readme_documents_no_key_the_loader_lacks(self):
        known = {(k.section, k.key) for k in CONFIG_SCHEMA}
        extra = sorted(pair for pair in readme_key_help() if pair not in known)
        self.assertEqual(extra, [], "configs/README.md documents keys CONFIG_SCHEMA does not have")

    def test_schema_is_json_and_ordered_by_section(self):
        import json
        exported = json.loads(json.dumps(config_schema()))
        names = [section["name"] for section in exported["sections"]]
        self.assertEqual(len(names), len(set(names)), "a section appears twice in CONFIG_SCHEMA")
        self.assertEqual(names[:len(REQUIRED_SECTIONS)], list(REQUIRED_SECTIONS))

    def test_cli_prints_the_schema(self):
        import io
        import json
        from contextlib import redirect_stdout
        out = io.StringIO()
        with redirect_stdout(out):
            code = dt4ag_config._main(["dt4ag_config.py", "--schema"])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out.getvalue()), config_schema())


if __name__ == "__main__":
    unittest.main()
