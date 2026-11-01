#!/usr/bin/env python3
"""Serve the run builder page on 127.0.0.1 and answer what a file cannot.

Usage, from the repository root, in the pipeline environment::

    uv run python pipeline/scripts/run-builder/serve.py [--port 8765] [--settings FILE]

The page (``run-builder.html``) builds the config and the commands itself and
also works opened directly as a file. This server adds five actions, each one
handler on ``Builder`` behind ``POST /api/<action>`` with a JSON body:

- ``info``: the local settings, the repository root, and whether the page's
  embedded schema matches ``dt4ag_config.config_schema()``;
- ``list``: the directories in one directory under the data root;
- ``capture``: one capture's photographs, camera folders and mask sets;
- ``discover``: the captures beneath a collection (``run_collection.discover_captures``);
- ``validate``: load each generated config with ``dt4ag_config.load_config``,
  then run ``run_pipeline.py --dry-run`` or ``run_collection.py --dry-run``;
- ``save``: validate, then write the configs into a configs directory, never
  overwriting a file.

It binds 127.0.0.1 only, refuses every path outside the data root, and never
deletes a file or starts a pipeline stage. The dry runs it starts execute
nothing. Machine-specific values come from a gitignored settings file or
environment variables (README.md, "Settings"). Standard library only.
"""

from __future__ import annotations

import argparse
import configparser
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Dict, List, Optional

TOOL_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = TOOL_DIR.parents[1]
REPO_ROOT = PIPELINE_DIR.parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import dt4ag_config  # noqa: E402
import run_collection  # noqa: E402
import run_pipeline  # noqa: E402

PAGE = TOOL_DIR / "run-builder.html"
DEFAULT_SETTINGS = TOOL_DIR / "settings.ini"
HOST = "127.0.0.1"
DEFAULT_PORT = 8765
#: Requests larger than this are refused: configs are a few KB.
MAX_BODY_BYTES = 1024 * 1024
#: Directory listings stop here, so a huge directory cannot stall the page.
MAX_ENTRIES = 2000
#: A dry run takes about a second; this is the ceiling, not the expectation.
DRY_RUN_TIMEOUT = 300
#: Dry-run output returned to the page is cut to this many characters.
MAX_OUTPUT_CHARS = 200_000
CONFIG_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*\.ini$")
SCHEMA_MARKER = '<script type="application/json" id="config-schema">'

#: Settings keys: (section, key, environment variable).
SETTINGS_KEYS = (
    ("run-builder", "data_root", "RUN_BUILDER_DATA_ROOT"),
    ("run-builder", "configs_dir", "RUN_BUILDER_CONFIGS_DIR"),
    ("run-builder", "catalog_root", "RUN_BUILDER_CATALOG_ROOT"),
    ("run-builder", "catalog_root_id", "RUN_BUILDER_CATALOG_ROOT_ID"),
    ("run-builder", "shell_setup", "RUN_BUILDER_SHELL_SETUP"),
    ("samask", "directory", "RUN_BUILDER_SAMASK_DIR"),
    ("samask", "command", "RUN_BUILDER_SAMASK_COMMAND"),
    ("samask", "mask_extension", "RUN_BUILDER_SAMASK_MASK_EXTENSION"),
)


class Refused(Exception):
    """A request the server will not carry out. The message goes to the page."""

    def __init__(self, message: str, status: HTTPStatus = HTTPStatus.BAD_REQUEST):
        super().__init__(message)
        self.status = status


# --------------------------------------------------------------------------
# settings
# --------------------------------------------------------------------------

@dataclass
class Settings:
    data_root: Path
    configs_dir: Path
    catalog_root: Path
    catalog_root_id: str = ""
    shell_setup: str = ""
    samask_directory: str = ""
    samask_command: str = ""
    samask_mask_extension: str = ""
    source: str = ""


def load_settings(path: Optional[Path] = None, environ: Optional[Dict[str, str]] = None) -> Settings:
    """Settings from the file (if any), then environment variables, later winning."""
    environ = os.environ if environ is None else environ
    values: Dict[str, str] = {}
    sources = []
    file = Path(path).expanduser() if path else DEFAULT_SETTINGS
    if file.is_file():
        parser = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=(";", "#"))
        try:
            parser.read(file, encoding="utf-8")
        except configparser.Error as exc:
            raise Refused(f"settings file {file} could not be parsed: {exc}") from exc
        known = {(s, k) for s, k, _ in SETTINGS_KEYS}
        for section in parser.sections():
            for key in parser.options(section):
                if (section, key) not in known:
                    raise Refused(f"unknown setting [{section}] {key} in {file}; "
                                  "see settings.example.ini for the keys")
        for section, key, _ in SETTINGS_KEYS:
            if parser.has_option(section, key):
                values[f"{section}.{key}"] = parser.get(section, key).strip()
        sources.append(str(file))
    elif path:
        raise Refused(f"settings file not found: {file}")
    for section, key, variable in SETTINGS_KEYS:
        if environ.get(variable, "").strip():
            values[f"{section}.{key}"] = environ[variable].strip()
            sources.append(variable)
    raw_root = values.get("run-builder.data_root", "")
    if not raw_root:
        raise Refused("no data root: set [run-builder] data_root in "
                      f"{DEFAULT_SETTINGS} or RUN_BUILDER_DATA_ROOT (README.md, \"Settings\")")
    data_root = Path(os.path.expanduser(raw_root)).resolve()
    if not data_root.is_dir():
        raise Refused(f"data root does not exist: {data_root}")

    def under_root(name: str, default: Path) -> Path:
        raw = values.get(name, "")
        candidate = Path(os.path.expanduser(raw)) if raw else default
        if not candidate.is_absolute():
            candidate = data_root / candidate
        candidate = candidate.resolve()
        if not _within(candidate, data_root):
            raise Refused(f"setting {name} = {candidate} is outside the data root {data_root}")
        return candidate

    return Settings(
        data_root=data_root,
        configs_dir=under_root("run-builder.configs_dir", data_root / "configs"),
        catalog_root=under_root("run-builder.catalog_root", data_root),
        catalog_root_id=values.get("run-builder.catalog_root_id", ""),
        shell_setup=values.get("run-builder.shell_setup", ""),
        samask_directory=values.get("samask.directory", ""),
        samask_command=values.get("samask.command", ""),
        samask_mask_extension=values.get("samask.mask_extension", ""),
        source=", ".join(sources),
    )


# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------

def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def embedded_schema(page: Path = PAGE) -> Optional[dict]:
    """The schema JSON embedded in the page, or None if there is none."""
    try:
        text = page.read_text(encoding="utf-8")
    except OSError:
        return None
    start = text.find(SCHEMA_MARKER)
    if start < 0:
        return None
    start += len(SCHEMA_MARKER)
    end = text.find("</script>", start)
    try:
        return json.loads(text[start:end])
    except ValueError:
        return None


def _mask_sets(capture: Path) -> List[dict]:
    """The capture's ``masks`` and ``masks_<variant>`` directories with what they hold."""
    found = []
    for entry in sorted(capture.iterdir()):
        if not entry.is_dir() or entry.is_symlink():
            continue
        if entry.name == "masks":
            variant = ""
        elif entry.name.startswith("masks_") and len(entry.name) > len("masks_"):
            variant = entry.name[len("masks_"):]
        else:
            continue
        files = run_pipeline.count_images(entry)
        found.append({
            "directory": entry.name,
            "variant": variant,
            "files": len(files),
            "extensions": sorted({p.suffix.lower() for p in files}),
        })
    return found


# --------------------------------------------------------------------------
# the actions, one handler each
# --------------------------------------------------------------------------

class Builder:
    """Every action the page can ask for. HTTP-free, so tests call it directly."""

    def __init__(self, settings: Settings,
                 runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
                 page: Path = PAGE):
        self.settings = settings
        self.runner = runner
        self.page = page

    # -- helpers ----------------------------------------------------------

    def _path(self, raw, base: Optional[Path] = None, label: str = "path") -> Path:
        """``raw`` resolved (absolute, or relative to ``base``), refused outside the data root."""
        if not isinstance(raw, str):
            raise Refused(f"{label} must be a string")
        base = base or self.settings.data_root
        candidate = Path(os.path.expanduser(raw.strip())) if raw.strip() else base
        if not candidate.is_absolute():
            candidate = base / candidate
        resolved = candidate.resolve()
        if not _within(resolved, self.settings.data_root):
            raise Refused(f"{label} {candidate} is outside the data root {self.settings.data_root}",
                          HTTPStatus.FORBIDDEN)
        return resolved

    def _datasets_dir(self, body: dict) -> Path:
        name = body.get("datasets_dirname") or "datasets"
        return self._path(str(name), label="datasets directory")

    def _rel(self, path: Path, base: Path) -> str:
        return path.relative_to(base).as_posix() if path != base else ""

    def _capture_summary(self, capture: Path, datasets_dir: Path) -> dict:
        images = capture / run_collection.IMAGES_DIRNAME
        photos = run_pipeline.count_images(images) if images.is_dir() else []
        cameras = sorted({p.parent.relative_to(images).as_posix() for p in photos})
        rel = capture.relative_to(datasets_dir) if _within(capture, datasets_dir) else None
        return {
            "path": str(capture),
            "rel": rel.as_posix() if rel else None,
            "capture_id": run_collection.capture_id_for(rel) if rel else None,
            "is_capture": images.is_dir(),
            "photographs": len(photos),
            "photo_extensions": sorted({p.suffix.lower() for p in photos}),
            "cameras": [c for c in cameras if c != "."],
            "mask_sets": _mask_sets(capture) if capture.is_dir() else [],
        }

    # -- actions ----------------------------------------------------------

    def info(self, body: dict) -> dict:
        embedded = embedded_schema(self.page)
        settings = asdict(self.settings)
        settings = {k: str(v) for k, v in settings.items()}
        return {
            "settings": settings,
            "repo_root": str(REPO_ROOT),
            "python": sys.executable,
            "schema_matches": embedded == dt4ag_config.config_schema(),
        }

    def list(self, body: dict) -> dict:
        folder = self._path(body.get("path", ""), label="directory")
        if not folder.is_dir():
            raise Refused(f"not a directory: {folder}", HTTPStatus.NOT_FOUND)
        entries = []
        for entry in sorted(folder.iterdir()):
            if entry.name.startswith(".") or not entry.is_dir():
                continue
            if len(entries) >= MAX_ENTRIES:
                break
            link = entry.is_symlink()
            entries.append({
                "name": entry.name,
                "path": str(entry),
                "symlink": link,
                "is_capture": not link and (entry / run_collection.IMAGES_DIRNAME).is_dir(),
            })
        parent = folder.parent if folder != self.settings.data_root else None
        return {"path": str(folder), "parent": str(parent) if parent else None,
                "entries": entries, "truncated": len(entries) >= MAX_ENTRIES}

    def capture(self, body: dict) -> dict:
        datasets_dir = self._datasets_dir(body)
        capture = self._path(body.get("path", ""), base=datasets_dir, label="capture")
        if not capture.is_dir():
            raise Refused(f"capture directory does not exist: {capture}", HTTPStatus.NOT_FOUND)
        return self._capture_summary(capture, datasets_dir)

    def discover(self, body: dict) -> dict:
        datasets_dir = self._datasets_dir(body)
        collection = self._path(body.get("path", ""), base=datasets_dir, label="collection")
        if not _within(collection, datasets_dir):
            raise Refused(f"collection {collection} is not inside the datasets directory {datasets_dir}")
        if not collection.is_dir():
            raise Refused(f"collection directory does not exist: {collection}", HTTPStatus.NOT_FOUND)
        try:
            paths, ignored = run_collection.discover_captures(collection)
        except run_collection.CollectionError as exc:
            raise Refused(str(exc)) from exc
        return {
            "collection": str(collection),
            "rel": self._rel(collection, datasets_dir),
            "captures": [self._capture_summary(p, datasets_dir) for p in paths],
            "ignored": [{"path": str(p), "why": why} for p, why in ignored],
        }

    def _configs(self, body: dict) -> List[dict]:
        configs = body.get("configs")
        if not isinstance(configs, list) or not configs:
            raise Refused("no configs given")
        for item in configs:
            if not isinstance(item, dict) or not isinstance(item.get("text"), str) \
                    or not isinstance(item.get("name"), str):
                raise Refused("each config needs a name and a text")
            if not CONFIG_NAME_RE.match(item["name"]):
                raise Refused(f"config name {item['name']!r} must be one file name ending in .ini "
                              "(letters, digits, '.', '_', '-')")
        names = [item["name"] for item in configs]
        if len(set(names)) != len(names):
            raise Refused("two configs have the same name")
        return configs

    def _load(self, path: Path, masks_pending: bool) -> Optional[str]:
        """Load one config file; return the loader's error message, or None.

        With ``masks_pending`` (masks are still to be generated) the config is
        loaded twice: as written with ``validate_paths=False``, which checks
        every key and value, and with ``use_masks`` turned off and paths
        checked, which proves the data root and photographs exist. Only the
        checks that need the mask directory are left to the dry run after
        samask has written it.
        """
        try:
            if not masks_pending:
                cfg = dt4ag_config.load_config(path)
            else:
                cfg = dt4ag_config.load_config(path, validate_paths=False)
                parser = configparser.ConfigParser(interpolation=None)
                parser.read(path, encoding="utf-8")
                if parser.has_section("dataset"):
                    parser.set("dataset", "use_masks", "false")
                probe = path.with_name(path.stem + ".no-masks-probe.ini")
                with probe.open("w", encoding="utf-8") as handle:
                    parser.write(handle)
                dt4ag_config.load_config(probe)
        except dt4ag_config.ConfigError as exc:
            return str(exc)
        if not _within(cfg.data_root.resolve(), self.settings.data_root):
            return (f"[paths] data_root {cfg.data_root} is outside the server's data root "
                    f"{self.settings.data_root}")
        return None

    def _dry_run(self, path: Path, body: dict) -> dict:
        kind = body.get("kind", "capture")
        if kind == "collection":
            collection = body.get("collection")
            if not isinstance(collection, str) or not collection.strip():
                raise Refused("a collection dry run needs the collection path")
            catalog_root = self._path(body.get("catalog_root") or str(self.settings.catalog_root),
                                      label="catalog root")
            root_id = body.get("catalog_root_id") or self.settings.catalog_root_id
            argv = [sys.executable, str(PIPELINE_DIR / "run_collection.py"), "--config", str(path),
                    "--collection", collection.strip(), "--catalog-root", str(catalog_root),
                    "--catalog-root-id", str(root_id), "--dry-run"]
        elif kind == "capture":
            argv = [sys.executable, str(PIPELINE_DIR / "run_pipeline.py"),
                    "--config", str(path), "--dry-run"]
        else:
            raise Refused(f"unknown kind {kind!r}; use capture or collection")
        try:
            done = self.runner(argv, cwd=str(REPO_ROOT), capture_output=True, text=True,
                               timeout=DRY_RUN_TIMEOUT, check=False)
        except subprocess.TimeoutExpired:
            return {"exit_code": None, "output": f"dry run did not finish in {DRY_RUN_TIMEOUT} s"}
        output = (done.stdout or "") + (done.stderr or "")
        return {"exit_code": done.returncode, "output": output[-MAX_OUTPUT_CHARS:]}

    def validate(self, body: dict) -> dict:
        configs = self._configs(body)
        masks_pending = bool(body.get("masks_pending"))
        results = []
        with tempfile.TemporaryDirectory(prefix="run-builder-") as folder:
            for item in configs:
                path = Path(folder) / item["name"]
                path.write_text(item["text"], encoding="utf-8")
                error = self._load(path, masks_pending=False)
                result = {"name": item["name"], "loads": error is None, "error": error}
                if error and masks_pending:
                    pending_error = self._load(path, masks_pending=True)
                    result["loads_without_masks"] = pending_error is None
                    if pending_error:
                        result["error"] = pending_error
                if error is None:
                    result["dry_run"] = self._dry_run(path, body)
                results.append(result)
        return {"results": results}

    def save(self, body: dict) -> dict:
        configs = self._configs(body)
        raw_dir = body.get("directory") or str(self.settings.configs_dir)
        directory = self._path(raw_dir, label="configs directory")
        targets = [directory / item["name"] for item in configs]
        existing = [str(t) for t in targets if t.exists() or t.is_symlink()]
        if existing:
            raise Refused("refusing to overwrite: " + ", ".join(existing)
                          + ". Choose another name.", HTTPStatus.CONFLICT)
        masks_pending = bool(body.get("masks_pending"))
        with tempfile.TemporaryDirectory(prefix="run-builder-") as folder:
            for item in configs:
                path = Path(folder) / item["name"]
                path.write_text(item["text"], encoding="utf-8")
                error = self._load(path, masks_pending=masks_pending)
                if error:
                    raise Refused(f"{item['name']} does not load, so nothing was saved:\n{error}")
        directory.mkdir(parents=True, exist_ok=True)
        written = []
        for item, target in zip(configs, targets):
            try:
                with target.open("x", encoding="utf-8") as handle:
                    handle.write(item["text"])
            except FileExistsError:
                raise Refused(f"refusing to overwrite {target}, which appeared while saving; "
                              f"saved so far: {', '.join(written) or 'nothing'}",
                              HTTPStatus.CONFLICT) from None
            written.append(str(target))
        return {"written": written, "masks_pending": masks_pending}


ACTIONS = ("info", "list", "capture", "discover", "validate", "save")


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------

def make_handler(builder: Builder, quiet: bool = False):
    class Handler(BaseHTTPRequestHandler):
        server_version = "run-builder"

        def log_message(self, fmt, *args):  # one short line per request
            if not quiet:
                sys.stderr.write("%s %s\n" % (self.command, self.path))

        def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: HTTPStatus, value: dict) -> None:
            self._send(status, json.dumps(value).encode("utf-8"), "application/json")

        def _allowed_hosts(self) -> set:
            port = self.server.server_address[1]
            return {f"{HOST}:{port}", f"localhost:{port}"}

        def _host_ok(self) -> bool:
            # A page on another site can make the browser send requests here
            # (DNS rebinding); a Host other than this server's own is refused.
            return self.headers.get("Host", "") in self._allowed_hosts()

        def do_GET(self):
            if not self._host_ok():
                return self._json(HTTPStatus.FORBIDDEN, {"error": "unexpected Host header"})
            if self.path in ("/", "/index.html", "/run-builder.html"):
                return self._send(HTTPStatus.OK, builder.page.read_bytes(), "text/html; charset=utf-8")
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self):
            if not self._host_ok():
                return self._json(HTTPStatus.FORBIDDEN, {"error": "unexpected Host header"})
            origin = self.headers.get("Origin")
            if origin and origin.split("://", 1)[-1] not in self._allowed_hosts():
                return self._json(HTTPStatus.FORBIDDEN, {"error": "cross-origin request refused"})
            if not self.headers.get("Content-Type", "").startswith("application/json"):
                return self._json(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": "send JSON"})
            action = self.path.removeprefix("/api/") if self.path.startswith("/api/") else ""
            if action not in ACTIONS:
                return self._json(HTTPStatus.NOT_FOUND, {"error": f"unknown action {action!r}"})
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = -1
            if length < 0 or length > MAX_BODY_BYTES:
                return self._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "request too large"})
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(body, dict):
                    raise ValueError("the body must be a JSON object")
            except ValueError as exc:
                return self._json(HTTPStatus.BAD_REQUEST, {"error": f"bad JSON: {exc}"})
            try:
                result = getattr(builder, action)(body)
            except Refused as exc:
                return self._json(exc.status, {"error": str(exc)})
            except OSError as exc:
                return self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})
            self._json(HTTPStatus.OK, result)

    return Handler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="serve.py",
        description="Serve the run builder on 127.0.0.1. It lists folders, validates and saves "
                    "configs under the data root; it never deletes anything or starts a run.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"default {DEFAULT_PORT}")
    parser.add_argument("--settings", type=Path,
                        help=f"settings file (default {DEFAULT_SETTINGS.name} beside this script)")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        settings = load_settings(args.settings)
    except Refused as exc:
        print(f"SETTINGS ERROR: {exc}", file=sys.stderr)
        return 2
    builder = Builder(settings)
    if not builder.info({})["schema_matches"]:
        print("warning: the page's embedded config schema differs from dt4ag_config; run "
              "python pipeline/scripts/run-builder/build.py", file=sys.stderr)
    server = ThreadingHTTPServer((HOST, args.port), make_handler(builder))
    print(f"run builder: http://{HOST}:{args.port}/   (data root {settings.data_root}; Ctrl-C stops)",
          flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
