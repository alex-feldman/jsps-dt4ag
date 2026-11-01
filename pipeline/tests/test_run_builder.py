"""Tests for ``pipeline/scripts/run-builder/``: the page's output and the local server.

The page builds configs and commands in the browser, in the ``builder-core``
script block of ``run-builder.html``. These tests run that block under Node
(``node`` on PATH; the page-output tests are skipped without it) and check
what it produces with the pipeline itself: every generated config loads with
``dt4ag_config.load_config``, and every generated command parses with the real
``run_pipeline``, ``run_collection`` and ``video-frames`` argument parsers.
The server tests call ``serve.Builder`` directly, and one class talks to a real
server on a free 127.0.0.1 port.

Stdlib ``unittest`` only and hermetic: every data tree is a temporary
directory, no GPU, no COLMAP binary (the version is pinned), no samask.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

PIPELINE_DIR = Path(__file__).resolve().parent.parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import run_collection  # noqa: E402
import run_pipeline  # noqa: E402
from dt4ag_config import config_schema, load_config  # noqa: E402

TOOL_DIR = PIPELINE_DIR / "scripts" / "run-builder"
PAGE = TOOL_DIR / "run-builder.html"


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses look their module up here
    spec.loader.exec_module(module)
    return module


serve = _module("run_builder_serve", TOOL_DIR / "serve.py")
build = _module("run_builder_build", TOOL_DIR / "build.py")
video_frames = _module("video_frames_for_builder", PIPELINE_DIR / "scripts" / "video-frames" / "video-frames.py")

NODE = shutil.which("node")
SAMASK_TEMPLATE = ("uv run python 01_segmask_sam3_hq.py --images {images} --masks {masks} "
                   "--prompt {prompt} --device cpu")

# Kept in step with test_dt4ag_config.FORBIDDEN_TERMS. The pipeline
# reconstructs arbitrary scenes; no committed file may imply a subject.
FORBIDDEN_TERMS = [
    "plant",
    "tomato",
    "tanashi",
    "soybean",
    "houseplant",
    "arabidopsis",
]

NODE_HARNESS = r"""
const vm = require("vm"), fs = require("fs");
const html = fs.readFileSync(process.argv[1], "utf8");
const core = html.split('<script id="builder-core">')[1].split("</script>")[0];
const schema = JSON.parse(html.split('<script type="application/json" id="config-schema">')[1].split("</script>")[0]);
const context = {};
vm.createContext(context);
vm.runInContext(core, context);
const states = JSON.parse(fs.readFileSync(0, "utf8"));
process.stdout.write(JSON.stringify(states.map((s) => context.RunBuilder.plan(schema, s))));
"""


def plans(*states):
    """Run the page's ``RunBuilder.plan`` on each state, under Node."""
    done = subprocess.run([NODE, "-e", NODE_HARNESS, str(PAGE)], input=json.dumps(list(states)),
                          capture_output=True, text=True, timeout=60, check=False)
    if done.returncode != 0:
        raise AssertionError(f"node failed: {done.stderr}")
    return json.loads(done.stdout)


def plan(state):
    return plans(state)[0]


# --------------------------------------------------------------------------
# a data tree
# --------------------------------------------------------------------------

class DataTree:
    """``data/datasets/<captures>`` with photographs and optional mask sets."""

    def __init__(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.data = self.root / "data"
        self.datasets = self.data / "datasets"
        self.datasets.mkdir(parents=True)
        (self.data / "configs").mkdir()

    def capture(self, rel: str, photos=3, cameras=("cam-a",), mask_sets=()) -> Path:
        capture = self.datasets / rel
        for camera in cameras:
            for index in range(photos):
                photo = capture / "images" / camera / f"frame_{index:03d}.jpg"
                photo.parent.mkdir(parents=True, exist_ok=True)
                photo.write_bytes(b"jpg")
                for directory in mask_sets:
                    mask = capture / directory / camera / f"frame_{index:03d}.png"
                    mask.parent.mkdir(parents=True, exist_ok=True)
                    mask.write_bytes(b"png")
        return capture

    def state(self, **changes) -> dict:
        state = {
            "mode": "capture", "dataRoot": str(self.data), "capture": "", "collection": "", "captures": [],
            "masks": {"mode": "none", "existing": [], "prompts": [], "use": "", "configFor": "one",
                      "prefixVariant": True},
            # Pinned so no COLMAP binary is needed to derive a run id.
            "values": {"train": {"quit_on_train_completion": "true"}, "run": {"colmap_version": "312"}},
            "configName": "test-run", "configsDir": str(self.data / "configs"), "repoRoot": "",
            "shellSetup": "",
            "samask": {"directory": str(self.root / "samask"), "command": SAMASK_TEMPLATE,
                       "maskExtension": ".png"},
            "catalog": {"enabled": False, "root": "", "rootId": "", "captureId": ""},
            "video": {"source": "", "reviewRoot": "", "ffmpegPrefix": ""}, "today": "2026-11-01",
        }
        for key, value in changes.items():
            if isinstance(value, dict) and isinstance(state.get(key), dict):
                state[key] = {**state[key], **value}
            else:
                state[key] = value
        return state

    def write(self, config: dict) -> Path:
        path = self.data / "configs" / config["name"]
        path.write_text(config["text"], encoding="utf-8")
        return path

    def cleanup(self):
        self._tmp.cleanup()


def pipeline_args(step):
    """A pipeline step's argv after ``uv run python <script>``."""
    argv = step["argv"]
    assert argv[:3] == ["uv", "run", "python"], argv
    return argv[3], argv[4:]


# --------------------------------------------------------------------------
# the embedded schema
# --------------------------------------------------------------------------

class TestEmbeddedSchema(unittest.TestCase):
    def test_page_schema_matches_the_loader(self):
        self.assertEqual(
            serve.embedded_schema(PAGE), config_schema(),
            "run-builder.html carries a stale copy of dt4ag_config.config_schema(); "
            "run python3 pipeline/scripts/run-builder/build.py")

    def test_build_check_reports_up_to_date(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(build.main(["--check"]), 0)

    def test_embedding_escapes_script_end(self):
        with mock.patch.object(build, "config_schema", return_value={"help": "</script><b>"}):
            text = build.embed(build.MARKER + "{}</script>")
        self.assertNotIn("</script><b>", text)
        self.assertEqual(json.loads(text[len(build.MARKER):text.index("</script>")]), {"help": "</script><b>"})


class TestObjectAgnostic(unittest.TestCase):
    def test_run_builder_files_name_no_object(self):
        files = sorted(TOOL_DIR.glob("*")) + [Path(__file__)]
        for path in files:
            if path.is_dir() or path.name == "settings.ini":
                continue
            text = path.read_text(encoding="utf-8")
            head, _, rest = text.partition("FORBIDDEN_TERMS = [")
            text = (head + rest.partition("]")[2]) if rest else text
            text = text.replace("pl" + "ant_catalog", "catalog").replace("PL" + "ANT_CATALOG", "CATALOG").lower()
            for term in FORBIDDEN_TERMS:
                self.assertNotIn(term, text, f"{path.name} must stay object-agnostic but mentions {term!r}")

    def test_no_em_dash_in_new_files(self):
        for path in sorted(TOOL_DIR.glob("*")) + [Path(__file__)]:
            if path.is_file() and path.name != "settings.ini":
                self.assertNotIn(chr(0x2014), path.read_text(encoding="utf-8"), path.name)


# --------------------------------------------------------------------------
# what the page generates, checked with the pipeline itself
# --------------------------------------------------------------------------

@unittest.skipUnless(NODE, "node not found: install Node.js to run the page-output tests")
class TestGeneratedConfigs(unittest.TestCase):
    def setUp(self):
        self.tree = DataTree()
        self.addCleanup(self.tree.cleanup)

    def test_one_capture_without_masks(self):
        self.tree.capture("scene-01/capture-a", cameras=("cam-a", "cam-b"))
        p = plan(self.tree.state(capture="scene-01/capture-a"))
        self.assertEqual([c["name"] for c in p["configs"]], ["test-run.ini"])
        cfg = load_config(self.tree.write(p["configs"][0]))
        self.assertEqual(cfg.images_path, self.tree.datasets / "scene-01/capture-a/images")
        self.assertFalse(cfg.use_masks)
        self.assertTrue(cfg.quit_on_train_completion)
        self.assertFalse(p["masksPending"])

    def test_capture_typed_as_absolute_path_inside_datasets(self):
        capture = self.tree.capture("scene-01/capture-a")
        p = plan(self.tree.state(capture=str(capture)))
        cfg = load_config(self.tree.write(p["configs"][0]))
        self.assertEqual(cfg.images_rel, Path("scene-01/capture-a/images"))

    def test_every_setting_reaches_the_config(self):
        self.tree.capture("scene-01/capture-a")
        values = {"train": {"max_num_iterations": "10000", "downscale_factor": "4", "eval_mode": "all",
                            "quit_on_train_completion": "true"},
                  "export": {"checkpoint_interval": "5000", "env_label": "uv-env", "platform_label": "wsl"},
                  "run": {"id_prefix": "test", "colmap_version": "312"},
                  "host": {"memory_high": "8g", "process_cpus": "0,1"}}
        p = plan(self.tree.state(capture="scene-01/capture-a", values=values))
        cfg = load_config(self.tree.write(p["configs"][0]))
        self.assertEqual((cfg.max_num_iterations, cfg.downscale_factor, cfg.eval_mode), (10000, 4, "all"))
        self.assertEqual((cfg.checkpoint_interval, cfg.env_label, cfg.platform_label), (5000, "uv-env", "wsl"))
        self.assertEqual((cfg.run_id_prefix, cfg.memory_high, cfg.process_cpus), ("test", "8G", "0,1"))

    def test_untouched_keys_take_the_loader_defaults(self):
        self.tree.capture("scene-01/capture-a")
        p = plan(self.tree.state(capture="scene-01/capture-a"))
        cfg = load_config(self.tree.write(p["configs"][0]))
        self.assertEqual(cfg.steps_per_save, 2500)
        self.assertEqual(cfg.platform_label, sys.platform)
        self.assertEqual(cfg.memory_high, "")

    def test_existing_mask_set_chosen_by_name(self):
        self.tree.capture("scene-01/capture-a", mask_sets=("masks", "masks_silver-car"))
        p = plan(self.tree.state(capture="scene-01/capture-a",
                                 masks={"mode": "existing", "existing": ["masks", "masks_silver-car"],
                                        "use": "silver-car"},
                                 values={"dataset": {"mask_extensions": ".png"},
                                         "train": {"quit_on_train_completion": "true"},
                                         "run": {"colmap_version": "312"}}))
        self.assertEqual(len(p["configs"]), 1)
        cfg = load_config(self.tree.write(p["configs"][0]))
        self.assertTrue(cfg.use_masks)
        self.assertEqual(cfg.mask_variant, "silver-car")
        self.assertEqual(cfg.masks_path.name, "masks_silver-car")

    def test_default_masks_directory(self):
        self.tree.capture("scene-01/capture-a", mask_sets=("masks",))
        p = plan(self.tree.state(capture="scene-01/capture-a",
                                 masks={"mode": "existing", "existing": ["masks"], "use": ""},
                                 values={"dataset": {"mask_extensions": ".png"},
                                         "train": {"quit_on_train_completion": "true"},
                                         "run": {"colmap_version": "312"}}))
        cfg = load_config(self.tree.write(p["configs"][0]))
        self.assertEqual((cfg.mask_variant, cfg.masks_path.name), ("", "masks"))

    def test_each_generated_variant_is_its_own_config(self):
        self.tree.capture("scene-01/capture-a", mask_sets=("masks_silver-car", "masks_wheel"))
        p = plan(self.tree.state(capture="scene-01/capture-a",
                                 masks={"mode": "generate", "prompts": ["Silver car", "wheel"],
                                        "configFor": "each"}))
        self.assertEqual([c["name"] for c in p["configs"]], ["test-run-silver-car.ini", "test-run-wheel.ini"])
        self.assertTrue(p["masksPending"])
        for config, variant in zip(p["configs"], ("silver-car", "wheel")):
            with self.subTest(variant=variant):
                cfg = load_config(self.tree.write(config))
                self.assertEqual(cfg.mask_variant, variant)
                self.assertEqual(cfg.masks_path, self.tree.datasets / f"scene-01/capture-a/masks_{variant}")
                self.assertEqual(cfg.mask_extensions, frozenset({".png"}))
                self.assertEqual(cfg.run_id_prefix, f"run-{variant}")
                self.assertTrue(str(cfg.masked_images_path).endswith(f"masked/{variant}/scene-01/capture-a"))

    def test_variants_are_never_merged(self):
        self.tree.capture("scene-01/capture-a", mask_sets=("masks_silver-car", "masks_wheel"))
        p = plan(self.tree.state(capture="scene-01/capture-a",
                                 masks={"mode": "generate", "prompts": ["silver car", "wheel"],
                                        "configFor": "one", "use": "wheel"}))
        self.assertEqual(len(p["configs"]), 1)
        cfg = load_config(self.tree.write(p["configs"][0]))
        self.assertEqual(cfg.mask_variant, "wheel")
        masks = [s["fill"]["masks"] for s in p["steps"] if s["kind"] == "samask"]
        self.assertEqual([Path(m).name for m in masks], ["masks_silver-car", "masks_wheel"])

    def test_variant_prefix_can_be_left_off(self):
        self.tree.capture("scene-01/capture-a", mask_sets=("masks_silver-car", "masks_wheel"))
        p = plan(self.tree.state(capture="scene-01/capture-a",
                                 masks={"mode": "generate", "prompts": ["silver car", "wheel"],
                                        "configFor": "each", "prefixVariant": False}))
        self.assertEqual({load_config(self.tree.write(c)).run_id_prefix for c in p["configs"]}, {"run"})

    def test_duplicate_prompts_warn_and_make_one_variant(self):
        p = plan(self.tree.state(capture="scene-01/capture-a",
                                 masks={"mode": "generate", "prompts": ["silver car", "Silver  car"]}))
        self.assertEqual(len([s for s in p["steps"] if s["kind"] == "samask"]), 1)
        self.assertTrue(any("both make masks_silver-car" in w for w in p["warnings"]))

    def test_collection_base_config_loads_and_dry_runs(self):
        for name in ("capture-a", "capture-b"):
            self.tree.capture(f"scene-01/{name}")
        state = self.tree.state(mode="collection", collection="scene-01",
                                captures=["scene-01/capture-a", "scene-01/capture-b"],
                                catalog={"rootId": "test-data"})
        p = plan(state)
        path = self.tree.write(p["configs"][0])
        cfg = load_config(path)
        self.assertEqual(cfg.run_count, "")
        self.assertTrue(cfg.quit_on_train_completion)
        dry = [s for s in p["steps"] if s["kind"] == "dry-run"][0]
        script, args = pipeline_args(dry)
        self.assertEqual(script, "pipeline/run_collection.py")
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            code = run_collection.main(args)
        self.assertEqual(code, 0, out.getvalue())
        self.assertIn("captures found    : 2", out.getvalue())

    def test_collection_surfaces_the_runners_refusal(self):
        self.tree.capture("scene-01/capture-a")
        values = {"train": {"quit_on_train_completion": "true"}, "run": {"run_count": "03", "colmap_version": "312"}}
        p = plan(self.tree.state(mode="collection", collection="scene-01", captures=["scene-01/capture-a"],
                                 catalog={"rootId": "test-data"}, values=values))
        self.tree.write(p["configs"][0])
        _, args = pipeline_args([s for s in p["steps"] if s["kind"] == "dry-run"][0])
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            code = run_collection.main(args)
        self.assertEqual(code, 1)
        self.assertIn("run_count is pinned", out.getvalue())

    def test_collection_variants_get_one_samask_command_per_capture_and_prompt(self):
        for name in ("capture-a", "capture-b"):
            self.tree.capture(f"scene-01/{name}", mask_sets=("masks_silver-car", "masks_wheel"))
        p = plan(self.tree.state(mode="collection", collection="scene-01",
                                 captures=["scene-01/capture-a", "scene-01/capture-b"],
                                 masks={"mode": "generate", "prompts": ["silver car", "wheel"], "configFor": "each"},
                                 catalog={"rootId": "test-data"}))
        samask = [s for s in p["steps"] if s["kind"] == "samask"]
        self.assertEqual(len(samask), 4)
        for config in p["configs"]:
            self.tree.write(config)
        for dry in [s for s in p["steps"] if s["kind"] == "dry-run"]:
            _, args = pipeline_args(dry)
            out = io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(run_collection.main(args), 0, out.getvalue())

    def test_single_capture_dry_run_succeeds(self):
        self.tree.capture("scene-01/capture-a")
        p = plan(self.tree.state(capture="scene-01/capture-a"))
        self.tree.write(p["configs"][0])
        _, args = pipeline_args([s for s in p["steps"] if s["kind"] == "dry-run"][0])
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(run_pipeline.main(args), 0, out.getvalue())
        self.assertIn("images found      : 3", out.getvalue())

    def test_value_with_line_break_is_joined(self):
        p = plan(self.tree.state(capture="scene-01/capture-a",
                                 values={"colmap": {"extra_args": "--gpu_index 0\n--num_threads 8"}}))
        self.assertIn("extra_args = --gpu_index 0 --num_threads 8", p["configs"][0]["text"])
        self.assertTrue(any("line break" in w for w in p["warnings"]))


@unittest.skipUnless(NODE, "node not found: install Node.js to run the page-output tests")
class TestGeneratedCommands(unittest.TestCase):
    def setUp(self):
        self.tree = DataTree()
        self.addCleanup(self.tree.cleanup)

    def test_capture_commands_parse_with_run_pipeline(self):
        p = plan(self.tree.state(capture="scene-01/capture-a"))
        dry, real = [s for s in p["steps"] if s["kind"] in ("dry-run", "run")]
        for step, is_dry in ((dry, True), (real, False)):
            script, args = pipeline_args(step)
            self.assertEqual(script, "pipeline/run_pipeline.py")
            parsed = run_pipeline.parse_args(args)
            self.assertEqual(parsed.dry_run, is_dry)
            self.assertEqual(parsed.config, str(self.tree.data / "configs" / "test-run.ini"))
            self.assertEqual(parsed.stages, list(run_pipeline.STAGES))
            self.assertEqual(shlex.split(step["command"]), step["argv"])

    def test_catalog_flags_parse(self):
        p = plan(self.tree.state(capture="scene-01/capture-a",
                                 catalog={"enabled": True, "rootId": "test-data", "captureId": "scene-01-capture-a"}))
        _, args = pipeline_args([s for s in p["steps"] if s["kind"] == "run"][0])
        parsed = run_pipeline.parse_args(args)
        self.assertEqual((str(parsed.catalog_root), parsed.catalog_root_id, parsed.catalog_capture_id),
                         (str(self.tree.data), "test-data", "scene-01-capture-a"))

    def test_collection_commands_parse_with_run_collection(self):
        p = plan(self.tree.state(mode="collection", collection="scene-01", captures=["scene-01/capture-a"],
                                 catalog={"rootId": "test-data"}))
        steps = [s for s in p["steps"] if s["kind"] in ("dry-run", "run")]
        self.assertEqual([s["kind"] for s in steps], ["dry-run", "run"])
        for step in steps:
            script, args = pipeline_args(step)
            self.assertEqual(script, "pipeline/run_collection.py")
            parsed = run_collection.build_parser().parse_args(args)
            self.assertEqual((parsed.collection, parsed.catalog_root_id), ("scene-01", "test-data"))
            self.assertEqual(parsed.dry_run, step["kind"] == "dry-run")

    def test_dry_runs_come_before_real_runs(self):
        p = plan(self.tree.state(capture="scene-01/capture-a",
                                 masks={"mode": "generate", "prompts": ["silver car", "wheel"], "configFor": "each"}))
        kinds = [s["kind"] for s in p["steps"]]
        self.assertEqual(kinds, ["note", "samask", "samask", "dry-run", "dry-run", "run", "run"])

    def test_samask_commands_quote_the_prompt_and_name_each_variant(self):
        p = plan(self.tree.state(capture="scene-01/capture a",
                                 masks={"mode": "generate", "prompts": ["driver's door"]}))
        step = [s for s in p["steps"] if s["kind"] == "samask"][0]
        words = shlex.split(step["command"])
        self.assertEqual(words[:3], ["(cd", str(self.tree.root / "samask"), "&&"])
        self.assertEqual(words[words.index("--prompt") + 1], "driver's door")
        self.assertEqual(words[words.index("--masks") + 1],
                         str(self.tree.datasets / "scene-01/capture a/masks_driver-s-door"))
        self.assertEqual(words[words.index("--images") + 1], str(self.tree.datasets / "scene-01/capture a/images"))

    def test_missing_samask_template_warns_and_prints_no_command(self):
        p = plan(self.tree.state(capture="scene-01/capture-a", samask={"command": "", "directory": ""},
                                 masks={"mode": "generate", "prompts": ["wheel"]}))
        self.assertEqual([s["command"] for s in p["steps"] if s["kind"] == "samask"], [""])
        self.assertTrue(any("samask command template" in w for w in p["warnings"]))

    def test_unknown_template_placeholder_warns(self):
        p = plan(self.tree.state(capture="scene-01/capture-a", samask={"command": SAMASK_TEMPLATE + " {weights}"},
                                 masks={"mode": "generate", "prompts": ["wheel"]}))
        self.assertTrue(any("{weights}" in w for w in p["warnings"]))

    def test_video_review_command_parses_with_video_frames(self):
        p = plan(self.tree.state(mode="video", capture="clip-0500ms",
                                 video={"source": "/videos/clip one.mp4", "reviewRoot": str(self.tree.data / "evals/clip"),
                                        "ffmpegPrefix": "/opt/ffmpeg/bin"}))
        review = [s for s in p["steps"] if s["kind"] == "video"][0]
        self.assertEqual(review["argv"][:2], ["python3", "pipeline/scripts/video-frames/video-frames.py"])
        parsed = video_frames.build_parser().parse_args(review["argv"][2:])
        self.assertEqual((parsed.command, parsed.source, parsed.ffmpeg_prefix),
                         ("review", "/videos/clip one.mp4", "/opt/ffmpeg/bin"))
        self.assertEqual(shlex.split(review["command"]), review["argv"])
        note = [s for s in p["steps"] if s["kind"] == "note" and "extract" in s["what"]][0]
        self.assertIn(str(self.tree.datasets / "clip-0500ms"), note["what"])
        self.assertTrue([s for s in p["steps"] if s["kind"] == "dry-run"], "a video continues as a capture")

    def test_setup_step_quotes_the_repository(self):
        p = plan(self.tree.state(capture="scene-01/capture-a", repoRoot="/home/someone/my repo",
                                 shellSetup='export PATH="$HOME/bin:$PATH"'))
        setup = p["steps"][0]
        self.assertEqual(setup["kind"], "setup")
        self.assertEqual(setup["command"].splitlines(), ["cd '/home/someone/my repo'", 'export PATH="$HOME/bin:$PATH"'])

    def test_shell_quoting_round_trips(self):
        awkward = ["plain", "with space", "it's", "", "$HOME", "a\"b", "semi;colon", "x*y"]
        p = plan(self.tree.state(capture="scene-01/capture-a", configsDir="/data/odd dir/it's"))
        argv = [s for s in p["steps"] if s["kind"] == "run"][0]["argv"]
        self.assertEqual(shlex.split([s for s in p["steps"] if s["kind"] == "run"][0]["command"]), argv)
        harness = r"""
const vm = require("vm"), fs = require("fs");
const html = fs.readFileSync(process.argv[1], "utf8");
const context = {}; vm.createContext(context);
vm.runInContext(html.split('<script id="builder-core">')[1].split("</script>")[0], context);
process.stdout.write(context.RunBuilder.render(JSON.parse(fs.readFileSync(0, "utf8"))));
"""
        done = subprocess.run([NODE, "-e", harness, str(PAGE)], input=json.dumps(awkward),
                              capture_output=True, text=True, timeout=60, check=True)
        self.assertEqual(shlex.split(done.stdout), awkward)


# --------------------------------------------------------------------------
# the server's handlers
# --------------------------------------------------------------------------

def _completed(argv, **kwargs):
    return subprocess.CompletedProcess(argv, 0, stdout="dry run output\n", stderr="")


class ServerCase(unittest.TestCase):
    def setUp(self):
        self.tree = DataTree()
        self.addCleanup(self.tree.cleanup)
        self.settings = self._settings()
        self.calls = []

        def runner(argv, **kwargs):
            self.calls.append(argv)
            return _completed(argv)

        self.builder = serve.Builder(self.settings, runner=runner)

    def _settings(self):
        with mock.patch.object(serve, "DEFAULT_SETTINGS", self.tree.root / "absent.ini"):
            return serve.load_settings(environ={"RUN_BUILDER_DATA_ROOT": str(self.tree.data),
                                                "RUN_BUILDER_CATALOG_ROOT_ID": "test-data"})

    def config_text(self, images_subpath="scene-01/capture-a/images", data_root=None, extra=""):
        return (f"[paths]\ndata_root = {data_root or self.tree.data}\n[dataset]\nimages_subpath = {images_subpath}\n"
                f"{extra}\n[run]\ncolmap_version = 312\n[colmap]\n[nerfstudio]\n"
                "[train]\nquit_on_train_completion = true\n[export]\n")


class TestSettings(unittest.TestCase):
    def setUp(self):
        self.tree = DataTree()
        self.addCleanup(self.tree.cleanup)

    def test_file_then_environment(self):
        settings = self.tree.root / "settings.ini"
        settings.write_text(f"[run-builder]\ndata_root = {self.tree.data}\ncatalog_root_id = from-file\n"
                            "[samask]\ncommand = run {images}\nmask_extension = .png\n", encoding="utf-8")
        loaded = serve.load_settings(settings, environ={"RUN_BUILDER_CATALOG_ROOT_ID": "from-env"})
        self.assertEqual(loaded.data_root, self.tree.data.resolve())
        self.assertEqual(loaded.configs_dir, (self.tree.data / "configs").resolve())
        self.assertEqual(loaded.catalog_root, self.tree.data.resolve())
        self.assertEqual((loaded.catalog_root_id, loaded.samask_command, loaded.samask_mask_extension),
                         ("from-env", "run {images}", ".png"))

    def test_no_data_root_is_refused(self):
        with mock.patch.object(serve, "DEFAULT_SETTINGS", self.tree.root / "absent.ini"):
            with self.assertRaises(serve.Refused):
                serve.load_settings(environ={})

    def test_unknown_setting_is_refused(self):
        settings = self.tree.root / "settings.ini"
        settings.write_text(f"[run-builder]\ndata_root = {self.tree.data}\ndata_rot = x\n", encoding="utf-8")
        with self.assertRaises(serve.Refused) as caught:
            serve.load_settings(settings, environ={})
        self.assertIn("data_rot", str(caught.exception))

    def test_configs_dir_outside_the_data_root_is_refused(self):
        with self.assertRaises(serve.Refused):
            serve.load_settings(None, environ={"RUN_BUILDER_DATA_ROOT": str(self.tree.data),
                                               "RUN_BUILDER_CONFIGS_DIR": str(self.tree.root)})

    def test_server_binds_loopback_only(self):
        self.assertEqual(serve.HOST, "127.0.0.1")
        self.assertNotIn("--host", serve.build_parser().format_help())


class TestListingAndDiscovery(ServerCase):
    def test_list_marks_captures_and_skips_hidden(self):
        self.tree.capture("scene-01/capture-a")
        (self.tree.datasets / "scene-01" / ".hidden").mkdir()
        (self.tree.datasets / "scene-01" / "notes").mkdir()
        listing = self.builder.list({"path": "datasets/scene-01"})
        self.assertEqual([(e["name"], e["is_capture"]) for e in listing["entries"]],
                         [("capture-a", True), ("notes", False)])

    def test_paths_outside_the_data_root_are_refused(self):
        outside = self.tree.root / "elsewhere"
        outside.mkdir()
        (self.tree.data / "link-out").symlink_to(outside)
        for path in ("..", str(self.tree.root), "/", "link-out", "datasets/../../elsewhere"):
            with self.subTest(path=path):
                with self.assertRaises(serve.Refused) as caught:
                    self.builder.list({"path": path})
                self.assertEqual(caught.exception.status, 403)
        with self.assertRaises(serve.Refused):
            self.builder.capture({"path": "../../elsewhere"})
        with self.assertRaises(serve.Refused):
            self.builder.discover({"path": str(outside)})

    def test_capture_summary(self):
        self.tree.capture("scene-01/capture-a", cameras=("cam-a", "cam-b"), mask_sets=("masks", "masks_wheel"))
        summary = self.builder.capture({"path": "scene-01/capture-a"})
        self.assertTrue(summary["is_capture"])
        self.assertEqual((summary["photographs"], summary["cameras"]), (6, ["cam-a", "cam-b"]))
        self.assertEqual([(m["directory"], m["variant"], m["files"], m["extensions"]) for m in summary["mask_sets"]],
                         [("masks", "", 6, [".png"]), ("masks_wheel", "wheel", 6, [".png"])])
        self.assertEqual(summary["capture_id"], run_collection.capture_id_for(Path("scene-01/capture-a")))

    def test_discover_uses_the_collection_runner(self):
        for name in ("capture-b", "capture-a"):
            self.tree.capture(f"scene-01/{name}")
        found = self.builder.discover({"path": "scene-01"})
        self.assertEqual([c["rel"] for c in found["captures"]], ["scene-01/capture-a", "scene-01/capture-b"])
        self.assertEqual(found["rel"], "scene-01")

    def test_discover_surfaces_the_runners_refusal(self):
        self.tree.capture("scene-01/capture-a")
        self.tree.capture("scene-01/capture-a/images/inner")
        with self.assertRaises(serve.Refused) as caught:
            self.builder.discover({"path": "scene-01"})
        self.assertIn("inside another capture", str(caught.exception))

    def test_info(self):
        info = self.builder.info({})
        self.assertTrue(info["schema_matches"])
        self.assertEqual(info["settings"]["data_root"], str(self.tree.data.resolve()))


class TestValidate(ServerCase):
    def test_good_config_loads_and_dry_runs(self):
        self.tree.capture("scene-01/capture-a")
        result = self.builder.validate({"configs": [{"name": "a.ini", "text": self.config_text()}]})["results"][0]
        self.assertTrue(result["loads"], result)
        self.assertEqual(result["dry_run"]["exit_code"], 0)
        argv = self.calls[0]
        self.assertEqual(argv[1:2], [str(PIPELINE_DIR / "run_pipeline.py")])
        self.assertEqual(argv[-1], "--dry-run")

    def test_the_loaders_error_is_returned(self):
        self.tree.capture("scene-01/capture-a")
        text = self.config_text(extra="use_masks = maybe")
        result = self.builder.validate({"configs": [{"name": "a.ini", "text": text}]})["results"][0]
        self.assertFalse(result["loads"])
        self.assertIn("use_masks", result["error"])
        self.assertNotIn("dry_run", result)
        self.assertEqual(self.calls, [])

    def test_masks_still_to_generate(self):
        self.tree.capture("scene-01/capture-a")
        text = self.config_text(extra="use_masks = true\nmask_variant = wheel\nmask_extensions = .png")
        result = self.builder.validate({"masks_pending": True,
                                        "configs": [{"name": "a.ini", "text": text}]})["results"][0]
        self.assertFalse(result["loads"])
        self.assertTrue(result["loads_without_masks"])
        self.assertIn("masks_wheel", result["error"])

    def test_collection_dry_run_argv(self):
        self.tree.capture("scene-01/capture-a")
        self.builder.validate({"kind": "collection", "collection": "scene-01",
                               "configs": [{"name": "a.ini", "text": self.config_text()}]})
        argv = self.calls[0]
        self.assertEqual(argv[1], str(PIPELINE_DIR / "run_collection.py"))
        self.assertEqual(argv[argv.index("--collection") + 1], "scene-01")
        self.assertEqual(argv[argv.index("--catalog-root-id") + 1], "test-data")
        self.assertEqual(argv[argv.index("--catalog-root") + 1], str(self.tree.data.resolve()))

    def test_config_pointing_outside_the_data_root_is_refused(self):
        other = DataTree()
        self.addCleanup(other.cleanup)
        other.capture("scene-01/capture-a")
        result = self.builder.validate({"configs": [{"name": "a.ini",
                                                     "text": self.config_text(data_root=other.data)}]})["results"][0]
        self.assertFalse(result["loads"])
        self.assertIn("outside the server's data root", result["error"])

    def test_real_dry_run(self):
        self.tree.capture("scene-01/capture-a")
        builder = serve.Builder(self.settings)
        result = builder.validate({"configs": [{"name": "a.ini", "text": self.config_text()}]})["results"][0]
        self.assertEqual(result["dry_run"]["exit_code"], 0, result["dry_run"]["output"])
        self.assertIn("dry run", result["dry_run"]["output"])

    def test_bad_names_are_refused(self):
        for name in ("../a.ini", "a.txt", "sub/a.ini", ".hidden.ini", ""):
            with self.subTest(name=name):
                with self.assertRaises(serve.Refused):
                    self.builder.validate({"configs": [{"name": name, "text": self.config_text()}]})


class TestSave(ServerCase):
    def test_saves_into_the_configs_directory(self):
        self.tree.capture("scene-01/capture-a")
        saved = self.builder.save({"configs": [{"name": "a.ini", "text": self.config_text()}]})
        target = self.tree.data.resolve() / "configs" / "a.ini"
        self.assertEqual(saved["written"], [str(target)])
        self.assertEqual(target.read_text(encoding="utf-8"), self.config_text())

    def test_never_overwrites(self):
        self.tree.capture("scene-01/capture-a")
        existing = self.tree.data / "configs" / "b.ini"
        existing.write_text("keep me", encoding="utf-8")
        with self.assertRaises(serve.Refused) as caught:
            self.builder.save({"configs": [{"name": "a.ini", "text": self.config_text()},
                                           {"name": "b.ini", "text": self.config_text()}]})
        self.assertEqual(caught.exception.status, 409)
        self.assertEqual(existing.read_text(encoding="utf-8"), "keep me")
        self.assertFalse((self.tree.data / "configs" / "a.ini").exists(), "nothing is saved when one name exists")

    def test_refuses_a_directory_outside_the_data_root(self):
        self.tree.capture("scene-01/capture-a")
        with self.assertRaises(serve.Refused) as caught:
            self.builder.save({"directory": str(self.tree.root), "configs": [{"name": "a.ini", "text": self.config_text()}]})
        self.assertEqual(caught.exception.status, 403)
        self.assertFalse((self.tree.root / "a.ini").exists())

    def test_refuses_a_config_that_does_not_load(self):
        self.tree.capture("scene-01/capture-a")
        with self.assertRaises(serve.Refused) as caught:
            self.builder.save({"configs": [{"name": "a.ini", "text": self.config_text(images_subpath="missing/images")}]})
        self.assertIn("does not exist", str(caught.exception))
        self.assertFalse((self.tree.data / "configs" / "a.ini").exists())

    def test_saves_with_masks_still_to_generate(self):
        self.tree.capture("scene-01/capture-a")
        text = self.config_text(extra="use_masks = true\nmask_variant = wheel\nmask_extensions = .png")
        with self.assertRaises(serve.Refused):
            self.builder.save({"configs": [{"name": "a.ini", "text": text}]})
        saved = self.builder.save({"masks_pending": True, "configs": [{"name": "a.ini", "text": text}]})
        self.assertEqual(len(saved["written"]), 1)

    def test_masks_pending_still_checks_the_photographs(self):
        text = self.config_text(images_subpath="missing/images",
                                extra="use_masks = true\nmask_variant = wheel\nmask_extensions = .png")
        with self.assertRaises(serve.Refused):
            self.builder.save({"masks_pending": True, "configs": [{"name": "a.ini", "text": text}]})

    def test_creates_a_missing_configs_subdirectory(self):
        self.tree.capture("scene-01/capture-a")
        saved = self.builder.save({"directory": "configs/new", "configs": [{"name": "a.ini", "text": self.config_text()}]})
        self.assertTrue(Path(saved["written"][0]).is_file())


class TestHttp(ServerCase):
    def setUp(self):
        super().setUp()
        self.server = serve.ThreadingHTTPServer((serve.HOST, 0), serve.make_handler(self.builder, quiet=True))
        self.port = self.server.server_address[1]
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    def request(self, path, body=None, headers=None):
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(f"http://{serve.HOST}:{self.port}{path}", data=data,
                                         headers={"Content-Type": "application/json", **(headers or {})})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as error:
            return error.code, error.read()

    def test_serves_the_page(self):
        status, body = self.request("/")
        self.assertEqual(status, 200)
        self.assertIn(b'id="builder-core"', body)

    def test_api_round_trip(self):
        self.tree.capture("scene-01/capture-a")
        status, body = self.request("/api/capture", {"path": "scene-01/capture-a"})
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["photographs"], 3)

    def test_refusal_is_json_with_its_status(self):
        status, body = self.request("/api/list", {"path": "/"})
        self.assertEqual(status, 403)
        self.assertIn("outside the data root", json.loads(body)["error"])

    def test_foreign_host_and_origin_are_refused(self):
        self.assertEqual(self.request("/api/info", {}, {"Host": "evil.example:80"})[0], 403)
        self.assertEqual(self.request("/api/info", {}, {"Origin": "http://evil.example"})[0], 403)
        self.assertEqual(self.request("/api/info", {}, {"Origin": f"http://127.0.0.1:{self.port}"})[0], 200)

    def test_unknown_action_and_non_json(self):
        self.assertEqual(self.request("/api/delete", {})[0], 404)
        self.assertEqual(self.request("/api/info", {}, {"Content-Type": "text/plain"})[0], 415)


if __name__ == "__main__":
    unittest.main()
