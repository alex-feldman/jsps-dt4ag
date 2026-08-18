"""Tests for ``pipeline/run_pipeline.py``.

Stdlib ``unittest`` only, matching ``test_dt4ag_config.py``. Every test is
hermetic: no GPU, no COLMAP, no nerfstudio, no network, no data drive. Nothing
here executes a subprocess. What is covered is the runner's own logic, which is
where the mistakes it can make on its own live:

* argument parsing and stage selection
* the checkpoint-discovery rule (a ``config.yml`` alone is NOT enough)
* the commands built for each stage
* export verification, including a PLY that exists but holds no geometry

The stages themselves are not covered here; proving those work needs a real
run, which is the point of the end-to-end verification, not of a unit test.
"""

from __future__ import annotations

import json

import sys
import tempfile
import unittest
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent.parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import run_pipeline  # noqa: E402
from dt4ag_config import load_config  # noqa: E402
from run_pipeline import (  # noqa: E402
    STAGES,
    StageError,
    colmap_command,
    count_images,
    discover_checkpoint,
    export_command,
    export_filename,
    parse_args,
    process_command,
    read_ply_vertex_count,
    resolve_stages,
    train_command,
    verify_export,
)

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

CONFIG_TEMPLATE = """\
[paths]
data_root = {data_root}
datasets_dirname = datasets
colmap_dirname = colmap
outputs_dirname = outputs
exports_dirname = exports

[dataset]
images_subpath = {images_subpath}
{dataset_extra}

[run]
id_prefix = run
date = 260101
run_count = 03
colmap_version = 3120
log_file = run-log.csv

[colmap]
data_type = individual
single_camera = 1
single_camera_per_folder = 1
dense = 0
extra_args = {extra_args}

[nerfstudio]
scene_type = images
video_path =
skip_colmap = {skip_colmap}
colmap_model_path = sparse/0

[train]
method = splatfacto
max_num_iterations = 500
{train_extra}
use_scale_regularization = true
background_color = random
quit_on_train_completion = true
max_log_size = 0

[export]
format = gaussian-splat
env_label = my-env
platform_label = ubuntu
export_3dgs = false
gauss_to_pc_script =
"""


def make_config(root: Path, **overrides):
    """Build a data tree and a config file inside ``root``, and load it.

    Defaults to the pre-2026-08-18 layout (camera directories directly under
    the capture, masks beside the photographs), which is still supported and is
    what most of these tests exercise. Pass ``images_subpath`` ending in
    ``images`` for the canonical layout.
    """
    values = {
        "data_root": root,
        "extra_args": "",
        "skip_colmap": "true",
        "dataset_extra": "",
        "train_extra": "",
        "images_subpath": "scene-01/capture-a/session-001",
    }
    values.update(overrides)
    images = root / "datasets" / values["images_subpath"]
    images.mkdir(parents=True, exist_ok=True)
    config_path = root / "run.ini"
    config_path.write_text(CONFIG_TEMPLATE.format(**values), encoding="utf-8")
    return load_config(config_path)


class TempTreeTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)


# --------------------------------------------------------------------------
# stage selection
# --------------------------------------------------------------------------

class TestResolveStages(unittest.TestCase):
    def test_default_is_every_stage_in_order(self):
        self.assertEqual(resolve_stages(None, None), list(STAGES))

    def test_stage_order_is_canonical_not_the_order_given(self):
        self.assertEqual(resolve_stages(["export,colmap"], None), ["colmap", "export"])

    def test_repeated_stage_flags_accumulate(self):
        self.assertEqual(
            resolve_stages(["train", "process"], None), ["process", "train"]
        )

    def test_duplicates_are_collapsed(self):
        self.assertEqual(resolve_stages(["train,train", "train"], None), ["train"])

    def test_whitespace_and_case_are_tolerated(self):
        self.assertEqual(resolve_stages([" Train , EXPORT "], None), ["train", "export"])

    def test_from_stage_runs_the_tail_of_the_pipeline(self):
        self.assertEqual(resolve_stages(None, "train"), ["train", "export"])

    def test_from_stage_colmap_is_the_whole_pipeline(self):
        self.assertEqual(resolve_stages(None, "colmap"), list(STAGES))

    def test_unknown_stage_is_rejected_by_name(self):
        with self.assertRaises(SystemExit) as ctx:
            resolve_stages(["trian"], None)
        self.assertIn("trian", str(ctx.exception))

    def test_stage_and_from_stage_are_mutually_exclusive(self):
        with self.assertRaises(SystemExit) as ctx:
            resolve_stages(["train"], "train")
        self.assertIn("--from-stage", str(ctx.exception))

    def test_empty_stage_value_is_rejected_rather_than_meaning_everything(self):
        with self.assertRaises(SystemExit):
            resolve_stages([","], None)


class TestParseArgs(unittest.TestCase):
    def test_defaults(self):
        args = parse_args([])
        self.assertIsNone(args.config)
        self.assertIsNone(args.run_id)
        self.assertFalse(args.dry_run)
        self.assertFalse(args.allow_viewer_hang)
        self.assertEqual(args.stages, list(STAGES))

    def test_config_and_dry_run(self):
        args = parse_args(["--config", "/tmp/x.ini", "--dry-run"])
        self.assertEqual(args.config, "/tmp/x.ini")
        self.assertTrue(args.dry_run)

    def test_stages_are_resolved_onto_the_namespace(self):
        args = parse_args(["--stage", "export", "--stage", "colmap"])
        self.assertEqual(args.stages, ["colmap", "export"])

    def test_from_stage_rejects_an_unknown_name(self):
        with self.assertRaises(SystemExit):
            parse_args(["--from-stage", "nope"])

    def test_run_id_override_is_carried_through(self):
        args = parse_args(["--run-id", "run_260101-03-3120"])
        self.assertEqual(args.run_id, "run_260101-03-3120")


# --------------------------------------------------------------------------
# checkpoint discovery
# --------------------------------------------------------------------------

class TestDiscoverCheckpoint(TempTreeTestCase):
    def make_run_dir(self, name, config=True, checkpoints=()):
        path = self.root / "splatfacto" / name
        path.mkdir(parents=True)
        if config:
            (path / "config.yml").write_text("method_name: splatfacto\n", encoding="utf-8")
        if checkpoints:
            models = path / "nerfstudio_models"
            models.mkdir()
            for step in checkpoints:
                (models / f"step-{step:09d}.ckpt").write_bytes(b"weights")
        return path

    def test_missing_method_directory_names_it(self):
        with self.assertRaises(StageError) as ctx:
            discover_checkpoint(self.root / "splatfacto")
        self.assertIn("splatfacto", str(ctx.exception))

    def test_no_run_directories_at_all(self):
        (self.root / "splatfacto").mkdir()
        with self.assertRaises(StageError) as ctx:
            discover_checkpoint(self.root / "splatfacto")
        self.assertIn("no training run directories", str(ctx.exception))

    def test_config_yml_without_weights_is_not_usable(self):
        # The real trap: a crashed ns-train leaves config.yml and no weights,
        # and ns-export against it dies inside the checkpoint loader.
        self.make_run_dir("2026-01-01_120000", config=True, checkpoints=())
        with self.assertRaises(StageError) as ctx:
            discover_checkpoint(self.root / "splatfacto")
        self.assertIn("config.yml and a checkpoint", str(ctx.exception))

    def test_weights_without_config_yml_are_not_usable(self):
        self.make_run_dir("2026-01-01_120000", config=False, checkpoints=(499,))
        with self.assertRaises(StageError):
            discover_checkpoint(self.root / "splatfacto")

    def test_picks_the_newest_usable_run(self):
        self.make_run_dir("2026-01-01_120000", checkpoints=(499,))
        newer = self.make_run_dir("2026-01-01_130000", checkpoints=(499,))
        run_dir, config_yml, checkpoint = discover_checkpoint(self.root / "splatfacto")
        self.assertEqual(run_dir, newer)
        self.assertEqual(config_yml, newer / "config.yml")
        self.assertEqual(checkpoint.name, "step-000000499.ckpt")

    def test_skips_a_newer_but_unusable_run(self):
        good = self.make_run_dir("2026-01-01_120000", checkpoints=(499,))
        self.make_run_dir("2026-01-01_130000", checkpoints=())
        run_dir, _, _ = discover_checkpoint(self.root / "splatfacto")
        self.assertEqual(run_dir, good)

    def test_picks_the_highest_checkpoint_within_a_run(self):
        self.make_run_dir("2026-01-01_120000", checkpoints=(499, 1999, 29999))
        _, _, checkpoint = discover_checkpoint(self.root / "splatfacto")
        self.assertEqual(checkpoint.name, "step-000029999.ckpt")

    def test_not_before_rejects_a_run_that_predates_this_invocation(self):
        # Otherwise a training run that crashed without saving weights would be
        # papered over by exporting some earlier run's checkpoint.
        self.make_run_dir("2026-01-01_120000", checkpoints=(499,))
        cutoff = run_pipeline._run_dir_time(
            self.root / "splatfacto" / "2026-01-01_120000"
        ) + 3600
        with self.assertRaises(StageError) as ctx:
            discover_checkpoint(self.root / "splatfacto", not_before=cutoff)
        self.assertIn("NOT being exported", str(ctx.exception))
        self.assertIn("2026-01-01_120000", str(ctx.exception))

    def test_not_before_accepts_a_run_started_after_the_cutoff(self):
        path = self.make_run_dir("2026-01-01_120000", checkpoints=(499,))
        cutoff = run_pipeline._run_dir_time(path) - 60
        run_dir, _, _ = discover_checkpoint(self.root / "splatfacto", not_before=cutoff)
        self.assertEqual(run_dir, path)

    def test_run_dir_time_falls_back_to_mtime_for_an_odd_name(self):
        path = self.make_run_dir("not-a-timestamp", checkpoints=(1,))
        self.assertAlmostEqual(
            run_pipeline._run_dir_time(path), path.stat().st_mtime, places=3
        )


# --------------------------------------------------------------------------
# command construction
# --------------------------------------------------------------------------

class TestCommands(TempTreeTestCase):
    def test_colmap_command(self):
        cfg = make_config(self.root)
        workspace = cfg.colmap_workspace("run_260101-03-3120")
        self.assertEqual(
            colmap_command(cfg, workspace),
            [
                "colmap", "automatic_reconstructor",
                "--workspace_path", str(workspace),
                "--image_path", str(cfg.images_path),
                "--data_type", "individual",
                "--single_camera", "1",
                "--single_camera_per_folder", "1",
                "--dense", "0",
            ],
        )

    def test_colmap_extra_args_are_split_not_passed_as_one_blob(self):
        cfg = make_config(self.root, extra_args="--num_threads 4")
        command = colmap_command(cfg, self.root / "ws")
        self.assertEqual(command[-2:], ["--num_threads", "4"])

    def test_process_command_includes_skip_colmap(self):
        cfg = make_config(self.root)
        command = process_command(cfg, self.root / "ws")
        self.assertEqual(command[:2], ["ns-process-data", "images"])
        self.assertIn("--skip-colmap", command)
        self.assertEqual(command[-2:], ["--colmap-model-path", "sparse/0"])

    def test_process_command_omits_skip_colmap_when_disabled(self):
        cfg = make_config(self.root, skip_colmap="false")
        self.assertNotIn("--skip-colmap", process_command(cfg, self.root / "ws"))

    def test_train_command(self):
        cfg = make_config(self.root)
        workspace = self.root / "ws"
        self.assertEqual(
            train_command(cfg, workspace),
            [
                "ns-train", "splatfacto",
                "--data", str(workspace),
                "--pipeline.model.use_scale_regularization", "True",
                "--pipeline.model.background_color", "random",
                "--output-dir", str(cfg.output_parent),
                "--viewer.quit-on-train-completion", "True",
                "--max-num-iterations", "500",
                "--logging.local-writer.max-log-size", "0",
            ],
        )

    def test_export_filename_records_the_run_provenance(self):
        """The leading component is the CAPTURE, not the collection above it."""
        cfg = make_config(self.root)
        self.assertEqual(
            export_filename(cfg, "run_260101-03-3120"),
            "session-001_run_260101-03-3120_splat_ubuntu_my-env_500steps_individual.ply",
        )

    def test_export_filename_carries_the_resolution_when_known(self):
        """Two resolutions of one dataset must not collide on one filename."""
        cfg = make_config(self.root)
        self.assertEqual(
            export_filename(cfg, "run_260101-03-3120", 2),
            "session-001_run_260101-03-3120_splat_ubuntu_my-env_500steps_ds2_individual.ply",
        )
        self.assertNotEqual(
            export_filename(cfg, "run_260101-03-3120", 2),
            export_filename(cfg, "run_260101-03-3120", 4),
        )

    def test_export_command(self):
        cfg = make_config(self.root)
        command = export_command(
            cfg, self.root / "config.yml", self.root / "exports", "out.ply"
        )
        self.assertEqual(command[:2], ["ns-export", "gaussian-splat"])
        self.assertEqual(command[-2:], ["--output-filename", "out.ply"])


# --------------------------------------------------------------------------
# artefact inspection
# --------------------------------------------------------------------------

class TestCountImages(TempTreeTestCase):
    def test_finds_images_one_directory_per_camera(self):
        for camera in ("cam-a", "cam-b"):
            (self.root / camera).mkdir()
            for index in range(3):
                (self.root / camera / f"{index}.JPG").write_bytes(b"x")
        self.assertEqual(len(count_images(self.root)), 6)

    def test_ignores_non_image_files(self):
        (self.root / "notes.txt").write_text("x", encoding="utf-8")
        (self.root / "a.png").write_bytes(b"x")
        self.assertEqual([p.name for p in count_images(self.root)], ["a.png"])

    def test_empty_directory_yields_nothing(self):
        self.assertEqual(count_images(self.root), [])


class TestExportVerification(TempTreeTestCase):
    def write_ply(self, name, vertices, padding=4096):
        header = (
            "ply\nformat binary_little_endian 1.0\n"
            f"element vertex {vertices}\nproperty float x\nend_header\n"
        ).encode("ascii")
        path = self.root / name
        path.write_bytes(header + b"\0" * padding)
        return path

    def test_reads_the_vertex_count(self):
        self.assertEqual(read_ply_vertex_count(self.write_ply("a.ply", 69015)), 69015)

    def test_missing_file_lists_the_directory(self):
        (self.root / "exports").mkdir()
        (self.root / "exports" / "other.ply").write_bytes(b"x")
        with self.assertRaises(StageError) as ctx:
            verify_export(self.root / "exports" / "wanted.ply", self.root / "exports")
        self.assertIn("does not exist", str(ctx.exception))
        self.assertIn("other.ply", str(ctx.exception))

    def test_tiny_file_is_a_header_with_no_geometry(self):
        path = self.write_ply("small.ply", 0, padding=0)
        with self.assertRaises(StageError) as ctx:
            verify_export(path, self.root)
        self.assertIn("bytes", str(ctx.exception))

    def test_zero_vertices_is_rejected_even_at_a_plausible_size(self):
        path = self.write_ply("empty.ply", 0)
        with self.assertRaises(StageError) as ctx:
            verify_export(path, self.root)
        self.assertIn("0 vertices", str(ctx.exception))

    def test_a_real_looking_export_passes(self):
        verify_export(self.write_ply("good.ply", 69015), self.root)


# --------------------------------------------------------------------------
# the committed runner itself
# --------------------------------------------------------------------------

class TestObjectAgnostic(unittest.TestCase):
    """The runner, its tests and the pipeline README must imply no subject."""

    @staticmethod
    def _scannable(path: Path) -> str:
        """File text with this suite's own list of forbidden terms removed.

        Without this the check trips over the very literal that defines it.
        """
        text = path.read_text(encoding="utf-8")
        head, _, rest = text.partition("FORBIDDEN_TERMS = [")
        if not rest:
            return text.lower()
        return (head + rest.partition("]")[2]).lower()

    def test_committed_files_name_no_object(self):
        for relative in (
            "run_pipeline.py",
            "README.md",
            "tests/test_run_pipeline.py",
        ):
            path = PIPELINE_DIR / relative
            text = self._scannable(path)
            for term in FORBIDDEN_TERMS:
                self.assertNotIn(
                    term,
                    text,
                    f"{relative} must stay object-agnostic, but it mentions "
                    f"{term!r}",
                )


class GpuArchCheckTests(unittest.TestCase):
    """The startup guard that refuses an unsupported GPU.

    These tests stub the measured architecture set rather than reading the real
    gsplat binary, so they stay hermetic and keep passing on a machine with a
    different wheel installed. The set used here, {70, 75, 80, 86, 90} with no
    PTX, is what gsplat 1.4.0+pt24cu121 actually contains, measured from its
    fatbin on 2026-08-11.
    """

    SHIPPED = {70, 75, 80, 86, 90}
    _DEFAULT = object()   # so a test can pass archs=None to mean "unreadable"

    def _check(self, capability, archs=_DEFAULT, has_ptx=False):
        archs = self.SHIPPED if archs is self._DEFAULT else archs
        original = run_pipeline._gsplat_binary_archs
        run_pipeline._gsplat_binary_archs = lambda: (archs, has_ptx)
        try:
            problems = []
            run_pipeline._check_gpu_arch(problems, capability)
            return problems
        finally:
            run_pipeline._gsplat_binary_archs = original

    def test_exact_match_is_accepted(self):
        self.assertEqual(self._check((7, 5)), [])

    def test_higher_minor_within_major_is_accepted(self):
        # Ada is 8.9 and there is no sm_89 cubin, but sm_86 is binary
        # compatible with it. Rejecting Ada would exclude the RTX 40 series.
        self.assertEqual(self._check((8, 9)), [])

    def test_lower_minor_within_major_is_rejected(self):
        # Compatibility runs upward only: an sm_80 cubin does not run on 8.0's
        # predecessors, and there is no such thing here, but the rule must not
        # be implemented as "same major is fine".
        self.assertNotEqual(self._check((9, 0), archs={95}), [])

    def test_newer_major_is_rejected(self):
        problems = self._check((12, 0))
        self.assertEqual(len(problems), 1)
        self.assertIn("sm_120", problems[0])
        self.assertIn("Blackwell", problems[0])

    def test_older_major_is_rejected(self):
        self.assertNotEqual(self._check((6, 1)), [])

    def test_ptx_permits_an_otherwise_unsupported_gpu(self):
        # PTX is the driver's JIT path, so its presence reopens the closed set.
        self.assertEqual(self._check((12, 0), has_ptx=True), [])

    def test_unreadable_binary_does_not_block_the_run(self):
        # A binary we cannot parse is not evidence of an unsupported GPU, and
        # failing closed here would break every machine the parser cannot read.
        self.assertEqual(self._check((12, 0), archs=None), [])

    def test_message_names_the_supported_families_in_hardware_order(self):
        message = self._check((12, 0))[0]
        for family in ("Volta", "Turing", "Ampere", "Ada Lovelace", "Hopper"):
            self.assertIn(family, message)
        self.assertLess(message.index("Volta"), message.index("Hopper"))

    def test_message_explains_that_configuration_cannot_fix_it(self):
        message = self._check((12, 0))[0]
        self.assertIn("no PTX", message)
        self.assertIn("not of this repository", message)


class GsplatBinaryScanTests(unittest.TestCase):
    """The fatbin reader itself, against a synthetic binary.

    Building a fake ELF is cheaper and more honest than asserting against
    whatever wheel happens to be installed: it proves the parser reads the
    fields it claims to read.
    """

    def _cuda_elf(self, sm):
        import struct

        blob = bytearray(b"\x00" * 0x40)
        blob[0:4] = b"\x7fELF"
        blob[4] = 2       # EI_CLASS, 64-bit
        blob[7] = 0x33    # EI_OSABI, ELFOSABI_CUDA
        struct.pack_into("<I", blob, 0x30, sm)   # e_flags, low byte is the arch
        return bytes(blob)

    def _scan(self, blob):
        import tempfile as _tf

        with _tf.TemporaryDirectory() as tmp:
            target = Path(tmp) / "csrc.so"
            target.write_bytes(blob)

            class FakeGsplat:
                __file__ = str(Path(tmp) / "__init__.py")

            real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __import__

            def fake_import(name, *args, **kwargs):
                if name == "gsplat":
                    return FakeGsplat
                return real_import(name, *args, **kwargs)

            import builtins

            builtins.__import__ = fake_import
            try:
                return run_pipeline._gsplat_binary_archs()
            finally:
                builtins.__import__ = real_import

    def test_reads_architectures_from_cuda_elf_headers(self):
        blob = b"pad" + self._cuda_elf(75) + b"pad" + self._cuda_elf(86)
        archs, has_ptx = self._scan(blob)
        self.assertEqual(archs, {75, 86})
        self.assertFalse(has_ptx)

    def test_ignores_non_cuda_elfs(self):
        host = bytearray(self._cuda_elf(99))
        host[7] = 0x00          # a plain System V ELF, not a cubin
        archs, _ = self._scan(bytes(host) + self._cuda_elf(80))
        self.assertEqual(archs, {80})

    def test_returns_none_when_nothing_parses(self):
        archs, _ = self._scan(b"not an elf at all")
        self.assertIsNone(archs)


class MaskSupportTests(TempTreeTestCase):
    """Per-image masks: kept out of SfM, optionally wired into transforms.json.

    The dataset that motivated this stores every frame twice, RGB as .jpg and a
    binary subject mask as .png with the same stem, and both were reaching
    COLMAP as photographs.
    """

    DATASET = "image_extensions = .jpg\nmask_extensions = .png\nuse_masks = {use}"

    def _dataset(self, use_masks="false", pairs=2, unpaired=0):
        cfg = make_config(
            self.root, dataset_extra=self.DATASET.format(use=use_masks)
        )
        for index in range(1, pairs + 1):
            (cfg.images_path / f"shot_{index}.jpg").write_bytes(b"rgb%d" % index)
            (cfg.images_path / f"shot_{index}.png").write_bytes(b"mask%d" % index)
        for index in range(pairs + 1, pairs + unpaired + 1):
            (cfg.images_path / f"shot_{index}.jpg").write_bytes(b"rgb%d" % index)
        return cfg

    def test_default_config_filters_nothing(self):
        cfg = make_config(self.root)
        self.assertFalse(cfg.filters_dataset_files)

    def test_masks_default_to_beside_the_photographs(self):
        cfg = self._dataset()
        self.assertEqual(cfg.masks_path, cfg.images_path)
        self.assertEqual(
            cfg.mask_for(cfg.images_path / "shot_1.jpg"),
            cfg.images_path / "shot_1.png",
        )

    def _canonical(self, use_masks="false", with_masks=True):
        """A capture in the canonical layout: <capture>/images/ and masks/."""
        capture = self.root / "datasets" / "scene-01" / "capture-a"
        (capture / "images" / "cam-002").mkdir(parents=True)
        (capture / "images" / "cam-002" / "shot_1.jpg").write_bytes(b"rgb")
        if with_masks:
            (capture / "masks" / "cam-002").mkdir(parents=True)
            (capture / "masks" / "cam-002" / "shot_1.png").write_bytes(b"mask")
        return capture, make_config(
            self.root,
            images_subpath="scene-01/capture-a/images",
            dataset_extra=self.DATASET.format(use=use_masks),
        )

    def test_masks_are_found_in_the_canonical_sibling_tree(self):
        """<capture>/masks/ mirrors <capture>/images/, with no config key."""
        capture, cfg = self._canonical()
        self.assertEqual(cfg.masks_path, capture / "masks")
        self.assertNotEqual(cfg.masks_path, cfg.images_path)
        self.assertEqual(
            cfg.mask_for(cfg.images_path / "cam-002" / "shot_1.jpg"),
            capture / "masks" / "cam-002" / "shot_1.png",
        )

    def test_a_masks_sibling_is_only_read_beside_a_directory_named_images(self):
        """Otherwise it would belong to the collection, not to this capture.

        The sibling has to be created beside the CONFIGURED image directory,
        which is what the guard inspects. An earlier version of this test made
        one two levels up and passed with the guard deleted.
        """
        cfg = self._dataset()
        sibling = cfg.images_path.parent / "masks"
        sibling.mkdir(parents=True, exist_ok=True)
        # images_path is 'session-001', not 'images', so the sibling is not a
        # capture's mask tree and must be ignored.
        self.assertNotEqual(cfg.images_path.name, "images")
        self.assertEqual(load_config(self.root / "run.ini").masks_path, cfg.images_path)

    def test_use_masks_on_a_canonical_capture_without_masks_is_refused(self):
        with self.assertRaises(Exception):
            self._canonical(use_masks="true", with_masks=False)

    def test_the_retired_mask_subpath_key_is_refused_rather_than_ignored(self):
        """Silently not reading it would train against the wrong supervision."""
        with self.assertRaises(Exception):
            make_config(
                self.root,
                dataset_extra="mask_extensions = .png\nmask_subpath = anything",
            )

    def test_extensions_classify_photographs_and_masks(self):
        cfg = self._dataset()
        self.assertTrue(cfg.is_photograph(Path("a.JPG")))     # case insensitive
        self.assertFalse(cfg.is_photograph(Path("a.png")))
        self.assertTrue(cfg.is_mask(Path("a.png")))

    def test_an_extension_cannot_be_both(self):
        with self.assertRaises(Exception):
            make_config(
                self.root,
                dataset_extra="image_extensions = .jpg\nmask_extensions = .jpg",
            )

    def test_use_masks_without_mask_extensions_is_rejected(self):
        with self.assertRaises(Exception):
            make_config(self.root, dataset_extra="use_masks = true")

    def test_staged_tree_holds_only_photographs(self):
        cfg = self._dataset()
        workspace = cfg.colmap_workspace("run_260101-03-3120")
        staged = run_pipeline.sfm_input_path(cfg, workspace)
        run_pipeline.stage_sfm_inputs(cfg, staged)
        staged_names = sorted(p.name for p in staged.rglob("*") if p.is_file())
        self.assertEqual(staged_names, ["shot_1.jpg", "shot_2.jpg"])

    def test_colmap_and_process_are_pointed_at_the_staged_tree(self):
        cfg = self._dataset()
        workspace = cfg.colmap_workspace("run_260101-03-3120")
        staged = str(run_pipeline.sfm_input_path(cfg, workspace))
        self.assertIn(staged, colmap_command(cfg, workspace))
        self.assertIn(staged, process_command(cfg, workspace))

    def test_unfiltered_configs_still_point_at_the_dataset_directory(self):
        cfg = make_config(self.root)
        workspace = cfg.colmap_workspace("run_260101-03-3120")
        self.assertIn(str(cfg.images_path), colmap_command(cfg, workspace))

    def test_composites_default_under_derived_and_never_into_datasets(self):
        """datasets/ is input; composites are rebuildable and gigabytes big."""
        _, cfg = self._canonical(use_masks="true")
        self.assertEqual(
            cfg.masked_images_path,
            cfg.derived_dir / "masked" / "scene-01" / "capture-a",
        )
        self.assertFalse(
            str(cfg.masked_images_path).startswith(str(cfg.datasets_dir))
        )

    def test_an_override_pointing_into_datasets_is_refused(self):
        """Which is exactly what every pre-v0.2.0 config did."""
        with self.assertRaises(Exception):
            make_config(
                self.root,
                dataset_extra=(
                    "image_extensions = .jpg\nmask_extensions = .png\n"
                    "use_masks = true\n"
                    "masked_images_subpath = datasets/scene-01/masked-images"
                ),
            )

    def test_a_stale_extra_composite_is_refused_not_silently_reused(self):
        """A subset test cannot see files that should not be there.

        The reuse check asks whether the set MATCHES, not whether it contains
        what is needed, because anything extra in the composited directory is
        handed to COLMAP and ns-process-data as part of the capture.
        """
        cfg = self._dataset(use_masks="true")
        cfg.masked_images_path.mkdir(parents=True)
        for index in (1, 2):
            (cfg.masked_images_path / f"shot_{index}.png").write_bytes(b"ok")
        (cfg.masked_images_path / "left_over.png").write_bytes(b"stale")
        with self.assertRaises(Exception) as ctx:
            run_pipeline.composite_masked_images(cfg)
        self.assertIn("left_over.png", str(ctx.exception))

    def test_an_exactly_matching_set_is_still_reused(self):
        """The whole point of the check is to avoid recompositing needlessly."""
        cfg = self._dataset(use_masks="true")
        cfg.masked_images_path.mkdir(parents=True)
        for index in (1, 2):
            (cfg.masked_images_path / f"shot_{index}.png").write_bytes(b"ok")
        lines = run_pipeline.composite_masked_images(cfg)   # would shell out
        self.assertIn("reused", lines[0])

    def test_sfm_input_is_the_masked_directory_when_masking(self):
        """Every stage downstream sees masked images and nothing else."""
        cfg = self._dataset(use_masks="true")
        workspace = cfg.colmap_workspace("run_260101-03-3120")
        self.assertEqual(
            run_pipeline.sfm_input_path(cfg, workspace), cfg.masked_images_path
        )

    def test_colmap_and_process_get_the_same_directory(self):
        """--skip-colmap looks COLMAP's image names up in the copy rename map."""
        cfg = self._dataset(use_masks="true")
        workspace = cfg.colmap_workspace("run_260101-03-3120")
        target = str(cfg.masked_images_path)
        self.assertIn(target, colmap_command(cfg, workspace))
        self.assertIn(target, process_command(cfg, workspace))

    def test_a_photograph_with_no_mask_is_an_error(self):
        cfg = self._dataset(use_masks="true", pairs=1, unpaired=1)
        with self.assertRaises(StageError) as caught:
            run_pipeline.composite_masked_images(cfg)
        self.assertIn("no mask", str(caught.exception))

    def test_an_already_complete_output_is_reused_not_rebuilt(self):
        """Recompositing is minutes and gigabytes for a byte-identical result."""
        cfg = self._dataset(use_masks="true", pairs=2)
        cfg.masked_images_path.mkdir(parents=True)
        for index in (1, 2):
            (cfg.masked_images_path / f"shot_{index}.png").write_bytes(b"composited")
        lines = run_pipeline.composite_masked_images(cfg)   # would shell out if rebuilt
        self.assertIn("reused", lines[0])


class DownscaleFactorTests(TempTreeTestCase):
    def test_zero_leaves_the_flag_off_entirely(self):
        cfg = make_config(self.root)
        self.assertEqual(cfg.downscale_factor, 0)
        self.assertNotIn(
            "nerfstudio-data",
            train_command(cfg, self.root / "ws"),
        )

    def test_a_pinned_factor_is_passed_through(self):
        cfg = make_config(self.root, train_extra="downscale_factor = 4")
        command = train_command(cfg, self.root / "ws")
        # The dataparser is a tyro subcommand appended after the parent's
        # options, not a nested config path. Asserting the tail catches a
        # regression to the plausible-but-rejected --pipeline.datamanager.
        # dataparser.downscale-factor form.
        self.assertEqual(command[-3:], ["nerfstudio-data", "--downscale-factor", "4"])

    def test_non_power_of_two_is_rejected(self):
        with self.assertRaises(Exception):
            make_config(self.root, train_extra="downscale_factor = 3")

    def _pyramid(self, cfg, levels, width=5184, height=2912, first="frame_00047.jpg"):
        workspace = cfg.colmap_workspace("run_260101-03-3120")
        workspace.mkdir(parents=True, exist_ok=True)
        (workspace / "transforms.json").write_text(json.dumps({
            "frames": [{"file_path": f"images/{first}", "w": width, "h": height}]
        }))
        for level in levels:
            directory = workspace / f"images_{level}"
            directory.mkdir(exist_ok=True)
            (directory / first).write_bytes(b"")
        return workspace

    def test_a_pinned_factor_wins_without_touching_the_disk(self):
        cfg = make_config(self.root, train_extra="downscale_factor = 8")
        workspace = cfg.colmap_workspace("run_260101-03-3120")
        self.assertEqual(run_pipeline.resolve_downscale_factor(cfg, workspace), 8)

    def test_auto_resolves_to_the_level_nerfstudio_would_pick(self):
        cfg = make_config(self.root)
        workspace = self._pyramid(cfg, [2, 4, 8])
        # 5184 -> 2592 -> 1296, the first at or under 1600.
        self.assertEqual(run_pipeline.resolve_downscale_factor(cfg, workspace), 4)

    def test_auto_stops_where_the_pyramid_stops(self):
        """A level that exists on disk but lacks the probed frame reads as absent."""
        cfg = make_config(self.root)
        workspace = self._pyramid(cfg, [2])
        self.assertEqual(run_pipeline.resolve_downscale_factor(cfg, workspace), 2)

    def test_auto_is_native_when_there_is_no_pyramid(self):
        cfg = make_config(self.root)
        workspace = self._pyramid(cfg, [])
        self.assertEqual(run_pipeline.resolve_downscale_factor(cfg, workspace), 1)

    def test_small_images_need_no_downscale(self):
        cfg = make_config(self.root)
        workspace = self._pyramid(cfg, [2, 4], width=1280, height=720)
        self.assertEqual(run_pipeline.resolve_downscale_factor(cfg, workspace), 1)

    def test_undeterminable_returns_zero_rather_than_guessing(self):
        cfg = make_config(self.root)
        workspace = cfg.colmap_workspace("run_260101-03-3120")
        workspace.mkdir(parents=True, exist_ok=True)
        self.assertEqual(run_pipeline.resolve_downscale_factor(cfg, workspace), 0)


class DownscalePyramidTests(unittest.TestCase):
    """The check that catches ns-process-data's silent half-built pyramid.

    Regression cover for the 2026-08-17 CUDA OOM: a dataset of .jpg frames
    interleaved with .png masks left one file in each images_N/, nerfstudio
    read that as no pyramid at all, and training ran at 5184x2912.
    """

    def _workspace(self, frames, levels, width=5184, height=2912):
        root = Path(tempfile.mkdtemp())
        (root / "transforms.json").write_text(json.dumps({
            "frames": [
                {"file_path": f"images/{name}", "w": width, "h": height}
                for name in frames
            ]
        }))
        for level, present in levels.items():
            directory = root / f"images_{level}"
            directory.mkdir()
            for name in present:
                (directory / name).write_bytes(b"")
        return root

    def test_complete_pyramid_passes(self):
        frames = ["frame_00001.jpg", "frame_00003.jpg"]
        root = self._workspace(frames, {2: frames, 4: frames})
        lines = run_pipeline.verify_downscale_pyramid(root)
        self.assertEqual(len(lines), 2)
        self.assertIn("2/2", lines[0])

    def test_partial_pyramid_raises_and_names_the_mixed_extensions(self):
        frames = ["frame_00001.jpg", "frame_00003.jpg"]
        root = self._workspace(frames, {4: ["frame_00001.jpg"]})
        with self.assertRaises(StageError) as caught:
            run_pipeline.verify_downscale_pyramid(root)
        self.assertIn("frame_00003.jpg", str(caught.exception))

    def test_partial_pyramid_is_an_error_even_with_the_escape_hatch(self):
        frames = ["frame_00001.jpg", "frame_00003.jpg"]
        root = self._workspace(frames, {4: ["frame_00001.jpg"]})
        with self.assertRaises(StageError):
            run_pipeline.verify_downscale_pyramid(root, allow_full_resolution=True)

    def test_absent_pyramid_raises_when_frames_are_large(self):
        root = self._workspace(["frame_00001.jpg"], {})
        with self.assertRaises(StageError) as caught:
            run_pipeline.verify_downscale_pyramid(root)
        self.assertIn("native resolution", str(caught.exception))

    def test_absent_pyramid_is_fine_for_small_frames(self):
        root = self._workspace(["frame_00001.jpg"], {}, width=1280, height=720)
        self.assertEqual(len(run_pipeline.verify_downscale_pyramid(root)), 1)

    def test_absent_pyramid_can_be_allowed_explicitly(self):
        root = self._workspace(["frame_00001.jpg"], {})
        self.assertEqual(
            len(run_pipeline.verify_downscale_pyramid(root, allow_full_resolution=True)),
            1,
        )

    def test_unregistered_extra_files_do_not_matter(self):
        """The masks live in images/ but are not registered, so they are not required."""
        frames = ["frame_00001.jpg"]
        root = self._workspace(frames, {4: ["frame_00001.jpg", "frame_00002.png"]})
        self.assertEqual(len(run_pipeline.verify_downscale_pyramid(root)), 1)

    def test_mask_directories_do_not_confuse_the_image_check(self):
        """A masks_N/ directory beside images_N/ must not count as a pyramid level.

        Such a directory is left behind by older nerfstudio mask handling; the
        pipeline itself no longer writes one, since masks are composited into
        the alpha channel before any stage runs.
        """
        frames = ["frame_00001.jpg"]
        root = self._workspace(frames, {4: frames})
        (root / "masks_4").mkdir()
        lines = run_pipeline.verify_downscale_pyramid(root)
        self.assertEqual(len(lines), 1)
        self.assertIn("images_4", lines[0])

    def test_max_auto_resolution_matches_nerfstudio(self):
        try:
            from nerfstudio.data.dataparsers import nerfstudio_dataparser
        except Exception:
            self.skipTest("nerfstudio not importable")
        self.assertEqual(
            run_pipeline.MAX_AUTO_RESOLUTION,
            nerfstudio_dataparser.MAX_AUTO_RESOLUTION,
        )


if __name__ == "__main__":
    unittest.main()
