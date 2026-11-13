"""Tests for ``pipeline/colmap_threads.py``, ``pipeline/windows_host.py`` and their use by the runner.

Stdlib ``unittest`` only and hermetic. Image headers are written by the tests;
``/proc`` and ``/sys/fs/cgroup`` are fake trees in a temporary directory;
Windows is a scripted function, so no ``powershell.exe`` runs. The runner tests
replace ``read_host_memory`` with fixed figures and run no subprocess.

What only a real run can show (that the chosen count really fits, and that
COLMAP's per-thread cost is what the constants say) is not covered here; see
the measured constants in ``colmap_threads.py``.
"""

from __future__ import annotations

import configparser
import io
import struct
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

PIPELINE_DIR = Path(__file__).resolve().parent.parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import colmap_threads as ct  # noqa: E402
import run_pipeline  # noqa: E402
import windows_host  # noqa: E402
from dt4ag_config import ConfigError, load_config  # noqa: E402

GIB = 1024


def png_bytes(width: int, height: int) -> bytes:
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + ihdr
            + b"\0\0\0\0" + b"rest of the file")


def jpeg_bytes(width: int, height: int, sof: int = 0xC0, exif: int = 5000) -> bytes:
    def segment(marker: int, body: bytes) -> bytes:
        return bytes([0xFF, marker]) + struct.pack(">H", len(body) + 2) + body

    frame = struct.pack(">BHHB", 8, height, width, 3) + b"\x01\x22\x00\x02\x11\x01\x03\x11\x01"
    return (b"\xff\xd8"
            + segment(0xE0, b"JFIF\0\x01\x01\0\0\x01\0\x01\0\0")
            + segment(0xE1, b"Exif\0\0" + b"\0" * exif)        # a large EXIF block
            + segment(0xDB, b"\0" * 65)                          # quantization table
            + segment(0xC4, b"\0" * 30)                          # DHT: not a frame header
            + b"\xff"                                            # a fill byte
            + segment(sof, frame)
            + segment(0xDA, b"\0" * 10) + b"scan data")


def host(linux=None, cgroup=None, cgroup_label="", ram=None, commit=None, cpus=24, warnings=()):
    return ct.HostMemory(linux_available_mib=linux, cgroup_room_mib=cgroup,
                         cgroup_label=cgroup_label, windows_ram_mib=ram,
                         windows_commit_mib=commit, usable_cpus=cpus,
                         warnings=list(warnings))


def cost_18mp() -> ct.ImageCost:
    return ct.image_cost([Path("a.jpg")], size_of=lambda _: (5184, 3456))


class TempCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)


# --------------------------------------------------------------------------
# the constants live in one place
# --------------------------------------------------------------------------

class TestConstants(unittest.TestCase):
    def test_measured_values(self):
        self.assertEqual((ct.PER_THREAD_MIB, ct.MEASURED_PIXELS, ct.HIGH_QUALITY_MAX_IMAGE_SIZE,
                          ct.BASE_MIB), (1100, 3_840_000, 2400, 256))
        self.assertEqual(ct.MEASURED_PIXELS, 2400 * 1600)

    def test_windows_reserves_come_from_the_guard_thresholds(self):
        self.assertEqual(windows_host.WINDOWS_RAM_RESERVE_MIB, 2048)
        self.assertLess(windows_host.GUARD_STOP_MIB, windows_host.WINDOWS_RAM_RESERVE_MIB)
        self.assertLess(windows_host.WINDOWS_RAM_RESERVE_MIB, windows_host.GUARD_RELIEF_MIB)
        self.assertEqual(windows_host.WINDOWS_COMMIT_RESERVE_MIB,
                         windows_host.GUARD_STOP_COMMIT_MIB)


# --------------------------------------------------------------------------
# header reader
# --------------------------------------------------------------------------

class TestImageSize(TempCase):
    def _file(self, name, data):
        path = self.root / name
        path.write_bytes(data)
        return path

    def test_png(self):
        self.assertEqual(ct.image_size(self._file("a.png", png_bytes(4000, 3000))), (4000, 3000))

    def test_baseline_jpeg_past_a_large_exif_block_and_a_huffman_table(self):
        path = self._file("a.jpg", jpeg_bytes(5184, 3456, exif=60000))
        self.assertEqual(ct.image_size(path), (5184, 3456))

    def test_progressive_jpeg(self):
        self.assertEqual(ct.image_size(self._file("p.jpg", jpeg_bytes(3000, 4000, sof=0xC2))),
                         (3000, 4000))

    def test_unreadable_files_give_none(self):
        cases = {
            "missing.jpg": None,
            "empty.jpg": b"",
            "text.jpg": b"not an image at all",
            "truncated.jpg": jpeg_bytes(100, 100)[:40],
            "noframe.jpg": b"\xff\xd8" + b"\xff\xda\x00\x04\0\0",
            "photo.tif": b"II*\x00" + b"\0" * 40,
            "zero.png": png_bytes(0, 10),
        }
        for name, data in cases.items():
            with self.subTest(name=name):
                path = self.root / name
                if data is not None:
                    path.write_bytes(data)
                self.assertIsNone(ct.image_size(path))


# --------------------------------------------------------------------------
# per-thread cost
# --------------------------------------------------------------------------

class TestImageCost(unittest.TestCase):
    def cost(self, sizes, capped=True):
        table = dict(sizes)
        return ct.image_cost([Path(n) for n in table], capped=capped,
                             size_of=lambda path: table[path.name])

    def test_18_mp_caps_to_the_measured_size(self):
        cost = self.cost({"a": (5184, 3456)})
        self.assertEqual((cost.per_thread_mib, cost.extracted), (1100, (2400, 1600)))
        self.assertEqual(cost.describe(), "1100 MiB/thread for 5184x3456 capped to 2400x1600")

    def test_12_mp_4_by_3_costs_more_than_18_mp_3_by_2(self):
        cost = self.cost({"a": (4000, 3000)})
        self.assertEqual((cost.extracted, cost.per_thread_mib), ((2400, 1800), 1238))

    def test_portrait_and_small_photographs(self):
        self.assertEqual(self.cost({"a": (3456, 5184)}).extracted, (1600, 2400))
        small = self.cost({"a": (1920, 1080)})
        self.assertEqual((small.extracted, small.per_thread_mib), ((1920, 1080), 594))

    def test_largest_capped_photograph_sets_the_cost(self):
        cost = self.cost({"wide": (6000, 2000), "four-three": (4000, 3000), "small": (800, 600)})
        self.assertEqual(cost.size, (4000, 3000))
        self.assertEqual(cost.photographs, 3)

    def test_unreadable_header_counts_as_a_square_at_the_cap(self):
        cost = self.cost({"good": (5184, 3456), "bad": None})
        self.assertEqual((cost.extracted, cost.unreadable), ((2400, 2400), 1))
        self.assertEqual(cost.per_thread_mib, 1650)
        self.assertIn("1 header(s) unreadable", cost.describe())

    def test_another_quality_uses_uncapped_pixels(self):
        cost = self.cost({"a": (5184, 3456)}, capped=False)
        self.assertEqual(cost.extracted, (5184, 3456))
        self.assertEqual(cost.per_thread_mib, 5133)   # ceil(1100 * 17915904 / 3840000)
        unreadable = self.cost({"a": None}, capped=False)
        self.assertEqual(unreadable.extracted, (ct.UNCAPPED_FALLBACK_SIZE,) * 2)


# --------------------------------------------------------------------------
# extra_args
# --------------------------------------------------------------------------

class TestExtraArgs(unittest.TestCase):
    def test_forms(self):
        self.assertEqual(ct.extra_args_option("--gpu_index 0 --num_threads 4", "num_threads"), "4")
        self.assertEqual(ct.extra_args_option("--num_threads=6", "num_threads"), "6")
        self.assertEqual(ct.extra_args_option("--num_threads 2 --num_threads 3", "num_threads"), "3")
        self.assertEqual(ct.extra_args_option("--num_threads", "num_threads"), "")
        self.assertIsNone(ct.extra_args_option("--gpu_index 0", "num_threads"))
        self.assertIsNone(ct.extra_args_option("", "num_threads"))
        self.assertIsNone(ct.extra_args_option("--num_threadsx 4", "num_threads"))
        self.assertIsNone(ct.extra_args_option("'unterminated", "num_threads"))
        self.assertEqual(ct.extra_args_option("--quality extreme", "quality"), "extreme")


# --------------------------------------------------------------------------
# the formula
# --------------------------------------------------------------------------

class TestChooseAuto(unittest.TestCase):
    """The proposal's worked examples (18 MP 3:2 photographs, 1100 MiB per thread)."""

    def test_16_gb_machine_bound_by_windows_ram(self):
        decision = ct.choose_auto(cost_18mp(), host(linux=7300, ram=7168, commit=10240))
        self.assertEqual((decision.threads, decision.source), (4, "auto"))
        self.assertEqual(
            decision.log_line(),
            "colmap threads    : 4 (auto: budget 5120 MiB, bound by Windows available RAM "
            "7168 - 2048; 1100 MiB/thread for 5184x3456 capped to 2400x1600; 24 CPUs)")
        self.assertEqual(decision.command_args(), ["--num_threads", "4"])
        self.assertEqual(decision.recorded, "4")

    def test_less_windows_ram_fewer_threads(self):
        self.assertEqual(ct.choose_auto(cost_18mp(), host(linux=7300, ram=5120)).threads, 2)

    def test_busy_laptop_gets_one_thread(self):
        decision = ct.choose_auto(cost_18mp(), host(linux=7311, ram=4409, commit=12060))
        self.assertEqual(decision.threads, 1)
        self.assertIn("budget 2361 MiB", decision.summary)

    def test_commit_headroom_can_bind(self):
        decision = ct.choose_auto(cost_18mp(), host(linux=30000, ram=20000, commit=5000))
        self.assertEqual(decision.threads, 2)
        self.assertIn("bound by Windows commit headroom 5000 - 2048", decision.summary)

    def test_memory_high_binds(self):
        decision = ct.choose_auto(cost_18mp(), host(linux=7300, cgroup=6144,
                                                    cgroup_label="[host] memory_high 6G",
                                                    ram=9216))
        self.assertEqual(decision.threads, 5)
        self.assertIn("bound by [host] memory_high 6G", decision.summary)

    def test_linux_workstation_is_capped_by_its_cpus(self):
        decision = ct.choose_auto(cost_18mp(), host(linux=58000, cpus=16))
        self.assertEqual(decision.threads, 16)
        self.assertIn("capped at 16 usable CPUs", decision.summary)
        self.assertIn("bound by Linux MemAvailable 58000", decision.summary)

    def test_12_mp_photographs(self):
        cost = ct.image_cost([Path("a")], size_of=lambda _: (4000, 3000))
        self.assertEqual(ct.choose_auto(cost, host(linux=7300, ram=7168)).threads, 3)

    def test_exactly_one_thread_fits(self):
        self.assertEqual(ct.choose_auto(cost_18mp(), host(linux=256 + 1100)).threads, 1)

    def test_below_one_thread_refuses_with_figures_and_the_override(self):
        decision = ct.choose_auto(cost_18mp(), host(linux=7300, ram=3300, commit=9000),
                                  source_file="/configs/run.ini")
        self.assertIsNone(decision.threads)
        self.assertEqual(decision.command_args(), [])
        self.assertTrue(decision.summary.startswith("REFUSED"))
        for text in ("1100 MiB per thread", "budget is 1252 MiB",
                     "Windows available RAM 3300 - 2048", "Linux MemAvailable 7300",
                     "Windows commit headroom 9000 - 2048", "num_threads = 1",
                     "--num_threads 1", "/configs/run.ini"):
            self.assertIn(text, decision.refusal)

    def test_a_negative_budget_also_refuses(self):
        self.assertIsNotNone(ct.choose_auto(cost_18mp(), host(linux=7300, ram=1000)).refusal)

    def test_no_memory_figure_keeps_colmaps_default_with_a_warning(self):
        decision = ct.choose_auto(cost_18mp(), host())
        self.assertIsNone(decision.threads)
        self.assertIsNone(decision.refusal)
        self.assertEqual(decision.command_args(), [])
        self.assertTrue(decision.warnings)

    def test_host_warnings_are_passed_on(self):
        decision = ct.choose_auto(cost_18mp(), host(linux=7300, warnings=["probe failed"]))
        self.assertEqual(decision.warnings, ["probe failed"])


# --------------------------------------------------------------------------
# the key, extra_args and the conflict rule
# --------------------------------------------------------------------------

class TestDecideThreads(unittest.TestCase):
    def reader(self, figures):
        calls = []

        def read(memory_high):
            calls.append(memory_high)
            return figures
        return read, calls

    def test_extra_args_wins_over_every_setting_and_reads_nothing(self):
        read, calls = self.reader(host(linux=1))
        for setting in ("auto", "all", "8"):
            with self.subTest(setting=setting):
                decision = ct.decide_threads(setting, "--gpu_index 0 --num_threads 4", [],
                                             host_reader=read)
                self.assertEqual((decision.source, decision.recorded), ("extra_args", "4"))
                self.assertEqual(decision.command_args(), [])
                self.assertIn(f"overrides [colmap] num_threads = {setting}", decision.summary)
                self.assertIsNone(decision.refusal)
        self.assertEqual(calls, [])

    def test_all_adds_nothing(self):
        read, calls = self.reader(host(linux=1))
        decision = ct.decide_threads("all", "", [], host_reader=read)
        self.assertEqual((decision.threads, decision.source, decision.recorded),
                         (None, "config", "all"))
        self.assertEqual(decision.command_args(), [])
        self.assertEqual(calls, [])

    def test_a_number_is_used_as_given(self):
        read, calls = self.reader(host(linux=1))
        decision = ct.decide_threads("12", "", [], host_reader=read)
        self.assertEqual((decision.command_args(), decision.source), (["--num_threads", "12"], "config"))
        self.assertEqual(calls, [])

    def test_auto_reads_the_machine_with_memory_high(self):
        read, calls = self.reader(host(linux=7300, ram=7168))
        decision = ct.decide_threads("auto", "", [Path("a")], memory_high="6G", host_reader=read)
        self.assertEqual(calls, ["6G"])
        self.assertEqual(decision.source, "auto")

    def test_auto_with_another_quality_uses_uncapped_pixels(self):
        figures = host(linux=60000, cpus=64)
        with mock.patch.object(ct, "image_size", return_value=(5184, 3456)):
            high = ct.decide_threads("auto", "--quality high", [Path("a")],
                                     host_reader=lambda _: figures)
            extreme = ct.decide_threads("auto", "--quality extreme", [Path("a")],
                                        host_reader=lambda _: figures)
        self.assertEqual(high.threads, 54)
        self.assertEqual(extreme.threads, 11)
        self.assertIn("uncapped", extreme.summary)


# --------------------------------------------------------------------------
# cgroup room, on a fake /proc and /sys/fs/cgroup
# --------------------------------------------------------------------------

class TestCgroupRoom(TempCase):
    def tree(self, own: str, groups: dict):
        proc = self.root / "proc" / "self"
        proc.mkdir(parents=True)
        (proc / "cgroup").write_text(own)
        cgroup = self.root / "cgroup"
        for path, files in groups.items():
            directory = cgroup / path
            directory.mkdir(parents=True, exist_ok=True)
            for name, text in files.items():
                (directory / name).write_text(text)
        return str(self.root / "proc"), str(cgroup)

    def test_memory_high_is_the_room(self):
        self.assertEqual(ct.cgroup_room("6G", "/nonexistent", "/nonexistent"),
                         (6144, "[host] memory_high 6G"))
        self.assertEqual(ct.cgroup_room("6000M")[0], 6000)

    def test_emulated_vm_scope_limit_minus_anon(self):
        proc, cgroup = self.tree("0::/user.slice/user-1000.slice/run-x.scope\n", {
            "user.slice": {"memory.max": "max\n", "memory.high": "max\n"},
            "user.slice/user-1000.slice": {"memory.max": "max\n"},
            "user.slice/user-1000.slice/run-x.scope": {
                "memory.max": f"{8 * GIB * 1048576}\n", "memory.high": "max\n",
                "memory.stat": f"anon {300 * 1048576}\nfile {5000 * 1048576}\n"},
        })
        room, label = ct.cgroup_room("", proc, cgroup)
        self.assertEqual(room, 8192 - 300)
        self.assertEqual(label, "cgroup /user.slice/user-1000.slice/run-x.scope limit 8192 - anon 300")

    def test_smallest_ancestor_wins_and_memory_high_counts(self):
        proc, cgroup = self.tree("0::/a/b\n", {
            "a": {"memory.high": f"{4 * GIB * 1048576}\n", "memory.max": "max\n",
                  "memory.stat": f"anon {1000 * 1048576}\n"},
            "a/b": {"memory.max": f"{8 * GIB * 1048576}\n", "memory.stat": "anon 0\n"},
        })
        self.assertEqual(ct.cgroup_room("", proc, cgroup), (4096 - 1000, "cgroup /a limit 4096 - anon 1000"))

    def test_no_limits_cgroup_v1_and_missing_files_give_nothing(self):
        proc, cgroup = self.tree("0::/user.slice/session.scope\n", {
            "user.slice": {"memory.max": "max\n"},
            "user.slice/session.scope": {},
        })
        self.assertEqual(ct.cgroup_room("", proc, cgroup), (None, ""))
        (self.root / "proc" / "self" / "cgroup").write_text("12:memory:/user.slice\n")
        self.assertEqual(ct.cgroup_room("", proc, cgroup), (None, ""))
        self.assertEqual(ct.cgroup_room("", str(self.root / "none"), cgroup), (None, ""))

    def test_missing_memory_stat_counts_no_anon(self):
        proc, cgroup = self.tree("0::/s\n", {"s": {"memory.max": f"{2048 * 1048576}\n"}})
        self.assertEqual(ct.cgroup_room("", proc, cgroup)[0], 2048)


class TestReadHostMemory(TempCase):
    def setUp(self):
        super().setUp()
        proc = self.root / "proc" / "self"
        proc.mkdir(parents=True)
        (proc / "cgroup").write_text("0::/\n")
        (self.root / "proc" / "meminfo").write_text(
            "MemTotal:       8000000 kB\nMemFree:  100 kB\nMemAvailable:    7486464 kB\n")
        self.proc = str(self.root / "proc")
        self.cgroup = str(self.root / "cgroup")

    def read(self, **kwargs):
        defaults = dict(proc_root=self.proc, cgroup_root=self.cgroup, cpus=lambda: 8,
                        wsl=lambda: True, powershell=lambda: "/ps.exe",
                        sample=lambda ps: {"ram_mib": 4409, "commit_mib": 12060})
        defaults.update(kwargs)
        return ct.read_host_memory("", **defaults)

    def test_wsl_reads_windows(self):
        figures = self.read()
        self.assertEqual((figures.linux_available_mib, figures.windows_ram_mib,
                          figures.windows_commit_mib, figures.usable_cpus, figures.warnings),
                         (7311, 4409, 12060, 8, []))

    def test_not_wsl_skips_the_probe(self):
        def never(_):
            raise AssertionError("probed outside WSL")
        figures = self.read(wsl=lambda: False, sample=never)
        self.assertIsNone(figures.windows_ram_mib)
        self.assertEqual(figures.warnings, [])

    def test_a_failed_probe_warns_and_uses_linux(self):
        def fail(_):
            raise windows_host.WindowsProbeError("probe exited 1")
        figures = self.read(sample=fail)
        self.assertIsNone(figures.windows_ram_mib)
        self.assertIn("probe exited 1", figures.warnings[0])
        self.assertEqual(figures.linux_available_mib, 7311)
        missing = self.read(powershell=lambda: None)
        self.assertIn("powershell.exe is not reachable", missing.warnings[0])

    def test_unreadable_meminfo(self):
        (self.root / "proc" / "meminfo").write_text("MemTotal: 1 kB\n")
        self.assertIsNone(self.read(wsl=lambda: False).linux_available_mib)


# --------------------------------------------------------------------------
# the shared Windows probe
# --------------------------------------------------------------------------

class TestWindowsProbe(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(windows_host.parse_probe('﻿noise\r\n{"ram_mib":4409,"commit_mib":12060}\r\n', None),
                         {"ram_mib": 4409, "commit_mib": 12060})
        with self.assertRaises(windows_host.WindowsProbeError):
            windows_host.parse_probe("Access denied", None)
        with self.assertRaises(windows_host.WindowsProbeError):
            windows_host.parse_probe('{"commit_mib":1}', None)

    def test_sample_runs_powershell_once_and_reports_failures(self):
        done = mock.Mock(returncode=0, stdout='{"ram_mib":5000,"commit_mib":9000}', stderr="")
        with mock.patch.object(windows_host.subprocess, "run", return_value=done) as run:
            self.assertEqual(windows_host.sample_windows("/ps.exe"),
                             {"ram_mib": 5000, "commit_mib": 9000})
        self.assertEqual(run.call_args.args[0][:3], ["/ps.exe", "-NoProfile", "-NonInteractive"])
        failed = mock.Mock(returncode=1, stdout="", stderr="boom")
        with mock.patch.object(windows_host.subprocess, "run", return_value=failed):
            with self.assertRaises(windows_host.WindowsProbeError):
                windows_host.sample_windows("/ps.exe")
        with mock.patch.object(windows_host.subprocess, "run", side_effect=OSError("gone")):
            with self.assertRaises(windows_host.WindowsProbeError):
                windows_host.sample_windows("/ps.exe")


# --------------------------------------------------------------------------
# the config key
# --------------------------------------------------------------------------

CONFIG = """\
[paths]
data_root = {data_root}

[dataset]
images_subpath = scene-01/images

[run]
date = 260101
run_count = 01
colmap_version = 3120

[colmap]
{colmap}

[nerfstudio]

[train]
quit_on_train_completion = true

[export]
"""


def write_config(root: Path, colmap: str = "", photographs=None) -> Path:
    images = root / "data" / "datasets" / "scene-01" / "images"
    images.mkdir(parents=True, exist_ok=True)
    for name, data in (photographs or {"a.jpg": jpeg_bytes(5184, 3456)}).items():
        (images / name).write_bytes(data)
    path = root / "run.ini"
    path.write_text(CONFIG.format(data_root=root / "data", colmap=colmap), encoding="utf-8")
    return path


class TestConfigKey(TempCase):
    def test_default_is_auto(self):
        self.assertEqual(load_config(write_config(self.root)).colmap_num_threads, "auto")

    def test_accepted_values_are_normalized(self):
        for raw, expected in (("AUTO", "auto"), (" all ", "all"), ("4", "4"), ("08", "8")):
            with self.subTest(raw=raw):
                cfg = load_config(write_config(self.root, f"num_threads = {raw}"))
                self.assertEqual(cfg.colmap_num_threads, expected)

    def test_bad_values_are_refused_naming_the_key(self):
        for raw in ("0", "-2", "four", "1.5", "auto all"):
            with self.subTest(raw=raw):
                with self.assertRaises(ConfigError) as caught:
                    load_config(write_config(self.root, f"num_threads = {raw}"))
                self.assertIn("num_threads", str(caught.exception))
                self.assertIn("run.ini", str(caught.exception))

    def test_empty_means_the_default(self):
        self.assertEqual(load_config(write_config(self.root, "num_threads =")).colmap_num_threads,
                         "auto")

    def test_describe_shows_it(self):
        self.assertIn("colmap num_threads : auto", load_config(write_config(self.root)).describe())


# --------------------------------------------------------------------------
# the runner: dry run, refusal, run record
# --------------------------------------------------------------------------

class TestRunner(TempCase):
    def test_unmeasured_version_refuses_auto_even_when_run_id_is_pinned(self):
        cfg = load_config(write_config(self.root, "executable = /opt/other/bin/colmap"))
        with mock.patch.object(run_pipeline, "detect_colmap_version_parts", return_value=(4, 2, 1)) as detect:
            decision = run_pipeline.decide_colmap_threads(cfg, [])
        self.assertIsNotNone(decision.refusal)
        detect.assert_called_once_with("/opt/other/bin/colmap")
        self.assertEqual(run_pipeline.colmap_command(cfg, self.root)[0], "/opt/other/bin/colmap")
        self.assertIn("/opt/other/bin/colmap", run_pipeline.process_command(cfg, self.root))

    def test_auto_gate_distinguishes_compact_collisions_and_unknown_patch(self):
        cfg = load_config(write_config(self.root))
        for version in ((3, 1, 20), (3, 12, None), (4, 2, 1), None):
            with self.subTest(version=version), mock.patch.object(
                    run_pipeline, "detect_colmap_version_parts", return_value=version):
                self.assertIsNotNone(run_pipeline.decide_colmap_threads(cfg, []).refusal)

    def test_fake_executable_banners_are_unambiguous(self):
        for banner, expected in (("3.1.20", (3, 1, 20)), ("3.12.0", (3, 12, 0))):
            fake = self.root / ("colmap-" + banner)
            fake.write_text("#!/bin/sh\nprintf 'COLMAP " + banner + "\n'\n")
            fake.chmod(0o755)
            self.assertEqual(run_pipeline.detect_colmap_version_parts(str(fake)), expected)

    def test_explicit_threads_allow_unmeasured_binary(self):
        cfg = load_config(write_config(self.root, "num_threads = 2"))
        with mock.patch.object(run_pipeline, "detect_colmap_version_parts", return_value=(4, 2, 1)):
            self.assertEqual(run_pipeline.decide_colmap_threads(cfg, []).threads, 2)

    def run_pipeline(self, cfg, *argv, figures=None, version=(3, 12, 0)):
        args = run_pipeline.parse_args(["--config", str(cfg.source), *argv])
        out = io.StringIO()
        figures = figures or host(linux=7300, ram=7168, commit=10240)
        with mock.patch.object(run_pipeline, "detect_colmap_version_parts", return_value=version), \
                mock.patch.object(ct, "read_host_memory", return_value=figures) as read, \
                redirect_stdout(out):
            try:
                code = run_pipeline.run_pipeline(cfg, args)
            except run_pipeline.StageError as exc:
                code = exc
        return code, out.getvalue(), read

    def test_dry_run_shows_the_line_and_the_flag(self):
        cfg = load_config(write_config(self.root))
        code, out, read = self.run_pipeline(cfg, "--dry-run", "--stage", "colmap")
        self.assertEqual(code, 0)
        self.assertIn("colmap threads    : 4 (auto: budget 5120 MiB, bound by Windows "
                      "available RAM 7168 - 2048; 1100 MiB/thread for 5184x3456 capped to "
                      "2400x1600; 24 CPUs)", out)
        [command] = [line for line in out.splitlines() if line.strip().startswith("$ colmap")]
        self.assertIn("--dense 0 --num_threads 4", command)
        read.assert_called_once_with("")

    def test_dry_run_reports_a_refusal_without_failing(self):
        cfg = load_config(write_config(self.root))
        code, out, _ = self.run_pipeline(cfg, "--dry-run", "--stage", "colmap",
                                         figures=host(linux=7300, ram=2500))
        self.assertEqual(code, 0)
        self.assertIn("colmap threads    : REFUSED", out)
        self.assertIn("a real run would refuse to start now", out)

    def test_stages_without_colmap_decide_nothing(self):
        cfg = load_config(write_config(self.root))
        code, out, read = self.run_pipeline(cfg, "--dry-run", "--stage", "train,export")
        self.assertEqual(code, 0)
        read.assert_not_called()
        self.assertNotIn("colmap threads", out)

    def test_extra_args_override_is_logged(self):
        cfg = load_config(write_config(self.root, "extra_args = --num_threads 2"))
        code, out, read = self.run_pipeline(cfg, "--dry-run", "--stage", "colmap")
        read.assert_not_called()
        self.assertIn("colmap threads    : 2 (extra_args --num_threads; overrides "
                      "[colmap] num_threads = auto)", out)
        [command] = [line for line in out.splitlines() if line.strip().startswith("$ colmap")]
        self.assertEqual(command.count("--num_threads"), 1)

    def _real(self, cfg, figures, version=(3, 12, 0)):
        """A real (not dry) colmap stage whose external command is a fake."""
        commands = []

        def fake_run(command, label, dry_run):
            commands.append(list(command))
            sparse = cfg.colmap_workspace("run_260101-01-3120") / "sparse" / "0"
            sparse.mkdir(parents=True, exist_ok=True)
            (sparse / "cameras.bin").write_bytes(b"x")

        with mock.patch.object(run_pipeline, "check_prerequisites"), \
                mock.patch.object(run_pipeline, "run_command", side_effect=fake_run), \
                mock.patch.object(run_pipeline, "detect_nerfstudio_version", return_value="x"):
            code, out, _ = self.run_pipeline(cfg, "--stage", "colmap", figures=figures, version=version)
        return code, out, commands

    def test_run_records_actual_version_separately_from_pinned_id(self):
        cfg = load_config(write_config(self.root, "num_threads = 2\nexecutable = /opt/candidate/bin/colmap"))
        with mock.patch.object(run_pipeline.shutil, "which", return_value="/opt/candidate/bin/colmap"):
            code, out, commands = self._real(cfg, host(linux=7300), version=(3, 1, 20))
        self.assertEqual(code, 0, out)
        self.assertEqual(commands[0][0], "/opt/candidate/bin/colmap")
        [archive] = list(cfg.config_archive_dir.iterdir())
        record = configparser.ConfigParser(interpolation=None)
        record.read(archive, encoding="utf-8")
        self.assertEqual(record["run-record"]["colmap_version"], "3120")
        self.assertEqual(record["run-record"]["colmap_version_actual"], "3.1.20")
        self.assertEqual(record["run-record"]["colmap_executable_actual"], "/opt/candidate/bin/colmap")

    def test_real_run_refuses_below_one_thread_and_records_nothing(self):
        cfg = load_config(write_config(self.root))
        code, out, commands = self._real(cfg, host(linux=7300, ram=3000))
        self.assertIsInstance(code, run_pipeline.StageError)
        self.assertIn("not enough memory for one COLMAP feature-extraction thread", str(code))
        self.assertEqual(commands, [])
        self.assertFalse(cfg.run_log.exists())
        self.assertFalse(cfg.config_archive_dir.exists())

    def test_real_run_passes_the_count_and_records_it_next_to_memory_high(self):
        cfg = load_config(write_config(self.root))
        code, out, commands = self._real(cfg, host(linux=7300, ram=6000))
        self.assertEqual(code, 0, out)
        self.assertIn("--num_threads", commands[0])
        self.assertEqual(commands[0][commands[0].index("--num_threads") + 1], "3")
        [archive] = list(cfg.config_archive_dir.iterdir())
        record = configparser.ConfigParser(interpolation=None)
        record.read(archive, encoding="utf-8")
        self.assertEqual(record["run-record"]["colmap_threads"], "3")
        self.assertEqual(record["run-record"]["colmap_threads_source"], "auto")

    def test_real_run_with_all_records_the_config_source(self):
        cfg = load_config(write_config(self.root, "num_threads = all"))
        code, out, commands = self._real(cfg, host(linux=1))
        self.assertEqual(code, 0, out)
        self.assertNotIn("--num_threads", commands[0])
        [archive] = list(cfg.config_archive_dir.iterdir())
        record = configparser.ConfigParser(interpolation=None)
        record.read(archive, encoding="utf-8")
        self.assertEqual((record["run-record"]["colmap_threads"],
                          record["run-record"]["colmap_threads_source"]), ("all", "config"))


if __name__ == "__main__":
    unittest.main()
