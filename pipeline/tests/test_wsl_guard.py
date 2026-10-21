"""Tests for ``pipeline/scripts/wsl-guard/wsl-guard.py``.

Stdlib ``unittest`` only, matching the rest of the suite, and hermetic: no
WSL, no Windows, no powershell.exe and no sudo. The Windows side is a fake
``Host`` whose samples a test scripts, and ``subprocess.run`` is never reached
for a probe or a cache release. The two end-to-end cases start a real child,
but that child is a few lines of Python standing in for ``run_pipeline.py``,
so what is exercised is the guard's own process-group handling.

What only a real WSL2 run can show (that the probe reads what Task Manager
shows, that a release returns memory to Windows, that the thresholds are the
right ones) is out of scope here.
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import os
import signal
import sys
import tempfile
import textwrap
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

PIPELINE_DIR = Path(__file__).resolve().parent.parent

_SPEC = importlib.util.spec_from_file_location(
    "wsl_guard", PIPELINE_DIR / "scripts" / "wsl-guard" / "wsl-guard.py")
wg = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(wg)


def args_for(*argv):
    return wg.parse_args(list(argv))


class FakeHost:
    """Scripted Windows samples; counts cache releases instead of making them."""

    def __init__(self, ram, drive_gib=None, release_ok=True):
        self._ram = list(ram)
        self.drive_gib = drive_gib
        self.release_ok = release_ok
        self.releases = 0

    def sample(self):
        ram = self._ram.pop(0) if len(self._ram) > 1 else self._ram[0]
        if isinstance(ram, Exception):
            raise ram
        sample = {"ram_mib": ram}
        if self.drive_gib is not None:
            sample["drive_gib"] = self.drive_gib
        return sample

    def release_cache(self):
        self.releases += 1
        return self.release_ok


class TestArguments(unittest.TestCase):

    def test_everything_after_the_separator_passes_through(self):
        own, rest = wg.split_argv(["--stop-mib", "900", "--", "--config", "x.ini", "--", "y"])
        self.assertEqual(own, ["--stop-mib", "900"])
        self.assertEqual(rest, ["--config", "x.ini", "--", "y"])

    def test_no_separator_means_no_runner_arguments(self):
        self.assertEqual(wg.split_argv(["--quiet"]), (["--quiet"], []))

    def test_defaults(self):
        args = args_for("--", "--config", "x.ini", "--stage", "colmap,process")
        self.assertEqual((args.start_mib, args.relief_mib, args.stop_mib), (4096, 3072, 1024))
        self.assertEqual(args.interval, 5.0)
        self.assertEqual(args.settle, 90.0)
        self.assertEqual(args.relief_every, 60.0)
        self.assertEqual(args.kill_grace, 30.0)
        self.assertIsNone(args.drive)
        self.assertEqual(args.passthrough, ["--config", "x.ini", "--stage", "colmap,process"])

    def test_child_command_runs_the_runner_unbuffered_with_the_passthrough(self):
        args = args_for("--python", "/py", "--runner", "/r.py", "--", "--dry-run")
        self.assertEqual(wg.child_command(args), ["/py", "-u", "/r.py", "--dry-run"])

    def test_drive_letter_is_upper_cased(self):
        self.assertEqual(args_for("--drive", "d").drive, "D")

    def test_inconsistent_thresholds_are_usage_errors(self):
        for argv in (["--stop-mib", "5000"],                    # stop above start
                     ["--relief-mib", "500"],                   # relief below stop
                     ["--drive", "DD"],
                     ["--drive", "D", "--stop-drive-gib", "20"],
                     ["--interval", "0"],
                     ["--start-mib", "-1"]):
            with self.subTest(argv=argv), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    wg.parse_args(argv)
                self.assertEqual(caught.exception.code, 2)


class TestProbe(unittest.TestCase):

    def test_script_reads_available_memory_and_only_the_named_drive(self):
        self.assertIn("FreePhysicalMemory", wg.probe_script(None))
        self.assertNotIn("Get-PSDrive", wg.probe_script(None))
        self.assertIn("(Get-PSDrive D).Free", wg.probe_script("D"))

    def test_parse_plain_output(self):
        self.assertEqual(wg.parse_probe('{"ram_mib":6512}\r\n', None), {"ram_mib": 6512})

    def test_parse_tolerates_a_bom_and_stray_lines(self):
        text = "﻿WARNING: something\r\n{\"ram_mib\":59,\"drive_gib\":120.4}\r\n"
        self.assertEqual(wg.parse_probe(text, "D"), {"ram_mib": 59, "drive_gib": 120.4})

    def test_parse_refuses_output_without_json(self):
        with self.assertRaises(wg.GuardError):
            wg.parse_probe("Get-CimInstance : Access denied", None)

    def test_parse_refuses_a_missing_drive_value(self):
        with self.assertRaises(wg.GuardError):
            wg.parse_probe('{"ram_mib":100}', "D")

    def test_powershell_found_on_path_or_at_the_usual_place(self):
        self.assertEqual(wg.find_powershell(lambda _: "/x/powershell.exe", lambda _: False),
                         "/x/powershell.exe")
        self.assertEqual(wg.find_powershell(lambda _: None, lambda _: True),
                         wg.POWERSHELL_FALLBACK)
        self.assertIsNone(wg.find_powershell(lambda _: None, lambda _: False))

    def test_wsl_detected_from_the_kernel_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "osrelease"
            path.write_text("5.15.167.4-microsoft-standard-WSL2\n")
            self.assertTrue(wg.is_wsl(str(path)))
            path.write_text("6.8.0-45-generic\n")
            self.assertFalse(wg.is_wsl(str(path)))
            self.assertFalse(wg.is_wsl(str(Path(tmp) / "missing")))

    def test_host_probe_and_release_go_through_subprocess(self):
        host = wg.Host("/ps.exe", "D")
        done = mock.Mock(returncode=0, stdout='{"ram_mib":4000,"drive_gib":50}', stderr="")
        with mock.patch.object(wg.subprocess, "run", return_value=done) as run:
            self.assertEqual(host.sample(), {"ram_mib": 4000, "drive_gib": 50.0})
            self.assertEqual(run.call_args.args[0][:2], ["/ps.exe", "-NoProfile"])
            self.assertTrue(host.release_cache())
            self.assertEqual(run.call_args.args[0][:2], ["sudo", "-n"])

    def test_release_without_passwordless_sudo_is_quietly_false(self):
        refused = mock.Mock(returncode=1, stdout="", stderr="sudo: a password is required")
        with mock.patch.object(wg.subprocess, "run", return_value=refused):
            self.assertFalse(wg.Host("/ps.exe", None).release_cache())
        with mock.patch.object(wg.subprocess, "run", side_effect=FileNotFoundError("sudo")):
            self.assertFalse(wg.Host("/ps.exe", None).release_cache())


class TestDecisions(unittest.TestCase):

    def test_start_needs_ram_and_disk(self):
        args = args_for("--drive", "D")
        self.assertEqual(wg.start_problems({"ram_mib": 5000, "drive_gib": 50}, args), [])
        self.assertEqual(len(wg.start_problems({"ram_mib": 4095, "drive_gib": 50}, args)), 1)
        self.assertEqual(len(wg.start_problems({"ram_mib": 5000, "drive_gib": 11.9}, args)), 1)
        self.assertEqual(len(wg.start_problems({"ram_mib": 10, "drive_gib": 1}, args)), 2)

    def test_disk_is_ignored_without_a_drive(self):
        self.assertEqual(wg.start_problems({"ram_mib": 5000}, args_for()), [])

    def test_stop_below_the_stop_line_and_stop_wins_over_release(self):
        args = args_for()
        self.assertEqual(wg.poll_action({"ram_mib": 1023}, args, 100.0, None), "stop")
        self.assertIsNone(wg.poll_action({"ram_mib": 3072}, args, 100.0, None))

    def test_release_below_the_relief_line_at_most_once_per_period(self):
        args = args_for()
        self.assertEqual(wg.poll_action({"ram_mib": 2000}, args, 100.0, None), "release")
        self.assertIsNone(wg.poll_action({"ram_mib": 2000}, args, 159.0, 100.0))
        self.assertEqual(wg.poll_action({"ram_mib": 2000}, args, 160.0, 100.0), "release")

    def test_no_cache_release_never_releases_but_still_stops(self):
        args = args_for("--no-cache-release")
        self.assertIsNone(wg.poll_action({"ram_mib": 2000}, args, 100.0, None))
        self.assertEqual(wg.poll_action({"ram_mib": 10}, args, 100.0, None), "stop")

    def test_low_disk_stops(self):
        args = args_for("--drive", "D")
        self.assertEqual(wg.poll_action({"ram_mib": 9000, "drive_gib": 3.9}, args, 0, None),
                         "stop")
        self.assertIsNone(wg.poll_action({"ram_mib": 9000, "drive_gib": 4.0}, args, 0, None))

    def test_exit_code_for_a_signalled_child(self):
        self.assertEqual(wg.exit_code_for(0), 0)
        self.assertEqual(wg.exit_code_for(1), 1)
        self.assertEqual(wg.exit_code_for(-signal.SIGTERM), 128 + signal.SIGTERM)

    def test_cache_hint_only_when_the_cache_is_large(self):
        with tempfile.TemporaryDirectory() as tmp:
            meminfo = Path(tmp) / "meminfo"
            meminfo.write_text("MemTotal: 22000000 kB\nCached: 10066329 kB\n")
            self.assertIn("9.6 GiB", wg.cache_hint(str(meminfo)))
            meminfo.write_text("Cached: 1048576 kB\n")
            self.assertEqual(wg.cache_hint(str(meminfo)), "")
        self.assertEqual(wg.cache_hint("/nonexistent/meminfo"), "")

    def test_records_sit_beside_the_named_log(self):
        log, status, samples = wg.log_paths(args_for("--log", "/l/run-a.log"), "x")
        self.assertEqual((str(log), str(status), str(samples)),
                         ("/l/run-a.log", "/l/run-a-status.json", "/l/run-a-host.jsonl"))

    def test_default_log_goes_to_the_state_directory(self):
        with mock.patch.dict(os.environ, {"XDG_STATE_HOME": "/state"}):
            log, _, _ = wg.log_paths(args_for(), "20261020-120000")
        self.assertEqual(str(log), "/state/jsps-dt4ag/wsl-guard/wsl-guard-20261020-120000.log")
        log, _, _ = wg.log_paths(args_for("--log-dir", "/d"), "s")
        self.assertEqual(str(log), "/d/wsl-guard-s.log")


class TestWaitForHeadroom(unittest.TestCase):

    def _clock(self):
        now = [0.0]

        def clock():
            return now[0]

        def sleep(seconds):
            now[0] += seconds
        return clock, sleep, now

    def test_enough_ram_releases_nothing(self):
        host = FakeHost([9000])
        sample, released = wg.wait_for_headroom(host, args_for(), {"ram_mib": 9000})
        self.assertEqual((sample["ram_mib"], released, host.releases), (9000, False, 0))

    def test_release_then_poll_until_windows_sees_it(self):
        host = FakeHost([3000, 3500, 6000])
        clock, sleep, now = self._clock()
        with redirect_stderr(io.StringIO()):
            sample, released = wg.wait_for_headroom(host, args_for(), {"ram_mib": 2000},
                                                    clock, sleep)
        self.assertTrue(released)
        self.assertEqual(sample["ram_mib"], 6000)
        self.assertEqual(now[0], 15.0)        # three 5 s polls, not the full 90 s

    def test_gives_up_after_the_settle_time(self):
        host = FakeHost([2000])
        clock, sleep, now = self._clock()
        with redirect_stderr(io.StringIO()):
            sample, released = wg.wait_for_headroom(host, args_for(), {"ram_mib": 2000},
                                                    clock, sleep)
        self.assertTrue(released)
        self.assertEqual(sample["ram_mib"], 2000)
        self.assertEqual(now[0], 90.0)

    def test_without_sudo_it_does_not_wait(self):
        host = FakeHost([2000], release_ok=False)
        clock, sleep, now = self._clock()
        sample, released = wg.wait_for_headroom(host, args_for(), {"ram_mib": 2000},
                                                clock, sleep)
        self.assertFalse(released)
        self.assertEqual(now[0], 0.0)

    def test_no_cache_release_skips_the_attempt(self):
        host = FakeHost([2000])
        wg.wait_for_headroom(host, args_for("--no-cache-release"), {"ram_mib": 2000})
        self.assertEqual(host.releases, 0)


class TestPreflight(unittest.TestCase):

    def test_not_wsl_is_refused_with_a_reason(self):
        host, reason = wg.preflight(args_for(), wsl=lambda: False, powershell=lambda: "/ps")
        self.assertIsNone(host)
        self.assertIn("not WSL2", reason)

    def test_no_powershell_is_refused_with_a_reason(self):
        host, reason = wg.preflight(args_for(), wsl=lambda: True, powershell=lambda: None)
        self.assertIsNone(host)
        self.assertIn("powershell.exe", reason)

    def test_main_exits_76_off_wsl(self):
        with mock.patch.object(wg, "is_wsl", return_value=False), \
                redirect_stderr(io.StringIO()) as err:
            # preflight's defaults were bound at definition time, so patch it.
            with mock.patch.object(wg, "preflight",
                                   side_effect=lambda a: (None, "this is not WSL2")):
                self.assertEqual(wg.main(["--", "--dry-run"]), wg.EXIT_REFUSED)
        self.assertIn("REFUSED", err.getvalue())


@unittest.skipUnless(hasattr(os, "killpg"), "needs POSIX process groups")
class TestRun(unittest.TestCase):
    """End to end, with a stand-in for run_pipeline.py and a fake Windows."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def _runner(self, body):
        path = self.tmp / "fake_runner.py"
        path.write_text(textwrap.dedent(body))
        return path

    def _args(self, runner, *extra):
        return args_for("--runner", str(runner), "--log", str(self.tmp / "run.log"),
                        "--interval", "0.05", "--quiet", *extra, "--", "--config", "x.ini")

    def test_refusal_starts_nothing_and_exits_76(self):
        runner = self._runner("raise SystemExit(0)\n")
        host = FakeHost([2000], release_ok=False)
        with redirect_stderr(io.StringIO()) as err:
            code = wg.run(self._args(runner), host)
        self.assertEqual(code, wg.EXIT_REFUSED)
        self.assertIn("2000 MiB", err.getvalue())
        self.assertFalse((self.tmp / "run.log").exists())

    def test_a_clean_run_passes_the_exit_code_and_records_it(self):
        runner = self._runner("""
            import sys, time
            print("args", sys.argv[1:])
            time.sleep(0.3)
            raise SystemExit(3)
        """)
        host = FakeHost([9000])
        with redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
            code = wg.run(self._args(runner), host)
        self.assertEqual(code, 3)
        self.assertIn("args ['--config', 'x.ini']", (self.tmp / "run.log").read_text())
        status = json.loads((self.tmp / "run-status.json").read_text())
        self.assertEqual((status["returncode"], status["exit_code"]), (3, 3))
        self.assertIsNone(status["guard_stop"])
        self.assertTrue((self.tmp / "run-host.jsonl").read_text().strip())

    def test_low_ram_stops_its_own_group_and_exits_75(self):
        runner = self._runner("""
            import signal, sys, time
            def bye(*_):
                print("got SIGTERM", flush=True)
                sys.exit(1)
            signal.signal(signal.SIGTERM, bye)
            print("running", flush=True)
            time.sleep(60)
        """)
        host = FakeHost([9000, 9000, 500])
        with redirect_stderr(io.StringIO()):
            code = wg.run(self._args(runner, "--kill-grace", "5"), host)
        self.assertEqual(code, wg.EXIT_GUARD_STOP)
        status = json.loads((self.tmp / "run-status.json").read_text())
        self.assertEqual(status["guard_stop"]["ram_mib"], 500)
        self.assertEqual(status["exit_code"], wg.EXIT_GUARD_STOP)
        self.assertIn("got SIGTERM", (self.tmp / "run.log").read_text())

    def test_relief_releases_the_cache_during_the_run(self):
        runner = self._runner("import time\ntime.sleep(0.4)\n")
        host = FakeHost([9000, 2500])
        with redirect_stderr(io.StringIO()):
            code = wg.run(self._args(runner, "--relief-every", "60"), host)
        self.assertEqual(code, 0)
        self.assertEqual(host.releases, 1)      # once, however many low samples
        status = json.loads((self.tmp / "run-status.json").read_text())
        self.assertEqual(status["cache_releases"], 1)

    def test_scope_figures_reach_the_status_only_for_a_dedicated_cgroup(self):
        runner = self._runner("import time\ntime.sleep(0.4)\n")
        reading = {"cgroup": "/run-x.scope", "scope_peak_mib": 900, "scope_anon_mib": 300,
                   "scope_file_mib": 600, "scope_swap_mib": 5, "scope_oom_kills": 0}
        for own, dedicated in (("/shared.scope", True), ("/run-x.scope", False)):
            with self.subTest(own=own):
                log = self.tmp / f"run-{dedicated}.log"
                args = args_for("--runner", str(runner), "--log", str(log),
                                "--interval", "0.05", "--quiet", "--", "--config", "x.ini")
                with mock.patch.object(wg, "scope_memory", return_value=dict(reading)), \
                        mock.patch.object(wg, "cgroup_path", return_value=own), \
                        redirect_stderr(io.StringIO()):
                    code = wg.run(args, FakeHost([9000]))
                self.assertEqual(code, 0)
                status = json.loads((self.tmp / f"run-{dedicated}-status.json").read_text())
                self.assertEqual(status["scope_dedicated"], dedicated)
                lines = [json.loads(x) for x in
                         (self.tmp / f"run-{dedicated}-host.jsonl").read_text().splitlines()]
                self.assertTrue(all(line["scope_peak_mib"] == 900 for line in lines))
                if dedicated:
                    self.assertEqual((status["scope_peak_mib"], status["scope_anon_max_mib"],
                                      status["scope_swap_max_mib"], status["scope_oom_kills"]),
                                     (900, 300, 5, 0))
                    self.assertNotIn("scope_shared", lines[0])
                else:
                    self.assertNotIn("scope_peak_mib", status)
                    self.assertTrue(lines[0]["scope_shared"])

    def test_a_run_without_cgroup_files_still_works(self):
        runner = self._runner("import time\ntime.sleep(0.3)\n")
        with mock.patch.object(wg, "scope_memory", return_value={}), \
                redirect_stderr(io.StringIO()):
            code = wg.run(self._args(runner), FakeHost([9000]))
        self.assertEqual(code, 0)
        status = json.loads((self.tmp / "run-status.json").read_text())
        self.assertNotIn("scope_peak_mib", status)
        self.assertFalse(status["scope_dedicated"])
        self.assertFalse(status["killed_by_sigkill_unprompted"])

    def test_a_sigkill_the_guard_did_not_send_is_flagged(self):
        runner = self._runner("import os, signal, time\ntime.sleep(0.2)\n"
                              "os.kill(os.getpid(), signal.SIGKILL)\n")
        with redirect_stderr(io.StringIO()):
            code = wg.run(self._args(runner), FakeHost([9000]))
        self.assertEqual(code, 128 + signal.SIGKILL)
        status = json.loads((self.tmp / "run-status.json").read_text())
        self.assertTrue(status["killed_by_sigkill_unprompted"])

    def test_an_existing_log_is_never_overwritten(self):
        runner = self._runner("raise SystemExit(0)\n")
        (self.tmp / "run.log").write_text("earlier run\n")
        with redirect_stderr(io.StringIO()):
            code = wg.run(self._args(runner), FakeHost([9000]))
        self.assertEqual(code, wg.EXIT_REFUSED)
        self.assertEqual((self.tmp / "run.log").read_text(), "earlier run\n")


class FakeCgroup:
    """A fake /proc and /sys/fs/cgroup tree under a temporary directory."""

    def __init__(self, root: Path):
        self.proc = root / "proc"
        self.cg = root / "cgroup"
        self.proc.mkdir()
        self.cg.mkdir()

    def process(self, pid, path):
        (self.proc / str(pid)).mkdir(exist_ok=True)
        (self.proc / str(pid) / "cgroup").write_text(f"0::{path}\n")
        directory = self.cg / path.lstrip("/")
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    @staticmethod
    def fill(directory, current=100, peak=300, anon=40, file=60, swap=7, oom=0,
             skip=()):
        mib = 1048576
        files = {
            "memory.current": f"{current * mib}\n",
            "memory.peak": f"{peak * mib}\n",
            "memory.stat": f"anon {anon * mib}\nfile {file * mib}\nkernel 5\n",
            "memory.swap.current": f"{swap * mib}\n",
            "memory.events": f"low 0\nhigh 0\nmax 3\noom 1\noom_kill {oom}\n",
        }
        for name, text in files.items():
            if name not in skip:
                (directory / name).write_text(text)


class TestScopeMemory(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.fake = FakeCgroup(Path(self._tmp.name))

    def read(self, pid=10):
        return wg.scope_memory(pid, str(self.fake.proc), str(self.fake.cg))

    def test_reads_every_field_in_mib(self):
        directory = self.fake.process(10, "/user.slice/run-r1.scope")
        self.fake.fill(directory, current=100, peak=300, anon=40, file=60, swap=7, oom=2)
        self.assertEqual(self.read(), {
            "cgroup": "/user.slice/run-r1.scope", "scope_cur_mib": 100,
            "scope_peak_mib": 300, "scope_anon_mib": 40, "scope_file_mib": 60,
            "scope_swap_mib": 7, "scope_oom_kills": 2})

    def test_a_missing_file_drops_only_its_own_field(self):
        directory = self.fake.process(10, "/a.scope")
        self.fake.fill(directory, skip=("memory.peak", "memory.swap.current"))
        sample = self.read()
        self.assertNotIn("scope_peak_mib", sample)
        self.assertNotIn("scope_swap_mib", sample)
        self.assertEqual(sample["scope_anon_mib"], 40)
        self.assertEqual(sample["scope_oom_kills"], 0)

    def test_everything_missing_gives_just_the_path(self):
        self.fake.process(10, "/a.scope")
        self.assertEqual(self.read(), {"cgroup": "/a.scope"})

    def test_garbled_values_are_skipped(self):
        directory = self.fake.process(10, "/a.scope")
        self.fake.fill(directory)
        (directory / "memory.current").write_text("max\n")
        (directory / "memory.events").write_text("oom_kill many\n")
        sample = self.read()
        self.assertNotIn("scope_cur_mib", sample)
        self.assertNotIn("scope_oom_kills", sample)
        self.assertEqual(sample["scope_peak_mib"], 300)

    def test_no_such_process_or_no_cgroup_v2_gives_nothing(self):
        self.assertEqual(self.read(99), {})
        (self.fake.proc / "11").mkdir()
        (self.fake.proc / "11" / "cgroup").write_text("12:memory:/old\n")   # cgroup v1
        self.assertEqual(self.read(11), {})
        self.fake.process(12, "/gone.scope")
        (self.fake.cg / "gone.scope").rmdir()
        self.assertEqual(self.read(12), {})

    def test_the_root_cgroup_path(self):
        (self.fake.proc / "13").mkdir()
        (self.fake.proc / "13" / "cgroup").write_text("0::/\n")
        self.assertEqual(wg.cgroup_path(13, str(self.fake.proc)), "/")
        self.assertEqual(self.read(13), {"cgroup": "/"})

    def test_maxima_keep_the_largest_value_and_the_oom_count_only_grows(self):
        state = {}
        wg.fold_scope_maxima(state, {"cgroup": "/a.scope", "scope_peak_mib": 300,
                                     "scope_anon_mib": 40, "scope_swap_mib": 7,
                                     "scope_file_mib": 60, "scope_oom_kills": 0})
        wg.fold_scope_maxima(state, {"scope_peak_mib": 250, "scope_anon_mib": 90,
                                     "scope_swap_mib": 3, "scope_oom_kills": 1})
        self.assertEqual(state, {
            "scope_peak_mib": 300, "scope_anon_max_mib": 90, "scope_swap_max_mib": 7,
            "scope_file_max_mib": 60, "scope_oom_kills": 1, "scope_cgroup": "/a.scope"})

    def test_a_shared_cgroup_is_not_dedicated(self):
        directory = self.fake.process(10, "/session.scope")
        self.fake.fill(directory)
        for own, expected in (("/session.scope", False), ("/other.scope", True)):
            with self.subTest(own=own):
                sample, dedicated = wg.scope_sample(
                    10, own, str(self.fake.proc), str(self.fake.cg))
                self.assertEqual((bool(sample), dedicated), (True, expected))
        self.assertEqual(wg.scope_sample(99, "/x", str(self.fake.proc), str(self.fake.cg)),
                         ({}, False))


class TestEmulateVm(unittest.TestCase):

    def test_parse_memory_and_swap(self):
        self.assertEqual(wg.parse_emulate("8G:4G"), ("8G", "4G"))
        self.assertEqual(wg.parse_emulate("6000M:512M"), ("6000M", "512M"))
        self.assertEqual(wg.parse_emulate("8G"), ("8G", "0M"))
        self.assertEqual(wg.parse_emulate("8G:0M"), ("8G", "0M"))

    def test_bad_values_are_refused(self):
        for value in ("", "8", "8GB", "G", "8G:x", "-8G", "8.5G", "8G:4G:1G",
                      "8g", "100M", "0G", ":4G", "8G:0"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    wg.parse_emulate(value)

    def test_the_command_is_wrapped_in_a_hard_limit_scope(self):
        args = args_for("--emulate-vm", "8G:4G", "--python", "/py", "--runner", "/r.py",
                        "--", "--config", "x.ini", "--stage", "train,export")
        self.assertEqual(wg.child_command(args), [
            "systemd-run", "--user", "--scope", "--quiet",
            "-p", "MemoryMax=8G", "-p", "MemorySwapMax=4G",
            "/py", "-u", "/r.py", "--config", "x.ini", "--stage", "train,export"])

    def test_without_the_option_nothing_is_wrapped(self):
        args = args_for("--python", "/py", "--runner", "/r.py", "--", "--dry-run")
        self.assertEqual(wg.child_command(args), ["/py", "-u", "/r.py", "--dry-run"])
        self.assertIsNone(args.emulate_vm)

    def test_a_bad_value_is_a_usage_error(self):
        with redirect_stderr(io.StringIO()) as err, self.assertRaises(SystemExit) as caught:
            wg.parse_args(["--emulate-vm", "8GB", "--", "--dry-run"])
        self.assertEqual(caught.exception.code, 2)
        self.assertIn("--emulate-vm", err.getvalue())

    def test_a_config_with_memory_high_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "run.ini"
            config.write_text("[host]\nmemory_high = 6G\nprocess_cpus = 0,1\n")
            for passthrough in (["--config", str(config)], [f"--config={config}"]):
                with self.subTest(passthrough=passthrough):
                    with redirect_stderr(io.StringIO()) as err,                             self.assertRaises(SystemExit) as caught:
                        wg.parse_args(["--emulate-vm", "8G:4G", "--", *passthrough])
                    self.assertEqual(caught.exception.code, 2)
                    self.assertIn("memory_high", err.getvalue())
                    self.assertIn("escapes", err.getvalue())
            # Without the emulation the same config is fine.
            wg.parse_args(["--", "--config", str(config)])

    def test_a_config_without_memory_high_or_unreadable_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            plain = Path(tmp) / "plain.ini"
            plain.write_text("[host]\nprocess_cpus = 0,1\n")
            empty = Path(tmp) / "empty.ini"
            empty.write_text("[host]\nmemory_high =\n")
            for config in (plain, empty, Path(tmp) / "missing.ini"):
                with self.subTest(config=config.name):
                    args = wg.parse_args(["--emulate-vm", "8G:4G", "--", "--config", str(config)])
                    self.assertEqual(args.emulate_vm, "8G:4G")
        args = wg.parse_args(["--emulate-vm", "8G:4G", "--", "--dry-run"])
        self.assertEqual(args.emulate_vm, "8G:4G")

    def test_print_command_prints_and_exits_zero_without_any_check(self):
        out = io.StringIO()
        with redirect_stdout(out), mock.patch.object(wg, "preflight") as preflight:
            code = wg.main(["--emulate-vm", "8G:4G", "--print-command", "--python", "/py",
                            "--runner", "/r.py", "--", "--stage", "colmap"])
        self.assertEqual(code, 0)
        preflight.assert_not_called()
        self.assertEqual(
            out.getvalue().strip(),
            "systemd-run --user --scope --quiet -p MemoryMax=8G -p MemorySwapMax=4G "
            "/py -u /r.py --stage colmap")

    def test_scope_probe_accepts_a_scope_that_got_its_limit(self):
        done = mock.Mock(returncode=0, stdout=f"{8 * 1073741824}\n", stderr="")
        calls = []

        def run_probe(command, **kwargs):
            calls.append(command)
            return done
        self.assertEqual(wg.scope_problem("8G:4G", lambda _: "/usr/bin/systemd-run", run_probe), "")
        self.assertIn("MemoryMax=8G", calls[0])
        self.assertIn("MemorySwapMax=4G", calls[0])

    def test_scope_probe_refuses_when_the_limit_did_not_apply(self):
        silent = mock.Mock(returncode=0, stdout="max\n", stderr="")
        problem = wg.scope_problem("8G:4G", lambda _: "/x", lambda *a, **k: silent)
        self.assertIn("did not get the memory limit", problem)
        failed = mock.Mock(returncode=1, stdout="", stderr="Failed to connect to bus")
        self.assertIn("Failed to connect", wg.scope_problem("8G", lambda _: "/x",
                                                            lambda *a, **k: failed))

    def test_scope_probe_refuses_without_systemd_run_or_when_it_cannot_start(self):
        self.assertIn("systemd-run", wg.scope_problem("8G", lambda _: None))

        def boom(*a, **k):
            raise FileNotFoundError("systemd-run")
        self.assertIn("could not be started", wg.scope_problem("8G", lambda _: "/x", boom))

    def test_preflight_reports_a_scope_problem_only_when_emulating(self):
        runner = Path(__file__)       # any existing file
        with mock.patch.object(wg, "scope_problem", return_value="no scopes here") as probe:
            host, reason = wg.preflight(args_for("--runner", str(runner)),
                                        wsl=lambda: True, powershell=lambda: "/ps")
            self.assertIsNotNone(host)
            probe.assert_not_called()
            host, reason = wg.preflight(args_for("--emulate-vm", "8G:4G", "--runner", str(runner)),
                                        wsl=lambda: True, powershell=lambda: "/ps")
            self.assertIsNone(host)
            self.assertEqual(reason, "no scopes here")


if __name__ == "__main__":
    unittest.main()
