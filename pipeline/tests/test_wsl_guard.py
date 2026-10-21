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

    def test_an_existing_log_is_never_overwritten(self):
        runner = self._runner("raise SystemExit(0)\n")
        (self.tmp / "run.log").write_text("earlier run\n")
        with redirect_stderr(io.StringIO()):
            code = wg.run(self._args(runner), FakeHost([9000]))
        self.assertEqual(code, wg.EXIT_REFUSED)
        self.assertEqual((self.tmp / "run.log").read_text(), "earlier run\n")


if __name__ == "__main__":
    unittest.main()
