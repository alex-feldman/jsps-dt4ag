#!/usr/bin/env python3
"""Run ``run_pipeline.py`` on WSL2 under a Windows-RAM and disk guard.

WSL2's Linux VM grows into Windows' RAM to hold the Linux file cache, and does
not give it back while a job keeps running. A stage that copies or downscales
thousands of megapixels of images can take Windows from several GiB available
to almost nothing in under a minute, and Windows then pages or freezes. Nothing
inside Linux sees this: ``free`` reports a healthy VM. This wrapper looks from
the Windows side, through ``powershell.exe``, and:

* refuses to start while Windows has too little RAM available (after first
  trying to hand WSL's clean file cache back) or too little free disk;
* polls every few seconds during the run, releases the cache when RAM runs
  low, and stops the run before Windows runs out;
* stops ONLY the process group it started, never anything else;
* writes a host-sample JSONL and a status JSON beside the run's log.

Stdlib only. Usage::

    python pipeline/scripts/wsl-guard/wsl-guard.py [guard options] -- [run_pipeline.py arguments]

Exit codes: the pipeline's own (0, 1, 2, 130), 75 when the guard stopped the
run, 76 when it refused to start (including "this is not WSL2"), 2 for a usage
error. See README.md in this directory.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
RUNNER = HERE.parent.parent / "run_pipeline.py"

EXIT_GUARD_STOP = 75
EXIT_REFUSED = 76

#: Where powershell.exe usually lives when WSL interop is on but the Windows
#: PATH is not appended to the Linux one (``appendWindowsPath=false``).
POWERSHELL_FALLBACK = "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"

#: Drops only CLEAN cache, after a sync, so it is safe under running jobs. Run
#: through ``sudo -n``: without passwordless sudo it fails at once and the
#: guard carries on without it.
RELEASE_COMMAND = ["sudo", "-n", "sh", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches"]

_DRIVE_RE = re.compile(r"^[A-Za-z]$")


class GuardError(Exception):
    """The guard cannot do its job (not WSL, no PowerShell, bad probe output)."""


# --------------------------------------------------------------------------
# arguments
# --------------------------------------------------------------------------

def split_argv(argv: Sequence[str]) -> Tuple[List[str], List[str]]:
    """Guard options before the first ``--``, the runner's arguments after it."""
    argv = list(argv)
    if "--" in argv:
        index = argv.index("--")
        return argv[:index], argv[index + 1:]
    return argv, []


def default_log_dir() -> Path:
    state = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    return Path(state) / "jsps-dt4ag" / "wsl-guard"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wsl-guard.py",
        description=(
            "Run run_pipeline.py on WSL2 under a Windows-RAM and disk guard. "
            "Everything after -- is passed to run_pipeline.py unchanged."
        ),
        epilog=(
            "Exit codes: the pipeline's own, 75 guard stop, 76 refused to start, "
            "2 usage error."
        ),
    )
    parser.add_argument("--log", type=Path, default=None,
                        help="Log file for the run's output. The status JSON and host "
                             "samples are written beside it. Must not exist yet. "
                             "Default: a timestamped file in --log-dir.")
    parser.add_argument("--log-dir", type=Path, default=None,
                        help=f"Directory for the default log name. Default: "
                             f"$XDG_STATE_HOME/jsps-dt4ag/wsl-guard "
                             f"({default_log_dir()}).")
    parser.add_argument("--start-mib", type=int, default=4096,
                        help="Refuse to start below this much Windows available RAM, "
                             "in MiB (default 4096).")
    parser.add_argument("--relief-mib", type=int, default=3072,
                        help="During the run, release WSL's file cache below this much "
                             "Windows available RAM, in MiB (default 3072).")
    parser.add_argument("--stop-mib", type=int, default=1024,
                        help="Stop the run below this much Windows available RAM, in "
                             "MiB (default 1024).")
    parser.add_argument("--drive", default=None,
                        help="Also watch free space on this Windows drive letter, for "
                             "example D (the drive that holds the WSL disk or the data).")
    parser.add_argument("--start-drive-gib", type=float, default=12.0,
                        help="Refuse to start below this much free space on --drive, "
                             "in GiB (default 12).")
    parser.add_argument("--stop-drive-gib", type=float, default=4.0,
                        help="Stop the run below this much free space on --drive, in "
                             "GiB (default 4).")
    parser.add_argument("--interval", type=float, default=5.0,
                        help="Seconds between samples during the run (default 5).")
    parser.add_argument("--settle", type=float, default=90.0,
                        help="Seconds to wait, polling, for Windows to see a cache "
                             "release before refusing to start (default 90; Windows "
                             "has been measured to take about 40 s).")
    parser.add_argument("--relief-every", type=float, default=60.0,
                        help="At most one cache release per this many seconds during "
                             "the run (default 60).")
    parser.add_argument("--kill-grace", type=float, default=30.0,
                        help="Seconds between SIGTERM and SIGKILL on a guard stop "
                             "(default 30).")
    parser.add_argument("--no-cache-release", action="store_true",
                        help="Never try to release WSL's file cache.")
    parser.add_argument("--quiet", action="store_true",
                        help="Write the run's output to the log only, not also to the "
                             "terminal.")
    parser.add_argument("--python", default=sys.executable,
                        help="Interpreter for run_pipeline.py (default: this one).")
    parser.add_argument("--runner", type=Path, default=RUNNER,
                        help=f"Path to run_pipeline.py (default: {RUNNER}).")
    return parser


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    own, passthrough = split_argv(argv)
    parser = build_parser()
    args = parser.parse_args(own)
    args.passthrough = passthrough
    problems = validate_args(args)
    if problems:
        parser.error("; ".join(problems))
    if args.drive:
        args.drive = args.drive.upper()
    return args


def validate_args(args: argparse.Namespace) -> List[str]:
    problems = []
    if min(args.start_mib, args.relief_mib, args.stop_mib) < 0:
        problems.append("RAM thresholds cannot be negative")
    if not args.stop_mib < args.start_mib:
        problems.append("--stop-mib must be below --start-mib, or every start would "
                        "be stopped at once")
    if args.relief_mib < args.stop_mib:
        problems.append("--relief-mib must not be below --stop-mib, or the cache is "
                        "only released after the run was stopped")
    if args.drive is not None and not _DRIVE_RE.match(args.drive):
        problems.append(f"--drive must be one drive letter such as D, got {args.drive!r}")
    if args.drive is not None and not args.stop_drive_gib < args.start_drive_gib:
        problems.append("--stop-drive-gib must be below --start-drive-gib")
    for name in ("interval", "settle", "relief_every", "kill_grace"):
        if getattr(args, name) <= 0:
            problems.append(f"--{name.replace('_', '-')} must be positive")
    return problems


# --------------------------------------------------------------------------
# the host probe
# --------------------------------------------------------------------------

def probe_script(drive: Optional[str]) -> str:
    """The PowerShell that reports Windows available RAM and, optionally, free disk.

    ``FreePhysicalMemory`` is what Task Manager calls "Available": it includes
    standby (cache Windows can reclaim at once), which is the right figure for
    "will Windows run out". Without standby it would look alarming on any
    healthy machine.
    """
    script = ("$o=Get-CimInstance Win32_OperatingSystem;"
              "$r=[ordered]@{ram_mib=[math]::Floor($o.FreePhysicalMemory/1024)};")
    if drive:
        script += f"$r['drive_gib']=[math]::Round((Get-PSDrive {drive}).Free/1GB,1);"
    return script + "$r|ConvertTo-Json -Compress"


def parse_probe(text: str, drive: Optional[str]) -> dict:
    """Read the probe's JSON line, tolerating a BOM and stray lines around it."""
    for line in reversed(text.replace("﻿", "").splitlines()):
        line = line.strip()
        if not (line.startswith("{") and line.endswith("}")):
            continue
        try:
            raw = json.loads(line)
        except ValueError:
            continue
        try:
            sample = {"ram_mib": int(raw["ram_mib"])}
            if drive:
                sample["drive_gib"] = float(raw["drive_gib"])
        except (KeyError, TypeError, ValueError) as exc:
            raise GuardError(f"probe output lacks a usable value ({exc}): {line!r}")
        return sample
    raise GuardError(f"probe printed no JSON: {text.strip()[:200]!r}")


def find_powershell(which: Callable[[str], Optional[str]] = shutil.which,
                    exists: Callable[[str], bool] = os.path.exists) -> Optional[str]:
    found = which("powershell.exe")
    if found:
        return found
    return POWERSHELL_FALLBACK if exists(POWERSHELL_FALLBACK) else None


def is_wsl(osrelease_path: str = "/proc/sys/kernel/osrelease") -> bool:
    try:
        with open(osrelease_path, encoding="utf-8") as handle:
            return "microsoft" in handle.read().lower()
    except OSError:
        return False


class Host:
    """The Windows side, seen through powershell.exe. A seam for tests."""

    def __init__(self, powershell: str, drive: Optional[str]):
        self.powershell = powershell
        self.drive = drive
        self.script = probe_script(drive)

    def sample(self) -> dict:
        try:
            completed = subprocess.run(
                [self.powershell, "-NoProfile", "-NonInteractive", "-Command", self.script],
                capture_output=True, text=True, timeout=30,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise GuardError(f"powershell.exe could not run: {exc}") from exc
        if completed.returncode != 0:
            raise GuardError(
                f"probe exited {completed.returncode}: {completed.stderr.strip()[:200]}")
        return parse_probe(completed.stdout, self.drive)

    def release_cache(self) -> bool:
        """True when the cache was dropped; False, having done nothing, without sudo."""
        try:
            completed = subprocess.run(RELEASE_COMMAND, capture_output=True, text=True,
                                       timeout=120)
        except (OSError, subprocess.SubprocessError):
            return False
        return completed.returncode == 0


# --------------------------------------------------------------------------
# decisions (pure)
# --------------------------------------------------------------------------

def start_problems(sample: dict, args: argparse.Namespace) -> List[str]:
    """Why the run must not start, or an empty list."""
    problems = []
    if sample["ram_mib"] < args.start_mib:
        problems.append(f"Windows has {sample['ram_mib']} MiB RAM available "
                        f"(need {args.start_mib})")
    if args.drive and sample.get("drive_gib") is not None \
            and sample["drive_gib"] < args.start_drive_gib:
        problems.append(f"{args.drive}: has {sample['drive_gib']} GiB free "
                        f"(need {args.start_drive_gib:g})")
    return problems


def poll_action(sample: dict, args: argparse.Namespace, now: float,
                last_release: Optional[float]) -> Optional[str]:
    """``"stop"``, ``"release"`` or None for one in-run sample. Stop wins."""
    if sample["ram_mib"] < args.stop_mib:
        return "stop"
    if args.drive and sample.get("drive_gib") is not None \
            and sample["drive_gib"] < args.stop_drive_gib:
        return "stop"
    if (not args.no_cache_release and sample["ram_mib"] < args.relief_mib
            and (last_release is None or now - last_release >= args.relief_every)):
        return "release"
    return None


def cache_hint(meminfo_path: str = "/proc/meminfo") -> str:
    """Name the usual culprit when WSL is holding GiBs of file cache."""
    try:
        with open(meminfo_path, encoding="utf-8") as handle:
            cached_kib = next(int(line.split()[1]) for line in handle
                              if line.startswith("Cached:"))
    except (OSError, StopIteration, ValueError, IndexError):
        return ""
    cached_gib = cached_kib / 1048576
    if cached_gib < 3:
        return ""
    return (f" WSL is holding {cached_gib:.1f} GiB of file cache. Release it with "
            "sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' and give Windows "
            "about a minute to see it; if Windows' available RAM barely moves, close "
            "other WSL work and run wsl --shutdown in PowerShell.")


def log_paths(args: argparse.Namespace, stamp: str) -> Tuple[Path, Path, Path]:
    """The run log, its status JSON and its host-sample JSONL, side by side."""
    if args.log is not None:
        log = args.log
    else:
        log = (args.log_dir or default_log_dir()) / f"wsl-guard-{stamp}.log"
    stem = log.with_suffix("") if log.suffix else log
    return log, Path(f"{stem}-status.json"), Path(f"{stem}-host.jsonl")


def child_command(args: argparse.Namespace) -> List[str]:
    return [args.python, "-u", str(args.runner)] + list(args.passthrough)


def exit_code_for(returncode: int) -> int:
    """A child killed by signal N reports -N; a shell would say 128 + N."""
    return 128 - returncode if returncode < 0 else returncode


# --------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------

def _now_iso() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def _say(message: str) -> None:
    print(f"[wsl-guard] {message}", file=sys.stderr, flush=True)


def wait_for_headroom(host: Host, args: argparse.Namespace, sample: dict,
                      clock: Callable[[], float] = time.monotonic,
                      sleep: Callable[[float], None] = time.sleep) -> Tuple[dict, bool]:
    """Before starting: release the cache once and wait for Windows to see it.

    Returns the latest sample and whether a release happened. Polls every
    ``--interval`` seconds up to ``--settle`` and returns early once there is
    enough headroom. Without passwordless sudo nothing is released and the
    first sample is returned as it was.
    """
    if args.no_cache_release or sample["ram_mib"] >= args.start_mib:
        return sample, False
    if not host.release_cache():
        return sample, False
    _say(f"Windows has {sample['ram_mib']} MiB available; released WSL's file "
         f"cache, waiting up to {args.settle:g} s for Windows to take it back")
    deadline = clock() + args.settle
    while clock() < deadline:
        sleep(min(args.interval, max(0.0, deadline - clock())))
        try:
            sample = host.sample()
        except GuardError:
            continue
        if sample["ram_mib"] >= args.start_mib:
            break
    _say(f"Windows now has {sample['ram_mib']} MiB available")
    return sample, True


def stop_group(child: subprocess.Popen, grace: float, first: int = signal.SIGTERM) -> None:
    """Signal the group this guard started, and nothing else, then SIGKILL it."""
    try:
        os.killpg(child.pid, first)
    except ProcessLookupError:
        return
    try:
        child.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    child.wait()


def _pump(stream, log_handle, echo: bool) -> None:
    for line in stream:
        log_handle.write(line)
        log_handle.flush()
        if echo:
            sys.stdout.write(line)
            sys.stdout.flush()


def run(args: argparse.Namespace, host: Host) -> int:
    try:
        sample = host.sample()
    except GuardError as exc:
        _say(f"REFUSED: cannot read Windows memory: {exc}. Nothing was started.")
        return EXIT_REFUSED
    sample, released = wait_for_headroom(host, args, sample)
    problems = start_problems(sample, args)
    if problems:
        _say("REFUSED: not enough headroom to start: " + "; ".join(problems)
             + ". Close applications and rerun. Nothing was started." + cache_hint())
        return EXIT_REFUSED

    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    log, status_path, samples_path = log_paths(args, stamp)
    log.parent.mkdir(parents=True, exist_ok=True)
    command = child_command(args)
    thresholds = {k: getattr(args, k) for k in (
        "start_mib", "relief_mib", "stop_mib", "drive", "start_drive_gib",
        "stop_drive_gib", "interval", "relief_every", "kill_grace", "no_cache_release")}
    state = {
        "command": command, "cwd": os.getcwd(), "started": _now_iso(),
        "start_sample": sample, "released_cache_before_start": released,
        "thresholds": thresholds, "log": str(log), "host_samples": str(samples_path),
        "cache_releases": 0, "guard_stop": None, "interrupted": False,
    }

    def write_status() -> None:
        status_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")

    started = time.monotonic()
    last_release: Optional[float] = None
    try:
        log_handle = log.open("x", encoding="utf-8")
    except FileExistsError:
        _say(f"REFUSED: log {log} already exists; name a new one. Nothing was started.")
        return EXIT_REFUSED
    with log_handle, samples_path.open("a", encoding="utf-8") as samples:
        try:
            child = subprocess.Popen(command, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True,
                                     errors="replace", start_new_session=True)
        except OSError as exc:
            _say(f"REFUSED: could not start {command[0]!r}: {exc}")
            return EXIT_REFUSED
        state.update(pid=child.pid, pgid=child.pid)
        write_status()
        _say(f"started pgid {child.pid}; log {log}")
        pump = threading.Thread(target=_pump, args=(child.stdout, log_handle, not args.quiet),
                                daemon=True)
        pump.start()
        try:
            while True:
                try:
                    child.wait(timeout=args.interval)
                    break
                except subprocess.TimeoutExpired:
                    pass
                try:
                    current = host.sample()
                except GuardError as exc:  # a failed probe is not a trip
                    samples.write(json.dumps({"at": _now_iso(), "probe_error": str(exc)}) + "\n")
                    samples.flush()
                    continue
                current["at"] = _now_iso()
                action = poll_action(current, args, time.monotonic(), last_release)
                if action == "release":
                    last_release = time.monotonic()
                    current["released_cache"] = host.release_cache()
                    state["cache_releases"] += 1
                samples.write(json.dumps(current) + "\n")
                samples.flush()
                if action == "stop":
                    state["guard_stop"] = current
                    write_status()
                    _say(f"HOST GUARD STOP: {current}; stopping process group {child.pid}")
                    stop_group(child, args.kill_grace)
                    break
        except KeyboardInterrupt:
            # The child is in its own session, so the terminal's Ctrl-C never
            # reached it. Forward it as SIGINT, so the runner can report the
            # interruption itself, then escalate.
            state["interrupted"] = True
            stop_group(child, args.kill_grace, first=signal.SIGINT)
        returncode = child.wait()
        pump.join(timeout=10)

    state.update(returncode=returncode, finished=_now_iso(),
                 elapsed_s=round(time.monotonic() - started))
    if state["guard_stop"]:
        code = EXIT_GUARD_STOP
    elif state["interrupted"]:
        code = 130
    else:
        code = exit_code_for(returncode)
    state["exit_code"] = code
    write_status()
    if state["guard_stop"]:
        _say(f"stopped by the host guard ({state['guard_stop']}). Free RAM or disk, "
             f"then rerun the same command. Log: {log}" + cache_hint())
    else:
        _say(f"finished, exit {code}, in {state['elapsed_s']} s; log {log}")
    return code


def preflight(args: argparse.Namespace,
              wsl: Callable[[], bool] = is_wsl,
              powershell: Callable[[], Optional[str]] = find_powershell) -> Tuple[Optional[Host], str]:
    """A Host to probe, or None and the reason this machine cannot be guarded."""
    if not wsl():
        return None, ("this is not WSL2 (no 'microsoft' in the kernel release). The "
                      "guard watches Windows' memory from inside WSL and has nothing to "
                      "watch here; run run_pipeline.py directly.")
    found = powershell()
    if not found:
        return None, ("powershell.exe is not reachable, so Windows' memory cannot be "
                      "read. WSL interop may be off ([interop] enabled=false in "
                      "/etc/wsl.conf).")
    if not args.runner.is_file():
        return None, f"run_pipeline.py not found at {args.runner}; pass --runner."
    return Host(found, args.drive), ""


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    host, reason = preflight(args)
    if host is None:
        _say("REFUSED: " + reason + " Nothing was started.")
        return EXIT_REFUSED
    return run(args, host)


if __name__ == "__main__":
    raise SystemExit(main())
