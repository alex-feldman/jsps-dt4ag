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
* writes a host-sample JSONL and a status JSON beside the run's log, with the
  memory of the run's own cgroup (peak, program memory, swap, OOM kills) when it
  has one;
* with ``--emulate-vm 8G:4G``, runs the whole invocation inside a hard memory
  limit, to check that it fits on a smaller machine (a 16 GB Windows machine
  gives WSL 8 GB of RAM and 4 GB of swap by default).

Stdlib only. Usage::

    python pipeline/scripts/wsl-guard/wsl-guard.py [guard options] -- [run_pipeline.py arguments]

Exit codes: the pipeline's own (0, 1, 2, 130), 75 when the guard stopped the
run, 76 when it refused to start (including "this is not WSL2"), 2 for a usage
error. See README.md in this directory.
"""

from __future__ import annotations

import argparse
import configparser
import datetime as _dt
import json
import os
import re
import shlex
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
_SIZE_RE = re.compile(r"^(\d+)([MG])$")

#: Smallest ``--emulate-vm`` memory accepted, in MiB. Below this the run could
#: not even start and the result would say nothing about a real machine.
MIN_EMULATE_MIB = 512

PROC_ROOT = "/proc"
CGROUP_ROOT = "/sys/fs/cgroup"
#: Stages that import torch. The guard checks the interpreter can find it first.
TORCH_STAGES = {"train", "export"}


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
    parser.add_argument("--start-commit-mib", type=int, default=8192,
                        help="Refuse to start below this much Windows commit headroom (the "
                             "commit limit minus what is committed), in MiB (default 8192). "
                             "Windows stops the whole WSL VM when commit runs out, even with "
                             "RAM still available.")
    parser.add_argument("--stop-commit-mib", type=int, default=2048,
                        help="Stop the run below this much Windows commit headroom, in MiB "
                             "(default 2048).")
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
    parser.add_argument("--emulate-vm", default=None, metavar="VM[:SWAP]",
                        help="Run the whole invocation inside a HARD memory limit, as on a "
                             "smaller machine: VM is the memory (program plus file cache) and "
                             "SWAP the swap, each like 8G or 6000M, for example 8G:4G (what "
                             "WSL gives a 16 GB Windows machine by default). Overflow is an "
                             "OOM kill. Uses systemd-run --user --scope. Refused when the run "
                             "config sets [host] memory_high, because a scope started inside "
                             "this one escapes it.")
    parser.add_argument("--print-command", action="store_true",
                        help="Print the command that would be started (with any "
                             "--emulate-vm wrapping) and exit 0, checking and starting "
                             "nothing.")
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
    if min(args.start_commit_mib, args.stop_commit_mib) < 0:
        problems.append("commit thresholds cannot be negative")
    if not args.stop_commit_mib < args.start_commit_mib:
        problems.append("--stop-commit-mib must be below --start-commit-mib")
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
    if args.emulate_vm is not None:
        try:
            parse_emulate(args.emulate_vm)
        except ValueError as exc:
            problems.append(str(exc))
        else:
            capped = config_memory_high(args.passthrough)
            if capped:
                problems.append(
                    f"--emulate-vm cannot be combined with [host] memory_high (the run "
                    f"config sets {capped}): the runner would start its own scope inside "
                    "the emulated one, and a nested systemd-run --user --scope escapes the "
                    "outer limit, so nothing would be emulated. Remove memory_high from the "
                    "config, or use a copy of the config without it, for the emulation run")
    return problems


def _size_mib(text: str) -> int:
    match = _SIZE_RE.match(text)
    if not match:
        raise ValueError(f"{text!r} is not a size like 8G or 6000M")
    return int(match.group(1)) * (1024 if match.group(2) == "G" else 1)


def parse_emulate(value: str) -> Tuple[str, str]:
    """Validate ``--emulate-vm`` and return ``(memory, swap)`` as systemd reads them.

    ``8G:4G`` is 8 GiB of memory and 4 GiB of swap; ``8G`` alone means no swap.
    Only ``M`` and ``G`` suffixes are accepted, and the memory must be at least
    ``MIN_EMULATE_MIB``.
    """
    memory, _, swap = value.partition(":")
    swap = swap or "0M"
    try:
        memory_mib = _size_mib(memory)
        _size_mib(swap)
    except ValueError as exc:
        raise ValueError(f"--emulate-vm {value!r}: {exc}; use for example 8G:4G")
    if memory_mib < MIN_EMULATE_MIB:
        raise ValueError(f"--emulate-vm memory must be at least {MIN_EMULATE_MIB}M, "
                         f"got {memory}")
    return memory, swap


def config_memory_high(passthrough: Sequence[str]) -> str:
    """``[host] memory_high`` of the run config named by ``--config``, or ``""``.

    Only looks at an explicit ``--config {path}`` or ``--config={path}``. A config
    that cannot be read, or no ``--config``, gives ``""``: the runner reports
    those itself, and this check must never stop a run for a reason of its own.
    """
    path = None
    items = list(passthrough)
    for index, item in enumerate(items):
        if item == "--config" and index + 1 < len(items):
            path = items[index + 1]
        elif item.startswith("--config="):
            path = item.split("=", 1)[1]
    if not path:
        return ""
    parser = configparser.ConfigParser(interpolation=None)
    try:
        with open(path, encoding="utf-8") as handle:
            parser.read_file(handle)
    except (OSError, configparser.Error, UnicodeDecodeError):
        return ""
    return parser.get("host", "memory_high", fallback="").strip()


# --------------------------------------------------------------------------
# the host probe
# --------------------------------------------------------------------------

def probe_script(drive: Optional[str]) -> str:
    """The PowerShell that reports Windows available RAM, commit headroom and, optionally, free disk.

    ``FreePhysicalMemory`` is what Task Manager calls "Available": it includes
    standby (cache Windows can reclaim at once), which is the right figure for
    "will Windows run out". Without standby it would look alarming on any
    healthy machine.

    ``FreeVirtualMemory`` is the commit headroom: the commit limit (RAM plus page
    file) minus what is committed. When it runs out Windows stops the WSL VM
    outright, even with RAM available (PX13, 2026-11-01: two runs died within
    seconds of starting COLMAP with about 5 GiB of RAM available and about 7 GiB
    of commit headroom).
    """
    script = ("$o=Get-CimInstance Win32_OperatingSystem;"
              "$r=[ordered]@{ram_mib=[math]::Floor($o.FreePhysicalMemory/1024);"
              "commit_mib=[math]::Floor($o.FreeVirtualMemory/1024)};")
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
            if raw.get("commit_mib") is not None:
                sample["commit_mib"] = int(raw["commit_mib"])
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
    if sample.get("commit_mib") is not None and sample["commit_mib"] < args.start_commit_mib:
        problems.append(f"Windows has {sample['commit_mib']} MiB commit headroom "
                        f"(need {args.start_commit_mib}); close programs or enlarge the "
                        f"page file")
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
    if sample.get("commit_mib") is not None and sample["commit_mib"] < args.stop_commit_mib:
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


def emulation_prefix(value: str) -> List[str]:
    """The ``systemd-run`` words that put a command in a hard memory limit."""
    memory, swap = parse_emulate(value)
    return ["systemd-run", "--user", "--scope", "--quiet",
            "-p", f"MemoryMax={memory}", "-p", f"MemorySwapMax={swap}"]


def child_command(args: argparse.Namespace) -> List[str]:
    command = [args.python, "-u", str(args.runner)] + list(args.passthrough)
    if args.emulate_vm:
        command = emulation_prefix(args.emulate_vm) + command
    return command


# --------------------------------------------------------------------------
# the run's own cgroup (Linux side)
# --------------------------------------------------------------------------

def cgroup_path(pid: int, proc_root: str = PROC_ROOT) -> Optional[str]:
    """The cgroup v2 path of a process (``/user.slice/.../run-x.scope``), or None.

    None on cgroup v1, a process that has gone, or any unreadable file.
    """
    try:
        with open(f"{proc_root}/{pid}/cgroup", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("0::"):
                    return line.strip()[3:] or "/"
    except OSError:
        pass
    return None


def _read_text(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _key_values(text: Optional[str]) -> dict:
    values = {}
    for line in (text or "").splitlines():
        parts = line.split()
        if len(parts) == 2:
            values[parts[0]] = parts[1]
    return values


def _mib(raw) -> Optional[int]:
    try:
        return round(int(str(raw).strip()) / 1048576)
    except ValueError:
        return None


def scope_memory(pid: int, proc_root: str = PROC_ROOT,
                 cgroup_root: str = CGROUP_ROOT) -> dict:
    """Memory of the cgroup a process lives in, in MiB, or ``{}`` when unreadable.

    ``scope_cur_mib`` (``memory.current``), ``scope_peak_mib`` (``memory.peak``,
    kernel 5.19 and later), ``scope_anon_mib`` and ``scope_file_mib`` (program
    memory and file cache, from ``memory.stat``), ``scope_swap_mib``
    (``memory.swap.current``) and ``scope_oom_kills`` (``oom_kill`` in
    ``memory.events``). A file that is missing (older kernel, controller not
    enabled) leaves only its own field out; the rest are still reported.
    ``cgroup`` is the path read. ``proc_root`` and ``cgroup_root`` are seams for
    tests.
    """
    relative = cgroup_path(pid, proc_root)
    if relative is None:
        return {}
    base = Path(cgroup_root) / relative.lstrip("/")
    if not base.is_dir():
        return {}
    sample: dict = {"cgroup": relative}
    stat = _key_values(_read_text(base / "memory.stat"))
    fields = (
        ("scope_cur_mib", _read_text(base / "memory.current")),
        ("scope_peak_mib", _read_text(base / "memory.peak")),
        ("scope_anon_mib", stat.get("anon")),
        ("scope_file_mib", stat.get("file")),
        ("scope_swap_mib", _read_text(base / "memory.swap.current")),
    )
    for key, raw in fields:
        if raw is not None and _mib(raw) is not None:
            sample[key] = _mib(raw)
    events = _key_values(_read_text(base / "memory.events"))
    if "oom_kill" in events:
        try:
            sample["scope_oom_kills"] = int(events["oom_kill"])
        except ValueError:
            pass
    return sample


def fold_scope_maxima(state: dict, sample: dict) -> None:
    """Keep the largest value seen of each scope figure in ``state``.

    ``scope_peak_mib``, ``scope_anon_max_mib``, ``scope_file_max_mib`` and
    ``scope_swap_max_mib`` are maxima; ``scope_oom_kills`` is the largest count
    seen (the kernel's counter only grows). ``scope_cgroup`` is the path read.
    """
    for source, target in (("scope_peak_mib", "scope_peak_mib"),
                           ("scope_anon_mib", "scope_anon_max_mib"),
                           ("scope_file_mib", "scope_file_max_mib"),
                           ("scope_swap_mib", "scope_swap_max_mib")):
        if source in sample:
            state[target] = max(state.get(target, 0), sample[source])
    if "scope_oom_kills" in sample:
        state["scope_oom_kills"] = max(state.get("scope_oom_kills", 0),
                                       sample["scope_oom_kills"])
    if "cgroup" in sample:
        state["scope_cgroup"] = sample["cgroup"]


def scope_sample(pid: int, own_cgroup: Optional[str], proc_root: str = PROC_ROOT,
                 cgroup_root: str = CGROUP_ROOT) -> Tuple[dict, bool]:
    """One cgroup reading of the child, and whether the child has a cgroup of its own.

    A child in the SAME cgroup as the guard (the usual case, without
    ``--emulate-vm``) shares it with the terminal session and whatever else runs
    there, so its peak and OOM count are not the run's. Only a dedicated cgroup's
    figures are folded into the status maxima.
    """
    sample = scope_memory(pid, proc_root, cgroup_root)
    dedicated = bool(sample) and sample["cgroup"] != own_cgroup
    return sample, dedicated


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
        "start_mib", "relief_mib", "stop_mib", "start_commit_mib", "stop_commit_mib",
        "drive", "start_drive_gib",
        "stop_drive_gib", "interval", "relief_every", "kill_grace", "no_cache_release")}
    state = {
        "command": command, "cwd": os.getcwd(), "started": _now_iso(),
        "start_sample": sample, "released_cache_before_start": released,
        "thresholds": thresholds, "log": str(log), "host_samples": str(samples_path),
        "cache_releases": 0, "guard_stop": None, "interrupted": False,
        "emulate_vm": args.emulate_vm,
    }
    own_cgroup = cgroup_path(os.getpid())

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
                scope, dedicated = scope_sample(child.pid, own_cgroup)
                if dedicated:
                    fold_scope_maxima(state, scope)
                state["scope_dedicated"] = dedicated
                try:
                    current = host.sample()
                except GuardError as exc:  # a failed probe is not a trip
                    samples.write(json.dumps({"at": _now_iso(), "probe_error": str(exc),
                                              **scope}) + "\n")
                    samples.flush()
                    continue
                current["at"] = _now_iso()
                current.update(scope)
                if scope and not dedicated:
                    current["scope_shared"] = True
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
    # A scope's OOM counter is only read every --interval seconds, and the scope
    # is gone once the run exits, so a kill in the last seconds can be missed.
    # A run killed by SIGKILL that the guard did not stop is the other sign.
    state["killed_by_sigkill_unprompted"] = (
        returncode == -signal.SIGKILL and not state["guard_stop"]
        and not state["interrupted"])
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
    problem = interpreter_problem(args.python, args.passthrough)
    if problem:
        return None, problem
    if args.emulate_vm:
        problem = scope_problem(args.emulate_vm)
        if problem:
            return None, problem
    return Host(found, args.drive), ""


def passthrough_stages(passthrough: Sequence[str]) -> Optional[set]:
    """The stages ``--stage`` names in the run_pipeline.py arguments, or None.

    None means run_pipeline.py decides (all stages, or everything from
    ``--from-stage`` on), which always includes training.
    """
    for i, arg in enumerate(passthrough):
        if arg == "--stage" and i + 1 < len(passthrough):
            value = passthrough[i + 1]
        elif arg.startswith("--stage="):
            value = arg.split("=", 1)[1]
        else:
            continue
        return {part.strip() for part in value.split(",") if part.strip()}
    return None


def interpreter_problem(python: str, passthrough: Sequence[str],
                        run_probe: Callable[..., "subprocess.CompletedProcess"] = subprocess.run) -> str:
    """Why ``python`` cannot run the requested stages, or ``""``.

    Found 2026-10-21: the COLMAP prefix (QUICKSTART section 3) ships its own
    ``python``, so after ``export PATH=$HOME/opt/colmap-prefix/bin:$PATH`` a bare
    ``python`` or ``$(which python)`` is COLMAP's, which has no torch. The
    pipeline then fails its prerequisite check, with a hint about uv that does
    not name the interpreter. ``find_spec`` locates torch without importing it,
    so this costs well under a second.
    """
    if "--dry-run" in passthrough:
        return ""
    stages = passthrough_stages(passthrough)
    if stages is not None and not stages & TORCH_STAGES:
        return ""
    code = ("import importlib.util, sys; print(sys.prefix); "
            "sys.exit(0 if importlib.util.find_spec('torch') else 3)")
    try:
        done = run_probe([python, "-c", code], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        return f"the interpreter {python!r} (--python) could not be started: {exc}"
    if done.returncode == 0:
        return ""
    prefix = (done.stdout or "").strip() or "unknown"
    return (f"the interpreter that would run the pipeline, {python} (environment "
            f"{prefix}), cannot find torch, so training would fail. This is usually "
            "the wrong python: the COLMAP prefix ($HOME/opt/colmap-prefix/bin) ships "
            "its own, so after exporting it onto PATH a bare 'python' or "
            "'$(which python)' can be COLMAP's. Start the guard with 'uv run python "
            "pipeline/scripts/wsl-guard/wsl-guard.py ...', or pass --python with the "
            "venv's full path ({repository}/.venv/bin/python).")


def scope_problem(emulate_vm: str,
                  which: Callable[[str], Optional[str]] = shutil.which,
                  run_probe: Callable[..., "subprocess.CompletedProcess"] = subprocess.run) -> str:
    """Why ``--emulate-vm`` cannot work here, or ``""``.

    Starts a tiny command once in a scope with the same two limits and reads the
    limit back from the scope's cgroup. Without a systemd user session, or
    without the memory controller delegated to it, systemd-run can still exit 0
    and the limit would silently not apply.
    """
    if not which("systemd-run"):
        return "--emulate-vm needs systemd-run, which is not on PATH."
    memory, swap = parse_emulate(emulate_vm)
    probe = ["systemd-run", "--user", "--scope", "--quiet", "-p", f"MemoryMax={memory}",
             "-p", f"MemorySwapMax={swap}", "sh", "-c",
             "cat /sys/fs/cgroup$(sed -n 's/^0:://p' /proc/self/cgroup)/memory.max"]
    try:
        done = run_probe(probe, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        return f"--emulate-vm: a test scope could not be started: {exc}"
    wanted = str(_size_mib(memory) * 1048576)
    if done.returncode != 0 or done.stdout.strip() != wanted:
        return ("--emulate-vm: a test scope did not get the memory limit (systemd-run "
                f"said {(done.stderr or '').strip()[:160]!r}, memory.max read "
                f"{done.stdout.strip()!r}, wanted {wanted}). The systemd user session may "
                "lack a delegated memory controller; see QUICKSTART, \"Memory: WSL can "
                "empty Windows' RAM\".")
    return ""


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.print_command:
        print(shlex.join(child_command(args)))
        return 0
    host, reason = preflight(args)
    if host is None:
        _say("REFUSED: " + reason + " Nothing was started.")
        return EXIT_REFUSED
    return run(args, host)


if __name__ == "__main__":
    raise SystemExit(main())
