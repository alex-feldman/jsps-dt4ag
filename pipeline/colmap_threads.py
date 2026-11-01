"""Choose COLMAP's ``--num_threads`` from the memory the machine has right now.

``[colmap] num_threads`` takes ``auto`` (the default), a positive whole number,
or ``all``. An explicit ``--num_threads`` in ``[colmap] extra_args`` always
wins over the key. With ``auto`` the runner, just before the colmap stage::

    budget  = min(Linux MemAvailable, cgroup room, Windows room)   # each where it applies
    threads = floor((budget - BASE_MIB) / per_thread_mib), at most the usable CPUs
    threads < 1: refuse to start, with the figures and how to override

Why: at ``--quality high`` (``automatic_reconstructor``'s default) COLMAP 3.12.0
extracts features on the CPU, one extractor per thread, and each holds its own
scale space. At COLMAP's default of one thread per core, a many-core machine
with modest RAM runs out within seconds (QUICKSTART, "COLMAP threads and
memory"). The figure is a snapshot taken once: ``wsl-guard.py`` remains the
backstop while the run goes on.

Only the pipeline's own ``colmap`` stage is affected. With ``[nerfstudio]
skip_colmap = false``, ``ns-process-data`` runs its own COLMAP, untouched.

Stdlib only (no imaging library): image sizes come from a small JPEG and PNG
header reader.
"""

from __future__ import annotations

import math
import os
import shlex
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import windows_host

__all__ = [
    "PER_THREAD_MIB",
    "MEASURED_PIXELS",
    "HIGH_QUALITY_MAX_IMAGE_SIZE",
    "BASE_MIB",
    "image_size",
    "ImageCost",
    "image_cost",
    "extra_args_option",
    "HostMemory",
    "read_host_memory",
    "cgroup_room",
    "ThreadsDecision",
    "choose_auto",
    "decide_threads",
]

# ==========================================================================
# MEASURED CONSTANTS: COLMAP 3.12.0, `automatic_reconstructor --quality high`
# --------------------------------------------------------------------------
# Measured 2026-11-01 on 120 photographs at 5184x3456 under a hard 8 GiB
# limit: 4 threads peaked at 3790 MiB of program memory (948 MiB per thread),
# 2 threads at 1709 MiB (855 per thread). --quality high makes COLMAP shrink a
# copy of each photograph so its long side is at most 2400 px before feature
# extraction (ModifyForHighQuality sets max_image_size = 2400;
# ImageResizerThread does the shrinking), so the measured photographs were
# extracted at 2400x1600 = 3.84 MP.
#
# RE-MEASURE these after the COLMAP 4.2.1 upgrade. 4.2.1 derives the limit as
# 0.75 * EffMaxImageSize() instead of a literal 2400, and its per-thread cost
# has not been measured.
# ==========================================================================

#: Memory one extractor thread needs at MEASURED_PIXELS: the measured 948 MiB
#: plus about 16 percent margin.
PER_THREAD_MIB = 1100
#: The pixel count PER_THREAD_MIB was measured at: 2400x1600, 3.84 MP.
MEASURED_PIXELS = 3_840_000
#: The long side ``--quality high`` shrinks each photograph to before extraction.
HIGH_QUALITY_MAX_IMAGE_SIZE = 2400
#: Fixed cost beside the threads (COLMAP's own process, the database).
BASE_MIB = 256

# ==========================================================================

#: The long side assumed for a photograph whose header cannot be read when
#: ``extra_args`` sets a ``--quality`` other than high: COLMAP 3.12.0's
#: default ``max_image_size``, which no other quality raises.
UNCAPPED_FALLBACK_SIZE = 3200

#: Label width of the runner's log lines ("colmap threads    : ").
_LABEL = "colmap threads    : "

PROC_ROOT = "/proc"
CGROUP_ROOT = "/sys/fs/cgroup"
MIB = 1024 * 1024


# --------------------------------------------------------------------------
# image size, from the file header
# --------------------------------------------------------------------------

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
#: JPEG start-of-frame markers (baseline, progressive, lossless, arithmetic).
#: 0xC4 (DHT), 0xC8 (JPG) and 0xCC (DAC) are not frames.
_SOF_MARKERS = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}


def _jpeg_size(handle) -> Optional[Tuple[int, int]]:
    if handle.read(2) != b"\xff\xd8":
        return None
    for _ in range(1000):  # a real file reaches its frame header in a handful
        byte = handle.read(1)
        if not byte:
            return None
        if byte != b"\xff":
            return None
        marker = handle.read(1)
        while marker == b"\xff":  # fill bytes
            marker = handle.read(1)
        if not marker:
            return None
        code = marker[0]
        if code == 0x01 or 0xD0 <= code <= 0xD8:  # no length field
            continue
        if code in (0xD9, 0xDA):  # end of image, start of scan: no frame header found
            return None
        length_bytes = handle.read(2)
        if len(length_bytes) != 2:
            return None
        length = struct.unpack(">H", length_bytes)[0]
        if length < 2:
            return None
        if code in _SOF_MARKERS:
            body = handle.read(5)
            if len(body) != 5:
                return None
            height, width = struct.unpack(">HH", body[1:5])
            return (width, height) if width and height else None
        handle.seek(length - 2, os.SEEK_CUR)
    return None


def image_size(path: Path) -> Optional[Tuple[int, int]]:
    """``(width, height)`` from a JPEG or PNG header, or None when it cannot be read.

    Reads a few bytes per marker, never the image data. Any other format, or a
    damaged file, gives None and the caller assumes the worst case.
    """
    try:
        with open(path, "rb") as handle:
            head = handle.read(24)
            if head[:8] == _PNG_SIGNATURE and head[12:16] == b"IHDR":
                width, height = struct.unpack(">II", head[16:24])
                return (width, height) if width and height else None
            if head[:2] == b"\xff\xd8":
                handle.seek(0)
                return _jpeg_size(handle)
    except (OSError, struct.error):
        return None
    return None


def capped_size(width: int, height: int, cap: Optional[int]) -> Tuple[int, int]:
    """The size COLMAP extracts at: long side shrunk to ``cap`` when it exceeds it."""
    if cap is None or max(width, height) <= cap:
        return width, height
    scale = cap / max(width, height)
    return int(round(width * scale)), int(round(height * scale))


@dataclass
class ImageCost:
    """What one extractor thread will need, from the largest photograph."""

    per_thread_mib: int
    #: The photograph that sets the cost, as on disk and as extracted.
    size: Tuple[int, int]
    extracted: Tuple[int, int]
    #: The ``--quality high`` cap applies (False: another --quality in extra_args).
    capped: bool
    unreadable: int = 0
    photographs: int = 0

    def describe(self) -> str:
        w, h = self.size
        cw, ch = self.extracted
        if self.capped and (w, h) != (cw, ch):
            text = f"{w}x{h} capped to {cw}x{ch}"
        else:
            text = f"{w}x{h}" + ("" if self.capped else " uncapped")
        if self.unreadable:
            text += f"; {self.unreadable} header(s) unreadable, counted as {cw}x{ch}"
        return f"{self.per_thread_mib} MiB/thread for {text}"


def image_cost(photographs: Iterable[Path], capped: bool = True,
               size_of: Optional[Callable[[Path], Optional[Tuple[int, int]]]] = None) -> ImageCost:
    """Per-thread memory for these photographs: ``PER_THREAD_MIB * pixels / MEASURED_PIXELS``.

    ``pixels`` is the largest photograph's pixel count as COLMAP extracts it:
    after shrinking its long side to ``HIGH_QUALITY_MAX_IMAGE_SIZE`` when
    ``capped``, as is. An unreadable header counts as a square at the cap (the
    worst case for it), or at ``UNCAPPED_FALLBACK_SIZE`` when not capped.
    Rounded up.
    """
    size_of = size_of or image_size
    cap = HIGH_QUALITY_MAX_IMAGE_SIZE if capped else None
    fallback_side = HIGH_QUALITY_MAX_IMAGE_SIZE if capped else UNCAPPED_FALLBACK_SIZE
    best: Optional[Tuple[int, Tuple[int, int], Tuple[int, int]]] = None
    unreadable = 0
    count = 0
    for path in photographs:
        count += 1
        size = size_of(path)
        if size is None:
            unreadable += 1
            size = (fallback_side, fallback_side)
        extracted = capped_size(size[0], size[1], cap)
        pixels = extracted[0] * extracted[1]
        if best is None or pixels > best[0]:
            best = (pixels, size, extracted)
    if best is None:
        side = fallback_side
        best = (side * side, (side, side), (side, side))
        unreadable = 0
    pixels, size, extracted = best
    per_thread = math.ceil(PER_THREAD_MIB * pixels / MEASURED_PIXELS)
    return ImageCost(per_thread, size, extracted, capped, unreadable, count)


# --------------------------------------------------------------------------
# extra_args
# --------------------------------------------------------------------------

def extra_args_option(extra_args: str, name: str) -> Optional[str]:
    """The value ``extra_args`` gives an option (``--name V`` or ``--name=V``), or None.

    The last occurrence wins. An option given with no value reads as ``""``.
    Unparseable quoting reads as absent: COLMAP will report it.
    """
    try:
        words = shlex.split(extra_args or "")
    except ValueError:
        return None
    flag = f"--{name}"
    found = None
    for index, word in enumerate(words):
        if word == flag:
            found = words[index + 1] if index + 1 < len(words) else ""
        elif word.startswith(flag + "="):
            found = word[len(flag) + 1:]
    return found


# --------------------------------------------------------------------------
# the machine's memory
# --------------------------------------------------------------------------

@dataclass
class HostMemory:
    """Every memory figure the budget can use, in MiB; None where it does not apply."""

    linux_available_mib: Optional[int] = None
    #: Room the colmap stage's cgroup leaves, and where it comes from.
    cgroup_room_mib: Optional[int] = None
    cgroup_label: str = ""
    #: Windows' own figures, raw (before the reserves). WSL only.
    windows_ram_mib: Optional[int] = None
    windows_commit_mib: Optional[int] = None
    usable_cpus: int = 1
    warnings: List[str] = field(default_factory=list)


def _read(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def mem_available_mib(meminfo: Path = Path(PROC_ROOT) / "meminfo") -> Optional[int]:
    """``MemAvailable`` from ``/proc/meminfo``, in MiB, or None."""
    text = _read(meminfo)
    for line in (text or "").splitlines():
        if line.startswith("MemAvailable:"):
            try:
                return int(line.split()[1]) // 1024
            except (IndexError, ValueError):
                return None
    return None


def _size_bytes(value: str) -> int:
    """``6G`` -> bytes, as ``[host] memory_high`` is written (validated by the loader)."""
    value = value.strip().upper()
    suffix = value[-1:] if value[-1:] in "KMGT" else ""
    number = int(value[:-1] if suffix else value)
    return number * 1024 ** " KMGT".index(suffix or " ")


def _limit(text: Optional[str]) -> Optional[int]:
    """A cgroup ``memory.max`` / ``memory.high`` value in bytes; None for ``max`` or unreadable."""
    if text is None:
        return None
    text = text.strip()
    if not text or text == "max":
        return None
    try:
        return int(text)
    except ValueError:
        return None


def cgroup_room(memory_high: str = "", proc_root: str = PROC_ROOT,
                cgroup_root: str = CGROUP_ROOT) -> Tuple[Optional[int], str]:
    """The memory the colmap stage's cgroup leaves it, in MiB, and a label; ``(None, "")`` when unlimited.

    With ``[host] memory_high`` the stage runs in a fresh scope capped at that
    value, so the room is the cap. Otherwise the stage inherits the runner's own
    cgroup (``--emulate-vm`` puts it in a hard-limited scope): for the runner's
    cgroup and each ancestor with a finite ``memory.max`` or ``memory.high``,
    the lower of the two minus that cgroup's ``anon`` from ``memory.stat``, and
    the smallest of those. cgroup v1, or any file missing, adds no limit.
    """
    if memory_high:
        return _size_bytes(memory_high) // MIB, f"[host] memory_high {memory_high}"
    own = _read(Path(proc_root) / "self" / "cgroup")
    relative = None
    for line in (own or "").splitlines():
        if line.startswith("0::"):
            relative = line.strip()[3:] or "/"
    if not relative:
        return None, ""
    best: Tuple[Optional[int], str] = (None, "")
    parts = [p for p in relative.strip("/").split("/") if p]
    while parts:
        path = "/" + "/".join(parts)
        base = Path(cgroup_root) / "/".join(parts)
        limits = [v for v in (_limit(_read(base / "memory.max")),
                              _limit(_read(base / "memory.high"))) if v is not None]
        if limits:
            limit = min(limits)
            anon = 0
            for line in (_read(base / "memory.stat") or "").splitlines():
                words = line.split()
                if len(words) == 2 and words[0] == "anon" and words[1].isdigit():
                    anon = int(words[1])
            room = (limit - anon) // MIB
            if best[0] is None or room < best[0]:
                best = (room, f"cgroup {path} limit {limit // MIB} - anon {anon // MIB}")
        parts.pop()
    return best


def usable_cpus() -> int:
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0)) or 1
    return os.cpu_count() or 1


def read_host_memory(memory_high: str = "", proc_root: str = PROC_ROOT,
                     cgroup_root: str = CGROUP_ROOT,
                     wsl: Callable[[], bool] = windows_host.is_wsl,
                     powershell: Callable[[], Optional[str]] = windows_host.find_powershell,
                     sample: Callable[[str], dict] = windows_host.sample_windows,
                     cpus: Callable[[], int] = usable_cpus) -> HostMemory:
    """Read every figure once. Windows is probed only on WSL; a failed probe is a warning."""
    host = HostMemory(usable_cpus=cpus())
    host.linux_available_mib = mem_available_mib(Path(proc_root) / "meminfo")
    host.cgroup_room_mib, host.cgroup_label = cgroup_room(memory_high, proc_root, cgroup_root)
    if wsl():
        found = powershell()
        if not found:
            host.warnings.append(
                "powershell.exe is not reachable, so Windows' memory was not read; the "
                "thread count uses Linux's figures only, which overstate what WSL can use")
        else:
            try:
                figures = sample(found)
            except windows_host.WindowsProbeError as exc:
                host.warnings.append(
                    f"Windows' memory could not be read ({exc}); the thread count uses "
                    "Linux's figures only, which overstate what WSL can use")
            else:
                host.windows_ram_mib = figures.get("ram_mib")
                host.windows_commit_mib = figures.get("commit_mib")
    return host


# --------------------------------------------------------------------------
# the decision
# --------------------------------------------------------------------------

@dataclass
class ThreadsDecision:
    """What the colmap stage gets, why, and what the run records."""

    #: Threads to pass as ``--num_threads``; None adds nothing (COLMAP's default,
    #: or extra_args already says).
    threads: Optional[int]
    #: ``auto``, ``config`` or ``extra_args``.
    source: str
    #: The text after "colmap threads    : " in the run log.
    summary: str
    #: The value recorded in the run record: a count, ``all``, or extra_args' value.
    recorded: str
    #: Set when auto found less than one thread's worth of memory: the run must not start.
    refusal: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    def command_args(self) -> List[str]:
        if self.threads is None or self.source == "extra_args":
            return []
        return ["--num_threads", str(self.threads)]

    def log_line(self) -> str:
        return _LABEL + self.summary


def _budget_terms(host: HostMemory) -> List[Tuple[int, str]]:
    terms: List[Tuple[int, str]] = []
    if host.linux_available_mib is not None:
        terms.append((host.linux_available_mib,
                      f"Linux MemAvailable {host.linux_available_mib}"))
    if host.cgroup_room_mib is not None:
        terms.append((host.cgroup_room_mib, host.cgroup_label))
    if host.windows_ram_mib is not None:
        reserve = windows_host.WINDOWS_RAM_RESERVE_MIB
        terms.append((host.windows_ram_mib - reserve,
                      f"Windows available RAM {host.windows_ram_mib} - {reserve}"))
    if host.windows_commit_mib is not None:
        reserve = windows_host.WINDOWS_COMMIT_RESERVE_MIB
        terms.append((host.windows_commit_mib - reserve,
                      f"Windows commit headroom {host.windows_commit_mib} - {reserve}"))
    return terms


def choose_auto(cost: ImageCost, host: HostMemory, source_file: object = "the config") -> ThreadsDecision:
    """The ``auto`` rule: budget, per-thread cost, CPU ceiling, refuse below one."""
    terms = _budget_terms(host)
    if not terms:
        warning = ("no memory figure could be read (no /proc/meminfo), so COLMAP keeps "
                   "its default of one thread per CPU")
        return ThreadsDecision(None, "auto", f"all (auto: {warning})", "all",
                               warnings=list(host.warnings) + [warning])
    budget, bound = min(terms, key=lambda term: term[0])
    raw = (budget - BASE_MIB) // cost.per_thread_mib
    threads = min(raw, host.usable_cpus)
    cpus = f"{host.usable_cpus} CPUs"
    if raw > host.usable_cpus:
        cpus = f"capped at {host.usable_cpus} usable CPUs"
    detail = f"budget {budget} MiB, bound by {bound}; {cost.describe()}; {cpus}"
    if threads < 1:
        figures = "; ".join(label for _, label in terms)
        refusal = (
            f"not enough memory for one COLMAP feature-extraction thread: it needs "
            f"about {cost.per_thread_mib} MiB per thread plus {BASE_MIB} MiB "
            f"({cost.describe()}), and the budget is {budget} MiB, bound by {bound} "
            f"(all figures in MiB: {figures}).\n"
            f"  Free memory and run again: close Windows applications (browsers "
            f"first), or release WSL's file cache with "
            f"sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'.\n"
            f"  Or override the choice in {source_file}: set [colmap] num_threads = 1, "
            f"or put --num_threads 1 in [colmap] extra_args. The run may then run out "
            f"of memory.")
        return ThreadsDecision(None, "auto", f"REFUSED (auto: {detail})", "refused",
                               refusal=refusal, warnings=list(host.warnings))
    return ThreadsDecision(threads, "auto", f"{threads} (auto: {detail})", str(threads),
                           warnings=list(host.warnings))


def decide_threads(setting: str, extra_args: str, photographs: Sequence[Path],
                   memory_high: str = "", source_file: object = "the config",
                   host_reader: Optional[Callable[[str], HostMemory]] = None) -> ThreadsDecision:
    """Apply ``[colmap] num_threads`` and ``extra_args`` to this run's photographs.

    Only ``auto`` reads the machine (``host_reader``, default ``read_host_memory``)
    or the photographs. An explicit ``--num_threads`` in ``extra_args`` wins,
    and the summary says the key was overridden.
    """
    in_extra = extra_args_option(extra_args, "num_threads")
    if in_extra is not None:
        return ThreadsDecision(
            None, "extra_args",
            f"{in_extra or '(no value)'} (extra_args --num_threads; overrides "
            f"[colmap] num_threads = {setting})",
            in_extra)
    if setting == "all":
        return ThreadsDecision(None, "config",
                               "all ([colmap] num_threads = all: COLMAP's default, one "
                               "per CPU, chosen without regard to memory)", "all")
    if setting != "auto":
        return ThreadsDecision(int(setting), "config",
                               f"{setting} ([colmap] num_threads = {setting})", setting)
    quality = extra_args_option(extra_args, "quality")
    capped = quality is None or quality.strip().lower() == "high"
    cost = image_cost(photographs, capped=capped)
    reader = host_reader or (lambda high: read_host_memory(high))
    return choose_auto(cost, reader(memory_high), source_file)
