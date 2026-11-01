"""Windows' memory, read from inside WSL2 through ``powershell.exe``.

Shared by ``scripts/wsl-guard/wsl-guard.py`` (which polls it during a run) and
``colmap_threads.py`` (which reads it once, to choose COLMAP's thread count),
so the PowerShell text, its parsing and the guard's thresholds live in one
place. Stdlib only.

Inside WSL, Linux's own figures overstate what the VM can use: free pages of
the VM are not yet backed by Windows RAM, and Windows stops the whole VM when
its commit limit runs out. Only the Windows side shows either.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from typing import Callable, Optional

__all__ = [
    "POWERSHELL_FALLBACK",
    "WindowsProbeError",
    "GUARD_START_MIB",
    "GUARD_RELIEF_MIB",
    "GUARD_STOP_MIB",
    "GUARD_START_COMMIT_MIB",
    "GUARD_STOP_COMMIT_MIB",
    "WINDOWS_RAM_RESERVE_MIB",
    "WINDOWS_COMMIT_RESERVE_MIB",
    "probe_script",
    "parse_probe",
    "find_powershell",
    "is_wsl",
    "sample_windows",
]

#: Where powershell.exe usually lives when WSL interop is on but the Windows
#: PATH is not appended to the Linux one (``appendWindowsPath=false``).
POWERSHELL_FALLBACK = "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"

# --------------------------------------------------------------------------
# thresholds, in MiB
# --------------------------------------------------------------------------

#: wsl-guard's defaults (its ``--start-mib``, ``--relief-mib``, ``--stop-mib``,
#: ``--start-commit-mib`` and ``--stop-commit-mib``): refuse to start, release
#: WSL's file cache, and stop the run, by Windows available RAM and by commit
#: headroom.
GUARD_START_MIB = 4096
GUARD_RELIEF_MIB = 3072
GUARD_STOP_MIB = 1024
GUARD_START_COMMIT_MIB = 8192
GUARD_STOP_COMMIT_MIB = 2048

#: What the COLMAP thread choice leaves to Windows. RAM: halfway between the
#: guard's stop and relief lines (2048), so a run started at the chosen count
#: is not already in the guard's release zone and stays clear of its stop line.
#: Commit: the guard's own stop line, below which Windows stops the VM.
WINDOWS_RAM_RESERVE_MIB = (GUARD_STOP_MIB + GUARD_RELIEF_MIB) // 2
WINDOWS_COMMIT_RESERVE_MIB = GUARD_STOP_COMMIT_MIB


class WindowsProbeError(Exception):
    """Windows' memory could not be read (no PowerShell, a failed or garbled probe)."""


def probe_script(drive: Optional[str]) -> str:
    """The PowerShell that reports Windows available RAM, commit headroom and, optionally, free disk.

    ``FreePhysicalMemory`` is what Task Manager calls "Available": it includes
    standby (cache Windows can reclaim at once), which is the right figure for
    "will Windows run out". Without standby it would look alarming on any
    healthy machine.

    ``FreeVirtualMemory`` is the commit headroom: the commit limit (RAM plus page
    file) minus what is committed. When it runs out Windows stops the WSL VM
    outright, even with RAM available (measured 2026-11-01: two runs died within
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
            raise WindowsProbeError(f"probe output lacks a usable value ({exc}): {line!r}")
        return sample
    raise WindowsProbeError(f"probe printed no JSON: {text.strip()[:200]!r}")


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


def sample_windows(powershell: str, drive: Optional[str] = None, timeout: float = 30) -> dict:
    """One probe: ``{"ram_mib": .., "commit_mib": .., ["drive_gib": ..]}``.

    Raises ``WindowsProbeError`` when PowerShell cannot run, fails, or prints
    nothing usable. About half a second per call.
    """
    try:
        completed = subprocess.run(
            [powershell, "-NoProfile", "-NonInteractive", "-Command", probe_script(drive)],
            capture_output=True, text=True, timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise WindowsProbeError(f"powershell.exe could not run: {exc}") from exc
    if completed.returncode != 0:
        raise WindowsProbeError(
            f"probe exited {completed.returncode}: {completed.stderr.strip()[:200]}")
    return parse_probe(completed.stdout, drive)
