# wsl-guard

Runs `run_pipeline.py` on WSL2 while watching **Windows'** available RAM, and
optionally free space on a Windows drive, so a run cannot empty the Windows host.
Stdlib only.

```bash
python pipeline/scripts/wsl-guard/wsl-guard.py --drive D -- \
    --config "$DATA/configs/$NAME.ini" --stage colmap,process
```

Everything after `--` goes to `run_pipeline.py` unchanged. The guard starts it
with the same Python it runs under (`--python` to change that).

## Why it exists

WSL2's Linux VM grows into Windows' RAM to hold the Linux file cache, and it does
not hand that memory back while work keeps running. A stage that copies or
downscales full-resolution images fills the cache fast: `ns-process-data` on a
55-frame 3840x2880 capture took Windows' available RAM from 6.5 GiB to 59 MiB in
about 25 seconds, while `free` inside WSL looked healthy. Releasing the cache
*before* a run does not help, because the run refills it; WSL's own
`autoMemoryReclaim` only acts after several idle minutes. QUICKSTART.md, "Memory:
WSL can empty Windows' RAM", has the full mechanism.

The guard is the backstop. The fix that keeps the VM from growing in the first
place is `[host] memory_high` in the run config (`configs/README.md`), which caps
each stage, file cache included. Use both: the cap keeps the run small, the guard
stops it if something else (another program, another WSL job) eats the headroom.

## What it does

1. **Refuses to run off WSL2** (no `microsoft` in the kernel release, or no
   reachable `powershell.exe`), with exit 76.
2. **Before starting**, reads Windows' available RAM (`Win32_OperatingSystem.
   FreePhysicalMemory`, the figure Task Manager calls "Available", standby
   included) and, with `--drive`, the drive's free space (`Get-PSDrive`). Below
   `--start-mib` it first tries to release WSL's file cache
   (`sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'`; skipped quietly
   without passwordless sudo) and polls for up to `--settle` seconds, because
   Windows sees released cache only about 40 seconds later (measured: 4.5 GiB to
   13.1 GiB). Still short, or short on disk: it refuses with exit 76 and starts
   nothing.
3. **During the run**, samples every `--interval` seconds. Below `--relief-mib`
   it releases the cache again, at most once per `--relief-every` seconds. Below
   `--stop-mib`, or below `--stop-drive-gib` free on the drive, it sends SIGTERM
   to the process group it started, SIGKILL after `--kill-grace` seconds, and
   exits 75. A probe that fails is recorded and is not a trip.
4. **Never stops anything it did not start.** The runner is launched in a new
   session, and only that process group is signaled.
5. Ctrl-C is forwarded to the run as SIGINT (the run is in its own session, so the
   terminal's Ctrl-C would not reach it), escalating the same way; exit 130.

Dropping clean cache is safe under running jobs: only cache that is already on
disk is dropped, after a `sync`.

## Options

| Option | Default | Meaning |
|---|---|---|
| `--start-mib` | `4096` | Refuse to start below this much Windows available RAM (MiB). |
| `--relief-mib` | `3072` | Release WSL's file cache below this during the run (MiB). |
| `--stop-mib` | `1024` | Stop the run below this (MiB). |
| `--drive` | none | A Windows drive letter whose free space is also watched, for example the one holding the WSL disk. |
| `--start-drive-gib` | `12` | Refuse to start below this much free space on `--drive` (GiB). |
| `--stop-drive-gib` | `4` | Stop the run below this (GiB). |
| `--interval` | `5` | Seconds between samples. |
| `--settle` | `90` | Seconds to wait for a pre-start release to show on the Windows side. |
| `--relief-every` | `60` | Minimum seconds between in-run releases. |
| `--kill-grace` | `30` | Seconds between SIGTERM and SIGKILL. |
| `--no-cache-release` | off | Never release the cache (still refuses and stops). |
| `--log` | timestamped | The run's output. Must not exist yet. |
| `--log-dir` | `$XDG_STATE_HOME/jsps-dt4ag/wsl-guard` (else `~/.local/state/...`) | Where the default log name goes. |
| `--quiet` | off | Output to the log only, not also to the terminal. |
| `--python`, `--runner` | this interpreter, the repository's `run_pipeline.py` | What is started. |

Inconsistent thresholds (stop at or above start, relief below stop) are usage
errors, exit 2.

## What it writes

Beside the log, with the log's name minus its extension:

- `<name>.log`: everything the run printed.
- `<name>-host.jsonl`: one line per sample, `{"ram_mib": ..., "drive_gib": ...,
  "at": ...}`, plus `released_cache` when a release was attempted and
  `probe_error` lines for failed probes.
- `<name>-status.json`: the command, working directory, thresholds, the start
  sample, the pid and process group, cache releases, `guard_stop` (the sample
  that tripped it, or null), the run's return code and the guard's exit code.
  Written at start and rewritten at the end, so a status with no `returncode`
  means the guard itself was killed.

## Exit codes

| Code | Meaning |
|---|---|
| the run's own | `0`, `1`, `2`, `130` as `run_pipeline.py` returns them; a run killed by a signal reports 128 + the signal |
| `75` | the guard stopped the run (RAM or disk) |
| `76` | refused to start: not WSL2, no `powershell.exe`, a failed first probe, not enough headroom, or the named log exists |
| `2` | usage error |

## What it does not do

It does not check the GPU, whether another pipeline or training job is already
running, or the Linux side's memory. It does not cap anything itself; that is
`[host] memory_high`. The thresholds are defaults that worked on a 32 GB laptop,
not derived limits: on a machine with more or less RAM, set them.

Tests: `pipeline/tests/test_wsl_guard.py` (hermetic: no WSL, Windows or sudo).
