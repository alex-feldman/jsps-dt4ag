# wsl-guard

Runs `run_pipeline.py` on WSL2 while watching **Windows'** available RAM, and
optionally free space on a Windows drive, so a run cannot empty the Windows host.
Stdlib only.

```bash
export PATH="$HOME/opt/colmap-prefix/bin:$PATH"
uv run python pipeline/scripts/wsl-guard/wsl-guard.py --drive D -- \
    --config "$DATA/configs/$NAME.ini" --stage colmap,process
```

Everything after `--` goes to `run_pipeline.py` unchanged. The guard starts it
with the same Python it runs under (`--python` to change that), so start the
guard with `uv run python`, as above. A bare `python`, or `--python "$(which
python)"`, can pick up the COLMAP prefix's own `python`, which has no torch;
for a run that trains or exports, the guard checks the interpreter can find
torch and refuses (exit 76, naming the interpreter) if it cannot.

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

The Windows probe and the default thresholds live in `pipeline/windows_host.py`,
which the runner also reads once before the colmap stage to choose COLMAP's
thread count (`[colmap] num_threads = auto`; QUICKSTART, "COLMAP threads and
memory"). That choice leaves Windows 2 GiB of RAM (halfway between this guard's
default `--stop-mib` and `--relief-mib`) and 2 GiB of commit headroom (its
default `--stop-commit-mib`).

## What it does

1. **Refuses to run off WSL2** (no `microsoft` in the kernel release, or no
   reachable `powershell.exe`), with exit 76.
2. **Before starting**, reads Windows' available RAM (`Win32_OperatingSystem.
   FreePhysicalMemory`, the figure Task Manager calls "Available", standby
   included), the commit headroom (`FreeVirtualMemory`: the commit limit, RAM
   plus page file, minus what is committed) and, with `--drive`, the drive's free
   space (`Get-PSDrive`). Below `--start-commit-mib` (default 8192) it refuses
   with exit 76: when commit runs out Windows stops the whole WSL VM, even with
   RAM still available, and releasing WSL's cache does not raise it. Below
   `--start-mib` it first tries to release WSL's file cache
   (`sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'`; skipped quietly
   without passwordless sudo) and polls for up to `--settle` seconds, because
   Windows sees released cache only about 40 seconds later (measured: 4.5 GiB to
   13.1 GiB). Still short, or short on disk: it refuses with exit 76 and starts
   nothing.
3. **During the run**, samples every `--interval` seconds. Below `--relief-mib`
   it releases the cache again, at most once per `--relief-every` seconds. Below
   `--stop-mib`, below `--stop-commit-mib` commit headroom (default 2048), or
   below `--stop-drive-gib` free on the drive, it sends SIGTERM
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
| `--emulate-vm` | off | `VM[:SWAP]`, for example `8G:4G`: run inside a hard memory limit. See "Emulating a smaller machine". |
| `--print-command` | off | Print the command that would be started and exit 0. Checks and starts nothing. |
| `--python`, `--runner` | this interpreter, the repository's `run_pipeline.py` | What is started. |

Inconsistent thresholds (stop at or above start, relief below stop), a bad
`--emulate-vm` value, and `--emulate-vm` with a config that sets `[host]
memory_high` are usage errors, exit 2.

## Emulating a smaller machine

`--emulate-vm 8G:4G` starts the run as

```
systemd-run --user --scope --quiet -p MemoryMax=8G -p MemorySwapMax=4G {python} -u run_pipeline.py ...
```

so the whole run, programs plus the file cache they create, must fit in 8 GiB,
with 4 GiB of swap behind it. That is what WSL gives a 16 GB Windows machine by
default (half of the RAM for the VM, swap a quarter of the RAM). Overflow is an
OOM kill, not a slowdown, unlike `[host] memory_high`. Sizes are `{n}M` or
`{n}G`; memory must be at least 512M; swap defaults to none. Before starting it
creates one test scope with the same limits and reads `memory.max` back, and
refuses (exit 76) if the limit did not apply, for example without a systemd user
session or a delegated memory controller.

**It cannot be combined with `[host] memory_high`.** A `systemd-run --user
--scope` started inside another scope does not stay inside it: it becomes a
sibling under `app.slice`, outside the limit. The runner wraps each stage this
way when `memory_high` is set, so the stages would run unlimited while the
emulation looked in place. The guard reads `[host] memory_high` from the file
named by `--config` and refuses with exit 2. It cannot see a config the runner
finds by itself, so always pass `--config` when emulating.

Measured 2026-10-20 on a 32 GB laptop: a full colmap, process, train and export
run on a 55-frame 3840x2880 capture under `8G:4G` exited 0 in 27 minutes, with 0
OOM kills, no swap, a program (anon) peak of 2.4 GiB and a scope peak of 8192
MiB (cache filling the limit). Pass means exit 0 and `scope_oom_kills` 0.

## Memory of the run's own cgroup

Every sample also reads the memory of the child's cgroup (`/proc/{pid}/cgroup`,
then `/sys/fs/cgroup/{path}/`). Each file is optional: a missing one (older
kernel, no memory controller, cgroup v1, no systemd) leaves out only its own
field, and a run with no readable cgroup still works.

The sample gets `scope_cur_mib`, `scope_peak_mib` (`memory.peak`, kernel 5.19 or
later), `scope_anon_mib` (program memory), `scope_file_mib` (file cache),
`scope_swap_mib` and `scope_oom_kills`, plus the `cgroup` path. The status JSON
keeps `scope_peak_mib`, `scope_anon_max_mib`, `scope_file_max_mib` and
`scope_swap_max_mib` as maxima, and `scope_oom_kills` as the largest count seen.

These figures are the run's own only when the child has a cgroup of its own, as
it does under `--emulate-vm`. Otherwise it shares the guard's cgroup, with the
terminal session and anything else in it, and the figures would include all of
that. Then `scope_dedicated` is false, the samples are marked `scope_shared`, and
the status keeps no maxima. Stages started by the runner's own `[host]
memory_high` scopes are separate cgroups and are not read.

The OOM counter is read every `--interval` seconds and the scope disappears when
the run exits, so a kill in the last seconds can be missed. The status therefore
also sets `killed_by_sigkill_unprompted` when the run died of SIGKILL and the
guard did not send it.

## What it writes

Beside the log, with the log's name minus its extension:

- `<name>.log`: everything the run printed.
- `<name>-host.jsonl`: one line per sample, `{"ram_mib": ..., "drive_gib": ...,
  "at": ...}`, plus `released_cache` when a release was attempted and
  `probe_error` lines for failed probes.
- `<name>-status.json`: the command, working directory, thresholds, the start
  sample, the pid and process group, cache releases, `guard_stop` (the sample
  that tripped it, or null), `emulate_vm`, the cgroup memory figures described
  above, the run's return code and the guard's exit code.
  Written at start and rewritten at the end, so a status with no `returncode`
  means the guard itself was killed.

## Exit codes

| Code | Meaning |
|---|---|
| the run's own | `0`, `1`, `2`, `130` as `run_pipeline.py` returns them; a run killed by a signal reports 128 + the signal |
| `75` | the guard stopped the run (RAM or disk) |
| `76` | refused to start: not WSL2, no `powershell.exe`, an interpreter that cannot find torch for a run that trains or exports, a failed first probe, not enough headroom, a test scope that did not get the `--emulate-vm` limit, or the named log exists |
| `2` | usage error |

## What it does not do

It does not check the GPU, whether another pipeline or training job is already
running, or the Linux side's memory. It does not cap anything itself; that is
`[host] memory_high`. The thresholds are defaults that worked on a 32 GB laptop,
not derived limits: on a machine with more or less RAM, set them.

Tests: `pipeline/tests/test_wsl_guard.py` (hermetic: no WSL, Windows or sudo).
