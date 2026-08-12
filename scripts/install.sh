#!/usr/bin/env bash
#
# jsps-dt4ag: automated install for the Linux alpha (Ubuntu 24.04 x86-64).
#
# This script is the supported install route. `pipeline/QUICKSTART.md` describes
# the same procedure in prose and is the fallback, not the mechanism.
#
# It performs, in order:
#
#   1. apt prerequisites (the only step that needs root)
#   2. uv
#   3. micromamba
#   4. the conda-forge COLMAP 3.12.0 CUDA prefix, with ffmpeg beside it
#   5. uv sync --frozen
#   6. a verification pass that proves the result actually works
#
# Design rules, inherited from pipeline/run_pipeline.py:
#
#   * Fail loudly. Every step is checked by its ARTEFACT (a binary that runs, a
#     library the loader can find, a symbol in a wheel), never by the fact that
#     the installer printed something reassuring. This codebase has five
#     documented silent-success defects and this script is not allowed to add a
#     sixth.
#   * Idempotent. Every step detects what is already satisfied, says so, and
#     skips it. Safe to re-run after a partial failure.
#   * --dry-run prints the exact commands it would otherwise execute, and
#     nothing else. Everything else in that output is a shell comment, so the
#     output is itself a runnable script and doubles as the manual procedure.
#     The manual install instructions are GENERATED from it, which is what stops
#     the automated and manual paths from drifting apart.

# This script deliberately holds command strings with an UNEXPANDED `$HOME`, so
# that one string can be both printed as documentation and expanded by `eval`
# when run. SC2016 is exactly that pattern and is not a defect here.
# shellcheck disable=SC2016

set -euo pipefail

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

# Paths are fixed to match pipeline/QUICKSTART.md exactly. Each one is held
# twice: once expanded, for this script's own tests, and once with a literal
# `$HOME` for the command strings, which are printed verbatim by --dry-run and
# expanded by `eval` when actually run. One string, two uses, so the printed
# command and the executed command cannot disagree.
COLMAP_PREFIX="$HOME/opt/colmap-prefix"
COLMAP_PREFIX_DOC='$HOME/opt/colmap-prefix'
MAMBA_BIN="$HOME/opt/bin/micromamba"
MAMBA_BIN_DOC='$HOME/opt/bin/micromamba'
UV_BIN_DIR="$HOME/.local/bin"

APT_PACKAGES="build-essential libx11-6 libgl1 libgomp1 curl ca-certificates git"

# Pinned. The build string is the CUDA build; the default conda-forge colmap is
# not CUDA-enabled and is uselessly slow.
COLMAP_SPEC="colmap=3.12.0=cuda_126h825ca31_0"

# The GitHub releases URL serves the bare binary. Do NOT use micro.mamba.pm:
# that endpoint serves a .tar.bz2 and a minimal Ubuntu has no bzip2, so tar
# fails with `bzip2: Cannot exec`.
MICROMAMBA_URL="https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-linux-64"

DRY_RUN=false
SKIP_APT=false
FAILED_STEP=""

# ---------------------------------------------------------------------------
# output helpers
# ---------------------------------------------------------------------------

# In --dry-run every line of prose is a shell comment and every command is bare,
# so the whole output can be redirected to a file and run. In normal mode the
# same helpers produce readable progress output.

comment() {
    if [ -n "${1-}" ]; then
        printf '# %s\n' "$1"
    else
        printf '#\n'
    fi
}

section() {
    if $DRY_RUN; then
        printf '\n'
        printf '# %s\n' "---------------------------------------------------------------------------"
        printf '# %s\n' "$1"
        printf '# %s\n' "---------------------------------------------------------------------------"
    else
        printf '\n== %s\n' "$1"
    fi
}

note() {
    if $DRY_RUN; then
        comment "${1-}"
    elif [ -z "${1-}" ]; then
        printf '\n'
    else
        printf '   %s\n' "$1"
    fi
}

skip() {
    $DRY_RUN || printf '   [ok] %s\n' "$1"
}

die() {
    printf '\n' >&2
    printf 'install.sh: FAILED%s\n' "${FAILED_STEP:+ at step $FAILED_STEP}" >&2
    printf '  %s\n' "$@" >&2
    printf '\n' >&2
    printf 'Nothing further was attempted. Fix the above and re-run this script;\n' >&2
    printf 'it is idempotent and will skip whatever already succeeded.\n' >&2
    exit 1
}

# Print a command string, then run it unless this is a dry run.
#
# The string is the single source of truth: --dry-run prints it, a real run
# evals it. `eval` is what expands the literal `$HOME` the string carries.
run() {
    local cmd="$1"
    if $DRY_RUN; then
        printf '%s\n' "$cmd"
        return 0
    fi
    printf '   $ %s\n' "$cmd"
    eval "$cmd" || die "command failed: $cmd"
}

# ---------------------------------------------------------------------------
# probes (read-only; these decide what to skip and prove what was installed)
# ---------------------------------------------------------------------------

have_cmd() { command -v "$1" >/dev/null 2>&1; }

# Ask the dynamic loader whether a shared library is resolvable. Checked at the
# soname rather than the package name because the soname is what open3d's
# extension actually dlopens, and it is what a non-Debian system would need too.
have_lib() {
    local out
    if have_cmd ldconfig; then
        out="$(ldconfig -p 2>/dev/null || true)"
    elif [ -x /sbin/ldconfig ]; then
        out="$(/sbin/ldconfig -p 2>/dev/null || true)"
    else
        return 1
    fi
    case "$out" in
        *"$1"*) return 0 ;;
        *) return 1 ;;
    esac
}

apt_satisfied() {
    have_cmd cc && have_cmd c++ && have_cmd curl \
        && have_lib libX11.so.6 && have_lib libGL.so.1 && have_lib libgomp.so.1
}

# ---------------------------------------------------------------------------
# argument handling
# ---------------------------------------------------------------------------

usage() {
    cat <<'USAGE'
Usage: scripts/install.sh [--dry-run] [--skip-apt] [--help]

Install the jsps-dt4ag Linux alpha on Ubuntu 24.04 x86-64.

  --dry-run   Print the exact commands this would run, and nothing else.
              The output is a runnable shell script and is the source the
              manual install instructions are generated from. Executes nothing,
              and prints the COMPLETE bare-machine procedure rather than the
              subset this particular machine still needs.
  --skip-apt  Do not touch apt at all, even if the system packages are missing.
              The verification pass still runs, so a missing prerequisite is
              reported rather than hidden.
  --help      This message.

Only the apt step needs root, and it is skipped entirely when the packages are
already present, so a machine that already has build-essential, libx11-6,
libgl1 and libgomp1 can run this script with no sudo at all.
USAGE
}

while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=true ;;
        --skip-apt) SKIP_APT=true ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'install.sh: unknown argument: %s\n\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

# ---------------------------------------------------------------------------
# where we are
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname -- "$SCRIPT_DIR")"

for required in pyproject.toml uv.lock pipeline/run_pipeline.py; do
    [ -e "$REPO_ROOT/$required" ] || die \
        "this script must live in the jsps-dt4ag repository, but" \
        "  $REPO_ROOT/$required" \
        "does not exist. Clone the repository and run scripts/install.sh from it:" \
        "  git clone -b pipeline-alpha https://github.com/alex-feldman/jsps-dt4ag.git"
done
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# header
# ---------------------------------------------------------------------------

if $DRY_RUN; then
    cat <<'HEADER'
#!/usr/bin/env bash
#
# jsps-dt4ag: manual install procedure, Linux alpha, Ubuntu 24.04 x86-64.
#
# GENERATED by `scripts/install.sh --dry-run`. Do not edit this by hand: edit
# the script and regenerate, otherwise the manual and automated routes drift.
#
# Prefer running `scripts/install.sh` itself. These are the same commands, in
# the same order, for the case where you need to see or adapt them. The script
# additionally skips whatever is already installed and verifies every step.
#
# This is the COMPLETE procedure for a bare machine, not the subset any one
# machine still needs. Steps you have already done are safe to skip.
#
# The ONLY thing you need before this runs is `git`, to obtain the repository,
# plus a working NVIDIA driver (check with `nvidia-smi`). Step 1 installs
# everything else, curl included. A bare ubuntu:24.04 has none of it.
#
# Everything below runs from the repository root, the directory holding
# pyproject.toml and uv.lock. Get there first; the -b pipeline-alpha is not
# optional, because the default branch contains no files under pipeline/:
#
#     git clone -b pipeline-alpha https://github.com/alex-feldman/jsps-dt4ag.git
#     cd jsps-dt4ag

set -euo pipefail
HEADER
else
    printf 'jsps-dt4ag install (Linux alpha, Ubuntu 24.04 x86-64)\n'
    printf '  repository root : %s\n' "$REPO_ROOT"
    printf '  COLMAP prefix   : %s\n' "$COLMAP_PREFIX"
    printf '  dry run         : no\n'
    if ! have_cmd curl; then
        # Not fatal here: step 1 installs curl. It only becomes fatal if step 1
        # is skipped or cannot run, which the check after step 1 catches.
        printf '  note            : curl is missing and will be installed by step 1\n'
    fi
fi

# ---------------------------------------------------------------------------
# 1. system packages
# ---------------------------------------------------------------------------

FAILED_STEP="1 (system packages)"
section "1. System packages. This is the only step that needs root."
note "build-essential is needed twice: at install time, because nerfstudio pulls"
note "in pyliblzfse and fpsample, which publish no Linux wheel and are compiled"
note "from sdist, and again at the FIRST training step, because splatfacto runs"
note "torch.compile on get_viewmat and triton builds its driver module with the"
note "host compiler. Neither is gsplat compiling; gsplat stays prebuilt."
note ""
note "libx11-6, libgl1 and libgomp1 are the complete set open3d needs. nerfstudio"
note "imports open3d unconditionally, so stage 2 cannot start without them. A"
note "desktop Ubuntu already has all three; a server image or container has none."
note ""
note "curl and ca-certificates are needed by steps 2 and 3 to fetch uv and"
note "micromamba over HTTPS, and a bare ubuntu:24.04 has neither. git is listed"
note "for completeness: you needed it to clone this repository, so it is already"
note "present, but the generated procedure is then complete for a bare machine."

if $DRY_RUN; then
    run "sudo apt-get update"
    run "sudo apt-get install -y $APT_PACKAGES"
elif apt_satisfied; then
    skip "cc, c++, libX11.so.6, libGL.so.1 and libgomp.so.1 all present; apt not needed"
elif $SKIP_APT; then
    note "--skip-apt given, so apt is not touched. The check below still runs, so a"
    note "genuinely missing prerequisite stops the script rather than being hidden."
elif [ "$(id -u)" -eq 0 ]; then
    run "apt-get update"
    run "apt-get install -y $APT_PACKAGES"
elif have_cmd sudo; then
    note "some of these are missing, so this step needs sudo and may prompt."
    run "sudo apt-get update"
    run "sudo apt-get install -y $APT_PACKAGES"
else
    die "system packages are missing and there is no sudo on this machine." \
        "Ask an administrator to run:" \
        "  apt-get install -y $APT_PACKAGES" \
        "then re-run this script, which will need no root at all."
fi

if ! $DRY_RUN; then
    # Verify the artefacts, not apt's exit code: a compiler that runs and three
    # sonames the loader can actually resolve. Runs even under --skip-apt,
    # because the point is to know, not to install.
    have_cmd curl || die \
        "'curl' is still not on PATH after the system-package step, and steps 2" \
        "and 3 cannot fetch uv or micromamba without it. Install it:" \
        "  sudo apt-get install -y curl ca-certificates"
    have_cmd cc || die \
        "'cc' is not on PATH. uv sync will die partway through building" \
        "pyliblzfse, after the whole multi-minute download." \
        "  sudo apt-get install -y build-essential"
    have_cmd c++ || die \
        "'c++' is not on PATH. uv sync will die building fpsample, which" \
        "publishes no wheel for any platform." \
        "  sudo apt-get install -y build-essential"
    if have_cmd ldconfig || [ -x /sbin/ldconfig ]; then
        for soname in libX11.so.6 libGL.so.1 libgomp.so.1; do
            have_lib "$soname" || die \
                "the dynamic loader cannot find $soname." \
                "open3d imports unconditionally inside nerfstudio, so stage 2 will not start." \
                "  sudo apt-get install -y $APT_PACKAGES"
        done
    else
        note "NOT VERIFIED: no ldconfig on this system, so libX11.so.6, libGL.so.1"
        note "and libgomp.so.1 could not be checked. If stage 2 dies with an OSError"
        note "about libX11.so.6, that is this."
    fi
fi

# ---------------------------------------------------------------------------
# 2. uv
# ---------------------------------------------------------------------------

FAILED_STEP="2 (uv)"
section "2. uv. A single static binary; it needs no Python and installs none."
note "uv also supplies the interpreter: 'uv sync --frozen' downloads its own"
note "CPython 3.10, so a system with no python3 at all is fine. The installer"
note "writes \$HOME/.local/bin into your shell profile, so a shell opened before"
note "this ran will not see uv until you re-source it or open a new one."

if $DRY_RUN; then
    run "curl -LsSf https://astral.sh/uv/install.sh | sh"
    run 'export PATH="$HOME/.local/bin:$PATH"'
    run "uv --version"
else
    PATH="$UV_BIN_DIR:$PATH"
    export PATH
    if have_cmd uv; then
        skip "uv already installed: $(command -v uv) ($(uv --version 2>/dev/null || echo 'version unknown'))"
    else
        run "curl -LsSf https://astral.sh/uv/install.sh | sh"
        hash -r
        have_cmd uv || die \
            "the uv installer ran but 'uv' is still not on PATH." \
            "Expected it at $UV_BIN_DIR/uv."
    fi
    uv --version >/dev/null 2>&1 || die "'uv' is on PATH but does not execute."
fi

# ---------------------------------------------------------------------------
# 3. micromamba
# ---------------------------------------------------------------------------

FAILED_STEP="3 (micromamba)"
section "3. micromamba. One static binary, used only to fetch COLMAP."
note "This is not a conda distribution: no base environment, no activate.d, and"
note "nothing on your PATH afterwards."
note ""
note "Do NOT fetch it from micro.mamba.pm. That endpoint serves a .tar.bz2 and a"
note "minimal Ubuntu has no bzip2, so tar fails. The URL below is the bare binary."

# -f so an HTTP error is an error. Without it curl happily writes the error page
# into the file and exits 0, which is a silent success and the exact failure
# mode this script exists to prevent.
MAMBA_FETCH="curl -fLs $MICROMAMBA_URL \\
     -o \"$MAMBA_BIN_DOC\""

if $DRY_RUN; then
    run "mkdir -p \"$(dirname "$MAMBA_BIN_DOC")\""
    run "$MAMBA_FETCH"
    run "chmod +x \"$MAMBA_BIN_DOC\""
    run "\"$MAMBA_BIN_DOC\" --version"
elif [ -x "$MAMBA_BIN" ] && "$MAMBA_BIN" --version >/dev/null 2>&1; then
    skip "micromamba already present and executable: $MAMBA_BIN"
else
    run "mkdir -p \"$(dirname "$MAMBA_BIN_DOC")\""
    run "$MAMBA_FETCH"
    run "chmod +x \"$MAMBA_BIN_DOC\""
    # Running it is the check that catches a downloaded tarball or an HTML error
    # page saved under a binary's name, which a file-exists test would not.
    "$MAMBA_BIN" --version >/dev/null 2>&1 || die \
        "$MAMBA_BIN was downloaded but does not execute." \
        "It is most likely not a binary: check what the URL actually returned." \
        "  $MICROMAMBA_URL"
fi

# ---------------------------------------------------------------------------
# 4. COLMAP + ffmpeg
# ---------------------------------------------------------------------------

FAILED_STEP="4 (COLMAP prefix)"
section "4. COLMAP 3.12.0 (CUDA build) and ffmpeg, into one standalone prefix."
note "COLMAP is a C++ binary and cannot come from pip or uv. Do not use"
note "'apt install colmap': Ubuntu 24.04 ships 3.9.1, which is too old. Do not"
note "build from source: that takes hours and needs a full CUDA toolkit."
note ""
note "ns-process-data refuses to start without BOTH colmap and ffmpeg on PATH,"
note "even though this pipeline always passes --skip-colmap, so both are"
note "installed here into the same prefix. About 4 GB on disk, 2 GB to download."

COLMAP_CREATE="MAMBA_ROOT_PREFIX=\"\$HOME/opt/mamba-root\" \\
    \"$MAMBA_BIN_DOC\" create -y -p \"$COLMAP_PREFIX_DOC\" -c conda-forge \\
    $COLMAP_SPEC ffmpeg"

# Read line 2 of the bare banner. `colmap --version` does not work in 3.12.
# The trailing `|| true` is because a bare `colmap` prints its usage and exits
# nonzero, and because `head` closes the pipe early; neither is a failure. What
# the banner SAYS is checked below, which is the part that matters.
COLMAP_BANNER="PATH=\"$COLMAP_PREFIX_DOC/bin:\$PATH\" colmap 2>&1 | head -2 || true"
FFMPEG_BANNER="PATH=\"$COLMAP_PREFIX_DOC/bin:\$PATH\" ffmpeg -version | head -1 || true"

if $DRY_RUN; then
    run "$COLMAP_CREATE"
    comment "Confirm it is the CUDA build. Line 2 must end 'with CUDA'; a"
    comment "'without CUDA' build runs and is uselessly slow."
    run "$COLMAP_BANNER"
    run "$FFMPEG_BANNER"
else
    if [ -x "$COLMAP_PREFIX/bin/colmap" ] && [ -x "$COLMAP_PREFIX/bin/ffmpeg" ]; then
        skip "COLMAP prefix already populated: $COLMAP_PREFIX"
    else
        note "this is the slow step: a few minutes and about 2 GB of download."
        run "$COLMAP_CREATE"
    fi
    [ -x "$COLMAP_PREFIX/bin/colmap" ] || die \
        "$COLMAP_PREFIX/bin/colmap does not exist after the micromamba create." \
        "Re-run: micromamba resumes rather than starting over."
    [ -x "$COLMAP_PREFIX/bin/ffmpeg" ] || die \
        "$COLMAP_PREFIX/bin/ffmpeg does not exist after the micromamba create." \
        "ns-process-data checks for ffmpeg unconditionally and will refuse to start."
fi

# ---------------------------------------------------------------------------
# 5. uv sync
# ---------------------------------------------------------------------------

FAILED_STEP="5 (uv sync)"
section "5. The Python environment, from the committed lockfile."
note "pyproject.toml and uv.lock pin the whole Python side: Python 3.10,"
note "torch 2.4.1+cu121, gsplat 1.4.0+pt24cu121 (prebuilt, not compiled),"
note "nerfstudio 1.1.5, numpy 1.26.4. About 250 packages and 7.4 GB. Budget"
note "generously the first time: the download dominates and has taken hours on a"
note "throttled link."

if $DRY_RUN; then
    run "uv sync --frozen"
else
    # No cheap "already satisfied" test exists here, and inventing one would be
    # a worse lie than a redundant run: uv sync is itself idempotent and is a
    # no-op in seconds when the venv already matches the lockfile.
    note "uv sync is its own idempotency check; it is a no-op if nothing changed."
    run "uv sync --frozen"
    [ -x "$REPO_ROOT/.venv/bin/ns-train" ] || die \
        "'uv sync --frozen' reported success but .venv/bin/ns-train does not exist." \
        "The environment is not usable. Do not proceed."
fi

# ---------------------------------------------------------------------------
# 6. verification
# ---------------------------------------------------------------------------

FAILED_STEP="6 (verification)"
section "6. Verification. Nothing above is trusted; each claim is re-measured."

GPU_PROBE="nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader"

GSPLAT_PROBE=$(cat <<'EOF'
uv run --frozen python -c 'import gsplat; from gsplat.cuda._backend import _C; print(gsplat.__version__, _C.__file__)'
EOF
)

# Delegates the GPU-architecture decision to run_pipeline.py's own
# check_prerequisites, deliberately rather than reimplementing it. That function
# reads the CUDA architectures out of the installed gsplat binary instead of a
# hardcoded table, so this check cannot drift away from what the runner will do
# at the start of a real run. It also covers colmap, ns-process-data, ns-train,
# ns-export and torch.cuda availability in one pass.
PREREQ_PROBE=$(cat <<EOF
PATH="$COLMAP_PREFIX_DOC/bin:\$PATH" uv run --frozen python -c 'import sys; sys.path.insert(0, "pipeline"); import run_pipeline; run_pipeline.check_prerequisites(["colmap", "process", "train", "export"])'
EOF
)

if $DRY_RUN; then
    comment "The GPU must be visible and its compute capability must be one the"
    comment "prebuilt gsplat wheel contains code for: sm_70, sm_75, sm_80, sm_86"
    comment "or sm_90, and no PTX, so the supported set is closed. RTX 50 series"
    comment "(sm_120) does not work. See QUICKSTART section 1."
    run "$GPU_PROBE"
    comment ""
    comment "COLMAP must resolve from the prefix and must report 'with CUDA'."
    run "$COLMAP_BANNER"
    comment ""
    comment "ffmpeg must resolve too: ns-process-data refuses to start without it."
    run "$FFMPEG_BANNER"
    comment ""
    comment "gsplat must be the PREBUILT wheel, not the pure-Python PyPI one."
    comment "Expect:  1.4.0+pt24cu121 .../site-packages/gsplat/csrc.so"
    comment "A bare 1.4.0, or a path under torch_extensions/, means the PyPI wheel"
    comment "shadowed the prebuilt one and gsplat will try to compile itself."
    run "$GSPLAT_PROBE"
    comment ""
    comment "Finally, the runner's own prerequisite check, which reads the CUDA"
    comment "architectures out of the installed gsplat binary and compares them"
    comment "against this GPU. If this passes, a real run will get past its own"
    comment "startup checks."
    run "$PREREQ_PROBE"
else
    # 6a. GPU
    have_cmd nvidia-smi || die \
        "'nvidia-smi' is not on PATH, so no NVIDIA driver is installed or visible." \
        "This pipeline trains on the GPU and there is no CPU fallback."
    gpu_info="$(eval "$GPU_PROBE" 2>/dev/null || true)"
    [ -n "$gpu_info" ] || die \
        "'nvidia-smi' ran but reported no GPU. Check the driver with a bare 'nvidia-smi'."
    printf '   [ok] gpu             : %s\n' "$(printf '%s' "$gpu_info" | head -1)"

    # 6b. COLMAP, from the prefix, and it must be the CUDA build.
    colmap_banner="$(eval "$COLMAP_BANNER" 2>/dev/null || true)"
    case "$colmap_banner" in
        *"error while loading shared libraries"*)
            die "colmap is present but cannot load its shared libraries:" \
                "  $colmap_banner" \
                "The conda-forge build installed by step 4 carries an \$ORIGIN-relative" \
                "RPATH and needs nothing but PATH. Do not reach for LD_LIBRARY_PATH:" \
                "the prefix carries CUDA 12.9 and torch needs its own 12.1." ;;
    esac
    case "$colmap_banner" in
        *"COLMAP 3.12"*) ;;
        "") die "no COLMAP banner at $COLMAP_PREFIX/bin/colmap. Step 4 did not produce a working binary." ;;
        *) die "COLMAP is not 3.12.0. Banner was:" "  $colmap_banner" ;;
    esac
    case "$colmap_banner" in
        *"with CUDA"*) ;;
        *) die "this COLMAP is not the CUDA build. Banner was:" \
               "  $colmap_banner" \
               "A 'without CUDA' build runs and is uselessly slow. Remove the prefix" \
               "and re-run so step 4 installs $COLMAP_SPEC:" \
               "  rm -rf $COLMAP_PREFIX" ;;
    esac
    printf '   [ok] colmap          : %s\n' "$(printf '%s' "$colmap_banner" | tr '\n' ' ')"

    # 6c. ffmpeg
    ffmpeg_line="$(eval "$FFMPEG_BANNER" 2>/dev/null || true)"
    [ -n "$ffmpeg_line" ] || die \
        "ffmpeg does not run from $COLMAP_PREFIX/bin." \
        "ns-process-data checks for it unconditionally, at argument-parse time," \
        "even though this pipeline always passes --skip-colmap, and exits 1 without it."
    printf '   [ok] ffmpeg          : %s\n' "$ffmpeg_line"

    # 6d. gsplat wheel identity. This is the check that catches the pure-Python
    # PyPI wheel shadowing the prebuilt one, which fails much later and blames
    # the compiler rather than the wheel.
    gsplat_id="$(eval "$GSPLAT_PROBE" 2>/dev/null || true)"
    [ -n "$gsplat_id" ] || die \
        "gsplat could not be imported from the uv environment." \
        "Run this by hand to see the error:" \
        "  $GSPLAT_PROBE"
    case "$gsplat_id" in
        *"+pt24cu121"*) ;;
        *) die "the installed gsplat is not the prebuilt wheel:" \
               "  $gsplat_id" \
               "Expected 1.4.0+pt24cu121. A bare 1.4.0 means the pure-Python PyPI wheel" \
               "shadowed it, and gsplat will try to compile itself at training time." \
               "Do NOT work around this with --index-strategy unsafe-best-match; that" \
               "reopens dependency confusion for every package. Re-sync from the" \
               "lockfile: the explicit = true indexes in pyproject.toml prevent it." ;;
    esac
    case "$gsplat_id" in
        *csrc.so*) ;;
        *) die "gsplat reports the right version but its extension is not the" \
               "wheel's own csrc.so:" \
               "  $gsplat_id" \
               "A path under torch_extensions/ means it JIT-compiled itself." ;;
    esac
    printf '   [ok] gsplat          : %s\n' "$gsplat_id"

    # 6e. The runner's own check: GPU architecture against the architectures
    # actually baked into that gsplat binary, plus torch CUDA and the ns-* tools.
    printf '   $ %s\n' "$PREREQ_PROBE"
    eval "$PREREQ_PROBE" || die \
        "run_pipeline.py's own prerequisite check failed; see its output above." \
        "The install is not usable for a real run. This is the same check the" \
        "runner performs at startup, so working around it here would achieve nothing."
fi

# ---------------------------------------------------------------------------
# done
# ---------------------------------------------------------------------------

FAILED_STEP=""
section "Done. What to do next."
note "The PATH export is needed in EVERY new shell: it is what puts colmap and"
note "ffmpeg where ns-process-data looks for them. Nothing else has to be"
note "activated, because 'uv run' supplies the whole Python environment. Do NOT"
note "also put the prefix's lib/ on LD_LIBRARY_PATH: it carries CUDA 12.9 and"
note "torch needs its own bundled 12.1."

if $DRY_RUN; then
    run 'export PATH="$HOME/opt/colmap-prefix/bin:$PATH"'
    run "cp pipeline/configs/example.ini pipeline/configs/my-run.ini"
    comment "Edit my-run.ini: at minimum [paths] data_root, [dataset] images_subpath,"
    comment "and [train] quit_on_train_completion = true. See QUICKSTART section 4."
    run "uv run python pipeline/dt4ag_config.py pipeline/configs/my-run.ini"
    run "uv run python pipeline/run_pipeline.py --config pipeline/configs/my-run.ini --dry-run"
    run "uv run python pipeline/run_pipeline.py --config pipeline/configs/my-run.ini"
else
    cat <<NEXT

   export PATH="\$HOME/opt/colmap-prefix/bin:\$PATH"
   cp pipeline/configs/example.ini pipeline/configs/my-run.ini
   # edit my-run.ini: [paths] data_root, [dataset] images_subpath,
   #                  [train] quit_on_train_completion = true
   uv run python pipeline/dt4ag_config.py pipeline/configs/my-run.ini
   uv run python pipeline/run_pipeline.py --config pipeline/configs/my-run.ini --dry-run
   uv run python pipeline/run_pipeline.py --config pipeline/configs/my-run.ini

install.sh: all steps completed and verified.
NEXT
fi
