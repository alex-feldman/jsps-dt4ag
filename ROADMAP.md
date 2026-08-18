# Roadmap

What each version **means**, and what has to be true before it ships.

This file holds acceptance criteria, not tasks. Day-to-day execution state (what
is in progress, blocked, or due) lives in the maintainer's project tracker and
deliberately does not appear here: two lists of the same work drift, and the one
in the repository is the one that would go stale unnoticed. If you want to know
what a version promises, read this. If you want to know what is being worked on
today, read the commit log.

Versioning is [SemVer](https://semver.org/) while pre-1.0: MINOR for new
capability, PATCH for fixes. Milestones are annotated git tags.

---

## v0.1.0 — alpha: supports pre-made masked images

**Released 2026-08-13** (tagged retroactively 2026-08-18).

- [x] Four stages driven from one INI config: COLMAP, ns-process-data, ns-train,
      ns-export
- [x] Every subprocess return code checked and every claimed artefact verified on
      disk, including a PLY that exists but declares zero vertices
- [x] Nothing hardcoded; the environment is reproducible from `pyproject.toml`
      and `uv.lock`
- [x] Installs and runs on a clean Linux machine from the repository alone
- [x] GPU architecture verified against the installed gsplat before a run starts
- [x] Masking supported by *starting from* pre-made masked images; the pipeline
      does not create them

**Known limitation, by design:** the pipeline began at masked images, so
producing them was a manual step outside it.

---

## v0.2.0 — supports separate raw images and mask images

**Released 2026-08-18.** The pipeline accepts raw photographs plus separate mask files
and does the masking itself.

- [x] Masks are composited into the photograph's alpha channel as a pre-step,
      after which every stage is unchanged and unaware
- [x] Compositing reuses the existing batch script rather than reimplementing it
- [x] Alpha compositing verified to remove background geometry: 2,385 gaussians
      against 96,938 for the same capture via the rejected mask-file route
- [x] Masks may sit beside the photographs or in a parallel tree
- [x] The training resolution is carried in the export filename, so two
      resolutions of one capture cannot collide. The run log records the
      *configured* value, which is the resolved one whenever it is pinned and
      the literal `auto` when it is not: the row is written before the stages
      run, so it states intent, and a per-run outcome record is v0.3.0's job
- [x] The process stage verifies the downscale pyramid it depends on
- [x] Optional per-capture provenance (`capture.ini`), read and recorded, never
      load-bearing
- [x] Composited output written to `derived/`, not into the input tree
- [x] Canonical layout implemented: a capture is a directory containing
      `images/` (see `pipeline/LAYOUT.md`)
- [x] Verified end to end on a multi-capture collection: eleven captures across
      two collections reconstructed from the migrated layout on 2026-08-18, all
      succeeding, and the five with recorded reference counts landed within 6%
      of them. No composite was written under `datasets/` by any of the eleven

**Acceptance:** someone with photographs and masks, and no knowledge of this
pipeline's history, can reconstruct a masked subject without pre-processing
anything by hand.

---

## v0.3.0 — run a whole collection, and know what every run did

Not started. Two halves of one story: driving many captures without a hand-written
loop, and being able to answer afterwards what each run did and how far it got.
Today a batch is a shell loop over one config per capture, with no resume and no
record of outcome (`pipeline/SEQUENTIAL-RUNS.md`).

**Batching**

- [ ] Point the pipeline at a collection and every capture beneath it
      reconstructs, one at a time, discovered by the `images/` rule
      (`pipeline/LAYOUT.md`)
- [ ] One base config for a collection, instead of one near-identical config per
      capture
- [ ] Re-running a partly failed batch skips what already completed instead of
      silently duplicating it under new run ids
- [ ] Concurrent runs cannot silently contend for one GPU

**Run records**

- [ ] A per-run JSON record under `runs/`, written INCREMENTALLY so a crashed or
      killed run still shows how far it got
- [ ] An explicit milestone ladder (`config_ok` → `prereqs_ok` →
      `masks_composited` → `colmap_done` → `transforms_ok` → `pyramid_ok` →
      `trained` → `checkpoint_found` → `exported` → `verified`), with the
      furthest milestone reached surfaced as one field
- [ ] Failed runs get records too. A failure record is worth more than a success
      one
- [ ] The record embeds the RESOLVED config, not a path to a config file that may
      since have been edited, plus tool versions, per-stage timings and the
      artefacts produced
- [ ] `runs/index.csv`, a queryable summary that is REBUILDABLE by scanning the
      records, so a drifted index is never authoritative
- [ ] Batch records under `runs/batches/`, since a batch spans captures and has
      no home under any one of them

**Acceptance:** point it at a collection, walk away, and afterwards be able to
say for every capture whether it succeeded, how long it took, what it produced,
and if it failed, at which milestone.

**Superseded:** `run-log.csv` becomes a frozen beta log. It records intent only
(the row is written before the stages run), carries no outcome, and is not
back-filled, because the values it lacks were never captured.

---

## v0.4.0 — beta: runs on Windows as well as Linux

Not started. Scope is deliberately platform support plus the rough edges the
alpha deferred, and macOS is **out**: gsplat ships a CUDA backend and nothing
else, and there is no macOS COLMAP binary, so it cannot be made to work by
configuration.

- [ ] Installs and runs on Windows from the repository alone
- [ ] The install path is documented and verified by someone other than the
      author
- [ ] LICENSE and CITATION.cff present

**Acceptance:** a collaborator on either OS can install it, run it on their own
capture, and cite it.

---

## v1.0.0 — the full public release

Not started.

- [ ] Everything in 0.4.0, stable across at least two independent installs
- [ ] The published documentation describes a pipeline someone has actually run
      from it, start to finish, without the author present
- [ ] A dataset and its reconstruction are published together, each citable
- [ ] **Video input is supported end to end.** A capture can be a video rather
      than a directory of photographs: frames extracted, the canonical layout
      populated from them, and masks applied to the extracted frames. Today the
      plumbing exists but is unexercised and half-blind: `[nerfstudio]
      scene_type = video` and `[colmap] data_type = video` are settable and
      never tested, and the `auto` heuristic that infers video from a directory
      named `frames` cannot fire under the canonical layout, where that
      component is always `images` (`pipeline/LAYOUT.md`). Either make it work
      or delete the half-support, because a settable option that is neither
      tested nor reachable is worse than an absent one

**Acceptance:** the repository stands on its own as a research artefact.

---

## Deferred decisions

Considered, deliberately not done, recorded so they are not re-litigated from
scratch.

- **Architecture Decision Records** (numbered, immutable `docs/adr/NNNN-*.md`,
  Nygard format). Considered 2026-08-18 and **held off**: the reasoning they
  would capture is already in the commit history and in the topic documents, and
  ADRs carry a standing discipline (never edit a decision, write a superseding
  record instead) whose cost looked higher than the convenience bought. Worth
  revisiting if either becomes true: a second person starts making architectural
  decisions here, or a decision gets reversed twice and nobody can reconstruct
  why. Note that only genuine *decisions* would qualify: `LAYOUT.md` is a
  specification that must stay editable, and `SEQUENTIAL-RUNS.md` records gaps
  rather than a decision.

## Explicit non-goals

- **macOS support.** Blocked upstream, not by this code. Revisit only if gsplat
  publishes a non-CUDA backend.
- **Mask generation.** Producing masks is a separate concern with its own
  repository. This pipeline applies masks; it does not create them.
- **A GUI.** The notebook covers exploration and the CLI covers everything else.
