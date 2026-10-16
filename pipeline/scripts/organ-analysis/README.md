# Organ analysis (experimental, CPU)

Lift existing per-photo leaf instances onto an exported Gaussian splat, then extract a provisional leaf-associated skeleton. This is a separate post-reconstruction command. It does not retrain the model, generate masks, install packages, select MTG, or reconstruct missing tissue.

When an export holds more than one plant, separate them first with `cluster_plants.py` ([below](#cluster-plants-and-write-per-plant-derivatives)). It needs only the PLY. It reports 3D clusters and their separations, and writes a derivative PLY for each plant you select explicitly. `orient_plants.py` ([below](#orient-plants-up-direction-plant-axes-and-detached-clusters)) then estimates the scene's up direction and each plant's vertical axis, and tests detached clusters against those axes.

Run with the pipeline's existing Python environment, which supplies NumPy, SciPy, NetworkX and Pillow. Optional static plots also require Matplotlib. No Torch/CUDA model is loaded. The output viewer uses plain HTML/JavaScript Canvas 2D and needs no external libraries or server.

## Inputs

Copy `example.json` and replace every placeholder. Relative paths resolve against the JSON file. Supply the source splat PLY, COLMAP `transforms.json`, the corresponding trained run's `dataparser_transforms.json`, original photographs, production plant masks, individual leaf instances, and an explicit frame-to-source-name map. All camera/mask/photo pixels must share the original image domain. The supported camera model has shared intrinsics and only `k1` radial distortion; unsupported coefficients fail rather than being silently ignored.

`frame_map.json` maps each camera frame stem to its source photograph stem, for example `{"frame_00001":"IMG_0001"}`. Mapping must be exact and one-to-one; this command never guesses it from sorted filenames. Plant mask filenames use those source stems. Leaf instances use `{leaf_instances}/{source_stem}/instances.json`, containing `{"instances":[{"file":"leaf_01.png"}]}` plus any other samask metadata. Instance images must be source-size masks. An empty detection list is retained as unknown evidence; absence of the sidecar or an instance file is an error.

Generate the masks separately with samask's `plant` and `leaf --instances` runs if they do not exist. Keep the two outputs separate. The union alone cannot supply individual leaf identities. No tag/string/soil prompt is required for this initial workflow; it does retain some clutter and tendrils.

## Run

From the repository root, inspect validated inputs without creating an output:

```text
python pipeline/scripts/organ-analysis/analyze_organs.py --config /path/to/capture-organs.json --output /path/to/new-leaf-output --stage leaves --validate-only
```

For a new capture, run the leaf-only pass and inspect `viewer.html` and `leaf_ids2.json`:

```text
python pipeline/scripts/organ-analysis/analyze_organs.py --config /path/to/capture-organs.json --output /path/to/new-leaf-output --stage leaves
```

Choose the tip cluster from this output and add `"apex_leaf": {cluster_integer}` to the config. Then reuse the checked leaf output in a new directory:

```text
python pipeline/scripts/organ-analysis/analyze_organs.py --config /path/to/capture-organs.json --output /path/to/new-organ-output --stage skeleton --labels-dir /path/to/new-leaf-output
```

When the apex association is already known for these exact inputs, run both stages together:

```text
python pipeline/scripts/organ-analysis/analyze_organs.py --config /path/to/capture-organs.json --output /path/to/new-organ-output
```

Every output directory must be new, even for validation-only calls. Partial/failed work is retained with a failure record. A repeated call never overwrites outputs. Skeleton-only reuse requires a successful prior command and matching source hashes, parameters and stored label hashes. It may add the apex choice, plot option and annotations, but must not silently change the analysis inputs.

All numerical options are in the optional `parameters` object; `analyze_organs.DEFAULTS` owns their names and values. They originate in the exploratory Welzo baseline, not universally tuned settings. By default `apex_leaf` is explicit for skeleton extraction; it is never assumed to be 4. Numeric IDs are local to a particular model and mask association result. Optional `annotations` maps cluster-ID strings to user-supplied JSON metadata; it does not move connections or turn an inference into measured geometry. Set `plots` to true for static 3D and photo-overlay sheets.

### Optional computed base and apex candidates

Add top-level `"orientation": "/path/to/completed-orientation/orientation.json"` and
`"apex_mode": "morphology"` to request conservative candidate agreement. Keep the
same `--stage skeleton --labels-dir` command above. Orientation is skeleton-level
provenance; the reused leaf inputs and parameters must remain unchanged.

`computed_axis.orientation` checks the completed orientation artifact and its
run record, capture identity, current source digests and matching camera-frame
digests before creating output. Soil-primary up is required by default. Only
explicit `"allow_camera_fallback": true` permits a recorded camera fallback;
that case has no soil-plane distance. Relative orientation paths resolve against
the config. Orientation artifacts cannot live inside protected photo/mask trees.

`computed_axis.candidates` owns the candidate rule and tie tolerance, which is
at least the existing skeleton `BIN` width in scene units and is recorded in
the report. It records a candidate network root whose anatomical stem base is unvalidated,
supported leaf median-height and associated endpoint-geodesic ranks, and terminal
height ranks. It resolves only when all three unique extrema identify the same
terminal. Ties or disagreement retain `apex-candidates.json` and a failed run
record, then raise an error without producing a successful skeleton. This is
candidate agreement, not proof of a biological apex or a traced primary stem.

An explicit `apex_leaf` together with `orientation` overrides that disagreement,
uses the supplied up for the root, and records the requested decision and all
candidate diagnostics. The original config remains truthful in the run record;
the resolved leaf is in the graph. With no orientation opt-in, the existing
explicit-apex path and geometry remain unchanged. Leaf-only runs do not consume
orientation provenance.

Orientation is scene-level evidence, reusable across per-plant derivatives in
the same capture and coordinate frame. Its soil base-plant selection does not
establish the identity of the analyzed PLY. Selecting and confirming the target
plant derivative remains an explicit upstream responsibility. The reported
root-to-plane distance is diagnostic; it does not validate a botanical base or
impose an additional distance threshold.

Design direction: favor anatomy-informed base/apex decisions and eventually a
plant described by botanical nodes with attached leaves. Botanical nodes remain
distinct from today's geometric graph samples. Considering established MTG or
Helios formats is a preference, not an adoption decision or completion gate.
Detailed node/phytomer conventions remain proposed pending Alex's approval.

The longer-term plant-plus-environment direction includes local temperature,
light and physiological modeling. Environmental fields versus point, surface,
organ or region attributes need a separate FSPM literature review. This is a
future compass, outside current implementation; no environment schema or spatial
resolution is required here.

## Outputs and limits

- `leaf_ids2.npz`: full filtered point centers, RGB approximation, per-point leaf/non-leaf labels, votes and visibility evidence. `-1` is unassigned; `-2` is plant-minus-leaf evidence, not an anatomical stem class.
- `leaf_ids2.json`: settings, merges and source-photo/instance memberships.
- Tiny residual labeled clusters are preserved with their point counts and marked `small_unreviewed_cluster` in the evidence/viewer; they are not automatically additional biological leaves. The association summary identifies the supported clusters.
- `skeleton.json` in skeleton mode: nodes, edges, main path, leaf-base/insertion candidates and scale-free network lengths. Nodes are graph samples, not phytomer nodes.
- `organ-evidence.json`: losslessly serialized graph, leaf associations, source/camera hashes, explicit ID scope, user annotations and uncertainty. Missing/zero-length off-apex paths are marked unresolved.
- `viewer.html`: deterministic display sample of up to 300 centers per leaf, graph lines and insertion labels, embedded for offline inspection. Rotate by dragging; zoom with the wheel; toggle points/edges; reset. Gray means unresolved side paths, not confirmed strings. This is not a Gaussian-splat renderer and its drawing order is not full visibility testing.
- `input-manifest.json` and `run-record.json`: source/code hashes, effective settings, versions, command, stage status and output hashes. A leaf stage can complete without a valid skeleton; full mode fails explicitly if no usable network/apex exists.

This does not establish biological classification accuracy, petiole completeness, metric scale or population generalization. Leaf-free paths may be ties, tendrils, missed tissue or artifacts. A zero petiole length at the chosen apex follows from the endpoint construction; other zero-length fallbacks are missing extraction evidence. Preserve originals and human review.

## Identify leaf-free branch candidates

After an organ-analysis run completes, run this CPU-only review command against
its output folder. It does not rerun masks, training, leaf labeling or skeletons:

```text
python pipeline/scripts/organ-analysis/identify_branches.py --asset /path/to/completed-organ-output --output /path/to/new-branch-review
```

It protects the main axis and paths to saved leaf endpoints, then identifies
the remaining terminal subtrees as candidates. Shared axes carrying leaves
farther downstream are preserved. Terminals near leaf centers and uncertain
nearest-base connections are flagged as guarded: their candidate branches stay
visible but are not eligible for the pruning helper.

Open `viewer.html` for amber candidate edges and B-number labels. Dashed edges
have leaf-proximity/base uncertainty. The checkbox preview without candidates
changes only the display. `branch-candidates.json` and `.csv` retain original
node/edge IDs, attachment/terminal nodes, guard evidence, lengths in scene units
and blank human dispositions. The source graph/labels are preserved exactly in
this additive review folder. `review-provenance.json` links it to the input.

`branches.prune_candidates(graph, report, selected_ids)` is a pure Python helper
that returns a derivative graph for an explicit selection without changing the
input. It refuses mismatched graphs, unknown candidates and protected/guarded
paths. Candidate detection and the viewer do not call this helper or delete
anything. A leaf-free endpoint is a candidate, not proof of a string or false
organ: tendrils and missed tissue can satisfy the same criterion.

## Cluster plants and write per-plant derivatives

`cluster_plants.py` groups the opacity-kept Gaussian centers of one export into 3D clusters by single linkage. It reports each cluster's size and position, the exact join radius and direct gap between every pair of clusters, and a radius sweep. Given an explicit selection, it copies the chosen source rows byte for byte into per-plant derivative PLYs. It needs no cameras or masks, runs on CPU with NumPy and SciPy, and never builds a list of all point pairs within a large radius. `DESIGN.md` explains the method and why that matters.

**It never chooses a plant.** Cluster numbers are ordered by size, which is bookkeeping, not evidence: a walked-around target collects more Gaussians than a neighbor seen in passing. Decide identity from photographs, instance masks, tags or human review, then record that decision in the selection.

Copy `plant-clusters-example.json`. Relative paths resolve against the JSON file. `inputs.ply` is the export. Optional `parameters` override `plant_clusters.DEFAULTS`, which owns the names and values. The defaults come from the October 2026 Welzo M1 separation and are experimental, not universal tuning:
- radius = `radius_mult` times the median nearest-neighbor spacing d1;
- clusters need at least `min_cluster_size` centers;
- `opacity_cutoff` is the organ-analysis default.

Check the sweep in `clusters.json` before trusting the radius on a new capture: a good radius sits on a plateau where the clusters you care about stop changing.

First pass: report only. Leave out `derivatives`.

```text
python pipeline/scripts/organ-analysis/cluster_plants.py --config /path/to/plant-clusters.json --output /path/to/new-cluster-report
```

Second pass, after the identity decision: add the following to the config, then run it into another new directory.
- `source_ply_sha256`, copied from the first `clusters.json`. Cluster IDs belong to one model.
- `expected_cluster_sizes` for the clusters you rely on. The run refuses if they differ.
- `derivatives`. Each entry has:
  - a unique `name`;
  - a required `basis`, saying who decided and on what evidence;
  - `clusters` (IDs) and/or `rows` (opacity-kept source row indices, as a list or a `.npy` path);
  - optionally `expand_unclustered_within`, a scene-unit distance. Unclustered centers that close to the selected centers are added. Centers of other named clusters are never added; they are only counted in the record.

Faint Gaussians (opacity at or below the cutoff) are attached to a derivative when their nearest opacity-kept center lies within the clustering radius and is selected.

Outputs:
- `clusters.json`: everything the run reports, unchanged by any selection. That covers settings, d1 and radius, spanning-tree statistics, the sweep, named clusters (size, centroid, bounding box), pairwise join radii (opacity-kept, and over all rows including faint Gaussians), merge events and pairwise direct gaps.
- `labels.npz`: `kept_rows`, `labels` (named cluster, or -1 for unclustered) and `components_at_radius` (every component).
- With derivatives:
  - `{name}.ply`: every property of the selected rows; the header is unchanged except the vertex count.
  - `{name}_rows.npy`: the selected source rows.
  - `selection.json`: per derivative, the basis, counts, sha256, expansion record and all-row isolation. Also the row overlap between derivatives.
- `input-manifest.json` and `run-record.json`: input and code hashes, effective settings, versions, command, per-step wall time and peak RSS, and output hashes. A failed run keeps a failure record.

Existing outputs are refused, `--validate-only` creates nothing, and the source PLY is re-hashed after the run.

Limits:
- Distances are in scale-free scene units.
- Single linkage merges plants that touch, or that a chain of floaters bridges. Read the join radii and gaps, not just the labels.
- Plants that were photographed sparsely fragment at a radius that suits the target. Their pieces join at larger radii, which the join radii show.
- Degenerate point sets, such as coplanar ones, are handled exactly up to `COMPLETE_GRAPH_LIMIT` points and refused above it.
- Non-plant objects attached to a plant (clips, tags, strings) stay in its cluster.

On the 74,180-row October 15 Welzo export, the command took about 8.5 s and peaked at 260 MiB under `ulimit -v 2500000`. Its M1 and neighbor derivatives were byte-identical to the separations made by hand on October 15. Run it under a hard memory cap anyway. `guarded_query_pairs` estimates the pair count and refuses an over-budget radius query for code that still needs one.

## Orient plants: up direction, plant axes and detached clusters

`orient_plants.py` computes three things and records them; it edits no derivative and needs no GPU:
1. **Up direction** of the scene, in the export's dataparser frame.
2. **Each plant's vertical axis** and horizontal radius profile.
3. **Detached-cluster assignment**: which plant, if any, a small detached cluster belongs to by vertical stacking.

**The ground plane is orthogonal to up by definition**, so detected soil is the primary up. The cameras' mean image-up assumes the photographer held the camera upright on average; it is only a fallback and a cross-check.

Every choice follows from the recorded settings. No visual judgment selects anything. `plant_orientation.DEFAULTS` owns the setting names and values. They are experimental defaults, not universal tuning; `DESIGN.md` explains each rule.

### Inputs

Copy `plant-orientation-example.json`. Relative paths resolve against the JSON file.
- `inputs.transforms` and `inputs.dataparser_transforms`: the COLMAP `transforms.json` and the trained run's `dataparser_transforms.json`. Every frame's camera counts.
- `inputs.cluster_report` (optional): a completed `cluster_plants.py` output folder. Its `labels.npz` and hashed source PLY supply clusters by ID. The report's opacity cutoff must equal this run's.
- `plants`: one entry per plant, each with a unique `name` and exactly one of:
  - `ply`, a per-plant derivative, optionally with `rows` (its `{name}_rows.npy`, source rows of the report's PLY);
  - `clusters`, IDs in the cluster report.
- `detached` (optional): a list in the same form, or `"auto"`. Auto takes every report cluster smaller than `detached_max_share` of the largest, excluding clusters a plant names or whose rows a plant's `rows` contain. A plant given as a PLY without `rows` cannot exclude its own clusters, so prefer an explicit list.
- `soil` (optional, but needed for the primary up; without it the run falls back to the camera mean image-up and says so): `colmap_sparse` (folder with `points3D.bin` and `images.bin`), `soil_masks`, `plant_masks` (production masks, subtracted from soil masks in memory), `frame_map` (frame stem to source name, as for organ analysis), optional `mask_extension` (default `.png`) and `base_plant` (required with more than one plant). `transforms.json` frames must carry `colmap_im_id`. A view without a soil mask is unknown, not a negative.
- `parameters`: overrides of `plant_orientation.DEFAULTS`.

### Run

```text
python pipeline/scripts/organ-analysis/orient_plants.py --config /path/to/orientation.json --output /path/to/new-orientation --validate-only
python pipeline/scripts/organ-analysis/orient_plants.py --config /path/to/orientation.json --output /path/to/new-orientation
```

The output directory must be new; `--validate-only` creates nothing. Exit code 0 means complete. Exit code 3 means no up exists: there is no usable soil plane and no camera mean image-up (no frames, or image-up vectors that cancel). `orientation.json` then records why and holds no axes or assignments. Exit code 2 means invalid input or a failed run, which keeps its failure record.

### Outputs

- `orientation.json` (`schema: plant-orientation`, `schema_version` 1):
  - `up_soil_plane`: the soil normal, status, refusal reason and diagnostics. These cover vote, neighborhood and inlier counts, base and sign references, angle to dataparser z, every base/refit `iterations` entry, and `converged`.
  - `up_camera_mean_image_up`: the mean direction, its spread (`angular_rms_deg`, maximum, median) and its angle to dataparser z.
  - `up`: the chosen vector and `method`. Its `status` is `primary` (soil), `fallback` (camera mean image-up) or `refused`. Also `angle_b_c_deg`, `up_disagreement` (above `disagreement_deg`; reported, never a switch) and the reason.
  - `plants.{name}.axis`: slices (height, point count, median horizontal position, 3D axis point, radius), `polyline`, `radius_profile`, `height_range` and `straightness`;
  - `detached.{name}`: median height and horizontal position, and per plant `r`, `R`, `ratio`, how the axis was looked up and the height-range test. Also the `assignment` (a plant name or `unassigned`) and its reason.
  - `cross_check_camera_mean_image_up`, when soil is primary: plant height ranges, straightness and detached assignments recomputed under the camera mean image-up. It is a cross-check only; no result above uses it.
  - Also the effective `parameters`, `horizontal_basis`, input and code hashes, repository revision and version.
- `input-manifest.json` and `run-record.json`, as for the other stages, with per-step wall time and peak RSS.

Limits:
- Up is a geometric estimate, not measured gravity. The soil plane is a bed-orientation cue: a tilted bed or plug makes it differ from gravity. A raised plug also separates bed level from the stem-emergence level. The camera mean image-up is only as upright as the photographer held the camera.
- Units are scale-free.
- A plant axis is a sequence of slice medians: leaves, tendrils and a neighbor's overlap all move it.
- An `unassigned` cluster is not evidence that it belongs to no plant.

## Plant asset package

After branch/endpoint review, `package_plant_asset.py` turns a reviewed organ result into one versioned, capture-scoped folder: `plant-asset/`. It reuses the review outputs and human decisions; it does not rerun masks, leaf lifting or skeletonization. `plant_asset.py` owns the contract (`SCHEMA`, `SCHEMA_VERSION`, `validate_asset`) and the single-leaf export; `DESIGN.md` explains the rules.

### Inputs

- `--review`: a `review_branches.py` (or `identify_branches.py`) output folder holding `organ-evidence.json`, `leaf_ids2.npz` and `branch-candidates.json` with endpoint IDs. The evidence point-label hash and graph digest are checked.
- `--endpoint-review` (optional): the endpoint CSV downloaded from that review viewer. Rows marked `false-branch` are rejected; every other endpoint keeps the root path it depends on. Graph digest, endpoint IDs, node IDs and coordinates must match.
- `--decisions`: a `plant-asset-decisions` JSON, version 1. Copy `plant-asset-decisions-example.json`. It binds to the source graph by `source_graph_sha256` and records:
  - `plant_id` (prefix of every organ ID) and `source` (who decided, and when);
  - `rejected_endpoints`: further endpoints to reject, each with a reason;
  - `leaves`: one entry per leaf cluster to package, keyed by cluster ID, with `role` (`true-leaf`, `cotyledon` or `combined-apex-region`) and `petiole`:
    - `{"source": "algorithm"}` keeps the skeleton's own leaf-tip path; zero-length fallbacks are refused;
    - `{"source": "endpoint", "endpoint_id": "b2-a"}` uses the retained graph path from that reviewed endpoint to the main axis;
    - `{"source": "inferred", "method": "nearest-point-mean-reference-angle"}` builds a straight inferred connection from the leaf point nearest the stem, at the mean initial angle of two extracted petioles. Optional `reference_leaf_ids` fixes those two; otherwise the two nearest along the stem are used. Optional `human_review` records acceptance;
    - `null` for the apex region, or with `petiole_unresolved_reason` for any other leaf;
    - optional `geometry_status` text (must start with `extracted`) and `biological_insertion_status` notes, plus `attachment.biological_insertion_status`;
  - optional `cotyledon_pairs`, `pair_insertions`, `review_summary` and `additional_limits`.

  Every skeleton leaf record needs a decision. Clusters without one, such as small unreviewed clusters, are not packaged.
- `--evidence` (repeatable): extra review records, copied into `source-evidence/` with hashes.

### Run

```text
python pipeline/scripts/organ-analysis/package_plant_asset.py --review /path/to/branch-review --endpoint-review /path/to/endpoint-review.csv --decisions /path/to/decisions.json --output /path/to/new-package --validate-only
python pipeline/scripts/organ-analysis/package_plant_asset.py --review /path/to/branch-review --endpoint-review /path/to/endpoint-review.csv --decisions /path/to/decisions.json --output /path/to/new-package
python pipeline/scripts/organ-analysis/validate_plant_asset.py /path/to/new-package/plant-asset
```

The output directory must be new. A failed run keeps `package-record.json` with the error. The same inputs give byte-identical `plant-asset/` files.

### Package contract (`plant-asset`, `schema_version` 1)

Treat the whole `plant-asset/` folder as the object. Every path in `plant.json` is a relative POSIX path inside it, so a copied folder stays valid; `source_files` keeps the absolute input paths only as provenance and is never read.

| Field | Content |
|---|---|
| `schema`, `schema_version` | `plant-asset`, `1`. The validator refuses any other version. |
| `plant_id`, `capture_id`, `id_scope` | Organ IDs are `{plant_id}:leaf:{cluster}`, `:petiole:`, `:attachment:`, `:internode:` and `{plant_id}:axis:main`. They are scoped to this capture and graph, not to other dates. |
| `source_graph_sha256` | `branches.graph_digest` of the unreviewed source graph the decisions refer to. |
| `units`, `metric_scale` | Always `scale_free_scene_units` and `null`. No metric scale is established. |
| `generator` | Command name and release version. |
| `point_labels` | `source-evidence/leaf_ids2.npz`: hash, point count and row-ID scope. Row IDs are rows of this opacity-filtered analysis array, not raw PLY vertex indices. |
| `reviewed_graph` | `skeleton-extracted-reviewed.json`: the retained graph, its hash and edge classification counts. Graph nodes are geometric samples, not phytomers. |
| `main_axis` | Polyline and source node IDs; `radius` is `null`. |
| `leaves[]` | `id`, `source_cluster_id`, `role`, `petiole_id` (or `null`), `point_geometry` (`leaves/leaf-{cluster}.npz`, hash, count, status) and `surface_mesh: null`. Each NPZ holds every source array for the whole cluster plus `source_point_row_ids`. |
| `petioles[]` | `polyline`, `attachment_id`, `classification`, `geometry_status`, `radius: null`, `inference`. Extracted: `extracted-petiole-candidate`, `source_node_ids` on the reviewed graph, source endpoint ID and node, blade-to-path gap. Inferred: `inferred-petiole`, `geometry_status: inferred`, `source_node_ids: null` and an `inference` record (method, references, angles, anchor row, confidence, optional human review). |
| `attachment_nodes[]` | Stem segment, fraction, `xyz` on the axis, arc length and `geometry_status`: `extracted-candidate` at a source node, or `inferred` with `source_node_id: null`. |
| `ordered_attachment_ids`, `internodes[]` | Attachments sorted by (arc length, cluster ID) and the intervals between neighbors. A zero interval is marked unresolved; intervals are not validated botanical internodes. |
| `cotyledon_pairs`, `pair_insertions`, `review_update` | Human annotations from the decisions; `null` or empty when absent. |
| `source_evidence[]`, `source_files` | Copied evidence (point labels, original evidence, branch report, decisions, CSV, extras) with hashes; absolute input paths with hashes. |
| `limits` | What the package does not establish. |

`validate_asset` checks all of these: every hash, each leaf NPZ against its source rows, extracted petioles on reviewed-graph edges, no extracted edge owned by two organs, attachments on the axis, and inferred geometry keeping its inferred labels. No status may claim measured geometry.

Beside `plant-asset/`, the output holds `viewer.html`, `checks.json`, `edge-classifications.csv`, `plant-asset-manifest.json` and `package-record.json` (command, code and input hashes).

### Select and export one leaf

In the viewer:

1. Open `viewer.html` from the package output in a browser. It works offline from a local file.
2. Under **Select one leaf**, click a leaf's points in the canvas (a drag still rotates) or choose the leaf in the list. The panel shows its role, point count and petiole status; the leaf and its petiole are highlighted, inferred petioles cyan dashed.
3. Click **Download selected leaf (JSON)**. The file `{plant_id}-leaf-{cluster}.json` holds the leaf record with its full point coordinates and original row IDs, its petiole polyline and status, and its attachment record. Other organs and the package are not changed.

From the command line, the same record goes to a new folder as `leaf.json` plus `points.npz`, a copy of the leaf NPZ with every attribute:

```text
python pipeline/scripts/organ-analysis/export_leaf.py --asset /path/to/new-package/plant-asset --leaf welzo-M1:leaf:2 --output /path/to/leaf-2-export
python pipeline/scripts/organ-analysis/export_leaf.py --reload /path/to/leaf-2-export
python pipeline/scripts/organ-analysis/export_leaf.py --reload /path/to/welzo-M1-leaf-5.json
```

`--leaf` takes a leaf ID or its cluster integer. Export validates the asset first and refuses an existing output. `plant_asset.load_leaf_export` reloads either form and returns the record plus `X` and `source_point_row_ids` (all arrays for a folder export). Reloading refuses a record whose petiole status disagrees with its inference record, so an inferred petiole cannot come back labeled extracted. The viewer download carries coordinates and row IDs only; recover other attributes from `point_labels` by row ID.

Limits: the package is the reviewed interpretation of one capture, not a measurement. It has no metric scale, radii, blade surfaces, cross-date IDs or validated insertion order. Inferred petioles are straight synthetic connections. The viewer shows sampled Gaussian centers and graph lines, not rendered splats.

## Verification and design

### Record endpoint review

Every graph endpoint has a unique source-scoped ID: a group's sorted terminal
node IDs become `b5-a`, `b5-b`, etc.; other degree-one nodes use `e{node_id}`.
Group IDs are retained across review and are not renumbered after rejection.

Branch-review viewers list every endpoint with category and optional notes.
Choices are not reviewed, unclear, leaf, false-branch, tip/apex, tendril,
stem/base, string/support and other. Earlier confirmed-false groups and retained
leaf-connected groups are prefilled from human dispositions; algorithmic
guesses are not marked as reviewed. Browser-local autosave checks graph and
source-node identity. Saving failure is visible and does not block CSV export.
Use the sticky Download endpoint review CSV button before returning results:
it includes all endpoints, coordinates, original decisions and current human
fields, even when the review is incomplete. Browser storage is profile-local
convenience, not a file the analysis agent can directly read. The annotations
never modify geometry, candidate classification or pruning by themselves.

```text
python pipeline/scripts/organ-analysis/review_branches.py --review /path/to/branch-review --decisions /path/to/decisions.json --output /path/to/new-reviewed-output
```

The decisions JSON contains `candidate_dispositions` mapping existing B labels
to `confirmed_false`, `retain_leaf_connected`, or `unclear`; an optional `source`
records review provenance. `upstream_candidate_labels` optionally flags the
path from the main axis to a group's attachment. Such paths may overlap saved
leaf routes, so they are review-only and never automatically pruned.

Retained groups are removed from the candidate set and kept in a separate
retained-branches table. The viewer labels every endpoint and marks incoming
review-only paths purple; these stay in cleanup previews when leaf support is
unresolved. `endpoints.csv` maps human-readable IDs back to source node IDs.
Optional `--export-confirmed-pruned` creates `skeleton-confirmed-pruned.json`
for explicitly confirmed-false, unguarded selections; it never overwrites the
source graph or alters leaf masks/point labels. The viewer's confirmed-cleanup
preview corresponds to these confirmed branches, while candidate preview may
also hide unconfirmed groups for inspection only.

`DESIGN.md` describes the algorithms and provenance. Run focused tests in the existing environment:

```text
python -m unittest discover -s pipeline/scripts/organ-analysis/tests -v
```

The real-data regression additionally compares every NPZ array, leaf membership JSON and full skeleton JSON against the preserved Welzo baseline in a fresh directory. That dataset is private/machine-local and is not part of the repository. No public claim of multi-plant or other-species validation is made.
