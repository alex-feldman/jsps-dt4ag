# Organ analysis (experimental, CPU)

Lift existing per-photo leaf instances onto an exported Gaussian splat, then extract a provisional leaf-associated skeleton. This is a separate post-reconstruction command. It does not retrain the model, generate masks, install packages, select MTG, or reconstruct missing tissue.

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

All numerical options are in the optional `parameters` object; `analyze_organs.DEFAULTS` owns their names and values. They originate in the exploratory Welzo baseline, not universally tuned settings. `apex_leaf` is always explicit for skeleton extraction; it is never assumed to be 4. Numeric IDs are local to a particular model and mask association result. Optional `annotations` maps cluster-ID strings to user-supplied JSON metadata; it does not move connections or turn an inference into measured geometry. Set `plots` to true for static 3D and photo-overlay sheets.

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
