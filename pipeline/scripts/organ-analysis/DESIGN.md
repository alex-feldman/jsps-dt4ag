# Organ-analysis implementation design

Status: experimental extraction of the reproducible Welzo leaf/skeleton baseline. Exact code hashes are recorded by each run; the annotated release tag identifies the source revision. This design is updated with the code. The README owns commands and input/output shape.

## Provenance and stages

The functions in `leaf_lift.py` and `skeleton.py` were extracted from the preserved `leaf_ids2.py` and confirmed-tip `stem_skel_tip4.py`. Before extraction, the original scripts reproduced their saved arrays, association JSON and full graph exactly. Repository extraction removes machine-specific paths, implicit module search paths, environment-variable settings, and the hardcoded tip 4. It preserves their numerical algorithm. This is a prototype method composition, not a novel validated skeletonizer.

`analyze_organs.spec_from` resolves explicit paths and typed parameters; `preflight` verifies raw camera/frame membership, source image/mask sizes, instance sidecars and file hashes. `geometry.Geometry.cameras` maps the COLMAP poses into the exported model's dataparser frame using saved transform, applied transform, scale and OpenGL-to-OpenCV axis conversion. The command refuses distortion beyond the verified shared-intrinsics `k1` model. `geometry.load_ply` reads only finite, scalar, binary-little-endian Gaussian vertex records and refuses truncated/unsupported data.

`leaf_lift.run` selects opacity-supported Gaussian centers, computes the existing scale/appearance approximations, splats depth discs for approximate visibility, and intersects visible projected centers with each leaf-instance mask. Cross-photo sets are greedily joined by overlap while preventing two instances from one photo sharing a cluster. An optional coverage merge also requires 3D contact. Votes associate centers with leaf clusters; plant-minus-leaf votes identify the candidate non-leaf network. Cluster numbering is by resulting size, so it must not be treated as a cross-run organ ID. Source-photo/instance memberships provide the association evidence.

`skeleton.run` filters non-leaf candidates for greenness, recovers some line-like leaf-labeled points with neighborhood PCA, links nearby points using `scipy.spatial.cKDTree`, and retains the largest component. SciPy sparse connected-components and Dijkstra provide connectivity and root-geodesic distance. Connected pieces in geodesic-distance bands become reduced graph nodes; NetworkX minimum spanning tree gives a tree. The root is the lowest retained dataparser-z point, not a measured stem-emergence point. A user-selected apex leaf anchors the main path. The last common node between each root-to-leaf path and the main path supplies a candidate insertion, and the remaining path supplies candidate petiole length. Geometry cannot distinguish a true attachment from a shortcut at crossing organs by itself.

`analyze_organs.evidence` wraps raw outputs in `organ-evidence-v1`, with model/point/graph ID scope, camera provenance and user annotations separate from candidate geometry. The command checks a lossless JSON reload and validates graph references and the label-array hash before completing. Source hashes are rechecked afterward. Failed work stays in its fresh output directory with an explicit failure record; existing output is refused.

`viewer.write_viewer` samples centers deterministically by array position, embeds the graph and display sample directly into HTML, and uses JavaScript yaw/pitch rotation plus orthographic Canvas 2D projection. Edges are drawn as lines, points depth-sorted for display, and insertion positions labeled. This is an inspection aid, not scientific occlusion or splat rendering. Source data stay at full precision in NPZ/JSON; no displayed rotation changes them.

## Quantities and software

| Output | Source and meaning |
|---|---|
| Leaf counts and votes | `leaf_lift.run` association/label arrays at explicit `DEFAULTS` or config overrides; not ground-truth leaf recall |
| Skeleton node/edge counts | `skeleton.run` reduced tree, checked against raw JSON; not biological node counts |
| Insertion distance and petiole length | NetworkX weighted path lengths in scale-free scene units; candidate geometry only |
| File hashes and equality checks | SHA-256 and exact array/parsed-JSON equality; software provenance/reproducibility, not accuracy |

The baseline tip-first base rule and GREEN, TIPGAP and TOUCH/contact settings
were developed or checked with agent-viewed pilot overlays before extraction
into this package. They do not invoke agent vision at runtime. The subsequent
Welzo sensitivity work shows stability relative to the reviewed pilot, not
independent anatomical ground truth. The portable TOUCH investigation has not
changed repository defaults. Any new morphology rule remains experimental and
must record its choices and failures rather than tune to a reviewed answer.

The CPU path uses NumPy, SciPy, NetworkX and Pillow; optional plots use Matplotlib. Versions are recorded at run time. No SAM, Torch, Nerfstudio or CUDA model is loaded by this command. Reconstruction and mask generation remain external inputs. The algorithm's library calls and parameters are owned by the code, not duplicated as universal numeric recommendations here.

## Checks and changing the command

### Leaf-free terminal-subtree candidate milestone

`identify_branches.py` consumes a completed organ-analysis output without rerunning reconstruction, masks, leaf lifting or skeletonization. It verifies the recorded graph/point evidence hashes and writes a fresh additive review directory. `branches.identify_candidates` validates a tree, recovers the saved main path by exact node coordinates, and protects the union of root-to-main-tip and root-to-supported-leaf paths. For a `tip` association it requires the nearest terminal to the saved blade base to agree with the recorded gap; for a `nearest` fallback it protects the nearest graph path conservatively as unresolved leaf-base adjacency. Every terminal within the recorded TIPGAP of any supported leaf's source centers is also protected. This extra guard is geometric support, not a new anatomical identification.

The remaining edges after protecting the main axis and saved leaf-tip associations form leaf-free terminal subtrees, each connected to the protected tree at one attachment. Proximity and nearest-base guards do not hide candidates: those subtrees remain in the review but are marked guarded and ineligible for programmatic pruning. A branching subtree with several leaf-free terminals is one candidate, so shared segments are not counted or removed multiple times. Candidate identity is derived from sorted source edge IDs; short B1/B2 labels are review aids. JSON/CSV preserve source node/edge IDs, attachment, terminal IDs, scale-free network length, support guards and blank human dispositions. Length is the sum of source Euclidean edge lengths; candidate count is the number of remaining connected subtrees, not a count of confirmed false organs.

The HTML highlights candidates in amber, labels their terminals, and offers a display-only preview without those edges. `branches.prune_candidates` is a separate pure function returning a derivative graph for an explicit candidate selection; detection and preview never call it or delete data. The first real milestone delivers identification only. Tendrils, missed blades, hidden petioles and graph artifacts may still be candidates; protected paths can also be wrong. No classifier accuracy or automatic biological disposition is claimed.

Regression fixtures cover leaf-carrying shared axes, leaf proximity, nearest-base uncertainty, merged multi-terminal candidate subtrees, graph/report identity, preserved originals and pure-function pruning. A real cucumber review checks graph/NPZ hashes and inspects the candidate overlay alongside the user's marked screenshot. The screenshot's boxes are human visual guidance, not calibrated 3D node selections or truth segmentation.

`branches.endpoint_records` assigns every degree-one node a unique ID, using
stable source-node order for group suffixes, including groups later retained.
`apply_review` copies the report, records explicit user dispositions and moves
leaf-connected groups out of the candidate set without renumbering. Incoming
review paths are traced back to the last shared main-axis node. They remain
separate from leaf-free candidates and are not deletion-authorized when they
intersect the recorded leaf support. `review_branches.py` writes another fresh
review directory and can export a separate confirmed-only derivative graph;
raw source evidence remains intact. No source-photo or Gaussian-rendering
feature is part of this update.

The endpoint form keeps browser answers separate from raw graph fields. It
prefills only prior explicit human dispositions, whitelists stored categories,
and verifies graph plus endpoint source-node identity on reload. Notes and
categories autosave to browser localStorage with a visible failure status. CSV
export includes every source endpoint and original raw coordinates/decisions,
with CSV quoting for commas and quotes; partial review is allowed. Browser
checks exercise actual form events, reload, download Blob content, storage
failure and graph invariance. Test answers are cleared afterward.

Focused tests cover malformed/truncated PLY, camera-frame mapping, unsupported distortion, missing/wrong-size masks, explicit apex requirement, output protection and reused-label integrity. A synthetic tiny scene exercises the public command, leaf-only/skeleton-stage flow and checked failure path. The real Welzo run must reproduce the baseline arrays, leaf memberships and graph; inspect a generated overlay/viewer in a real browser. A raw hand-check can inspect graph endpoints in `skeleton.json` and source-instance membership in `leaf_ids2.json`, independently of the wrapper's status.

Any new camera model, visibility rule, threshold, topology repair or organ heuristic changes the analysis and requires new regression/evidence. Do not tune per reconstruction arm or silently replace source scripts inside a frozen experiment. New datasets need their own review, apex choice and parameter checks. Metric scale, measured radii, missing petiole centerlines, MTG serialization and Helios generation are outside this command's claims.

## Plant clustering and separation

`plant_clusters.py` (the library) and `cluster_plants.py` (the command) separate plants in a multi-plant export before organ analysis. The README owns the command, config and outputs. `plant_clusters.DEFAULTS` owns the settings.

### Why it is built this way

The October 2026 Welzo M1 separation clustered centers with a radius graph: `cKDTree.query_pairs` at r, then connected components. That is affordable when r is a few times the point spacing. A later connection check asked whether any chain of points joined M1 to its neighbor across gaps up to 0.4 scene units. It called `query_pairs` at r = 0.4 over all 74,180 rows. That radius is about 100 times the spacing and comparable to the plant itself. Inside such a radius the pair count grows with the square of the number of points: an exact recount gives 383 million pairs, 6.1 GB as an int64 list. `query_pairs` grows its buffer by doubling and then copies it out, and the process reached 11.7 to 14.5 GB RSS. Windows free memory fell to 730 MiB, and a concurrent GPU job's host guard stopped that job. The lesson is twofold. Never materialize all pairs within a radius that is large relative to the spacing. And a memory budget written in a brief is not enforcement; a hard cap is.

### Method

1. Exact duplicate centers are merged first. This is exact for single linkage, because duplicates join at distance 0.
2. The unique edges of a 3D `scipy.spatial.Delaunay` triangulation (about 7 per point) form the candidate graph.
   - The Euclidean minimum spanning tree (MST) is a subgraph of the Delaunay triangulation, so no short link is lost.
   - Points qhull leaves outside every simplex get k-NN fallback edges.
   - Inputs with four or fewer points, and degenerate inputs (for example coplanar) up to `COMPLETE_GRAPH_LIMIT` points, use the complete graph, which is also exact. Larger degenerate inputs are refused, not joggled: joggling would triangulate perturbed points.
3. `scipy.sparse.csgraph.minimum_spanning_tree` gives N - 1 edges. Memory is O(N) throughout.

Every question is then answered from that one tree:
- **Clusters at radius r.** Deleting tree edges longer than r gives exactly the connected components of the radius-r graph. A radius sweep therefore only thresholds one array.
- **Join radius.** `join_heights` replays the tree edges in increasing length with union-find, tracking every row of every group. The join radius of two groups is the first length at which a row of one and a row of the other share a component. That is the minimax chain distance between them, exact at any radius. Computing it over all rows includes chains through faint Gaussians.
- **Direct gap.** `pairwise_gaps` builds a `cKDTree` per cluster and queries the other clusters with k = 1.
- **Guard.** `guarded_query_pairs` remains for callers that still need a pair list. It estimates the pair count from a 1,000-point sample (mean neighbor count times N / 2) and raises `PairBudgetExceeded` above a budget (default 4 million).

The radius is a multiple of the median nearest-neighbor spacing d1, because scenes have no metric scale. Cluster numbering is by decreasing size and carries no identity meaning.

Derivatives are explicit selections:
- **What a selection can name.** It can name cluster IDs and/or opacity-kept source rows. It can optionally add unclustered centers within a stated distance of the selected centers.
- **Faint Gaussians.** A faint Gaussian (sigmoid(opacity) at or below the cutoff) is attached when its nearest opacity-kept center is within the clustering radius and selected. That is the rule of the October 15 derivatives.
- **How rows are copied.** Rows are copied by index from a read-only memmap with the header's dtype, so every property is kept. The header is unchanged except the vertex count. The written file is re-read with `geometry.load_ply` and compared byte for byte with the source rows.
- **Identity stays outside.** The command records each selection's `basis` but never forms one. Identity remains a human or evidence decision.

### Provenance and verification

The method was proposed after the incident and benchmarked on 2026-10-16 against the October 15 Welzo M1 separation. It ran as a standalone `lean_cluster.py` under `ulimit -v 2500000`. It reproduced all 116 compared reference values with no tuning:
- d1;
- the radius sweep;
- all 4,795 components at 6 x d1;
- the ten named clusters, as a partition and in their numbering;
- every gap of M1 (c0) to the other clusters, opacity-kept and over all rows;
- the neighbor's internal merge radii.

It peaked at 236 MiB. The original radius-graph clustering step, re-measured under the same cap, took 1,451 MiB, and its forest was exact only up to 0.0777. The benchmark also turned earlier bounds into exact values. M1 joins its neighbor at 0.3107 over all rows, equal to the direct gap. The neighbor's c5 joins at 0.10147 opacity-kept. The guard's estimate was within 0.99 to 1.06 of exact `count_neighbors` counts.

This module is that code, with three changes:
- capture-specific constants removed and settings made explicit;
- the derivative writer of the October 15 scripts added;
- `join_heights` changed to track every row of a group. The benchmark tracked each group by its first row, which is exact only for joins at or above the clustering radius.

The release check ran `cluster_plants.py` on the same export into a fresh directory, under the same cap. Of 215 comparisons with the benchmark and the October 15 files, 214 matched exactly:
- sweep, partitions and numbering, spanning-tree statistics;
- all opacity-kept gaps and join radii;
- 44 of 45 all-row join radii;
- the all-row gaps from M1's component.

The M1 (c0 + c6) and neighbor (seven clusters plus unclustered centers within 0.3) derivatives had row sets and file bytes identical to the October 15 files. The one difference is the all-row c1-c9 join: 0.01412 here against the benchmark's 0.02078. A direct radius graph, which does not use the tree, confirmed 0.01412: c1 and c9 connect at that radius and not just below it, while their first rows do not. The benchmark value was the representative-row approximation. The run peaked at 260 MiB in about 8.5 s.

Synthetic tests check the following:
- components against a brute-force radius graph;
- join radii against brute-force minimax distances;
- known gaps and join radii, including a chain of faint points and the degenerate fallback;
- duplicate merging;
- the guard's refusal and its exactness at full sampling;
- byte-identical row copies with mixed property types;
- refusal of existing outputs, foreign source hashes, size mismatches, faint rows given as kept rows, unknown clusters and selections without a basis.

### Limits and changing it

Single linkage chains: two plants that touch, or a line of floaters between them, merge at a small radius. The join radii and direct gaps are reported so that this is visible rather than hidden in a label. A sparsely photographed plant fragments at a radius suited to the walked-around target. HDBSCAN was tried on the Welzo neighbor as a density-based second opinion. It agreed with the fixed-radius structure and does not rescue such fragmentation, so it is not part of the command. A different linkage, radius rule, opacity rule or faint-attachment rule changes the separation and needs new regression evidence like the above.

## Plant orientation: up, plant axes and detached clusters

`plant_orientation.py` (the library) and `orient_plants.py` (the command) estimate the scene's up direction, each plant's vertical axis and radius profile, and whether a small detached cluster stacks inside one plant's column. Alex decided on 2026-10-17 that up and plant axes belong in the pipeline whether or not detached clusters exist, so that later stages (skeleton root, apex rules) have a principled vertical. The README owns the command, config and outputs. `plant_orientation.DEFAULTS` owns the settings.

**The numeric settings are experimental defaults.** They were set on 2026-10-17 from the Welzo M1 captures and are not tuned or validated across captures. Each run records the values it used. **No agent visual judgment selects anything here.** Every status, choice and assignment follows from code and recorded settings; images and overlays are not read. The stage reports and records; like plant clustering, it never edits a derivative.

**Principle (Alex, 2026-10-17).** The ground plane is orthogonal to up by definition, so the detected soil plane (B) is the primary up. The cameras' mean image-up (C) assumes the camera was held upright on average, so it is not assumed. It serves as a fallback when no usable soil plane exists, and as a cross-check otherwise. A first design used the normal of the camera-center plane ("camera ring"). That design was removed after it refused both Welzo captures, whose cameras spread over a dome; see "Checks" below.

### Up, method C: camera mean image-up (fallback and cross-check)

Image-up vectors come from `transforms.json` mapped into the dataparser frame. The mapping is the one `Geometry.cameras` uses; `geometry.dataparser_frame` and `geometry.camera_to_world` were factored out of it unchanged. Image up is minus the OpenCV camera y axis.

C is the normalized mean of the unit image-up vectors. It never refuses because of camera layout. Its quality figure is the spread of the individual vectors about the mean: angular RMS, maximum and median in degrees. It is unavailable only with no frames or with vectors that cancel. With nerfstudio's `orientation_method: up`, C equals dataparser +z by construction, and it did on both Welzo captures.

### Up, method B: soil plane (primary)

This is the October 2 pilot's method (`clutter_lift.py` in the Welzo clutter investigation, `posthoc-clutter-lifting-design.md`), ported:
- **Coordinates.** Raw COLMAP `points3D.bin` coordinates are mapped with the saved dataparser transform, then scaled including translation. Raw points lack `applied_transform`, which the saved transform already contains.
- **Votes.** Each point is voted on at its own feature-track observations (`images.bin` keypoints), not at every pixel it projects behind. A point is soil when it has at least `soil_min_hits` (2) hits and a hit share of at least `soil_hit_share` (0.6) among tracked views that have a soil mask. A view votes at most once per point.
- **Masks.** Production plant masks are subtracted from soil masks in memory; the originals stay untouched. SAM2-HQ filled one pilot soil mask across leaves.
- **Neighborhood.** Points within `soil_local_radius` (0.3) of a base estimate are kept, and exact duplicate coordinates are merged for fitting only.
- **Fit.** Deterministic three-point RANSAC (seed 42, 1,000 trials, tolerance 0.004), then SVD of the largest consensus. At least 30 candidates and 30 inliers are required.

The command differs from the pilot in three places:
- **Base.** The base estimate is the axis point of the lowest slice (at least `min_slice_points` centers) of `base_plant`. The pilot used its skeleton root.
- **Iteration, so B's result does not depend on C.** C only bootstraps the neighborhood:
  1. The first base is taken along C. The command selects, fits and signs the plane.
  2. The base is recomputed along that soil normal, and the neighborhood is reselected and refitted.
  3. This repeats until the deduplicated inlier coordinates equal the previous iteration's, at most `max_soil_iterations` (5) fits.

  Every iteration is recorded: base, the direction used, counts, normal, angle to the previous normal and to C, and whether the inlier set changed. `converged` says whether the set stopped changing. Without convergence the last fit is used and a note says so. A synthetic test checks that two different bootstrap directions converge to the same normal.
- **Sign.** The normal points toward the base plant's mean opacity-kept center. The pilot pointed it toward leaf cluster 4.

The binary readers are local, so neither nerfstudio nor torch is imported.

The soil plane is a bed-orientation cue, and it can differ from gravity when the bed or plug is tilted. On October 2 the rockwool plug around the stem was excluded by the soil prompt, so bed level and stem-emergence level are different references.

### Choice

- If B exists and meets its minimums, B is used (`status: primary`). The B-versus-C angle is reported. Above `disagreement_deg` (10), `up_disagreement` is set, but the choice never switches away from B.
- If B is absent, refused or not run, C is used (`status: fallback`), and the reason names B's state.
- If neither exists, no up is chosen. The command writes the refusal and exits with code 3; it does not fall back to dataparser z.
- When B is primary, plant axes and detached assignments are also computed under C and written as `cross_check_camera_mean_image_up`. No result uses them.

### Plant axis

Opacity-kept centers (sigmoid(opacity) above `opacity_cutoff`) are projected on up. They are cut into `slice`-thick (0.02) bins starting at the lowest center. A bin with at least `min_slice_points` (30) centers gives:
- the component-wise median of its horizontal coordinates, in a deterministic basis perpendicular to up;
- the `radius_percentile` (95th) distance from that median, the radius.

The axis polyline places each median at its bin's center height. Bins below the minimum are skipped and counted as gaps. The height range runs from the lowest kept bin's floor to the highest kept bin's ceiling. Straightness is the largest horizontal deviation of the bin medians from a least-squares line in height; the RMS deviation and the line's tilt from up are reported as well.

### Detached-cluster assignment

For each detached cluster:
1. Take its median height along up and its median horizontal position.
2. For every plant, look up the axis position and radius R at that height: linear interpolation between bin centers, or the nearest bin, flagged, outside them. Compute r, the horizontal distance from the cluster to the axis.
3. Assign to plant P when all three hold:
   - r_P / R_P is at most `assign_max_ratio` (1.0);
   - the cluster's height is within P's height range widened by `height_margin_slices` (1) slices;
   - every other plant Q has r_Q / R_Q of at least `other_min_ratio` (2.0).

Otherwise the cluster is `unassigned`, and so is any cluster when some plant has no axis. Every ratio, lookup and test result is written.

### Checks and changing it

The synthetic tests cover:
- the camera mean image-up of a tilted ring and its spread, and refusal only for an empty or cancelling set;
- a tilted synthetic soil plane with outliers, recovered and signed toward the plant;
- the vote rule and its refusals;
- the binary COLMAP readers and one-vote-per-view track voting;
- soil iteration from a 25-degree-off bootstrap: convergence, the iteration record, and independence from the bootstrap direction, plus the cap and a refused fit;
- the choice rule: soil primary, disagreement reported and not switched, fallback to C with the reason, and no up;
- a vertical cylinder giving a straight axis with the expected radius;
- a blob inside one plant's column assigned, and blobs between plants or above them unassigned;
- the command end to end:
  - soil primary with the C cross-check;
  - a 20-degree soil-versus-camera disagreement that keeps soil;
  - the fallback to C;
  - a no-up refusal (exit 3);
  - config refusals, including the removed `prefer_soil`.

**Pilot regression (2026-10-17).** The port was given the pilot's base (skeleton root) and sign reference (leaf cluster 4) on the October 2 capture. It reproduced the pilot's `analysis-full` outputs exactly:
- point IDs, mapped coordinates (zero difference), hits, seen counts, and the soil and local flags;
- the deduplicated fit points and the inlier set, with counts 933, 479, 467 and 234;
- the normal (0.137, -0.359, 0.923), 22.6 degrees from dataparser z, and the root's 0.029 offset above the plane.

**Removed camera-ring method (2026-10-17).** The first design's camera-ring normal refused both Welzo captures at its planarity limit of 0.25:
- October 2: planarity 0.59 over 24 cameras;
- October 15: planarity 0.43 over 61 cameras.

In both, the cameras spread over a dome. Alex then set the ground-plane principle above.

**Welzo real-data checks (2026-10-17, commit 4b7d4e2).** The October 2 run used the four pilot soil masks.

| Quantity | October 2 | October 15 |
|---|---|---|
| Soil masks | 4 pilot masks | 60 of 61 views; one view had no detection and counts as unknown |
| Soil-supported points | 933 | 3,716 |
| Iterations to convergence | 3 | 3 |
| Final local / deduplicated / inlier points | 491 / 478 / 243 | 832 / 775 / 304 |
| Soil normal | (0.092, -0.349, 0.933) | (0.070, -0.057, 0.996) |
| Soil normal to dataparser z (= C) | 21.1 degrees, so `up_disagreement` is set | 5.2 degrees |
| C angular spread (RMS) | 55 degrees | 23 degrees |

The October 15 masks were made with the pilot's prompt `soil` and threshold 0.50, by the low-memory hybrid runner.

The October 15 check also gives the detached cluster c6 (469 centers, at M1's base):
- **Under the soil up:** assigned to M1. r/R is 0.544 for M1 and 2.606 for the neighbor 2M; 2M's lookup is its nearest slice, flagged.
- **Under C (cross-check only):** unassigned. r/R is 0.887 for M1 and 1.408 for 2M, below the 2.0 the rule requires.

The assignment therefore depends on the up estimate.

The first soil run on October 15 failed under a 2.4 GiB address-space cap, because it held all 60 masks at once. Votes now stream one mask at a time, and the pilot regression still matches exactly.

Changing a vote rule, threshold, slice rule, radius percentile or assignment ratio changes results and needs new regression evidence and a new output directory.

## Computed base and conservative apex candidates

The opt-in integration lives in `computed_axis.orientation` (artifact validation)
and `computed_axis.candidates` (rankings, ties, and decision). `skeleton.run`
uses the supplied up to select the lowest retained network point as root and
otherwise retains its existing graph construction and leaf associations. The
base is a candidate network root; the anatomical stem base is unvalidated.
It is not a measured stem base or soil-plane projection;
the report records its point-label array row, coordinates, height and signed
plane distance when available. Per-height slice medians from plant orientation
are not a traced stem and are not used to claim one here.

Candidate agreement is deliberately conservative. Supported leaf median height,
associated endpoint geodesic, and terminal height are separate diagnostics.
Their unique extrema must agree on one terminal. The owning function declares
the tolerance, bounded below by the existing skeleton BIN width in scene units
and protected against floating precision loss at larger scene scales. Equal
ranks remain ambiguity rather than an ID
tie-break. Disagreement preserves a candidate report and refuses success. A
supplied leaf override proceeds with its decision basis and the disagreement
recorded. Candidate agreement still cannot distinguish primary stem from
strings, tendrils, lateral branches or missed tissue in general.

Orientation enters skeleton provenance, not leaf parameters. A completed
artifact and run record bind capture identity and camera-frame source digests;
both and all recorded orientation sources are hashed. Invalid or changed
orientation is rejected before output creation. Soil up remains primary, with
camera fallback available only by explicit config. Failure records attempt
source-hash verification even when candidate resolution refuses. The legacy
path with no orientation remains unchanged.

Orientation is deliberately scene-level and reusable for per-plant derivatives
in its capture and coordinate frame. The soil fit's base-plant reference does
not bind or establish the identity of the analyzed PLY. Confirming the target
derivative remains an explicit upstream responsibility; no new plant-identity
schema or soil-distance gate is introduced. Signed plane distance is retained
as a diagnostic, not evidence that the root is an anatomical base.

The design preference is anatomy-informed base/apex decisions and eventually
node-first plant semantics with leaves attached. Botanical nodes and phytomers
remain distinct from current geometric graph samples and leaf segmentation
labels. Existing coordinates, topology and provenance are retained for future
mapping. Considering established MTG/OpenAlea or Helios formats is a preference,
not an adoption decision or a requirement for this feature. Alex approved stable anatomical IDs, separate base-to-apex ranks starting at
zero when the basal node is known, and explicit internode endpoints with
below-node phytomer grouping. Zero-to-many leaf attachments are provisionally
allowed for early development. Species-specific botanical, morphological and
anatomical fidelity remains the future target; a permissive container does not
assert actual leaf/branch cardinality. An axillary bud and
a developed branch need distinct treatment in that investigation, alongside
petioles and observation loss. The partial semantic identity prototype below adds
source-bound opaque identities and candidate queries. It accepts no new real
anatomical nodes or internodes, and implements no backend adapter.

MTG/Helios remains the initial candidate for evaluating botanical semantics and
derived geometry. OpenUSD is an additional scene/geometry exchange candidate.
Other options remain open. No representation or backend is selected. Evaluate identity,
uncertainty, topology, geometry preservation, round trips and implementation cost
before adoption. Automatic base/apex recovery does not block the reviewed-plant
partial semantic prototype described below. Backend adoption remains design direction.

### Future plant-plus-environment direction

The longer-term compass includes plant structure coupled with local environment
and physiology, including canopy temperature and light. This is outside the
current plant-representation feature. A virtual spatial field, point-associated
properties, surface or organ properties and regional descriptions are alternatives
for future FSPM literature work. No environmental grid, sampling resolution,
required property schema or physiological model is selected here. Preserve the
distinction between observations, inferred geometry and future model output;
do not turn this direction into a requirement for current extraction.

### Real-data boundary check, 2026-10-17

The unchanged explicit-apex path reproduced the October 2 saved skeleton
exactly (104 nodes, 103 edges). With soil-primary orientation, automatic
selection refused: highest median leaf was cluster 4 and farthest associated
leaf endpoint was cluster 1. The initial numerical-tolerance trial ranked
terminal sample 96 highest; the final BIN-resolution trial retained samples
91 and 96 as tied highest terminals. The existing
human override for cluster 4 completed with this disagreement retained.
October 15 also refused: highest median leaf was 11, farthest associated
leaves tied at 7, 11, 12 and 20, and highest terminal was sample 201.

The October 15 lowest retained network point was 0.4103 scene units above the
soil plane. This is a warning about the connected-network base estimate, not
a measured emergence point. Retaining the largest network component can omit
lower tissue. Component selection, mask/green filtering and scale-dependent
network linking require investigation before claiming anatomical recovery;
these checks did not change them. Disagreement and unchanged source hashes
were recorded even on refusal. Initial trials predate the release version bump; the final review-r2 trials
use alpha.8 code. All retain their actual source/code provenance and dirty
worktree status. The final legacy result remains exactly equal to the original.

The candidate signals share input geometry and associations; agreement is an
internal consistency test, not independent anatomical validation. The automatic
October 2 acceptance goal remains unmet, even though refusal is correct software
behavior. This release is candidate diagnostics, not a finished apex detector.

## Clicked image points and ruler scale

Origin: the October 2 Welzo M1 export has no metric scale, and the only ruler is a tape measure in one photograph (frame_00024), so its marks cannot be triangulated. The design keeps human input to clicks (procedure 6b) and makes every later step a recorded rule.

- **Why COLMAP frame first.** Clicked pixels refer to COLMAP's cameras. Working in that frame and mapping with `colmap_to_dataparser` avoids the transforms.json route, where using the saved dataparser transform directly is the applied_transform trap. Camera centers through both routes must agree; this also checks the frame-to-image mapping.
- **Projection.** Verified against the model: the reprojection of every COLMAP point equals COLMAP's stored mean error to 1e-12 px. This model's own keypoint residuals reach 4 px, so the multi-view flag is set at 4 px (COLMAP's default filter).
- **Single-view support.** Only points observed in the clicked image, so support is what that camera saw near the click. The ladder exists because keypoints are sparse (93% of frame_00024 has fewer than 4 within 40 px). The plane is a RANSAC consensus with a tolerance relative to depth; mixed surfaces, grazing planes and extrapolated hits are flagged, not hidden. The along-ray sigma is a prediction sigma; on held-out real points its ratio to the actual depth error has median 0.8 and p90 2.5.
- **Surface offset.** A tape resting on raised bed edges is above the soil it crosses. The offset is Alex's stated height, applied along the plane normal toward the camera; because it is metric, it is converted with the current estimate and iterated. Scale sensitivity on October 2: about 1.2% per cm of offset.
- **Scale combination.** All pairs, weighted median by known length: robust to one bad mark (synthetic: a mark misplaced by 4 cm moved the estimate by less than 1%), and longer baselines dominate. Pairs are not independent, so the spread is a check, not a standard error.
- **Checks.** Synthetic cameras with OPENCV distortion (exact recovery, calibrated covariance, refusals, suspended-mark behavior, offset recovery); real-camera regression with COLMAP keypoints as fake clicks (multi-view median error 2e-11 scene units; single-view held-out median 0.16% of depth). Investigation: `/home/alex/data/investigations/click-points-20261018/`.

## Plant asset package and single-leaf export

`plant_asset.py` owns the versioned contract and its validator; `package_plant_asset.py` emits it; `export_leaf.py` and the viewer's leaf control export one leaf. The README owns commands, the decisions file and the field table.

### Origin

The contract generalizes the reviewed Welzo M1 (October 2 capture) pilot `plant-asset/`. That pilot was assembled by four capture-specific scripts beside its review outputs: endpoint-CSV cleanup, a leaf-1 correction, inferred petioles plus packaging, and a final leaf-2 correction. Their decisions were hard-coded, such as node IDs, the endpoint for each leaf, and which leaves to infer. The emitter reads those decisions from a `plant-asset-decisions` file bound to the source graph digest, and reuses the scripts' logic:

- Endpoint review retains the union of root paths to every endpoint not explicitly rejected, so rejecting a tip removes only edges no retained endpoint needs.
- An algorithm petiole is the reviewed-graph path from the terminal nearest the saved blade base back to the main axis. The saved gap must still match, so a tip removed by review is refused rather than silently replaced.
- An endpoint petiole walks from the reviewed endpoint to the first main-axis node, which is the insertion.
- The inference is the pilot's construction unchanged. It takes the leaf point nearest the stem, averages the initial angles of two extracted petioles against centered root-to-apex tangents, and solves for an attachment on the existing stem with an exact angle, choosing the solution nearest along the stem. Stem points never move. When the decisions name the reference leaves, they are used; otherwise the two nearest extracted petioles along the stem.

Regenerating the Welzo pilot from its review folder, endpoint CSV and transcribed decisions reproduced every leaf array and row ID, the 63-node reviewed graph, all petiole and attachment geometry with zero float deviation, the six internode lengths and the 62-edge classification. The intentional differences from the pilot are these. The schema `welzo-organ-asset-pilot-v1` became `plant-asset` with `schema_version` 1. `unassigned_extracted_graph` became `reviewed_graph`, with a hash and edge counts. `generator` and `point_labels` were added. Every extracted petiole now records its source endpoint and blade gap, where the pilot did so only for leaf 2. Inferred petioles add `source_node_ids: null`, `method_id`, `reference_selection` and the recorded human review. Records are listed by cluster ID, and the equal-arc tie between leaves 2 and 6 is ordered by cluster ID. That swaps them in `ordered_attachment_ids` and in the endpoints of internodes 2 to 4; both attachments are the same point. Leaf NPZ bytes differ because of the deterministic writer, while their arrays are identical. The branch report, the decisions file and the earlier rejection record were added to the evidence. The graph records for endpoint-sourced leaves use one generic `base_rule` and `review_status`.

### Rules and their reasons

- **Whole folder, relative paths.** Every path the contract reads is a relative POSIX path that must resolve inside the folder, so a copied folder validates unchanged. Absolute input paths are kept only as `source_files` provenance and are never read.
- **Extracted versus inferred.** An inferred petiole must carry `inferred-petiole`, `geometry_status: inferred`, an inferred attachment, no source nodes and an inference record whose anchor is the cited leaf row. An extracted one must lie on reviewed-graph edges and sit on a cited stem node. No status may claim measurement; "not measured" disclaimers are allowed. Leaf exports repeat the check on reload, so an inferred petiole cannot be relabeled extracted.
- **Exclusive edges.** Each reviewed edge belongs to the main axis, one extracted petiole, or nothing. A shared extracted edge is refused rather than counted twice.
- **Row IDs.** Leaf NPZs hold the complete cluster from the opacity-filtered analysis array, with `source_point_row_ids` into `point_labels`. They are not raw PLY vertex indices.
- **Deterministic files.** Leaf NPZs are written with fixed entry order and zip timestamps, and JSON with fixed formatting, so the same inputs give byte-identical packages.
- **Ties.** Attachments with equal arc length are ordered by cluster ID. This order carries no biological meaning, and the zero interval between them is marked unresolved. The pilot happened to order a tie by packaging sequence.
- **Scale and unknowns.** Units stay scale-free; `metric_scale`, radii and surface meshes are `null` in version 1.

A new field may be added only with a schema version bump and a validator update. So may a new role, petiole source or inference method, or any change to these rules.

### Single-leaf export

`leaf_export_document` copies the leaf, petiole and attachment records unchanged and adds the source asset identity, including the `plant.json` hash, units and limits. The command-line export writes `leaf.json` and a byte copy of the leaf NPZ. The viewer embeds the same record for every leaf, with `X` and row IDs inline, and downloads it as JSON. `load_leaf_export` accepts both forms. Exporting never writes into the asset.

The viewer stays one offline HTML file. Clicking selects the frontmost displayed center within 8 px, and a press that moves more than 4 px is a rotation instead. The highlighted petiole is drawn from the embedded export record, so what is shown is what is downloaded.

Synthetic tests cover the emitter and its refusals, byte-identical repeats, validator refusals, export and reload, and the viewer's embedded payload. The Welzo pilot was additionally regenerated and compared, and the viewer was driven in headless Chrome.

## Partial semantic identity prototype

`plant_semantics.py` owns the separately versioned semantic/registry documents,
source-derived validation, queries and generic accepted-topology consistency
checks. The README owns usage and the format's limits. This is an additive
experiment, not a permanent source-of-truth or backend selection. The source
schema-1 document and every source file stay byte-identical; no schema-1 rule
changes. Complete source-tree hashes bind registry reuse and copied evidence.

Opaque identities are allocated once and saved. Source reference keys locate
observations within an exact hash-bound asset; they are not anatomical IDs.
Fresh registries start fresh identities. Changed sources are refused rather than
matched by labels or geometry. The writer uses deterministic UTF-8 sorted JSON
without timestamps or output paths; generator version is recorded provenance,
not a requirement to validate with that same software release.

Source attachment observations remain separate unresolved candidates. Equal
source arcs form geometric query groups; shared samples annotate ambiguity,
never merge anatomical nodes. Combined apex tissue remains a region. The
source-derived builder emits no accepted anatomy and null absolute ranks;
validation compares the semantic claims against exactly that derivation.
Adding status/evidence text cannot promote these observations. Inference status
and human acceptance of an approximation remain distinct from acceptance of a
botanical insertion.

A separate small topology helper tests future explicit accepted records without
making them eligible for source-package promotion. Accepted internodes require
distinct accepted endpoints on one axis, consistent acyclic succession and
explicit evidence; unknown basal identity requires null absolute ranks. The
below-node view joins the incoming internode at its apical endpoint and marks a
missing basal boundary incomplete. Synthetic fixtures demonstrate these rules,
not the anatomy of the reviewed cucumber.
