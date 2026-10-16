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
