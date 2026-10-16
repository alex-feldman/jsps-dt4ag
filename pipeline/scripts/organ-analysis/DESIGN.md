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

Focused tests cover malformed/truncated PLY, camera-frame mapping, unsupported distortion, missing/wrong-size masks, explicit apex requirement, output protection and reused-label integrity. A synthetic tiny scene exercises the public command, leaf-only/skeleton-stage flow and checked failure path. The real Welzo run must reproduce the baseline arrays, leaf memberships and graph; inspect a generated overlay/viewer in a real browser. A raw hand-check can inspect graph endpoints in `skeleton.json` and source-instance membership in `leaf_ids2.json`, independently of the wrapper's status.

Any new camera model, visibility rule, threshold, topology repair or organ heuristic changes the analysis and requires new regression/evidence. Do not tune per reconstruction arm or silently replace source scripts inside a frozen experiment. New datasets need their own review, apex choice and parameter checks. Metric scale, measured radii, missing petiole centerlines, MTG serialization and Helios generation are outside this command's claims.
