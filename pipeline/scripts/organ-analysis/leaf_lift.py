"""Cross-view leaf lift, extracted from the verified Welzo prototype.

See DESIGN.md for provenance, algorithm limits and regression checks.
"""
import json
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from pathlib import Path

def read_json(path):
    return json.loads(Path(path).read_text())

def run(geom, output, params):
    OUT = str(output)
    RD, RM, MAXR, MIN_G = (params[k] for k in ['RD','RM','MAXR','MIN_G'])
    ZTOL, JOIN, MERGE, TOUCH, TOUCH_SHARE = (params[k] for k in ['ZTOL','JOIN','MERGE','TOUCH','TOUCH_SHARE'])
    C0 = 0.28209479177387814
    fmap = read_json(OUT + "/frame_map.json")
    data = geom.load_ply()
    op = 1 / (1 + np.exp(-data["opacity"].astype(np.float64)))
    keep = op > params["opacity_cutoff"]
    d = data[keep]
    X = np.stack([d["x"], d["y"], d["z"]], 1).astype(np.float64)
    smid = np.median(np.exp(np.stack([d["scale_0"], d["scale_1"], d["scale_2"]], 1).astype(np.float64)), 1)
    rgb = np.clip(0.5 + C0 * np.stack([d["f_dc_0"], d["f_dc_1"], d["f_dc_2"]], 1).astype(np.float64), 0, 1)
    N = len(X)
    if N < 16:
        raise ValueError("At least 16 finite opacity-filtered centers are required")
    cams = geom.cameras(fmap)

    offs = {}
    for r in range(0, MAXR + 1):
        g = np.arange(-r, r + 1)
        dx, dy = np.meshgrid(g, g)
        m = dx ** 2 + dy ** 2 <= max(r, 0.5) ** 2
        offs[r] = (dx[m], dy[m])

    VIS = np.zeros((len(cams), N), bool)
    inst = []            # (photo, file, idx)
    plant_in = np.zeros((len(cams), N), bool)
    leaf_any = np.zeros((len(cams), N), bool)
    for vi, c in enumerate(cams):
        u, v, z = geom.project(c, X)
        W, Hh = c["w"] // RD, c["h"] // RD
        ok = (z > 0) & (u >= 0) & (u < c["w"]) & (v >= 0) & (v < c["h"])
        ud, vd = (u / RD).astype(int), (v / RD).astype(int)
        rad = np.clip(np.round(c["K"][0, 0] / RD * smid / np.maximum(z, 1e-6)), 0, MAXR).astype(int)
        depth = np.full(Hh * W, np.inf)
        for r in np.unique(rad[ok]):
            sel = np.flatnonzero(ok & (rad == r))
            dx, dy = offs[r]
            xx = (ud[sel][:, None] + dx[None]).ravel()
            yy = (vd[sel][:, None] + dy[None]).ravel()
            zz = np.repeat(z[sel], len(dx))
            inb = (xx >= 0) & (xx < W) & (yy >= 0) & (yy < Hh)
            np.minimum.at(depth, yy[inb] * W + xx[inb], zz[inb])
        vis = np.zeros(N, bool)
        vis[ok] = z[ok] <= depth[vd[ok] * W + ud[ok]] + ZTOL
        VIS[vi] = vis
        um, vm = (u / RM).astype(int), (v / RM).astype(int)
        plant = np.asarray(Image.open(geom.plant_path(c["name"])).convert("L").reduce(RM)) > 127
        plant_in[vi, vis] = plant[vm[vis], um[vis]]
        dd = f"{geom.LEAF}/{c['name']}"
        for ins in read_json(dd + "/instances.json")["instances"]:
            m = np.asarray(Image.open(f"{dd}/{ins['file']}").convert("L").reduce(RM)) > 127
            hit = np.zeros(N, bool)
            hit[vis] = m[vm[vis], um[vis]]
            leaf_any[vi] |= hit
            idx = np.flatnonzero(hit)
            if len(idx) >= MIN_G:
                inst.append((vi, ins["file"], idx))
        print(f"{c['name']}: visible {vis.sum()} (disc-depth visibility)")

    M = len(inst)
    if M == 0:
        raise ValueError("No leaf instances have enough visible 3D support")
    sets = [set(s[2].tolist()) for s in inst]
    edges = []
    for a in range(M):
        for b in range(a + 1, M):
            if inst[a][0] != inst[b][0]:
                k = len(sets[a] & sets[b])
                if k:
                    w = k / min(len(sets[a]), len(sets[b]))
                    if w >= JOIN:
                        edges.append((w, a, b))
    edges.sort(reverse=True)
    parent = list(range(M))
    views = [{inst[a][0]} for a in range(M)]


    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a


    for w, a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb and not (views[ra] & views[rb]):
            parent[rb] = ra
            views[ra] |= views[rb]
    cl = np.array([find(a) for a in range(M)])


    def labels(cl):
        ids = sorted(set(cl.tolist()))
        votes = np.zeros((N, len(ids)))
        pos = {r: i for i, r in enumerate(ids)}
        for a, (vi, f, idx) in enumerate(inst):
            votes[idx, pos[cl[a]]] += 1
        lab = np.full(N, -1)
        has = votes.sum(1) > 0
        lab[has] = np.array(ids)[votes[has].argmax(1)]
        return lab, votes


    # merge pass
    merged = []
    while True:
        lab, _ = labels(cl)
        ids = [r for r in set(cl.tolist()) if (lab == r).sum() >= 20]
        best = None
        for B in ids:
            pb = np.flatnonzero(lab == B)
            own = {inst[a][0] for a in np.flatnonzero(cl == B)}
            for A in ids:
                if A == B:
                    continue
                covs = []
                for a in np.flatnonzero(cl == A):
                    vi = inst[a][0]
                    if vi in own:
                        continue
                    vb = pb[VIS[vi, pb]]
                    if len(vb) >= 10:
                        covs.append(np.isin(vb, inst[a][2]).mean())
                if len(covs) >= 2 and np.median(covs) >= MERGE:
                    # occlusion also gives high cover (a leaf behind another projects into its mask),
                    # so require the two clusters to touch in 3D
                    pa = np.flatnonzero(lab == A)
                    dist, _ = cKDTree(X[pa]).query(X[pb])
                    touch = (dist < TOUCH).mean()
                    print(f"  candidate: {len(pb)} into {len(pa)}, cover {np.median(covs):.2f}, touching {touch:.2f}")
                    if touch < TOUCH_SHARE:
                        continue
                    s = (np.median(covs), len(covs), A, B)
                    if best is None or s > best:
                        best = s
        if best is None:
            break
        cov, n, A, B = best
        merged.append({"into": int(A), "folded": int(B), "median_cover": float(cov), "photos": int(n),
                       "folded_size": int((lab == B).sum())})
        print(f"merge: cluster of {(lab == B).sum()} gaussians folded into one of {(lab == A).sum()} "
              f"(median cover {cov:.2f} over {n} photos)")
        cl[cl == B] = A

    lab, votes = labels(cl)
    order = sorted(set(lab[lab >= 0].tolist()), key=lambda r: -(lab == r).sum())
    remap = {r: i for i, r in enumerate(order)}
    label = np.array([remap.get(l, -1) for l in lab])
    inst_cluster = np.array([remap.get(r, -1) for r in cl])
    leaf_votes = votes.sum(1)
    nonleaf_votes = (plant_in & ~leaf_any).sum(0)
    nonleaf = (nonleaf_votes > leaf_votes) & (nonleaf_votes >= 2)
    label[nonleaf] = -2
    conf = np.where(leaf_votes > 0, votes.max(1) / np.maximum(leaf_votes, 1), 0)

    summary = []
    print("\nleaf ID | photos | gaussians | median vote share")
    for k in range(len(order)):
        g = np.flatnonzero(label == k)
        mem = np.flatnonzero(inst_cluster == k)
        if len(g) < 20:
            continue
        ph = len({inst[a][0] for a in mem})
        summary.append({"leaf_id": k, "photos": ph, "gaussians": int(len(g)), "median_share": float(np.median(conf[g])),
                        "members": [[cams[inst[a][0]]["name"], inst[a][1]] for a in mem]})
        print(f"{k:7d} | {ph:6d} | {len(g):9d} | {np.median(conf[g]):.2f}")
    print(f"non-leaf {nonleaf.sum()}, unlabeled {(label == -1).sum()}, leaf IDs with >=20 gaussians {len(summary)}")
    np.savez(OUT + "/leaf_ids2.npz", X=X, label=label, conf=conf, rgb=rgb, smid=smid, nonleaf_votes=nonleaf_votes,
             leaf_votes=leaf_votes, vis_count=VIS.sum(0))
    with open(OUT + "/leaf_ids2.json", "w") as output:
        json.dump({"params": {"RD": RD, "RM": RM, "ZTOL": ZTOL, "MAXR": MAXR, "MIN_G": MIN_G, "JOIN": JOIN, "MERGE": MERGE, "TOUCH": TOUCH, "TOUCH_SHARE": TOUCH_SHARE},
                   "merges": merged, "clusters": summary}, output, indent=1, allow_nan=False)
