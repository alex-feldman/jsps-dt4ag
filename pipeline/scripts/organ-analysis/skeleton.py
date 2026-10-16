"""Confirmed-apex geodesic level-set skeleton; see DESIGN.md.

Nodes are geometric samples, not validated biological phytomers.
"""
import json
import numpy as np
import networkx as nx
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

def run(geom, output, params, apex_leaf, make_plots=False):
    OUT = str(output)
    LINK, BIN, LINP, GREEN = (params[k] for k in ['LINK','BIN','LINP','GREEN'])
    z = np.load(OUT + "/leaf_ids2.npz")
    X, label, rgb = z["X"], z["label"], z["rgb"]
    z.close()
    nl = np.flatnonzero(label == -2)
    if len(nl) == 0:
        raise ValueError("No plant-minus-leaf candidate points available")
    P, C = X[nl], rgb[nl]
    green = C[:, 1] - np.maximum(C[:, 0], C[:, 2])
    sat = C.max(1) - C.min(1)
    print("non-leaf", len(nl), "greenness quantiles 10/25/50/75/90:", np.quantile(green, [.1, .25, .5, .75, .9]).round(3))
    print("brightness median", np.median(C.mean(1)).round(3), "sat median", np.median(sat).round(3))
    keepg = green > GREEN
    print(f"green > {GREEN}: {keepg.sum()} kept, {(~keepg).sum()} dropped")

    tr = cKDTree(X)
    _, nb = tr.query(X, k=16)
    A = X[nb] - X[nb].mean(1, keepdims=True)
    ev = np.sort(np.linalg.eigvalsh(np.einsum("nki,nkj->nij", A, A) / 16), 1)[:, ::-1]
    sv = np.sqrt(np.clip(ev, 0, None))
    lin = (sv[:, 0] - sv[:, 1]) / sv[:, 0]
    gall = rgb[:, 1] - np.maximum(rgb[:, 0], rgb[:, 2])
    petc = np.flatnonzero((label >= 0) & (lin > LINP) & (gall > GREEN))
    print(f"petiole candidates (leaf-labeled, green, linearity > {LINP}): {len(petc)}")
    Q = np.concatenate([P[keepg], X[petc]])
    if len(Q) < 2:
        raise ValueError("Too few green candidate network points")
    idxQ = np.concatenate([nl[keepg], petc])
    t = cKDTree(Q)
    pairs = t.query_pairs(LINK, output_type="ndarray")
    w = np.linalg.norm(Q[pairs[:, 0]] - Q[pairs[:, 1]], axis=1)
    G = coo_matrix((w, (pairs[:, 0], pairs[:, 1])), shape=(len(Q), len(Q))).tocsr()
    G = G + G.T
    n, comp = connected_components(G, directed=False)
    big = np.argmax(np.bincount(comp))
    sel = np.flatnonzero(comp == big)
    print(f"components {n}; largest {len(sel)} of {len(Q)}")
    Q, idxQ = Q[sel], idxQ[sel]
    in_net = np.zeros(len(X), bool)
    in_net[idxQ] = True
    pet_pts = {int(L): int(((label == L) & in_net).sum()) for L in set(label[label >= 0].tolist())}
    print("petiole points recovered per leaf:", {k: v for k, v in sorted(pet_pts.items()) if v})
    G = G[sel][:, sel]
    up = np.array([0, 0, 1.0])
    h = Q @ up
    root = int(np.argmin(h))
    gd = dijkstra(G, indices=root)
    print(f"height span {h.max()-h.min():.3f}, geodesic max {gd[np.isfinite(gd)].max():.3f}")

    # level-set nodes
    binid = np.floor(gd / BIN).astype(int)
    node_of = np.full(len(Q), -1)
    nodes = []
    for b in np.unique(binid):
        m = np.flatnonzero(binid == b)
        sub = G[m][:, m]
        k, lab = connected_components(sub, directed=False)
        for j in range(k):
            mm = m[lab == j]
            node_of[mm] = len(nodes)
            nodes.append({"bin": int(b), "pts": mm, "c": Q[mm].mean(0)})
    Gc = G.tocoo()
    T = nx.Graph()
    for i, nd in enumerate(nodes):
        T.add_node(i, pos=nd["c"], bin=nd["bin"], n=len(nd["pts"]))
    for a, b in zip(Gc.row, Gc.col):
        na, nb = node_of[a], node_of[b]
        if na != nb:
            T.add_edge(na, nb, weight=float(np.linalg.norm(nodes[na]["c"] - nodes[nb]["c"])))
    T = nx.minimum_spanning_tree(T)
    rootn = int(node_of[root])
    deg = dict(T.degree())
    print(f"skeleton nodes {T.number_of_nodes()}, edges {T.number_of_edges()}, tips {sum(1 for d in deg.values() if d == 1)}, "
          f"branch nodes {sum(1 for d in deg.values() if d >= 3)}")

    # leaves -> base, insertion
    leaf_ids = sorted(set(label[label >= 0].tolist()))
    TIPGAP = params["TIPGAP"]
    tips = [t for t, dg in T.degree() if dg == 1 and t != rootn]
    qt = cKDTree(Q)
    info = {}
    for L in leaf_ids:
        pl = X[(label == L) & ~in_net]
        if len(pl) < 20:
            continue
        bt = cKDTree(pl)
        td = [(bt.query(nodes[t]["c"])[0], t) for t in tips]
        dmin, tbest = min(td) if td else (np.inf, -1)
        if dmin < TIPGAP:
            k = bt.query(nodes[tbest]["c"])[1]
            info[L] = {"gap": float(dmin), "base": pl[k], "node": int(tbest), "size": len(pl), "rule": "tip"}
        else:
            dist, j = qt.query(pl)
            i = int(np.argmin(dist))
            info[L] = {"gap": float(dist[i]), "base": pl[i], "node": int(node_of[j[i]]), "size": len(pl), "rule": "nearest"}
    apex = apex_leaf
    if apex not in info:
        raise RuntimeError("Selected apex cluster is not available in the derived leaf info")
    main = nx.shortest_path(T, rootn, info[apex]["node"], weight="weight")
    mainset = set(main)
    dist_main = {nd: nx.shortest_path_length(T, rootn, nd, weight="weight") for nd in main}
    print(f"\napex leaf (user-confirmed cluster): {apex}; main stem nodes {len(main)}, "
          f"length {dist_main[main[-1]]:.3f}")
    rows = []
    for L, d in info.items():
        path = nx.shortest_path(T, rootn, d["node"], weight="weight")
        ins = [p for p in path if p in mainset][-1]
        pet = nx.shortest_path_length(T, ins, d["node"], weight="weight")
        rows.append((dist_main[ins], L, ins, pet, d))
    rows.sort()
    print(f"BIN {BIN}: leaf | blade size | insertion along stem from base | petiole length (network) | blade-to-network gap")
    for s, L, ins, pet, d in rows:
        print(f"{L:4d} | {d['size']:5d} | {s:.3f} | {pet:.3f} | {d['gap']:.4f} | {d['rule']}")

    if make_plots:
        from PIL import Image, ImageEnhance
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        # plots
        cols = plt.get_cmap("tab10").colors
        c0 = X - np.median(X, 0)
        _, _, vt = np.linalg.svd(c0 - c0.mean(0), full_matrices=False)
        proj = lambda A: (A - np.median(X, 0)) @ vt.T
        fig, axes = plt.subplots(1, 3, figsize=(24, 9), dpi=90)
        qa, pa_ = proj(X), proj(P)
        for ax, (i, j) in zip(axes, [(1, 0), (2, 0), (1, 2)]):
            lc = np.array([cols[l % 10] if l >= 0 else (0.85, 0.85, 0.85) for l in label])
            ax.scatter(qa[:, i], qa[:, j], s=0.6, c=lc, alpha=0.25, linewidths=0)
            ax.scatter(pa_[~keepg, i], pa_[~keepg, j], s=3, c="k", linewidths=0, label="dropped (not green)")
            for a, b in T.edges():
                pp = proj(np.array([nodes[a]["c"], nodes[b]["c"]]))
                on = a in mainset and b in mainset
                ax.plot(pp[:, i], pp[:, j], c="red" if on else "magenta", lw=2.5 if on else 1.5)
            for s, L, ins, pet, d in rows:
                pi = proj(nodes[ins]["c"][None])[0]
                ax.scatter(pi[i], pi[j], s=60, c=[cols[L % 10]], edgecolors="k", zorder=5)
                pb = proj(d["base"][None])[0]
                ax.scatter(pb[i], pb[j], s=40, marker="x", c=[cols[L % 10]], zorder=5)
            ax.set_aspect("equal")
            lo, hi = np.quantile(qa[:, [i, j]], [0.005, 0.995], axis=0)
            ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
        axes[0].legend(loc="upper left")
        fig.suptitle("skeleton: red = main stem, magenta = petioles and other branches; dots = leaf insertion nodes, x = leaf bases")
        fig.savefig(OUT + "/skeleton-3d.png", bbox_inches="tight")
        plt.close(fig)

        fmap = json.load(open(OUT + "/frame_map.json"))
        cams = geom.cameras(fmap)
        fig, axes = plt.subplots(2, 2, figsize=(22, 17), dpi=80)
        for ax, vi in zip(axes.ravel(), np.linspace(0, len(cams)-1, min(4,len(cams)), dtype=int)):
            c = cams[vi]
            im = ImageEnhance.Brightness(Image.open(geom.image_path(c["name"])).reduce(4)).enhance(0.5)
            ax.imshow(im)
            u, v, _ = geom.project(c, P)
            ax.scatter(u[~keepg] / 4, v[~keepg] / 4, s=3, c="cyan", linewidths=0)
            pp = np.flatnonzero(in_net & (label >= 0))
            up_, vp_, _ = geom.project(c, X[pp])
            ax.scatter(up_ / 4, vp_ / 4, s=5, c=[cols[l % 10] for l in label[pp]], edgecolors="w", linewidths=0.3, zorder=4)
            for a, b in T.edges():
                uu, vv, _ = geom.project(c, np.array([nodes[a]["c"], nodes[b]["c"]]))
                on = a in mainset and b in mainset
                ax.plot(uu / 4, vv / 4, c="red" if on else "magenta", lw=2.5 if on else 1.5)
            for s, L, ins, pet, d in rows:
                uu, vv, _ = geom.project(c, nodes[ins]["c"][None])
                ax.scatter(uu / 4, vv / 4, s=70, c=[cols[L % 10]], edgecolors="w", zorder=5)
                ax.text(uu[0] / 4 + 8, vv[0] / 4, str(L), color="w", fontsize=15, weight="bold")
            uX, vX, _ = geom.project(c, X)
            lo = np.quantile(np.stack([uX, vX], 1) / 4, 0.002, 0) - 40
            hi = np.quantile(np.stack([uX, vX], 1) / 4, 0.998, 0) + 40
            ax.set_xlim(lo[0], hi[0]); ax.set_ylim(hi[1], lo[1]); ax.axis("off")
            ax.set_title(c["name"] + "  (cyan = dropped as not green; numbered dots = insertion nodes)", fontsize=13)
        fig.tight_layout()
        fig.savefig(OUT + "/skeleton-photos.png", bbox_inches="tight")
    result = {"params": {"LINK": LINK, "BIN": BIN, "GREEN": GREEN, "LINP": LINP}, "petiole_points": {str(k):v for k,v in pet_pts.items()}, "apex_leaf": int(apex),
               "skeleton_nodes": [{"id": int(i), "xyz": nodes[i]["c"].tolist()} for i in T.nodes()],
               "skeleton_edges": [{"a": int(a), "b": int(b), "main": bool(a in mainset and b in mainset)} for a, b in T.edges()],
               "main_stem": [nodes[n]["c"].tolist() for n in main],
               "leaves": [{"leaf_id": int(L), "insertion_along_stem": float(s), "insertion_xyz": nodes[ins]["c"].tolist(),
                           "petiole_network_length": float(pet), "base_xyz": d["base"].tolist(), "gap": d["gap"], "base_rule": d["rule"]}
                          for s, L, ins, pet, d in rows]}
    with open(OUT + "/skeleton.json", "w") as f:
        json.dump(result, f, indent=1, allow_nan=False)
    return result
