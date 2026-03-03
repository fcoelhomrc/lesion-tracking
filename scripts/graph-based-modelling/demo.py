# ============================================================
# Multi-Scenario 2D Mass Transport Demo
# 4 timepoints · shrink / grow / split / merge
# ============================================================

import argparse
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import ot
from matplotlib.patches import Rectangle
from rich import box
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from rich.table import Table
from scipy.ndimage import gaussian_filter
from scipy.ndimage import label as nd_label
from scipy.spatial.distance import cdist

_console = Console()

# ─── Configuration ────────────────────────────────────────────

GRID = 64
GAP = 22
X_STEP = GRID + GAP
N_TIMEPOINTS = 4

BG = "#0d1117"
PANEL_BG = "#161b22"
TEXT_COL = "#8b949e"
T_COLORS = ["#9BB7D4", "#7ECDC8", "#8DB596", "#C8A882"]  # blue / teal / sage / amber
EDGE_COL = "#C9B8D8"  # soft lavender

MARGIN = 10
MIN_R = 1.8
MAX_R = 9.0

# ─── Spec ID counter (for ground-truth provenance) ────────────

_SPEC_ID = [0]


def _next_id():
    _SPEC_ID[0] += 1
    return _SPEC_ID[0]


# ─── Blob rendering ───────────────────────────────────────────


def _add_blob(
    grid,
    center,
    radius,
    elongation=1.0,
    angle=0.0,
    roughness=0.55,
    bump_scale=0.6,
    seed=None,
):
    """Ellipsoidal blob with two-scale smooth surface noise for organic shapes."""
    rng = np.random.default_rng(seed)
    X, Y = np.indices(grid.shape)
    cx, cy = center
    # Rotated ellipsoidal distance: elongation stretches the perpendicular axis
    ca, sa = np.cos(angle), np.sin(angle)
    dx = (X - cx) * ca + (Y - cy) * sa
    dy = -(X - cx) * sa + (Y - cy) * ca
    dist = np.sqrt(dx**2 + (dy / elongation) ** 2)

    # Two-scale noise: coarse gives large bumps, medium adds secondary waviness.
    # Both sigmas are large enough to avoid sharp corners.
    sigma_c = max(2.5, radius * bump_scale)
    sigma_m = max(1.5, radius * 0.28)

    n1 = gaussian_filter(rng.standard_normal(grid.shape), sigma=sigma_c)
    n1 /= n1.std() + 1e-8
    n2 = gaussian_filter(rng.standard_normal(grid.shape), sigma=sigma_m)
    n2 /= n2.std() + 1e-8

    noise = (0.65 * n1 + 0.35 * n2) * roughness
    surface_weight = np.exp(-((dist - radius) ** 2) / (2 * sigma_c**2))
    noise *= surface_weight
    grid[dist <= radius + noise] = 1.0


def _new_spec(center, radius, rng, elongation=None, angle=None, parent_ids=None):
    return {
        "id": _next_id(),
        "parent_ids": list(parent_ids) if parent_ids else [],
        "center": center,
        "radius": radius,
        "elongation": float(
            elongation if elongation is not None else rng.uniform(0.55, 2.4)
        ),
        "angle": float(angle if angle is not None else rng.uniform(0, np.pi)),
        "seed": int(rng.integers(10000)),
        "roughness": 0.55,
        "bump_scale": 0.6,
    }


def render_state(specs):
    grid = np.zeros((GRID, GRID))
    for s in specs:
        _add_blob(
            grid,
            s["center"],
            s["radius"],
            elongation=s.get("elongation", 1.0),
            angle=s.get("angle", 0.0),
            roughness=s.get("roughness", 0.55),
            bump_scale=s.get("bump_scale", 0.6),
            seed=s.get("seed"),
        )
    return grid


# ─── Phenomena ────────────────────────────────────────────────


def _cc(c):
    return (
        float(np.clip(c[0], MARGIN, GRID - MARGIN)),
        float(np.clip(c[1], MARGIN, GRID - MARGIN)),
    )


def _cr(r):
    return float(np.clip(r, MIN_R, MAX_R))


def _ph_stable(specs, rng):
    return [dict(s) for s in specs]


def _ph_shrink(specs, rng):
    specs = [dict(s) for s in specs]
    i = int(rng.integers(len(specs)))
    specs[i] = dict(
        specs[i],
        radius=_cr(specs[i]["radius"] * rng.uniform(0.45, 0.7)),
        seed=int(rng.integers(10000)),
    )
    return specs


def _ph_grow(specs, rng):
    specs = [dict(s) for s in specs]
    i = int(rng.integers(len(specs)))
    specs[i] = dict(
        specs[i],
        radius=_cr(specs[i]["radius"] * rng.uniform(1.3, 1.7)),
        seed=int(rng.integers(10000)),
    )
    return specs


def _ph_split(specs, rng):
    specs = [dict(s) for s in specs]
    i = int(rng.integers(len(specs)))
    s = specs.pop(i)
    k = int(rng.integers(2, 4))  # split into 2 or 3
    cx, cy = s["center"]
    sub_r = _cr(s["radius"] / k * 0.9)
    # Guarantee disjoint sub-blobs: adjacent centres must be > 2*sub_r apart.
    # On a ring of radius `offset`, adjacent gap = 2*offset*sin(π/k) > 2*sub_r
    # → offset > sub_r/sin(π/k). Use 1.6× margin to account for noise.
    offset = sub_r / np.sin(np.pi / k) * 1.6
    angles = np.linspace(0, 2 * np.pi, k, endpoint=False) + rng.uniform(0, 2 * np.pi)
    for a in angles:
        nc = _cc((cx + offset * np.cos(a), cy + offset * np.sin(a)))
        # Sub-blobs get mild elongation so they stay clearly separated
        specs.append(
            _new_spec(
                nc,
                sub_r,
                rng,
                elongation=rng.uniform(0.8, 1.5),
                angle=rng.uniform(0, np.pi),
                parent_ids=[s["id"]],
            )
        )
    return specs


def _ph_spawn(specs, rng):
    specs = [dict(s) for s in specs]
    for _ in range(40):
        cx = rng.uniform(MARGIN + 6, GRID - MARGIN - 6)
        cy = rng.uniform(MARGIN + 6, GRID - MARGIN - 6)
        r = rng.uniform(3.0, 6.0)
        if all(
            np.hypot(cx - s["center"][0], cy - s["center"][1]) > r + s["radius"] + 6
            for s in specs
        ):
            specs.append(_new_spec((float(cx), float(cy)), float(r), rng))
            return specs
    return specs  # couldn't place without overlap — leave unchanged


def _ph_kill(specs, rng):
    if len(specs) <= 1:
        return _ph_shrink(specs, rng)
    specs = [dict(s) for s in specs]
    specs.pop(int(rng.integers(len(specs))))
    return specs


def _ph_merge(specs, rng):
    if len(specs) < 2:
        return _ph_grow(specs, rng)
    specs = [dict(s) for s in specs]
    # Merge the two closest blobs to minimise translation distance
    best_i, best_j, best_d = 0, 1, float("inf")
    for a in range(len(specs)):
        for b in range(a + 1, len(specs)):
            d = np.hypot(
                specs[a]["center"][0] - specs[b]["center"][0],
                specs[a]["center"][1] - specs[b]["center"][1],
            )
            if d < best_d:
                best_d, best_i, best_j = d, a, b
    si, sj = specs[best_i], specs[best_j]
    cx = (si["center"][0] + sj["center"][0]) / 2
    cy = (si["center"][1] + sj["center"][1]) / 2
    new_r = _cr(np.sqrt(si["radius"] ** 2 + sj["radius"] ** 2))
    specs = [s for k, s in enumerate(specs) if k not in (best_i, best_j)]
    specs.append(_new_spec(_cc((cx, cy)), new_r, rng, parent_ids=[si["id"], sj["id"]]))
    return specs


_PH_FNS = {
    "stable": _ph_stable,
    "shrink": _ph_shrink,
    "grow": _ph_grow,
    "split": _ph_split,
    "merge": _ph_merge,
    "spawn": _ph_spawn,
    "kill": _ph_kill,
}
_PH_NAMES = list(_PH_FNS.keys())
_PH_WEIGHTS = np.array([0.20, 0.40, 0.40, 1.00, 1.00, 1.00, 1.00])
_PH_WEIGHTS /= _PH_WEIGHTS.sum()


def generate_scenario(rng):
    n_init = int(rng.integers(1, 3))
    specs = []
    for _ in range(n_init):
        for _ in range(40):
            cx = rng.uniform(MARGIN + 6, GRID - MARGIN - 6)
            cy = rng.uniform(MARGIN + 6, GRID - MARGIN - 6)
            r = rng.uniform(4.0, 7.0)
            if all(
                np.hypot(cx - s["center"][0], cy - s["center"][1]) > r + s["radius"] + 6
                for s in specs
            ):
                specs.append(_new_spec((float(cx), float(cy)), float(r), rng))
                break
    if not specs:
        specs = [_new_spec((32.0, 32.0), 5.0, rng)]

    all_specs = [specs]
    phenomena = []
    for _ in range(N_TIMEPOINTS - 1):
        ph = rng.choice(_PH_NAMES, p=_PH_WEIGHTS)
        new_specs = _PH_FNS[ph](all_specs[-1], rng)
        if not new_specs:
            new_specs = [dict(all_specs[-1][0])]
            ph = "stable"
        all_specs.append(new_specs)
        phenomena.append(ph)

    return [render_state(s) for s in all_specs], all_specs, phenomena


# ─── OT ──────────────────────────────────────────────────────


def _spawn_mask(rho_src, rho_dst):
    """Boolean mask of rho_dst pixels belonging to components with no overlap with rho_src."""
    lbl, n = nd_label(rho_dst > 0)
    src_support = rho_src > 0
    mask = np.zeros_like(rho_dst, dtype=bool)
    for i in range(1, n + 1):
        comp = lbl == i
        if not np.any(comp & src_support):
            mask |= comp
    return mask


def run_ot(rhos, reg_m=None, exclude_spawn=False):
    """Compute unbalanced OT for consecutive timepoints.

    reg_m: marginal-relaxation penalty. Lower = more eager to destroy/create mass.
           If None, an adaptive value is derived from the mass ratio per transition.
    exclude_spawn: if True, pixels in rho_{t+1} with no spatial overlap with rho_t
                   are excluded from OT (treated as spawn candidates).
    """
    all_coords = [np.array(np.nonzero(r)).T.astype(float) for r in rhos]
    Ps = []
    ot_dst_coords = []
    for i in range(len(rhos) - 1):
        c0 = all_coords[i]
        if exclude_spawn:
            mask = _spawn_mask(rhos[i], rhos[i + 1])
            full = all_coords[i + 1].astype(int)
            keep = ~mask[full[:, 0], full[:, 1]]
            c1 = all_coords[i + 1][keep]
        else:
            c1 = all_coords[i + 1]
        ot_dst_coords.append(c1)
        if len(c0) == 0 or len(c1) == 0:
            Ps.append(np.zeros((max(1, len(c0)), max(1, len(c1)))))
            continue
        C = cdist(c0, c1) ** 2
        C /= C.max() + 1e-8
        a = np.ones(len(c0)) / len(c0)
        b = np.ones(len(c1)) / len(c1)
        if reg_m is None:
            ratio = len(c1) / (len(c0) + 1e-8)
            effective_reg_m = float(np.clip(min(ratio, 1.0 / ratio) * 2.0, 0.08, 2.0))
        else:
            effective_reg_m = reg_m
        P = ot.unbalanced.sinkhorn_knopp_unbalanced(
            a, b, C, reg=0.05, reg_m=effective_reg_m
        )
        Ps.append(P)
    return Ps, all_coords, ot_dst_coords


# ─── Temporal graph ───────────────────────────────────────────


def build_temporal_graph(rhos, Ps, all_coords, ot_dst_coords=None):
    nodes, edges, labeled = {}, [], []
    for t, rho in enumerate(rhos):
        lbl, n = nd_label(rho > 0)
        labeled.append(lbl)
        for i in range(1, n + 1):
            px = np.array(np.nonzero(lbl == i)).T
            nodes[(t, i)] = {"centroid": px.mean(axis=0), "mass": len(px), "t": t}

    for ti, P in enumerate(Ps):
        ls, ld = labeled[ti], labeled[ti + 1]
        sc = all_coords[ti].astype(int)
        dc = (ot_dst_coords[ti] if ot_dst_coords is not None else all_coords[ti + 1]).astype(int)
        if len(sc) == 0 or len(dc) == 0:
            continue
        sl = ls[sc[:, 0], sc[:, 1]]
        dl = ld[dc[:, 0], dc[:, 1]]
        P_tot = P.sum()
        if P_tot == 0:
            continue
        for si in np.unique(sl[sl > 0]):
            for dj in np.unique(dl[dl > 0]):
                w = P[np.ix_(sl == si, dl == dj)].sum()
                if w > P_tot * 5e-4:
                    edges.append(
                        {
                            "src": (ti, int(si)),
                            "dst": (ti + 1, int(dj)),
                            "weight": float(w),
                        }
                    )

    return nodes, edges, labeled


# ─── Graph pruning ────────────────────────────────────────────


def prune_graph_edges(edges, threshold=0.5):
    """N:1 pruning: for each destination node that receives multiple incoming edges,
    if the relative gap between the best and second-best weight exceeds `threshold`,
    keep only the strongest edge (drop the weaker ones as spurious).

    gap = (w_max - w_second) / w_max  →  prune when gap > threshold
    """
    incoming = defaultdict(list)
    for e in edges:
        incoming[e["dst"]].append(e)

    kept = []
    for inc in incoming.values():
        if len(inc) <= 1:
            kept.extend(inc)
            continue
        inc_sorted = sorted(inc, key=lambda e: e["weight"], reverse=True)
        gap = (inc_sorted[0]["weight"] - inc_sorted[1]["weight"]) / (
            inc_sorted[0]["weight"] + 1e-8
        )
        if gap > threshold:
            kept.append(inc_sorted[0])
        else:
            kept.extend(inc)
    return kept


# ─── Evaluation (L1 + L2) ────────────────────────────────────


def build_gt_edges(all_specs):
    """Extract ground-truth edges from spec provenance.

    Each edge dict has:
        src_id  – spec id at t, or None for spawns
        dst_id  – spec id at t+1, or None for kills
        etype   – "continue" | "split" | "merge" | "spawn" | "kill"
        t       – source timepoint index
    """
    gt_per_t = []
    for t in range(len(all_specs) - 1):
        ids_t = {s["id"] for s in all_specs[t]}
        gt, referenced = [], set()

        for s1 in all_specs[t + 1]:
            if s1["id"] in ids_t:  # preserved id → continue
                gt.append(
                    {
                        "src_id": s1["id"],
                        "dst_id": s1["id"],
                        "etype": "continue",
                        "t": t,
                    }
                )
                referenced.add(s1["id"])
            elif len(s1["parent_ids"]) == 1:  # one parent → split child
                gt.append(
                    {
                        "src_id": s1["parent_ids"][0],
                        "dst_id": s1["id"],
                        "etype": "split",
                        "t": t,
                    }
                )
                referenced.add(s1["parent_ids"][0])
            elif len(s1["parent_ids"]) > 1:  # multiple parents → merge
                for pid in s1["parent_ids"]:
                    gt.append(
                        {"src_id": pid, "dst_id": s1["id"], "etype": "merge", "t": t}
                    )
                    referenced.add(pid)
            else:  # no parents → spawn
                gt.append(
                    {"src_id": None, "dst_id": s1["id"], "etype": "spawn", "t": t}
                )

        for s in all_specs[t]:  # unreferenced at t → kill
            if s["id"] not in referenced:
                gt.append({"src_id": s["id"], "dst_id": None, "etype": "kill", "t": t})

        gt_per_t.append(gt)
    return gt_per_t


def _spec_to_comp(all_specs, labeled_grids):
    """Map (t, spec_id) → component label from nd_label."""
    out = []  # one dict per timepoint: spec_id → comp_label
    for specs, lbl in zip(all_specs, labeled_grids):
        m = {}
        for s in specs:
            cx = int(np.clip(round(s["center"][0]), 0, lbl.shape[0] - 1))
            cy = int(np.clip(round(s["center"][1]), 0, lbl.shape[1] - 1))
            comp = int(lbl[cx, cy])
            if comp > 0:
                m[s["id"]] = comp
        out.append(m)
    return out


def evaluate(gt_per_t, pred_edges, all_specs, labeled_grids):
    """Compute L1 (edge precision/recall/F1) and L2 (per-etype recovery).

    L1 covers association edges only (continue/split/merge).
    L2 covers all five event types; for spawn/kill recovery means
    'correctly no edge predicted'.

    Returns (l1_dict, l2_dict).
    """
    id_to_comp = _spec_to_comp(all_specs, labeled_grids)
    pred_set = {(e["src"], e["dst"]) for e in pred_edges}

    pred_out_deg = defaultdict(int)
    pred_in_deg = defaultdict(int)
    for e in pred_edges:
        pred_out_deg[e["src"]] += 1
        pred_in_deg[e["dst"]] += 1

    tp_e = defaultdict(int)
    fn_e = defaultdict(int)

    _etypes = ("continue", "split", "merge", "spawn", "kill")
    cm = {gt: {pr: 0 for pr in (*_etypes, "missed")} for gt in _etypes}

    assoc_gt_pairs = set()  # (node_src, node_dst) for continue/split/merge

    for t, gt_list in enumerate(gt_per_t):
        m_t = id_to_comp[t]
        m_t1 = id_to_comp[t + 1]

        for g in gt_list:
            etype = g["etype"]

            if etype == "spawn":
                dst_id = g["dst_id"]
                if dst_id not in m_t1:
                    continue
                dst_node = (t + 1, m_t1[dst_id])
                in_deg = pred_in_deg[dst_node]
                (fn_e if in_deg else tp_e)["spawn"] += 1
                pred_class = "spawn" if in_deg == 0 else ("merge" if in_deg > 1 else "continue")
                cm["spawn"][pred_class] += 1

            elif etype == "kill":
                src_id = g["src_id"]
                if src_id not in m_t:
                    continue
                src_node = (t, m_t[src_id])
                out_deg = pred_out_deg[src_node]
                (fn_e if out_deg else tp_e)["kill"] += 1
                pred_class = "kill" if out_deg == 0 else ("split" if out_deg > 1 else "continue")
                cm["kill"][pred_class] += 1

            else:  # continue / split / merge
                src_id, dst_id = g["src_id"], g["dst_id"]
                if src_id not in m_t or dst_id not in m_t1:
                    fn_e[etype] += 1
                    cm[etype]["missed"] += 1
                    continue
                pair = ((t, m_t[src_id]), (t + 1, m_t1[dst_id]))
                assoc_gt_pairs.add(pair)
                if pair in pred_set:
                    tp_e[etype] += 1
                    od = pred_out_deg[pair[0]]
                    id_ = pred_in_deg[pair[1]]
                    pred_class = "split" if od > 1 else ("merge" if id_ > 1 else "continue")
                    cm[etype][pred_class] += 1
                else:
                    fn_e[etype] += 1
                    cm[etype]["missed"] += 1

    fp = sum(1 for e in pred_edges if (e["src"], e["dst"]) not in assoc_gt_pairs)

    # L1: association edges (continue + split + merge)
    assoc_tp = tp_e["continue"] + tp_e["split"] + tp_e["merge"]
    assoc_fn = fn_e["continue"] + fn_e["split"] + fn_e["merge"]
    prec = assoc_tp / (assoc_tp + fp + 1e-9)
    rec = assoc_tp / (assoc_tp + assoc_fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    l1 = {
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "tp": assoc_tp,
        "fp": fp,
        "fn": assoc_fn,
    }

    # L2: per-etype recovery rate
    l2 = {}
    for etype in ("continue", "split", "merge", "spawn", "kill"):
        tp, fn = tp_e[etype], fn_e[etype]
        l2[etype] = {"recovery": tp / (tp + fn + 1e-9), "tp": tp, "fn": fn}

    return l1, l2, cm


def _cf(v, lo=0.5, hi=0.8):
    """Color-format a 0–1 float: green / yellow / red."""
    c = "green" if v >= hi else ("yellow" if v >= lo else "red")
    return f"[{c}]{v:.3f}[/{c}]"


def print_metrics(l1, l2, title=""):
    if title:
        _console.rule(f"[bold]{title}[/bold]")

    # L1: single-row summary table
    t1 = Table(
        title="L1  Association  (continue · split · merge)",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold dim",
        title_style="bold",
        min_width=52,
    )
    for col in ("Precision", "Recall", "F1", "TP", "FP", "FN"):
        t1.add_column(col, justify="right")
    t1.add_row(
        _cf(l1["precision"]),
        _cf(l1["recall"]),
        _cf(l1["f1"]),
        str(l1["tp"]),
        f"[red]{l1['fp']}[/red]" if l1["fp"] else "[dim]0[/dim]",
        f"[red]{l1['fn']}[/red]" if l1["fn"] else "[dim]0[/dim]",
    )
    _console.print(t1)

    # L2: one row per event type
    t2 = Table(
        title="L2  Per-event recovery",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold dim",
        title_style="bold",
        min_width=52,
    )
    t2.add_column("Event", style="bold", min_width=10)
    t2.add_column("Recovery", justify="right")
    t2.add_column("TP", justify="right")
    t2.add_column("FN", justify="right")
    t2.add_column("n", justify="right", style="dim")
    for et, v in l2.items():
        t2.add_row(
            et,
            _cf(v["recovery"]),
            str(v["tp"]),
            f"[red]{v['fn']}[/red]" if v["fn"] else "[dim]0[/dim]",
            str(v["tp"] + v["fn"]),
        )
    _console.print(t2)


def print_confusion_matrix(cm):
    _etypes = ("continue", "split", "merge", "spawn", "kill")
    cols = (*_etypes, "missed")

    t = Table(
        title="Confusion Matrix  (rows = GT · cols = predicted)",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold dim",
        title_style="bold",
    )
    t.add_column("GT \\ pred", style="bold", min_width=10)
    for c in cols:
        t.add_column(c, justify="right")

    for gt in _etypes:
        row = []
        for pr in cols:
            v = cm[gt][pr]
            if v == 0:
                row.append("[dim]0[/dim]")
            elif pr == gt:
                row.append(f"[green]{v}[/green]")
            elif pr == "missed":
                row.append(f"[yellow]{v}[/yellow]")
            else:
                row.append(f"[red]{v}[/red]")
        t.add_row(gt, *row)
    _console.print(t)


# ─── Drawing ──────────────────────────────────────────────────


def _draw_row(ax, rhos, nodes, edges):
    ax.set_facecolor(BG)
    xs, ys = np.arange(GRID), np.arange(GRID)
    XX, YY = np.meshgrid(xs, ys)

    # Blob contours
    for t, rho in enumerate(rhos):
        xo = t * X_STEP
        color = T_COLORS[t]
        ax.add_patch(
            Rectangle((xo, 0), GRID, GRID, facecolor=PANEL_BG, linewidth=0, zorder=0)
        )
        ax.text(
            xo + GRID / 2,
            -4,
            f"t = {t}",
            color=TEXT_COL,
            ha="center",
            va="top",
            fontsize=9,
        )
        if rho.max() > 0:
            ax.contourf(
                XX + xo,
                YY,
                rho.T,
                levels=[0.5, 1.01],
                colors=[color],
                alpha=0.75,
                zorder=2,
            )
            sm = gaussian_filter(rho.astype(float), sigma=2.0)
            sm /= sm.max()
            ax.contour(
                XX + xo,
                YY,
                sm.T,
                levels=[0.3],
                colors=["white"],
                linewidths=0.8,
                alpha=0.3,
                zorder=3,
            )

    # Graph edges (arrows spanning the gap between panels)
    if edges:
        max_w = max(e["weight"] for e in edges)
        for e in edges:
            nd_s = nodes.get(e["src"])
            nd_d = nodes.get(e["dst"])
            if nd_s is None or nd_d is None:
                continue
            sx = nd_s["centroid"][0] + e["src"][0] * X_STEP
            sy = nd_s["centroid"][1]
            dx = nd_d["centroid"][0] + e["dst"][0] * X_STEP
            dy = nd_d["centroid"][1]
            w = e["weight"] / max_w
            ax.annotate(
                "",
                xy=(dx, dy),
                xytext=(sx, sy),
                zorder=5,
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=EDGE_COL,
                    lw=0.8 + 3.0 * w,
                    alpha=0.35 + 0.55 * w,
                    connectionstyle="arc3,rad=0.15",
                ),
            )

    # Graph nodes: circles at centroids, radius ∝ blob mass
    if nodes:
        max_mass = max(d["mass"] for d in nodes.values())
        for nid, data in nodes.items():
            x = data["centroid"][0] + nid[0] * X_STEP
            y = data["centroid"][1]
            r = 1.5 + 3.5 * data["mass"] / max_mass
            ax.add_patch(
                plt.Circle(
                    (x, y),
                    r,
                    fill=False,
                    edgecolor="white",
                    lw=1.0,
                    zorder=6,
                    alpha=0.55,
                )
            )

    total_w = N_TIMEPOINTS * X_STEP - GAP
    ax.set_xlim(-3, total_w + 3)
    ax.set_ylim(-10, GRID + 3)
    ax.set_aspect("equal")
    ax.axis("off")


# ─── Sweep ────────────────────────────────────────────────────

_SWEEP_REG_M = [0.05, 0.10, None]
_SWEEP_PRUNE = [(False, None), (True, 0.50), (True, 0.10), (True, 0.05)]
_SWEEP_EXCLUDE_SPAWN = [False, True]


def _run_one_scenario(rng, reg_m, prune, prune_threshold, exclude_spawn):
    grids, all_specs, _ = generate_scenario(rng)
    Ps, coords, ot_dst = run_ot(grids, reg_m=reg_m, exclude_spawn=exclude_spawn)
    nodes, edges, labeled = build_temporal_graph(grids, Ps, coords, ot_dst)
    if prune:
        edges = prune_graph_edges(edges, threshold=prune_threshold)
    gt_per_t = build_gt_edges(all_specs)
    return evaluate(gt_per_t, edges, all_specs, labeled)


def _cfg_label(reg_m, prune, pt, excl):
    return (
        "auto" if reg_m is None else f"{reg_m:.2f}",
        "off" if not prune else f"{pt:.2f}",
        "on" if excl else "off",
    )


def _print_sweep_summary(results, n_seeds):
    _etypes = ("continue", "split", "merge", "spawn", "kill")
    t = Table(
        title=f"Sweep Summary  (median over {n_seeds} seeds)",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold dim",
        title_style="bold",
    )
    for col in ("reg_m", "prune", "excl_spawn", "F1", "Prec", "Rec"):
        t.add_column(col, justify="right")
    for et in _etypes:
        t.add_column(et[:4], justify="right")
    for (reg_m, prune, pt, excl), l1, l2, _ in results:
        reg_s, prune_s, spawn_s = _cfg_label(reg_m, prune, pt, excl)
        t.add_row(
            reg_s,
            prune_s,
            spawn_s,
            _cf(l1["f1"]),
            _cf(l1["precision"]),
            _cf(l1["recall"]),
            *[_cf(l2[et]["recovery"]) for et in _etypes],
        )
    _console.print(t)


def _print_sweep_cms(results):
    for (reg_m, prune, pt, excl), _, _, cm in results:
        reg_s, prune_s, spawn_s = _cfg_label(reg_m, prune, pt, excl)
        _console.rule(f"[dim]reg={reg_s}  prune={prune_s}  excl_spawn={spawn_s}[/dim]")
        print_confusion_matrix(cm)


def run_sweep(n_rows=20, n_seeds=5):
    _etypes = ("continue", "split", "merge", "spawn", "kill")
    _pred_classes = (*_etypes, "missed")
    configs = [
        (reg_m, prune, pt, excl)
        for excl in _SWEEP_EXCLUDE_SPAWN
        for reg_m in _SWEEP_REG_M
        for prune, pt in _SWEEP_PRUNE
    ]
    results = []
    with Progress(
        SpinnerColumn(),
        "[progress.description]{task.description}",
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        console=_console,
    ) as progress:
        sweep_task = progress.add_task("Sweeping…", total=len(configs) * n_seeds)
        for ci, (reg_m, prune, pt, excl) in enumerate(configs):
            seed_l1s, seed_l2s, seed_cms = [], [], []
            for seed in range(n_seeds):
                reg_s, prune_s, spawn_s = _cfg_label(reg_m, prune, pt, excl)
                progress.update(
                    sweep_task,
                    description=(
                        f"cfg {ci + 1}/{len(configs)}  seed {seed + 1}/{n_seeds}  "
                        f"reg={reg_s} prune={prune_s} excl={spawn_s}"
                    ),
                )
                rng = np.random.default_rng(seed)
                run_l1s, run_l2s, run_cms = [], [], []
                for _ in range(n_rows):
                    l1, l2, cm = _run_one_scenario(rng, reg_m, prune, pt, excl)
                    run_l1s.append(l1)
                    run_l2s.append(l2)
                    run_cms.append(cm)
                # aggregate n_rows scenarios (match main's aggregation)
                agg_l1 = {
                    k: float(np.mean([x[k] for x in run_l1s]))
                    for k in ("precision", "recall", "f1", "tp", "fp", "fn")
                }
                agg_l2 = {
                    et: {
                        "recovery": float(np.mean([x[et]["recovery"] for x in run_l2s])),
                        "tp": sum(x[et]["tp"] for x in run_l2s),
                        "fn": sum(x[et]["fn"] for x in run_l2s),
                    }
                    for et in _etypes
                }
                agg_cm = {
                    gt: {pr: sum(c[gt][pr] for c in run_cms) for pr in _pred_classes}
                    for gt in _etypes
                }
                seed_l1s.append(agg_l1)
                seed_l2s.append(agg_l2)
                seed_cms.append(agg_cm)
                progress.advance(sweep_task)

            med_l1 = {k: float(np.median([x[k] for x in seed_l1s])) for k in seed_l1s[0]}
            med_l2 = {
                et: {
                    k: float(np.median([x[et][k] for x in seed_l2s]))
                    for k in ("recovery", "tp", "fn")
                }
                for et in _etypes
            }
            med_cm = {
                gt: {pr: int(np.median([x[gt][pr] for x in seed_cms])) for pr in _pred_classes}
                for gt in _etypes
            }
            results.append(((reg_m, prune, pt, excl), med_l1, med_l2, med_cm))

    _console.rule("[bold]Sweep Report[/bold]")
    _print_sweep_summary(results, n_seeds)
    _print_sweep_cms(results)


# ─── Entry point ──────────────────────────────────────────────


def main(
    n_rows=4,
    seed=0,
    reg_m=None,
    prune=False,
    prune_threshold=0.5,
    do_eval=False,
    verbose=False,
    exclude_spawn=False,
):
    rng = np.random.default_rng(seed)

    total_w = N_TIMEPOINTS * X_STEP - GAP  # 4*86 - 22 = 322
    row_h = GRID + 14  # data-units height per row
    fig_w = 14.0
    fig_h = fig_w * (row_h * n_rows) / total_w

    fig, axes = plt.subplots(n_rows, 1, figsize=(fig_w, fig_h))
    fig.patch.set_facecolor(BG)
    if n_rows == 1:
        axes = [axes]

    all_l1, all_l2, all_cms = [], [], []

    with Progress(
        SpinnerColumn(),
        "[progress.description]{task.description}",
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        console=_console,
    ) as progress:
        task = progress.add_task("Generating scenarios…", total=n_rows)
        for idx, ax in enumerate(axes):
            progress.update(task, description=f"Scenario {idx + 1}/{n_rows}")
            grids, all_specs, phenomena = generate_scenario(rng)
            Ps, coords, ot_dst = run_ot(grids, reg_m=reg_m, exclude_spawn=exclude_spawn)
            nodes, edges, labeled = build_temporal_graph(grids, Ps, coords, ot_dst)
            if prune:
                edges = prune_graph_edges(edges, threshold=prune_threshold)

            if not do_eval:
                _draw_row(ax, grids, nodes, edges)

            if do_eval:
                gt_per_t = build_gt_edges(all_specs)
                l1, l2, cm = evaluate(gt_per_t, edges, all_specs, labeled)
                all_l1.append(l1)
                all_l2.append(l2)
                all_cms.append(cm)
                if verbose:
                    ph_str = " → ".join(phenomena)
                    print_metrics(
                        l1, l2, title=f"Scenario {idx + 1}/{n_rows}   {ph_str}"
                    )

            progress.advance(task)

    if do_eval:
        agg_l1 = {
            k: float(np.mean([x[k] for x in all_l1]))
            for k in ("precision", "recall", "f1", "tp", "fp", "fn")
        }
        agg_l2 = {
            et: {
                "recovery": float(np.mean([x[et]["recovery"] for x in all_l2])),
                "tp": sum(x[et]["tp"] for x in all_l2),
                "fn": sum(x[et]["fn"] for x in all_l2),
            }
            for et in ("continue", "split", "merge", "spawn", "kill")
        }
        n = len(all_l1)
        print_metrics(
            agg_l1, agg_l2, title=f"Aggregate  ({n} scenario{'s' if n != 1 else ''})"
        )
        _etypes = ("continue", "split", "merge", "spawn", "kill")
        agg_cm = {
            gt: {pr: sum(c[gt][pr] for c in all_cms) for pr in (*_etypes, "missed")}
            for gt in _etypes
        }
        print_confusion_matrix(agg_cm)

    if not do_eval:
        plt.tight_layout(pad=0.3, h_pad=0.5)
        out = "demo_output.png"
        plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
        plt.show()
        _console.print(f"[dim]Saved {out}[/dim]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2D mass transport demo")
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--reg-m",
        type=float,
        default=None,
        help="OT marginal penalty (lower = more eager to destroy mass). "
        "Default: adaptive per transition.",
    )
    parser.add_argument(
        "--prune", action="store_true", help="Prune spurious N:1 graph edges."
    )
    parser.add_argument(
        "--prune-threshold",
        type=float,
        default=0.5,
        help="Relative weight gap above which weaker N:1 edges are pruned (default 0.5).",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Compute L1/L2 association metrics (prints aggregate summary).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="With --eval: also print per-scenario breakdown (default: aggregate only).",
    )
    parser.add_argument(
        "--exclude-spawn",
        action="store_true",
        help="Exclude components with no spatial overlap with the previous frame from OT.",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run parameter sweep (reg_m × prune × exclude_spawn) and print report.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=5,
        help="Number of seeds per config for median computation in sweep mode (default: 5).",
    )
    args = parser.parse_args()
    if args.sweep:
        run_sweep(n_rows=args.rows, n_seeds=args.seeds)
    else:
        main(
            n_rows=args.rows,
            seed=args.seed,
            reg_m=args.reg_m,
            prune=args.prune,
            prune_threshold=args.prune_threshold,
            do_eval=args.eval,
            verbose=args.verbose,
            exclude_spawn=args.exclude_spawn,
        )
