"""
SemGraph-Route — Phase 3: Semantic Scene Graph Construction
===========================================================
Aggregates spatially-close keyframes of the same space type into
single space nodes, connects them with traversal edges, attaches
priors AND ground-truth labels for fair evaluation.

Usage:
  python build_scene_graph.py

Reads:
  data/processed/space_labeled_records.json
  data/priors/space_type_priors.json
  data/labels/training_labels.json          ← GT labels for evaluation

Writes:
  data/graphs/semantic_scene_graph.json
  data/graphs/semgraph_stats.json
"""

import json
import sys
import time
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime

import numpy as np
import networkx as nx

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).parent.resolve()
DATA_DIR       = BASE_DIR / "data"
SPACE_JSON     = DATA_DIR / "processed" / "space_labeled_records.json"
PRIORS_JSON    = DATA_DIR / "priors"    / "space_type_priors.json"
LABELS_JSON    = DATA_DIR / "labels"    / "training_labels.json"
OUT_GRAPH_JSON = DATA_DIR / "graphs"    / "semantic_scene_graph.json"
STATS_JSON     = DATA_DIR / "graphs"    / "semgraph_stats.json"
LOGS_DIR       = BASE_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

VALID_TYPES        = ["corridor", "junction", "open_space", "confined_space", "dark_zone"]
AGGREGATION_RADIUS = 2.0
EDGE_RADIUS        = 6.0
ALPHA, BETA, GAMMA, DELTA = 0.35, 0.30, 0.20, 0.15

# ─── Logger ───────────────────────────────────────────────────────────────────
log_lines = []

def log(msg):
    ts   = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    log_lines.append(line)

def save_log():
    p = LOGS_DIR / "phase3_build_scene_graph.log"
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

# ─── Aggregation ──────────────────────────────────────────────────────────────
def aggregate_into_space_nodes(records, priors, label_map):
    """
    Greedy spatial aggregation of keyframes → space nodes.
    GT risk/uncertainty sourced from training_labels.json via label_map.
    """
    valid = []
    for r in records:
        stype = r.get("space_type")
        pos   = r.get("position")
        if stype in VALID_TYPES and pos is not None:
            try:
                p = [float(x) for x in pos[:3]]
                valid.append((stype, p, r))
            except Exception:
                continue

    log(f"Valid records: {len(valid)} / {len(records)}")

    space_nodes = []
    assigned    = [False] * len(valid)

    for i, (stype_i, pos_i, rec_i) in enumerate(valid):
        if assigned[i]:
            continue

        cluster = [rec_i]
        cluster_pos = [pos_i]
        assigned[i] = True

        for j, (stype_j, pos_j, rec_j) in enumerate(valid):
            if assigned[j] or stype_j != stype_i:
                continue
            if np.linalg.norm(np.array(pos_i) - np.array(pos_j)) <= AGGREGATION_RADIUS:
                cluster.append(rec_j)
                cluster_pos.append(pos_j)
                assigned[j] = True

        centroid = np.mean(cluster_pos, axis=0).tolist()

        # ── Pull GT labels from training_labels.json (priority 1) ─────────
        gt_risks, gt_uncerts = [], []
        for rc in cluster:
            kid = rc.get("keyframe_idx")
            if kid is not None and kid in label_map:
                lbl = label_map[kid]
                gt_risks.append(float(lbl.get("semantic_risk_gt",
                                lbl.get("risk", 0.3))))
                gt_uncerts.append(float(lbl.get("uncertainty_gt",
                                  lbl.get("uncertainty", 0.3))))
            else:
                # Priority 2: inline fields in the record
                r_val = rc.get("semantic_risk_gt") or rc.get("risk")
                u_val = rc.get("uncertainty_gt")   or rc.get("uncertainty")
                if r_val is not None:
                    gt_risks.append(float(r_val))
                if u_val is not None:
                    gt_uncerts.append(float(u_val))

        # Only use real GT values — don't fall back to 0.3 placeholder
        mean_gt_risk  = float(np.mean(gt_risks))   if gt_risks   else None
        mean_gt_uncert= float(np.mean(gt_uncerts)) if gt_uncerts else None

        # ── Space type prior (for routing cost) ───────────────────────────
        prior      = priors.get(stype_i, {})
        prior_risk = prior.get("risk_mean",   0.3)
        prior_unc  = prior.get("uncert_mean", 0.3)

        space_nodes.append({
            "node_id":        len(space_nodes),
            "space_type":     stype_i,
            "position":       centroid,
            "n_keyframes":    len(cluster),
            "prior_risk":     prior_risk,
            "prior_uncert":   prior_unc,
            "gt_risk_mean":   round(mean_gt_risk,   4) if mean_gt_risk   is not None else None,
            "gt_uncert_mean": round(mean_gt_uncert, 4) if mean_gt_uncert is not None else None,
            "has_gt_labels":  mean_gt_risk is not None,
            "keyframe_ids":   [r.get("keyframe_idx", -1) for r in cluster],
        })

    # Coverage report
    with_gt = sum(1 for n in space_nodes if n["has_gt_labels"])
    log(f"Nodes with GT labels: {with_gt} / {len(space_nodes)} "
        f"({100*with_gt/max(len(space_nodes),1):.1f}%)")

    return space_nodes

# ─── Edge construction ────────────────────────────────────────────────────────
def build_edges(space_nodes):
    edges     = []
    positions = [np.array(n["position"]) for n in space_nodes]
    n         = len(space_nodes)

    for i in range(n):
        for j in range(i + 1, n):
            dist = float(np.linalg.norm(positions[i] - positions[j]))
            if dist <= EDGE_RADIUS:
                same = space_nodes[i]["space_type"] == space_nodes[j]["space_type"]
                edge = {
                    "source":               i,
                    "target":               j,
                    "distance":             round(dist, 3),
                    "traversal_confidence": 1.0,
                    "type_transition":      "same" if same else "cross",
                }
                edges.append(edge)
                edges.append({**edge, "source": j, "target": i})
    return edges

# ─── Build NetworkX graph ─────────────────────────────────────────────────────
def build_nx_graph(space_nodes, edges):
    G = nx.DiGraph()

    for node in space_nodes:
        nid = node["node_id"]
        G.add_node(nid,
            space_type      = node["space_type"],
            position        = node["position"],
            n_keyframes     = node["n_keyframes"],
            prior_risk      = node["prior_risk"],
            prior_uncert    = node["prior_uncert"],
            gt_risk_mean    = node["gt_risk_mean"],
            gt_uncert_mean  = node["gt_uncert_mean"],
            has_gt_labels   = node["has_gt_labels"],
        )

    for edge in edges:
        i   = edge["source"]
        j   = edge["target"]
        d   = edge["distance"]
        tau = edge["traversal_confidence"]
        s_j = space_nodes[j]["prior_risk"]
        u_j = space_nodes[j]["prior_uncert"]
        cost = ALPHA * d + BETA * s_j + GAMMA * u_j + DELTA * (1.0 - tau)
        G.add_edge(i, j,
            distance             = d,
            traversal_confidence = tau,
            type_transition      = edge["type_transition"],
            cost                 = round(cost, 4),
        )
    return G

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    log("=" * 62)
    log("SemGraph-Route — Phase 3: Semantic Scene Graph Construction")
    log(f"Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 62)

    for path, name in [(SPACE_JSON, "space_labeled_records"),
                       (PRIORS_JSON, "space_type_priors")]:
        if not path.exists():
            log(f"ERROR: {path} not found."); sys.exit(1)

    with open(SPACE_JSON, "r") as f:
        records = json.load(f)
    log(f"Loaded {len(records)} labeled keyframes.")
    with open(SPACE_JSON, "r") as f:
        records = json.load(f)
    log(f"Loaded {len(records)} labeled keyframes.")

    # Filter to training environment only so the graph matches ablation scope
    TRAIN_ENV = "abandonedfactory"
    records = [r for r in records
           if (r.get("environment") or r.get("env")) == TRAIN_ENV]
    log(f"Filtered to {TRAIN_ENV}: {len(records)} records")

    with open(PRIORS_JSON, "r") as f:
        priors = json.load(f)["space_type_priors"]
    log(f"Loaded priors: {list(priors.keys())}")

    # Load GT labels
    label_map = {}
    if LABELS_JSON.exists():
        with open(LABELS_JSON, "r") as f:
            labels = json.load(f)
        label_map = {l["keyframe_idx"]: l for l in labels}
        log(f"Loaded {len(label_map)} GT labels from training_labels.json")
    else:
        log("WARNING: training_labels.json not found — GT metrics will be None")

    # Step 1
    log("")
    log("Step 1: Aggregating keyframes into space nodes ...")
    space_nodes = aggregate_into_space_nodes(records, priors, label_map)
    log(f"  {len(records)} keyframes → {len(space_nodes)} space nodes")

    type_counts = Counter(n["space_type"] for n in space_nodes)
    log("  Node type distribution:")
    for t in VALID_TYPES:
        n   = type_counts.get(t, 0)
        pct = 100 * n / max(len(space_nodes), 1)
        bar = "█" * int(pct / 3)
        log(f"    {t:<18} {n:>4}  ({pct:5.1f}%)  {bar}")

    # Step 2
    log("")
    log("Step 2: Building traversal edges ...")
    edges = build_edges(space_nodes)
    n_unique = len(edges) // 2
    cross    = sum(1 for e in edges if e["type_transition"] == "cross") // 2
    log(f"  {n_unique} unique edges ({len(edges)} directed)")
    log(f"  Same-type: {n_unique - cross}  Cross-type: {cross}")

    # Step 3
    log("")
    log("Step 3: Building NetworkX DiGraph ...")
    G = build_nx_graph(space_nodes, edges)
    log(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    G_und  = G.to_undirected()
    comps  = list(nx.connected_components(G_und))
    largest= max(len(c) for c in comps) if comps else 0
    log(f"  Connected components: {len(comps)}  "
        f"Largest: {largest} ({100*largest/max(G.number_of_nodes(),1):.1f}%)")

    # Verify GT label coverage
    with_gt = sum(1 for n in G.nodes()
                  if G.nodes[n].get("gt_risk_mean") is not None)
    log(f"  Nodes with GT labels: {with_gt} / {G.number_of_nodes()}")
    if with_gt > 0:
        sample_risks = [G.nodes[n]["gt_risk_mean"] for n in list(G.nodes())[:5]
                        if G.nodes[n].get("gt_risk_mean") is not None]
        log(f"  Sample GT risks (first 5 nodes): {sample_risks}")

    # Step 4
    log("")
    log("Step 4: Saving ...")
    OUT_GRAPH_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_GRAPH_JSON, "w") as f:
        json.dump(nx.node_link_data(G), f, indent=2)
    log(f"  ✓ Graph → {OUT_GRAPH_JSON}")

    costs = [G[u][v]["cost"] for u, v in G.edges()]
    stats = {
        "n_input_keyframes": len(records),
        "n_space_nodes":     G.number_of_nodes(),
        "n_directed_edges":  G.number_of_edges(),
        "n_unique_edges":    n_unique,
        "cross_type_edges":  cross,
        "components":        len(comps),
        "largest_cc_pct":    round(100*largest/max(G.number_of_nodes(),1), 2),
        "gt_label_coverage": round(100*with_gt/max(G.number_of_nodes(),1), 2),
        "node_type_counts":  dict(type_counts),
        "cost_stats": {
            "mean": round(float(np.mean(costs)), 4),
            "std":  round(float(np.std(costs)),  4),
            "min":  round(float(np.min(costs)),  4),
            "max":  round(float(np.max(costs)),  4),
        },
    }
    with open(STATS_JSON, "w") as f:
        json.dump(stats, f, indent=2)

    elapsed = time.time() - t0
    log("")
    log("=" * 62)
    log("PHASE 3 COMPLETE")
    log("=" * 62)
    log(f"  Space nodes     : {G.number_of_nodes()}")
    log(f"  Directed edges  : {G.number_of_edges()}")
    log(f"  GT coverage     : {stats['gt_label_coverage']}%")
    log(f"  Mean edge cost  : {stats['cost_stats']['mean']}")
    log(f"  Done in         : {elapsed:.1f}s")
    log("=" * 62)
    log("NEXT STEP: python phase3_semgraph_planner.py")
    save_log()


if __name__ == "__main__":
    main()
    import sys as _sys
    if _sys.platform == "win32":
        input("\nPress Enter to close...")
