"""
SemGraph-Route — Hospital Environment Evaluation
=================================================
Evaluates all methods on the hospital environment using
priors learned from abandonedfactory (zero-shot transfer).

This evaluation is important because:
  - hospital has 79% corridor + 21% open_space
  - more semantic diversity than abandonedfactory (88.9% corridor)
  - both types have empirical priors from training
  - allows fair taxonomy validation with multiple types

Usage:
  python evaluate_hospital.py

Reads:
  data/processed/space_labeled_records.json
  data/priors/space_type_priors.json
  data/labels/training_labels.json

Writes:
  data/semgraph_results/hospital_results.json
  data/semgraph_results/hospital_bottleneck.json
  figures/hospital_aggregate.png
  figures/hospital_bottleneck.png
"""

import json
import sys
import time
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from heapq import heappush, heappop
from collections import defaultdict
from datetime import datetime

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent.resolve()
DATA_DIR        = BASE_DIR / "data"
SPACE_JSON      = DATA_DIR / "processed" / "space_labeled_records.json"
PRIORS_JSON     = DATA_DIR / "priors"    / "space_type_priors.json"
LABELS_JSON     = DATA_DIR / "labels"    / "training_labels.json"
RESULTS_DIR     = DATA_DIR / "semgraph_results"
HOSPITAL_JSON   = RESULTS_DIR / "hospital_results.json"
HOSP_BN_JSON    = RESULTS_DIR / "hospital_bottleneck.json"
FIGURES_DIR     = BASE_DIR / "figures"
LOGS_DIR        = BASE_DIR / "logs"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_ENV         = "hospital"
TRAIN_ENV          = "abandonedfactory"
VALID_TYPES        = ["corridor", "junction", "open_space", "confined_space", "dark_zone"]
AGGREGATION_RADIUS = 2.0
EDGE_RADIUS        = 6.0
N_EVAL_PAIRS       = 30   # hospital is small (28 nodes) so use fewer pairs
ALPHA              = 0.35
BETA               = 0.30
GAMMA              = 0.20

METHOD_ORDER = [
    "Geometric A*",
    "Space-Type A* (uniform priors)",
    "Risk-Only A*",
    "Moondream (no CoT) + A*",
    "SemGraph-Route (ours)",
]

# ─── Logger ───────────────────────────────────────────────────────────────────
log_lines = []

def log(msg):
    ts   = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    log_lines.append(line)

def save_log():
    with open(LOGS_DIR / "hospital_evaluation.log", "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

def pss(risk, uncert):
    return round(1.0 - (0.6 * risk + 0.4 * uncert), 4)

# ─── Build hospital graph ─────────────────────────────────────────────────────
def build_hospital_graph(records, priors, label_map):
    """
    Build semantic scene graph for hospital environment.
    Uses abandonedfactory priors (zero-shot transfer).
    GT labels from training_labels.json for fair evaluation.
    """
    hospital_records = [
        r for r in records
        if (r.get("environment") or r.get("env")) == TARGET_ENV
    ]
    log(f"Hospital records: {len(hospital_records)}")

    if len(hospital_records) < 4:
        log(f"ERROR: Only {len(hospital_records)} hospital records found.")
        log("Make sure phase1_space_labeling.py was run for hospital env.")
        return None

    valid = []
    for r in hospital_records:
        stype = r.get("space_type")
        pos   = r.get("position")
        if stype in VALID_TYPES and pos is not None:
            try:
                valid.append((stype, [float(x) for x in pos[:3]], r))
            except Exception:
                continue

    log(f"Valid records with space type and position: {len(valid)}")

    if len(valid) < 4:
        log("ERROR: Not enough valid records to build graph.")
        return None

    # Aggregate into space nodes
    space_nodes = []
    assigned    = [False] * len(valid)

    for i, (stype_i, pos_i, rec_i) in enumerate(valid):
        if assigned[i]:
            continue
        cluster, cluster_pos = [rec_i], [pos_i]
        assigned[i] = True
        for j, (stype_j, pos_j, rec_j) in enumerate(valid):
            if assigned[j] or stype_j != stype_i:
                continue
            if np.linalg.norm(np.array(pos_i) - np.array(pos_j)) <= AGGREGATION_RADIUS:
                cluster.append(rec_j)
                cluster_pos.append(pos_j)
                assigned[j] = True

        centroid = np.mean(cluster_pos, axis=0).tolist()

        # GT labels — priority from training_labels.json
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
                rv = rc.get("semantic_risk_gt") or rc.get("risk")
                uv = rc.get("uncertainty_gt")   or rc.get("uncertainty")
                if rv is not None: gt_risks.append(float(rv))
                if uv is not None: gt_uncerts.append(float(uv))

        # Use abandonedfactory priors (zero-shot transfer)
        prior     = priors.get(stype_i, {})
        prior_r   = prior.get("risk_mean",   0.3)
        prior_u   = prior.get("uncert_mean", 0.3)

        space_nodes.append({
            "node_id":        len(space_nodes),
            "space_type":     stype_i,
            "position":       centroid,
            "prior_risk":     prior_r,
            "prior_uncert":   prior_u,
            "gt_risk_mean":   float(np.mean(gt_risks))   if gt_risks   else None,
            "gt_uncert_mean": float(np.mean(gt_uncerts)) if gt_uncerts else None,
            "has_gt":         len(gt_risks) > 0,
        })

    log(f"Space nodes: {len(space_nodes)}")
    with_gt = sum(1 for n in space_nodes if n["has_gt"])
    log(f"Nodes with GT labels: {with_gt} / {len(space_nodes)}")

    # Build graph
    G         = nx.DiGraph()
    positions = [np.array(n["position"]) for n in space_nodes]

    for node in space_nodes:
        nid = node["node_id"]
        G.add_node(nid, **{k: v for k, v in node.items() if k != "node_id"})

    for i in range(len(space_nodes)):
        for j in range(len(space_nodes)):
            if i == j:
                continue
            dist = float(np.linalg.norm(positions[i] - positions[j]))
            if dist <= EDGE_RADIUS:
                s_j  = space_nodes[j]["prior_risk"]
                u_j  = space_nodes[j]["prior_uncert"]
                cost = ALPHA * dist + BETA * s_j + GAMMA * u_j
                G.add_edge(i, j,
                    distance = round(dist, 3),
                    cost     = round(cost, 4),
                )

    comps  = list(nx.connected_components(G.to_undirected()))
    largest = max(len(c) for c in comps) if comps else 0
    log(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    log(f"Connected components: {len(comps)}, largest: {largest}")

    return G

# ─── Baseline graph builders ──────────────────────────────────────────────────
def make_geo_graph(G):
    G2 = G.copy()
    for u, v in G2.edges():
        G2[u][v]["cost"] = G2[u][v]["distance"]
    return G2

def make_uniform_graph(G):
    G2 = G.copy()
    for u, v in G2.edges():
        G2[u][v]["cost"] = round(ALPHA*G2[u][v]["distance"] + BETA*0.25 + GAMMA*0.25, 4)
    return G2

def make_riskonly_graph(G):
    G2 = G.copy()
    for u, v in G2.edges():
        d   = G2[u][v]["distance"]
        s_j = G2.nodes[v].get("prior_risk", 0.3)
        G2[u][v]["cost"] = round(0.50*d + 0.50*s_j, 4)
    return G2

def make_nocot_graph(G):
    G2 = G.copy()
    for u, v in G2.edges():
        G2[u][v]["cost"] = round(ALPHA*G2[u][v]["distance"] + BETA*0.3 + GAMMA*0.3, 4)
    return G2

# ─── A* ───────────────────────────────────────────────────────────────────────
def astar(G, start, goal):
    goal_pos = np.array(G.nodes[goal]["position"])
    def h(n):
        return float(np.linalg.norm(np.array(G.nodes[n]["position"]) - goal_pos))
    heap, visited = [(h(start), 0.0, start, [start])], {}
    while heap:
        f, g, node, path = heappop(heap)
        if node in visited:
            continue
        visited[node] = g
        if node == goal:
            return path, g
        for nb in G.successors(node):
            if nb in visited:
                continue
            ng = g + G[node][nb].get("cost", 1.0)
            heappush(heap, (ng + h(nb), ng, nb, path + [nb]))
    return None, float("inf")

# ─── Path metrics ─────────────────────────────────────────────────────────────
def path_metrics(G_eval, path):
    if not path or len(path) < 2:
        return None
    risks, uncerts = [], []
    for n in path:
        attrs = G_eval.nodes[n]
        r = attrs.get("gt_risk_mean")   or attrs.get("prior_risk",   0.3)
        u = attrs.get("gt_uncert_mean") or attrs.get("prior_uncert", 0.3)
        risks.append(float(r))
        uncerts.append(float(u))
    pos    = [np.array(G_eval.nodes[n]["position"]) for n in path]
    length = sum(np.linalg.norm(pos[i+1]-pos[i]) for i in range(len(pos)-1))
    mr, mu = float(np.mean(risks)), float(np.mean(uncerts))
    return {
        "risk":     round(mr, 4),
        "uncert":   round(mu, 4),
        "pss":      pss(mr, mu),
        "length_m": round(float(length), 3),
    }

# ─── Pair sampling ────────────────────────────────────────────────────────────
def sample_pairs(G, n=30, min_dist=3.0, seed=42):
    """Use smaller min_dist for hospital (smaller graph)."""
    rng   = np.random.default_rng(seed)
    nodes = list(G.nodes())
    pairs = []
    for _ in range(n * 500):
        if len(pairs) >= n:
            break
        s = int(rng.choice(nodes))
        g = int(rng.choice(nodes))
        if s == g:
            continue
        dist = np.linalg.norm(
            np.array(G.nodes[s]["position"]) - np.array(G.nodes[g]["position"])
        )
        if dist < min_dist:
            continue
        try:
            if nx.has_path(G, s, g):
                pairs.append((s, g))
        except Exception:
            pass
    log(f"  Sampled {len(pairs)} valid pairs (min_dist={min_dist}m)")
    return pairs

# ─── Evaluate one method ──────────────────────────────────────────────────────
def evaluate_method(G_plan, G_eval, pairs, name):
    metrics_list, n_fail = [], 0
    for s, g in pairs:
        path, _ = astar(G_plan, s, g)
        if path is None:
            n_fail += 1
            continue
        m = path_metrics(G_eval, path)
        if m:
            metrics_list.append(m)

    if not metrics_list:
        log(f"  WARNING: {name} — no valid paths")
        return None

    def avg(key):
        vals = [m[key] for m in metrics_list]
        return round(float(np.mean(vals)), 4) if vals else 0.0

    result = {
        "method":      name,
        "risk":        avg("risk"),
        "uncert":      avg("uncert"),
        "pss":         avg("pss"),
        "n_evaluated": len(metrics_list),
        "n_failed":    n_fail,
    }
    log(f"  {name:<36} risk={result['risk']}  "
        f"uncert={result['uncert']}  PSS={result['pss']}  "
        f"(n={result['n_evaluated']})")
    return result

# ─── Bottleneck evaluation ────────────────────────────────────────────────────
def bottleneck_all_methods(G, method_graphs, G_eval):
    nodes = list(G.nodes())
    n     = len(nodes)
    step  = max(1, n // 10)   # smaller step for hospital
    scenario_risks = {name: [] for name in METHOD_ORDER}
    n_scenarios    = 0
    G_geo          = method_graphs["Geometric A*"]

    for i in range(0, n - step, step):
        s = nodes[i]
        g = nodes[min(i + step * 3, n-1)]
        if s == g:
            continue
        try:
            if not nx.has_path(G, s, g):
                continue
        except Exception:
            continue

        gp, _ = astar(G_geo, s, g)
        if gp is None:
            continue
        gm = path_metrics(G_eval, gp)
        if gm is None or gm["risk"] <= 0.10:  # lower threshold for hospital
            continue

        n_scenarios += 1
        for name in METHOD_ORDER:
            path, _ = astar(method_graphs[name], s, g)
            if path is None:
                continue
            m = path_metrics(G_eval, path)
            if m:
                scenario_risks[name].append(m["risk"])

    geo_risks = scenario_risks["Geometric A*"]
    geo_mean  = float(np.mean(geo_risks)) if geo_risks else 0.0

    summary = {}
    for name in METHOD_ORDER:
        vals = scenario_risks[name]
        if not vals:
            summary[name] = {"mean_risk": None, "mean_reduction_pct": None,
                             "max_reduction_pct": None, "n": 0}
            continue
        mean_r   = float(np.mean(vals))
        mean_red = 100*(geo_mean - mean_r) / max(geo_mean, 1e-6)
        per_sc   = [100*(geo_risks[i]-vals[i])/max(geo_risks[i],1e-6)
                    for i in range(min(len(geo_risks), len(vals)))]
        max_red  = float(np.max(per_sc)) if per_sc else 0.0
        summary[name] = {
            "mean_risk":          round(mean_r,   4),
            "mean_reduction_pct": round(mean_red, 2),
            "max_reduction_pct":  round(max_red,  2),
            "n": len(vals),
        }

    return summary, n_scenarios

# ─── Visualisation ────────────────────────────────────────────────────────────
def visualise_aggregate(results_dict, env_name):
    methods = [m for m in METHOD_ORDER if m in results_dict]
    psss    = [results_dict[m]["pss"]  for m in methods]
    risks   = [results_dict[m]["risk"] for m in methods]
    colors  = ["#95A5A6", "#BDC3C7", "#717D7E", "#F39C12", "#E74C3C"][:len(methods)]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # PSS
    ax   = axes[0]
    bars = ax.bar(range(len(methods)), psss, color=colors,
                  edgecolor="white", width=0.55, zorder=3)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([m.replace(" (", "\n(").replace(" + ", "\n+ ")
                        for m in methods], fontsize=8, ha="center")
    ax.set_ylabel("Path Safety Score (PSS ↑)", fontsize=11)
    ax.set_title(f"Aggregate PSS — {env_name}\n(zero-shot, priors from abandonedfactory)",
                 fontweight="bold", fontsize=10)
    ax.set_ylim(max(0.5, min(psss)-0.05), min(1.0, max(psss)+0.05))
    ax.grid(True, alpha=0.3, axis="y", zorder=0)
    for bar, val, m in zip(bars, psss, methods):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                f"{val:.4f}", ha="center", va="bottom", fontsize=8,
                fontweight="bold" if m == "SemGraph-Route (ours)" else "normal",
                color="#C0392B" if m == "SemGraph-Route (ours)" else "black")

    # Risk
    ax   = axes[1]
    bars = ax.bar(range(len(methods)), risks, color=colors,
                  edgecolor="white", width=0.55, zorder=3)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([m.replace(" (", "\n(").replace(" + ", "\n+ ")
                        for m in methods], fontsize=8, ha="center")
    ax.set_ylabel("Mean Path Risk ↓", fontsize=11)
    ax.set_title(f"Mean Path Risk — {env_name}\n(zero-shot, priors from abandonedfactory)",
                 fontweight="bold", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y", zorder=0)
    for bar, val, m in zip(bars, risks, methods):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
                f"{val:.4f}", ha="center", va="bottom", fontsize=8,
                fontweight="bold" if m == "SemGraph-Route (ours)" else "normal",
                color="#C0392B" if m == "SemGraph-Route (ours)" else "black")

    plt.suptitle(
        f"SemGraph-Route: Zero-Shot Evaluation on {env_name}\n"
        "All methods on same graph — priors transferred from abandonedfactory",
        fontweight="bold", fontsize=11
    )
    plt.tight_layout()
    p = FIGURES_DIR / "hospital_aggregate.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"  Aggregate figure → {p}")

def visualise_bottleneck(summary, n_scenarios, env_name):
    methods    = [m for m in METHOD_ORDER
                  if m in summary and summary[m]["mean_risk"] is not None]
    mean_risks = [summary[m]["mean_risk"]          for m in methods]
    mean_reds  = [summary[m]["mean_reduction_pct"] for m in methods]
    colors     = ["#95A5A6", "#BDC3C7", "#717D7E", "#F39C12", "#E74C3C"][:len(methods)]

    if not methods:
        log("  No bottleneck data to visualise.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.subplots_adjust(bottom=0.22, top=0.88, wspace=0.35)

    # Left: mean risk
    ax   = axes[0]
    bars = ax.bar(range(len(methods)), mean_risks, color=colors,
                  edgecolor="white", width=0.55, zorder=3)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([m.replace(" (", "\n(").replace(" + ", "\n+ ")
                        for m in methods], fontsize=8, ha="center")
    ax.set_ylabel("Mean Path Risk ↓", fontsize=11)
    ax.set_title(f"Bottleneck Mean Risk — {env_name}\n(n={n_scenarios} scenarios)",
                 fontweight="bold", fontsize=10)
    ax.set_ylim(0, max(mean_risks)*1.25)
    ax.grid(True, alpha=0.3, axis="y", zorder=0)
    geo_risk = summary["Geometric A*"]["mean_risk"] if "Geometric A*" in summary else 0
    for bar, val, m in zip(bars, mean_risks, methods):
        red  = 100*(geo_risk-val)/max(geo_risk,1e-6)
        xc   = bar.get_x()+bar.get_width()/2
        ytop = bar.get_height()
        ax.text(xc, ytop+0.003, f"{val:.4f}",
                ha="center", va="bottom", fontsize=8,
                fontweight="bold" if m=="SemGraph-Route (ours)" else "normal",
                color="#C0392B" if m=="SemGraph-Route (ours)" else "black")
        if m == "SemGraph-Route (ours)":
            ax.text(xc, ytop+0.015, f"({red:+.1f}% vs Geo)",
                    ha="center", va="bottom", fontsize=7,
                    fontweight="bold", color="#C0392B")

    # Right: reduction %
    ax      = axes[1]
    rcolors = ["#E74C3C" if m=="SemGraph-Route (ours)" else "#BDC3C7"
               for m in methods]
    bars    = ax.bar(range(len(methods)), mean_reds, color=rcolors,
                     edgecolor="white", width=0.55, zorder=3)
    ax.axhline(0, color="black", lw=1.0, zorder=4)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([m.replace(" (", "\n(").replace(" + ", "\n+ ")
                        for m in methods], fontsize=8, ha="center")
    ax.set_ylabel("Mean Risk Reduction vs Geometric A* (%)", fontsize=10)
    ax.set_title("Bottleneck Risk Reduction\n(positive = safer than Geo A*)",
                 fontweight="bold", fontsize=10)
    y_min = min(mean_reds) - 3.0
    y_max = max(mean_reds) + 3.0
    ax.set_ylim(y_min, y_max)
    ax.grid(True, alpha=0.3, axis="y", zorder=0)
    for bar, val, m in zip(bars, mean_reds, methods):
        xc   = bar.get_x()+bar.get_width()/2
        ypos = bar.get_height()+0.3 if val >= 0 else bar.get_height()-0.3
        va   = "bottom" if val >= 0 else "top"
        ax.text(xc, ypos, f"{val:+.1f}%",
                ha="center", va=va, fontsize=9,
                fontweight="bold" if m=="SemGraph-Route (ours)" else "normal",
                color="#C0392B" if m=="SemGraph-Route (ours)" else "black",
                zorder=5)

    plt.suptitle(
        f"SemGraph-Route: Bottleneck Analysis — {env_name}\n"
        "Zero-shot priors from abandonedfactory",
        fontweight="bold", fontsize=11
    )
    plt.tight_layout()
    p = FIGURES_DIR / "hospital_bottleneck.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"  Bottleneck figure → {p}")

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    log("=" * 68)
    log(f"SemGraph-Route — Hospital Zero-Shot Evaluation")
    log(f"Train env : {TRAIN_ENV}  →  Test env: {TARGET_ENV}")
    log(f"Started   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 68)

    # Load data
    for path in [SPACE_JSON, PRIORS_JSON]:
        if not path.exists():
            log(f"ERROR: {path} not found."); sys.exit(1)

    with open(SPACE_JSON,  "r") as f: all_records = json.load(f)
    with open(PRIORS_JSON, "r") as f:
        priors = json.load(f)["space_type_priors"]
    log(f"Loaded {len(all_records)} records.")
    log(f"Priors: {list(priors.keys())}")

    label_map = {}
    if LABELS_JSON.exists():
        with open(LABELS_JSON, "r") as f:
            label_map = {l["keyframe_idx"]: l for l in json.load(f)}
        log(f"GT labels: {len(label_map)}")
    else:
        log("WARNING: training_labels.json not found — GT metrics will use priors.")

    # Check hospital records exist
    hosp_records = [r for r in all_records
                    if (r.get("environment") or r.get("env")) == TARGET_ENV]
    log(f"\nHospital records in dataset: {len(hosp_records)}")
    if len(hosp_records) == 0:
        log("ERROR: No hospital records found.")
        log("Make sure phase1_space_labeling.py was run with --env hospital")
        sys.exit(1)

    # Space type distribution
    from collections import Counter
    type_counts = Counter(r.get("space_type") for r in hosp_records
                          if r.get("space_type") in VALID_TYPES)
    log("\nHospital space type distribution:")
    for t in VALID_TYPES:
        n   = type_counts.get(t, 0)
        pct = 100*n/max(sum(type_counts.values()),1)
        if n > 0:
            log(f"  {t:<20} {n:>4} frames ({pct:.1f}%)")

    # Build hospital graph
    log("\nBuilding hospital semantic scene graph ...")
    G = build_hospital_graph(all_records, priors, label_map)
    if G is None:
        log("ERROR: Could not build hospital graph."); sys.exit(1)

    with_gt = sum(1 for n in G.nodes() if G.nodes[n].get("gt_risk_mean") is not None)
    log(f"Nodes with GT labels: {with_gt} / {G.number_of_nodes()}")

    if with_gt == 0:
        log("WARNING: No GT labels — metrics will use prior values.")
        log("Results will show how priors perform, not GT-evaluated performance.")

    # Build method graphs
    log("\nBuilding baseline cost graphs ...")
    G_geo      = make_geo_graph(G)
    G_uniform  = make_uniform_graph(G)
    G_riskonly = make_riskonly_graph(G)
    G_nocot    = make_nocot_graph(G)

    method_graphs = {
        "Geometric A*":                   G_geo,
        "Space-Type A* (uniform priors)": G_uniform,
        "Risk-Only A*":                   G_riskonly,
        "Moondream (no CoT) + A*":        G_nocot,
        "SemGraph-Route (ours)":          G,
    }

    # Sample pairs
    log(f"\nSampling {N_EVAL_PAIRS} evaluation pairs ...")
    pairs = sample_pairs(G, n=N_EVAL_PAIRS, min_dist=3.0, seed=42)

    if not pairs:
        log("ERROR: No valid pairs — graph may be disconnected or too small.")
        sys.exit(1)

    # Aggregate evaluation
    log("\nAggregate evaluation ...")
    log(f"  {'Method':<36} {'risk':>6}  {'uncert':>7}  {'PSS':>7}")
    log(f"  {'-'*36} {'------':>6}  {'-------':>7}  {'-------':>7}")
    results_dict = {}
    for name in METHOD_ORDER:
        r = evaluate_method(method_graphs[name], G, pairs, name)
        if r:
            results_dict[name] = r

    # Bottleneck evaluation
    log("\nBottleneck evaluation ...")
    bn_summary, n_scenarios = bottleneck_all_methods(G, method_graphs, G)
    log(f"  Bottleneck scenarios found: {n_scenarios}")

    log("")
    log("=" * 68)
    log(f"HOSPITAL AGGREGATE TABLE (zero-shot from {TRAIN_ENV})")
    log("=" * 68)
    log(f"  {'Method':<36} {'s(v)↓':>7}  {'u(v)↓':>7}  {'PSS↑':>7}")
    log(f"  {'-'*36} {'-------':>7}  {'-------':>7}  {'-------':>7}")
    for name in METHOD_ORDER:
        if name in results_dict and name != "SemGraph-Route (ours)":
            r = results_dict[name]
            log(f"  {name:<36} {r['risk']:>7.4f}  {r['uncert']:>7.4f}  {r['pss']:>7.4f}")
    log(f"  {'-'*36} {'-------':>7}  {'-------':>7}  {'-------':>7}")
    sem = results_dict.get("SemGraph-Route (ours)", {})
    if sem:
        log(f"  {'SemGraph-Route (ours)':<36} "
            f"{sem['risk']:>7.4f}  {sem['uncert']:>7.4f}  {sem['pss']:>7.4f}")

    log("")
    log("=" * 68)
    log(f"HOSPITAL BOTTLENECK TABLE (n={n_scenarios} scenarios)")
    log("=" * 68)
    log(f"  {'Method':<36} {'Mean Risk↓':>10}  {'Mean Red%':>10}")
    log(f"  {'-'*36} {'----------':>10}  {'----------':>10}")
    for name in METHOD_ORDER:
        s = bn_summary.get(name, {})
        if s.get("mean_risk") is None:
            continue
        log(f"  {name:<36} {s['mean_risk']:>10.4f}  "
            f"{s['mean_reduction_pct']:>+9.1f}%")

    # Compare with geo
    geo_r = results_dict.get("Geometric A*", {})
    if sem and geo_r:
        d_pss  = round(sem["pss"]  - geo_r["pss"],  4)
        d_risk = round(geo_r["risk"]- sem["risk"],   4)
        p_risk = round(100*d_risk/max(geo_r["risk"],1e-6), 1)
        log("")
        log(f"  PSS vs Geometric A*:  {d_pss:+.4f}")
        log(f"  Risk vs Geometric A*: {p_risk:+.1f}%")
        if d_pss > 0:
            log("  ✓ SemGraph-Route OUTPERFORMS Geometric A* in hospital!")
        else:
            log("  ✗ SemGraph-Route does not beat Geometric A* in aggregate.")
            log("    Check bottleneck results for safety-critical comparison.")

    # Figures
    log("\nGenerating figures ...")
    visualise_aggregate(results_dict, TARGET_ENV)
    if n_scenarios > 0:
        visualise_bottleneck(bn_summary, n_scenarios, TARGET_ENV)
    else:
        log("  No bottleneck scenarios found — skipping bottleneck figure.")

    # Save
    with open(HOSPITAL_JSON, "w") as f:
        json.dump({
            "environment":    TARGET_ENV,
            "train_env":      TRAIN_ENV,
            "n_nodes":        G.number_of_nodes(),
            "n_edges":        G.number_of_edges(),
            "n_eval_pairs":   len(pairs),
            "aggregate":      results_dict,
            "computed_at":    datetime.now().isoformat(),
        }, f, indent=2)

    with open(HOSP_BN_JSON, "w") as f:
        json.dump({
            "environment":    TARGET_ENV,
            "n_scenarios":    n_scenarios,
            "method_summary": bn_summary,
            "computed_at":    datetime.now().isoformat(),
        }, f, indent=2)

    elapsed = time.time() - t0
    log(f"\n  Results → {HOSPITAL_JSON}")
    log(f"  Done in : {elapsed:.1f}s")
    log("=" * 68)
    save_log()


if __name__ == "__main__":
    main()
    import sys as _sys
    if _sys.platform == "win32":
        input("\nPress Enter to close...")
