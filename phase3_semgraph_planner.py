"""
SemGraph-Route — Phase 4: Semantic Graph A* Planner + Baseline Comparison
=========================================================================
Runs A* on the semantic scene graph and compares against four baselines,
all evaluated on the SAME 141-node semantic scene graph with GT labels.

Baselines:
  1. Geometric A*                  distance-only cost
  2. Space-Type A* (uniform prior) space types used but uniform cost
  3. Risk-Only A*                  semantic risk cost, no uncertainty term
  4. Moondream (no CoT) + A*       uniform semantic cost, no type awareness

Evaluations:
  - Aggregate path quality (all methods, same 50 pairs)
  - Bottleneck scenario analysis (all methods, geo risk > 0.15)

Usage:
  python phase3_semgraph_planner.py

Reads:
  data/graphs/semantic_scene_graph.json

Writes:
  data/semgraph_results/semgraph_results.json
  data/semgraph_results/main_results_table.json
  data/semgraph_results/bottleneck_all_methods.json
  figures/semgraph_results.png
  figures/bottleneck_comparison.png
"""

import json
import sys
import time
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from heapq import heappush, heappop
from datetime import datetime

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent.resolve()
DATA_DIR        = BASE_DIR / "data"
SEM_GRAPH_JSON  = DATA_DIR / "graphs"           / "semantic_scene_graph.json"
RESULTS_DIR     = DATA_DIR / "semgraph_results"
RESULTS_JSON    = RESULTS_DIR / "semgraph_results.json"
TABLE_JSON      = RESULTS_DIR / "main_results_table.json"
BOTTLENECK_JSON = RESULTS_DIR / "bottleneck_all_methods.json"
FIGURES_DIR     = BASE_DIR / "figures"
LOGS_DIR        = BASE_DIR / "logs"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

N_EVAL_PAIRS   = 50
ALPHA          = 0.35
BETA           = 0.30
GAMMA          = 0.20

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
    p = LOGS_DIR / "phase4_semgraph_planner.log"
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

def pss(risk, uncert):
    return round(1.0 - (0.6 * risk + 0.4 * uncert), 4)

# ─── A* ───────────────────────────────────────────────────────────────────────
def astar(G, start, goal):
    goal_pos = np.array(G.nodes[goal]["position"])
    def h(n):
        return float(np.linalg.norm(
            np.array(G.nodes[n]["position"]) - goal_pos
        ))
    heap    = [(h(start), 0.0, start, [start])]
    visited = {}
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

# ─── Baseline graph builders ──────────────────────────────────────────────────
def make_geo_graph(G):
    """Geometric A*: distance only."""
    G2 = G.copy()
    for u, v in G2.edges():
        G2[u][v]["cost"] = G2[u][v]["distance"]
    return G2

def make_uniform_prior_graph(G):
    """Space-Type A* (uniform priors): taxonomy but no prior differentiation."""
    G2 = G.copy()
    for u, v in G2.edges():
        d = G2[u][v]["distance"]
        G2[u][v]["cost"] = round(ALPHA*d + BETA*0.25 + GAMMA*0.25, 4)
    return G2

def make_risk_only_graph(G):
    """Risk-Only A*: learned risk prior, no uncertainty term."""
    G2 = G.copy()
    for u, v in G2.edges():
        d   = G2[u][v]["distance"]
        s_j = G2.nodes[v].get("prior_risk", 0.3)
        G2[u][v]["cost"] = round(0.50*d + 0.50*s_j, 4)
    return G2

def make_nocot_graph(G):
    """Moondream (no CoT)+A*: uniform cost, no space-type awareness."""
    G2 = G.copy()
    for u, v in G2.edges():
        d = G2[u][v]["distance"]
        G2[u][v]["cost"] = round(ALPHA*d + BETA*0.3 + GAMMA*0.3, 4)
    return G2

# ─── Path evaluation ──────────────────────────────────────────────────────────
def path_metrics(G_eval, path):
    """Always evaluate on GT labels for fair comparison."""
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
        "n_nodes":  len(path),
    }

# ─── Pair sampling ────────────────────────────────────────────────────────────
def sample_pairs(G, n=50, min_dist=10.0, seed=42):
    rng   = np.random.default_rng(seed)
    nodes = list(G.nodes())
    pairs = []
    for _ in range(n * 200):
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

# ─── Aggregate evaluation ─────────────────────────────────────────────────────
def evaluate_method(G_plan, G_eval, pairs, method_name):
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
        log(f"  WARNING: {method_name} — no valid paths")
        return None

    def avg(key):
        vals = [m[key] for m in metrics_list if m[key] is not None]
        return round(float(np.mean(vals)), 4) if vals else 0.0

    result = {
        "method":      method_name,
        "risk":        avg("risk"),
        "uncert":      avg("uncert"),
        "pss":         avg("pss"),
        "n_evaluated": len(metrics_list),
        "n_failed":    n_fail,
    }
    log(f"  {method_name:<36} risk={result['risk']}  "
        f"uncert={result['uncert']}  PSS={result['pss']}  "
        f"(n={result['n_evaluated']})")
    return result

# ─── Bottleneck: all methods ──────────────────────────────────────────────────
def bottleneck_all_methods(G, method_graphs, G_eval):
    """
    Find bottleneck scenarios where geo path risk > 0.15.
    Evaluate ALL methods on each scenario.
    Returns per-method mean risk and reduction vs Geometric A*.
    """
    nodes = list(G.nodes())
    n     = len(nodes)
    step  = max(1, n // 30)

    # Collect per-scenario per-method risk
    scenario_risks = {name: [] for name in METHOD_ORDER}
    n_scenarios    = 0

    G_geo = method_graphs["Geometric A*"]

    for i in range(0, n - step, step):
        s = nodes[i]
        g = nodes[min(i + step * 5, n-1)]
        if s == g:
            continue
        try:
            if not nx.has_path(G, s, g):
                continue
        except Exception:
            continue

        # Check geo path risk
        gp, _ = astar(G_geo, s, g)
        if gp is None:
            continue
        gm = path_metrics(G_eval, gp)
        if gm is None or gm["risk"] <= 0.15:
            continue

        n_scenarios += 1

        # Evaluate all methods on this scenario
        for name in METHOD_ORDER:
            Gm       = method_graphs[name]
            path, _  = astar(Gm, s, g)
            if path is None:
                continue
            m = path_metrics(G_eval, path)
            if m:
                scenario_risks[name].append(m["risk"])

    # Compute summary stats
    geo_risks  = scenario_risks["Geometric A*"]
    geo_mean   = float(np.mean(geo_risks)) if geo_risks else 0.0
    geo_max    = float(np.max(geo_risks))  if geo_risks else 0.0

    summary = {}
    for name in METHOD_ORDER:
        vals = scenario_risks[name]
        if not vals:
            summary[name] = {"mean_risk": None, "max_risk": None,
                             "mean_reduction_pct": None,
                             "max_reduction_pct": None, "n": 0}
            continue
        mean_r = float(np.mean(vals))
        max_r  = float(np.max(vals))
        mean_red = 100*(geo_mean - mean_r) / max(geo_mean, 1e-6)
        # Per-scenario max reduction
        per_scenario = [100*(geo_risks[i]-vals[i])/max(geo_risks[i],1e-6)
                        for i in range(min(len(geo_risks), len(vals)))]
        max_red = float(np.max(per_scenario)) if per_scenario else 0.0
        summary[name] = {
            "mean_risk":          round(mean_r,   4),
            "max_risk":           round(max_r,    4),
            "mean_reduction_pct": round(mean_red, 2),
            "max_reduction_pct":  round(max_red,  2),
            "n":                  len(vals),
        }

    log("")
    log("=" * 68)
    log(f"BOTTLENECK COMPARISON — ALL METHODS  (n={n_scenarios} scenarios, geo risk>0.15)")
    log("=" * 68)
    log(f"  {'Method':<36} {'Mean Risk↓':>10}  {'Max Risk↓':>10}  {'Mean Red%':>10}")
    log(f"  {'-'*36} {'----------':>10}  {'----------':>10}  {'----------':>10}")
    for name in METHOD_ORDER:
        s = summary[name]
        if s["mean_risk"] is None:
            log(f"  {name:<36} {'N/A':>10}")
            continue
        log(f"  {name:<36} {s['mean_risk']:>10.4f}  "
            f"{s['max_risk']:>10.4f}  {s['mean_reduction_pct']:>+9.1f}%")

    return summary, n_scenarios

# ─── Visualisation: aggregate ─────────────────────────────────────────────────
def visualise_aggregate(G, results_dict):
    TYPE_COLORS = {
        "corridor":       "#3498DB",
        "junction":       "#E67E22",
        "open_space":     "#2ECC71",
        "confined_space": "#E74C3C",
        "dark_zone":      "#8E44AD",
    }
    nodes   = list(G.nodes())
    pos_arr = np.array([G.nodes[n]["position"] for n in nodes])
    types   = [G.nodes[n].get("space_type", "corridor") for n in nodes]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Left: space type graph
    ax = axes[0]
    for t, c in TYPE_COLORS.items():
        mask = [i for i, nt in enumerate(types) if nt == t]
        if mask:
            ax.scatter(pos_arr[mask, 0], pos_arr[mask, 1],
                       c=c, s=18, alpha=0.75, label=t, zorder=2)
    ax.set_title("Semantic Scene Graph — Space Type Nodes",
                 fontweight="bold", fontsize=11)
    ax.legend(fontsize=8)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.2)

    # Right: aggregate PSS bar chart
    ax    = axes[1]
    methods = [m for m in METHOD_ORDER if m in results_dict]
    psss    = [results_dict[m]["pss"] for m in methods]
    colors  = ["#BDC3C7", "#AAB7B8", "#717D7E", "#F39C12", "#E74C3C"][:len(methods)]

    bars = ax.barh(methods, psss, color=colors, edgecolor="white", height=0.5)
    ax.set_xlabel("Path Safety Score (PSS ↑)", fontsize=11)
    ax.set_title(
        "Aggregate Path Quality — All Methods\n"
        "(TartanAir abandonedfactory, 141-node graph, GT labels)",
        fontweight="bold", fontsize=10
    )
    min_pss = max(0.75, min(psss) - 0.01)
    max_pss = min(1.00, max(psss) + 0.01)
    ax.set_xlim(min_pss, max_pss)
    ax.grid(True, alpha=0.3, axis="x")
    for bar, val in zip(bars, psss):
        ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    p = FIGURES_DIR / "semgraph_results.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"  Aggregate figure → {p}")

# ─── Visualisation: bottleneck ────────────────────────────────────────────────
def visualise_bottleneck(summary, n_scenarios):
    methods     = [m for m in METHOD_ORDER if summary[m]["mean_risk"] is not None]
    mean_risks  = [summary[m]["mean_risk"]          for m in methods]
    mean_reds   = [summary[m]["mean_reduction_pct"] for m in methods]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: mean risk in bottleneck scenarios
    ax     = axes[0]
    colors = ["#BDC3C7", "#AAB7B8", "#717D7E", "#F39C12", "#E74C3C"][:len(methods)]
    bars   = ax.bar(range(len(methods)), mean_risks,
                    color=colors, edgecolor="white", width=0.6)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([m.replace(" (", "\n(").replace(" + ", "\n+ ")
                        for m in methods],
                       rotation=0, ha="center", fontsize=7.5)
    ax.set_ylabel("Mean Path Risk ↓", fontsize=11)
    ax.set_title(f"Mean Path Risk in Bottleneck Scenarios\n"
                 f"(n={n_scenarios}, geo risk > 0.15)",
                 fontweight="bold", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    geo_risk = summary["Geometric A*"]["mean_risk"]
    for bar, val, m in zip(bars, mean_risks, methods):
        red = 100*(geo_risk - val)/max(geo_risk, 1e-6)
        label = f"{val:.4f}"
        if m == "SemGraph-Route (ours)":
            label += f"\n({red:+.1f}% vs Geo)"
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.003,
                    label, ha="center", va="bottom",
                    fontsize=7, fontweight="bold", color="#C0392B")
        else:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.003,
                    label, ha="center", va="bottom", fontsize=7)

    # Right: mean risk reduction vs Geometric A*
    ax      = axes[1]
    sem_idx = methods.index("SemGraph-Route (ours)") if "SemGraph-Route (ours)" in methods else -1
    rcolors = ["#E74C3C" if i == sem_idx else "#BDC3C7"
               for i in range(len(methods))]
    bars    = ax.bar(range(len(methods)), mean_reds,
                     color=rcolors, edgecolor="white", width=0.6)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([m.replace(" (", "\n(").replace(" + ", "\n+ ")
                        for m in methods],
                       rotation=0, ha="center", fontsize=7.5)
    ax.set_ylabel("Mean Risk Reduction vs Geometric A* (%)", fontsize=10)
    ax.set_title("Risk Reduction in Bottleneck Scenarios\n"
                 "Relative to Geometric A* (higher is better)",
                 fontweight="bold", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, mean_reds):
        ypos = bar.get_height() + 0.3 if val >= 0 else bar.get_height() - 1.5
        ax.text(bar.get_x() + bar.get_width()/2, ypos,
                f"{val:+.1f}%", ha="center", va="bottom", fontsize=8,
                fontweight="bold")

    plt.suptitle(
        "SemGraph-Route: Bottleneck Scenario Analysis\n"
        "All methods evaluated on same 141-node graph with GT labels",
        fontweight="bold", fontsize=11
    )
    plt.tight_layout()
    p = FIGURES_DIR / "bottleneck_comparison.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"  Bottleneck figure → {p}")

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    log("=" * 68)
    log("SemGraph-Route — Phase 4: A* Planner + Baseline Comparison")
    log(f"Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 68)

    if not SEM_GRAPH_JSON.exists():
        log(f"ERROR: {SEM_GRAPH_JSON} not found."); sys.exit(1)

    with open(SEM_GRAPH_JSON, "r") as f:
        G = nx.node_link_graph(json.load(f))
    log(f"Semantic graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    with_gt = sum(1 for n in G.nodes() if G.nodes[n].get("gt_risk_mean") is not None)
    log(f"Nodes with GT labels: {with_gt} / {G.number_of_nodes()}")
    if with_gt == 0:
        log("WARNING: No GT labels — re-run build_scene_graph.py first.")

    # Build all cost graphs
    log("\nBuilding baseline cost graphs ...")
    G_geo      = make_geo_graph(G)
    G_uniform  = make_uniform_prior_graph(G)
    G_riskonly = make_risk_only_graph(G)
    G_nocot    = make_nocot_graph(G)

    method_graphs = {
        "Geometric A*":                   G_geo,
        "Space-Type A* (uniform priors)": G_uniform,
        "Risk-Only A*":                   G_riskonly,
        "Moondream (no CoT) + A*":        G_nocot,
        "SemGraph-Route (ours)":          G,
    }
    log("  Done: Geometric | Uniform-Prior | Risk-Only | Moondream(no CoT) | SemGraph")

    # ── Aggregate evaluation ───────────────────────────────────────────────────
    log(f"\nSampling {N_EVAL_PAIRS} evaluation pairs ...")
    pairs = sample_pairs(G, n=N_EVAL_PAIRS, min_dist=10.0, seed=42)
    if not pairs:
        log("ERROR: No valid pairs found."); sys.exit(1)

    log("\nAggregate evaluation (all methods, same pairs) ...")
    log(f"  {'Method':<36} {'risk':>6}  {'uncert':>7}  {'PSS':>7}")
    log(f"  {'-'*36} {'------':>6}  {'-------':>7}  {'-------':>7}")
    results_dict = {}
    for name in METHOD_ORDER:
        r = evaluate_method(method_graphs[name], G, pairs, name)
        if r:
            results_dict[name] = r

    # ── Bottleneck evaluation — ALL methods ────────────────────────────────────
    log("\nBottleneck evaluation — all methods ...")
    bn_summary, n_scenarios = bottleneck_all_methods(G, method_graphs, G)

    # ── Figures ────────────────────────────────────────────────────────────────
    log("\nGenerating figures ...")
    visualise_aggregate(G, results_dict)
    visualise_bottleneck(bn_summary, n_scenarios)

    # ── Save ───────────────────────────────────────────────────────────────────
    sem = results_dict.get("SemGraph-Route (ours)", {})
    geo = results_dict.get("Geometric A*", {})

    with open(TABLE_JSON, "w") as f:
        json.dump({
            "aggregate":      results_dict,
            "computed_at":    datetime.now().isoformat(),
        }, f, indent=2)

    with open(RESULTS_JSON, "w") as f:
        json.dump({
            "semgraph":    sem,
            "geometric":   geo,
            "all_methods": results_dict,
        }, f, indent=2)

    with open(BOTTLENECK_JSON, "w") as f:
        json.dump({
            "n_scenarios":    n_scenarios,
            "method_summary": bn_summary,
            "computed_at":    datetime.now().isoformat(),
        }, f, indent=2)

    # ── Paper tables ───────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    log("")
    log("=" * 68)
    log("AGGREGATE RESULTS TABLE (Paper — secondary)")
    log("PSS = 1-(0.6·risk + 0.4·uncert), higher is better")
    log("=" * 68)
    log(f"  {'Method':<36} {'s(v)↓':>7}  {'u(v)↓':>7}  {'PSS↑':>7}")
    log(f"  {'-'*36} {'-------':>7}  {'-------':>7}  {'-------':>7}")
    for name in METHOD_ORDER:
        if name in results_dict and name != "SemGraph-Route (ours)":
            r = results_dict[name]
            log(f"  {name:<36} {r['risk']:>7.4f}  {r['uncert']:>7.4f}  {r['pss']:>7.4f}")
    log(f"  {'-'*36} {'-------':>7}  {'-------':>7}  {'-------':>7}")
    if sem:
        log(f"  {'SemGraph-Route (ours)':<36} "
            f"{sem['risk']:>7.4f}  {sem['uncert']:>7.4f}  {sem['pss']:>7.4f}")

    log("")
    log("=" * 68)
    log("BOTTLENECK RESULTS TABLE (Paper — PRIMARY result)")
    log("=" * 68)
    log(f"  {'Method':<36} {'Mean Risk↓':>10}  {'Mean Red%':>10}  {'Max Red%':>10}")
    log(f"  {'-'*36} {'----------':>10}  {'----------':>10}  {'----------':>10}")
    for name in METHOD_ORDER:
        s = bn_summary.get(name, {})
        if not s or s.get("mean_risk") is None:
            continue
        log(f"  {name:<36} {s['mean_risk']:>10.4f}  "
            f"{s['mean_reduction_pct']:>+9.1f}%  "
            f"{s['max_reduction_pct']:>+9.1f}%")

    log("")
    if sem and geo:
        d_pss  = round(sem["pss"] - geo["pss"], 4)
        d_risk = round(geo["risk"]- sem["risk"], 4)
        p_risk = round(100*d_risk/max(geo["risk"],1e-6), 1)
        log(f"  Aggregate PSS vs Geo A*:  {d_pss:+.4f}")
        log(f"  Aggregate risk vs Geo A*: {p_risk:+.1f}%")

    sem_bn = bn_summary.get("SemGraph-Route (ours)", {})
    if sem_bn and sem_bn.get("mean_reduction_pct") is not None:
        log(f"  Bottleneck mean reduction: {sem_bn['mean_reduction_pct']:+.1f}%")
        log(f"  Bottleneck max  reduction: {sem_bn['max_reduction_pct']:+.1f}%")

    log(f"\n  Figures → {FIGURES_DIR}")
    log(f"  Results → {RESULTS_JSON}")
    log(f"  Done in : {elapsed:.1f}s")
    log("=" * 68)
    log("NEXT STEP: python validate_prior_stability.py")
    save_log()


if __name__ == "__main__":
    main()
    import sys as _sys
    if _sys.platform == "win32":
        input("\nPress Enter to close...")
