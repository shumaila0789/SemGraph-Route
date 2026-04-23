"""
SemGraph-Route — Regenerate Bottleneck Figure (fixed labels)
=============================================================
Loads existing bottleneck results and regenerates the figure
with properly placed labels — no re-evaluation needed.

Usage:
  python regenerate_bottleneck_figure.py

Reads:
  data/semgraph_results/bottleneck_all_methods.json

Writes:
  figures/bottleneck_comparison.png
"""

import json
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

BASE_DIR        = Path(__file__).parent.resolve()
BOTTLENECK_JSON = BASE_DIR / "data" / "semgraph_results" / "bottleneck_all_methods.json"
FIGURES_DIR     = BASE_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

METHOD_ORDER = [
    "Geometric A*",
    "Space-Type A* (uniform priors)",
    "Risk-Only A*",
    "Moondream (no CoT) + A*",
    "SemGraph-Route (ours)",
]

# Short display labels for x-axis — avoids clipping
SHORT_LABELS = [
    "Geometric\nA*",
    "Space-Type A*\n(uniform priors)",
    "Risk-Only\nA*",
    "Moondream\n(no CoT)+A*",
    "SemGraph-Route\n(ours)",
]

BAR_COLORS = {
    "Geometric A*":                   "#95A5A6",
    "Space-Type A* (uniform priors)": "#BDC3C7",
    "Risk-Only A*":                   "#717D7E",
    "Moondream (no CoT) + A*":        "#F39C12",
    "SemGraph-Route (ours)":          "#E74C3C",
}

def main():
    if not BOTTLENECK_JSON.exists():
        print(f"ERROR: {BOTTLENECK_JSON} not found.")
        print("Run phase3_semgraph_planner.py first.")
        return

    with open(BOTTLENECK_JSON) as f:
        data = json.load(f)

    summary     = data["method_summary"]
    n_scenarios = data["n_scenarios"]

    methods    = [m for m in METHOD_ORDER if m in summary
                  and summary[m]["mean_risk"] is not None]
    mean_risks = [summary[m]["mean_risk"]          for m in methods]
    mean_reds  = [summary[m]["mean_reduction_pct"] for m in methods]
    labels     = [SHORT_LABELS[METHOD_ORDER.index(m)] for m in methods]
    colors     = [BAR_COLORS[m] for m in methods]

    geo_risk = summary["Geometric A*"]["mean_risk"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.subplots_adjust(bottom=0.22, top=0.88, wspace=0.35)

    # ── Left panel: mean risk ─────────────────────────────────────────────────
    ax   = axes[0]
    bars = ax.bar(range(len(methods)), mean_risks,
                  color=colors, edgecolor="white", width=0.55, zorder=3)

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(labels, fontsize=9, ha="center")
    ax.set_ylabel("Mean Path Risk \u2193", fontsize=11)
    ax.set_title(
        f"Mean Path Risk in Bottleneck Scenarios\n"
        f"(n\u2009=\u2009{n_scenarios} scenarios, geo path risk > 0.15)",
        fontweight="bold", fontsize=10, pad=10
    )
    ax.grid(True, alpha=0.3, axis="y", zorder=0)
    ax.set_ylim(0, max(mean_risks) * 1.22)

    for bar, val, method in zip(bars, mean_risks, methods):
        red   = 100 * (geo_risk - val) / max(geo_risk, 1e-6)
        xc    = bar.get_x() + bar.get_width() / 2
        ytop  = bar.get_height()
        # Value label just above bar
        ax.text(xc, ytop + 0.004, f"{val:.4f}",
                ha="center", va="bottom", fontsize=8,
                fontweight="bold" if method == "SemGraph-Route (ours)" else "normal",
                color="#C0392B" if method == "SemGraph-Route (ours)" else "black")
        # Reduction label further above (only for SemGraph)
        if method == "SemGraph-Route (ours)":
            ax.text(xc, ytop + 0.016, f"({red:+.1f}% vs Geo A*)",
                    ha="center", va="bottom", fontsize=7.5,
                    fontweight="bold", color="#C0392B")

    # ── Right panel: mean reduction % ────────────────────────────────────────
    ax      = axes[1]
    rcolors = [BAR_COLORS[m] for m in methods]
    bars    = ax.bar(range(len(methods)), mean_reds,
                     color=rcolors, edgecolor="white", width=0.55, zorder=3)

    ax.axhline(0, color="black", lw=1.0, zorder=4)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(labels, fontsize=9, ha="center")
    ax.set_ylabel("Mean Risk Reduction vs Geometric A* (%)", fontsize=10)
    ax.set_title(
        "Risk Reduction in Bottleneck Scenarios\n"
        "Relative to Geometric A* \u2191 (positive = safer than Geo A*)",
        fontweight="bold", fontsize=10, pad=10
    )
    ax.grid(True, alpha=0.3, axis="y", zorder=0)

    # Dynamic y limits with enough padding for labels
    y_min = min(mean_reds) - 2.5
    y_max = max(mean_reds) + 2.5
    ax.set_ylim(y_min, y_max)

    for bar, val, method in zip(bars, mean_reds, methods):
        xc = bar.get_x() + bar.get_width() / 2
        # Place label OUTSIDE the bar with fixed offset
        if val >= 0:
            # Positive bar: label above top of bar
            ypos = bar.get_height() + 0.3
            va   = "bottom"
        else:
            # Negative bar: label below bottom of bar
            ypos = bar.get_height() - 0.3
            va   = "top"

        ax.text(xc, ypos, f"{val:+.1f}%",
                ha="center", va=va, fontsize=9,
                fontweight="bold" if method == "SemGraph-Route (ours)" else "normal",
                color="#C0392B" if method == "SemGraph-Route (ours)" else "black",
                zorder=5)

    plt.suptitle(
        "SemGraph-Route: Bottleneck Scenario Analysis\n"
        "All methods evaluated on same 141-node graph with GT labels",
        fontweight="bold", fontsize=12, y=0.98
    )

    p = FIGURES_DIR / "bottleneck_comparison.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure saved → {p}")


if __name__ == "__main__":
    main()
    import sys
    if sys.platform == "win32":
        input("\nPress Enter to close...")
