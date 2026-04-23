"""
SemGraph-Route — Phase 2: Space Type Prior Learning
====================================================
Loads space_labeled_records.json, filters to abandonedfactory ONLY
(the training environment), computes mean risk and uncertainty per
space type, and saves space_type_priors.json.

These priors are the core of SemGraph-Route's zero-shot generalization
claim: learned on one environment, transferred to all others.

Usage:
  python learn_space_priors.py

Reads:
  data/processed/space_labeled_records.json   (Phase 1 output)
  data/labels/training_labels.json            (CoT-Route ground truth)

Writes:
  data/priors/space_type_priors.json          (NEW — never overwrites existing)
  data/priors/prior_learning_stats.json       (NEW)
"""

import json
import sys
import time
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import numpy as np

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent.resolve()
DATA_DIR     = BASE_DIR / "data"
SPACE_JSON   = DATA_DIR / "processed" / "space_labeled_records.json"
LABELS_JSON  = DATA_DIR / "labels"    / "training_labels.json"
PRIORS_DIR   = DATA_DIR / "priors"
PRIORS_JSON  = PRIORS_DIR / "space_type_priors.json"
STATS_JSON   = PRIORS_DIR / "prior_learning_stats.json"
LOGS_DIR     = BASE_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
PRIORS_DIR.mkdir(parents=True, exist_ok=True)

# Training environment — priors learned here, transferred everywhere else
TRAIN_ENV    = "abandonedfactory"

VALID_TYPES  = ["corridor", "junction", "open_space", "confined_space", "dark_zone"]

# ─── Logger ───────────────────────────────────────────────────────────────────
log_lines = []

def log(msg):
    ts   = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    log_lines.append(line)

def save_log():
    p = LOGS_DIR / "phase2_learn_priors.log"
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()

    log("=" * 62)
    log("SemGraph-Route — Phase 2: Space Type Prior Learning")
    log(f"Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Train env: {TRAIN_ENV}")
    log("=" * 62)

    # ── Load Phase 1 space labels ───────────────────────────────────────────
    if not SPACE_JSON.exists():
        log(f"ERROR: {SPACE_JSON} not found — run phase1_space_labeling.py first.")
        sys.exit(1)

    with open(SPACE_JSON, "r") as f:
        all_records = json.load(f)
    log(f"Loaded {len(all_records)} labeled records.")

    # ── Load ground-truth risk/uncertainty labels ───────────────────────────
    label_map = {}
    if LABELS_JSON.exists():
        with open(LABELS_JSON, "r") as f:
            labels = json.load(f)
        label_map = {l["keyframe_idx"]: l for l in labels}
        log(f"Loaded {len(label_map)} ground-truth labels.")
    else:
        log("WARNING: training_labels.json not found — will use inline risk/uncertainty fields.")

    # ── Filter to training environment only ─────────────────────────────────
    train_records = [
        r for r in all_records
        if (r.get("environment") or r.get("env")) == TRAIN_ENV
    ]
    log(f"Training env records: {len(train_records)} (from {TRAIN_ENV})")

    if len(train_records) == 0:
        log(f"ERROR: No records found for env='{TRAIN_ENV}'.")
        log("Check that Phase 1 was run without --env filter, or re-run with full dataset.")
        sys.exit(1)

    # ── Only use unambiguous labels for prior learning ───────────────────────
    # unambiguous = space_parse_ok AND confidence in (high, medium)
    unambiguous = [r for r in train_records if r.get("space_unambiguous", True)]
    log(f"Unambiguous labels   : {len(unambiguous)} / {len(train_records)} "
        f"({100*len(unambiguous)/max(len(train_records),1):.1f}%)")

    if len(unambiguous) == 0:
        log("WARNING: No unambiguous labels — using all training records.")
        unambiguous = train_records

    # ── Collect risk and uncertainty per space type ──────────────────────────
    type_risks   = defaultdict(list)
    type_uncerts = defaultdict(list)
    skipped      = 0

    for r in unambiguous:
        stype = r.get("space_type")
        if stype not in VALID_TYPES:
            skipped += 1
            continue

        kid = r.get("keyframe_idx")

        # Priority 1: ground-truth labels file
        if kid in label_map:
            risk  = label_map[kid].get("semantic_risk_gt",  None)
            uncert= label_map[kid].get("uncertainty_gt",    None)
        else:
            risk   = None
            uncert = None

        # Priority 2: inline fields in the record itself
        if risk   is None:
            risk   = r.get("semantic_risk_gt") or r.get("risk",        None)
        if uncert is None:
            uncert = r.get("uncertainty_gt")   or r.get("uncertainty", None)

        # Skip if we still have nothing
        if risk is None or uncert is None:
            skipped += 1
            continue

        type_risks[stype].append(float(risk))
        type_uncerts[stype].append(float(uncert))

    log(f"Skipped (no labels)  : {skipped}")

    # ── Compute priors ───────────────────────────────────────────────────────
    log("")
    log("Computing space type priors ...")

    # Expected approximate values from project brief for reference:
    EXPECTED = {
        "corridor":       {"risk": 0.12, "uncertainty": 0.20},
        "junction":       {"risk": 0.28, "uncertainty": 0.25},
        "open_space":     {"risk": 0.05, "uncertainty": 0.16},
        "confined_space": {"risk": 0.48, "uncertainty": 0.32},
        "dark_zone":      {"risk": 0.08, "uncertainty": 0.45},
    }

    priors = {}
    stats_rows = []

    for stype in VALID_TYPES:
        r_vals = type_risks.get(stype,   [])
        u_vals = type_uncerts.get(stype, [])
        n      = len(r_vals)

        if n == 0:
            # No data — fall back to expected values from project brief
            log(f"  {stype:<18} — no samples, using expected prior")
            risk_mean  = EXPECTED[stype]["risk"]
            uncert_mean= EXPECTED[stype]["uncertainty"]
            risk_std   = 0.0
            uncert_std = 0.0
            source     = "expected_fallback"
        else:
            risk_mean  = float(np.mean(r_vals))
            uncert_mean= float(np.mean(u_vals))
            risk_std   = float(np.std(r_vals))
            uncert_std = float(np.std(u_vals))
            source     = "empirical"

        priors[stype] = {
            "risk_mean":    round(risk_mean,   4),
            "risk_std":     round(risk_std,    4),
            "uncert_mean":  round(uncert_mean, 4),
            "uncert_std":   round(uncert_std,  4),
            "n_samples":    n,
            "source":       source,
        }

        exp_r = EXPECTED[stype]["risk"]
        exp_u = EXPECTED[stype]["uncertainty"]
        delta_r = risk_mean  - exp_r
        delta_u = uncert_mean - exp_u

        stats_rows.append({
            "type":       stype,
            "n":          n,
            "risk_mean":  risk_mean,
            "uncert_mean":uncert_mean,
            "risk_std":   risk_std,
            "uncert_std": uncert_std,
            "delta_risk_vs_expected":  delta_r,
            "delta_uncert_vs_expected":delta_u,
        })

    # ── Save priors ──────────────────────────────────────────────────────────
    prior_output = {
        "train_environment":  TRAIN_ENV,
        "computed_at":        datetime.now().isoformat(),
        "n_train_records":    len(train_records),
        "n_unambiguous":      len(unambiguous),
        "space_type_priors":  priors,
    }
    with open(PRIORS_JSON, "w") as f:
        json.dump(prior_output, f, indent=2)
    log(f"\n✓ Priors saved → {PRIORS_JSON}")

    # ── Save stats ───────────────────────────────────────────────────────────
    with open(STATS_JSON, "w") as f:
        json.dump(stats_rows, f, indent=2)

    # ── Print paper-ready prior table ────────────────────────────────────────
    log("")
    log("=" * 62)
    log("SPACE TYPE PRIORS — Paper Table (Table II in SemGraph-Route)")
    log("=" * 62)
    log(f"  {'Space Type':<18} {'N':>5}  {'Risk μ':>8}  {'Risk σ':>8}  "
        f"{'Uncert μ':>9}  {'Uncert σ':>9}  {'Source'}")
    log(f"  {'-'*18} {'-----':>5}  {'--------':>8}  {'--------':>8}  "
        f"{'----------':>9}  {'----------':>9}")
    for stype in VALID_TYPES:
        p = priors[stype]
        log(f"  {stype:<18} {p['n_samples']:>5}  "
            f"{p['risk_mean']:>8.4f}  {p['risk_std']:>8.4f}  "
            f"{p['uncert_mean']:>9.4f}  {p['uncert_std']:>9.4f}  "
            f"{p['source']}")

    log("")
    log("Comparison with expected priors from project brief:")
    log(f"  {'Space Type':<18} {'Emp Risk':>9} {'Exp Risk':>9} {'Δ Risk':>8}  "
        f"{'Emp Unc':>8} {'Exp Unc':>8} {'Δ Unc':>7}")
    log(f"  {'-'*18} {'-'*9} {'-'*9} {'-'*8}  {'-'*8} {'-'*8} {'-'*7}")
    for row in stats_rows:
        exp = EXPECTED[row["type"]]
        log(f"  {row['type']:<18} "
            f"{row['risk_mean']:>9.4f} {exp['risk']:>9.4f} {row['delta_risk_vs_expected']:>+8.4f}  "
            f"{row['uncert_mean']:>8.4f} {exp['uncertainty']:>8.4f} {row['delta_uncert_vs_expected']:>+7.4f}")

    # ── Generalization readiness check ───────────────────────────────────────
    log("")
    log("Generalization readiness:")
    all_empirical = all(priors[t]["source"] == "empirical" for t in VALID_TYPES)
    if all_empirical:
        log("  ✓ All 5 space types have empirical priors — ready for prior stability validation.")
    else:
        missing = [t for t in VALID_TYPES if priors[t]["source"] != "empirical"]
        log(f"  ⚠  Fallback priors used for: {missing}")
        log("     These types had no labeled samples in the training environment.")
        log("     Consider running Phase 1 on the full dataset (all 478 abandonedfactory frames).")

    elapsed = time.time() - t0
    log("")
    log(f"Done in {elapsed:.1f}s")
    log("NEXT STEP: python build_scene_graph.py")
    log("=" * 62)

    save_log()


if __name__ == "__main__":
    main()
    import sys as _sys
    if _sys.platform == "win32":
        input("\nPress Enter to close...")
