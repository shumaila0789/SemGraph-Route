"""
SemGraph-Route — Phase 1: Space Type Labeling
==============================================
Loads existing keyframe_records.json, runs VLM space classification
on each frame, and MERGES results into space_labeled_records.json.

SAFE MERGE: Never overwrites records from other environments.
  - On save, loads existing output file first
  - Replaces only records for the current --env
  - All other environments are preserved unchanged

Space taxonomy:
  corridor       – elongated, bounded, single movement axis
  junction       – multiple exits visible, branching point
  open_space     – large unobstructed area, low depth variance
  confined_space – high obstacle density, narrow passages
  dark_zone      – low luminance, poor texture (standalone type)

Usage:
  python phase1_space_labeling.py --env japanesealley --limit 100
  python phase1_space_labeling.py --env japanesealley --limit 100 --resume
  python phase1_space_labeling.py --mock  (heuristic mode, testing only)

Outputs:
  data/processed/space_labeled_records.json   (safely merged)
  data/processed/space_labeling_coverage.json
"""

import json
import re
import sys
import time
import warnings
import argparse
warnings.filterwarnings("ignore")
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime

import numpy as np
from PIL import Image

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent.resolve()
DATA_DIR   = BASE_DIR / "data"
INPUT_JSON = DATA_DIR / "processed" / "keyframe_records.json"
OUT_JSON   = DATA_DIR / "processed" / "space_labeled_records.json"
COVER_JSON = DATA_DIR / "processed" / "space_labeling_coverage.json"
LOGS_DIR   = BASE_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

VALID_TYPES = ["corridor", "junction", "open_space", "confined_space", "dark_zone"]

# ─── Logger ───────────────────────────────────────────────────────────────────
log_lines = []

def log(msg):
    ts   = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    log_lines.append(line)

def save_log():
    p = LOGS_DIR / "phase1_space_labeling.log"
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

# ─── VLM prompt ───────────────────────────────────────────────────────────────
SPACE_PROMPT = """Analyze this image carefully and answer in the exact format below.

STEP 1 - SPACE TYPE:
Classify this space as EXACTLY ONE of:
  corridor / junction / open_space / confined_space / dark_zone

Definitions:
  corridor       = elongated passage with walls on both sides, single axis of movement
  junction       = intersection or branching point with 2 or more visible exits
  open_space     = large unobstructed area with wide field of view, few walls nearby
  confined_space = tight area with high obstacle density, narrow clearance
  dark_zone      = very low lighting or heavily shadowed regardless of geometry

STEP 2 - CONFIDENCE:
Rate your classification confidence as: high / medium / low

STEP 3 - EVIDENCE:
One sentence explaining the key visual cues for your classification.

OUTPUT FORMAT (use exactly these labels):
space_type: <one of the five types>
confidence: <high|medium|low>
evidence: <one sentence>"""

# ─── Load VLM ─────────────────────────────────────────────────────────────────
def load_vlm():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    model_id = "vikhyatk/moondream2"
    revision  = "2025-01-09"

    log(f"Loading MoondreamV2 ({model_id} @ {revision}) ...")
    try:
        log("  Step 1/2: tokenizer ...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=revision, trust_remote_code=True,
        )
        log("  Step 2/2: model weights ...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, trust_remote_code=True,
            torch_dtype=torch.float32,
        )
        model.eval()
        model._tokenizer = tokenizer
        log("MoondreamV2 loaded successfully — using real VLM.")
        return model, "moondream"
    except Exception as e:
        log(f"VLM load failed: {e}")
        log("Falling back to mock classifier.")
        return None, "mock"

# ─── Mock classifier ──────────────────────────────────────────────────────────
def mock_classify(record, img_path):
    import random
    risk  = record.get("semantic_risk_gt") or record.get("risk",        0.3)
    uncert= record.get("uncertainty_gt")   or record.get("uncertainty", 0.3)
    if risk < 0.08:
        return "open_space"
    elif risk > 0.40:
        return "confined_space"
    else:
        rng = random.Random(hash(str(img_path)) % (2**32))
        if uncert > 0.30 and rng.random() < 0.25:
            return "junction"
        return "corridor"

# ─── VLM inference ────────────────────────────────────────────────────────────
def query_vlm(model, pil_img):
    try:
        enc = model.encode_image(pil_img)
        raw = model.answer_question(enc, SPACE_PROMPT, model._tokenizer)
        return str(raw) if not isinstance(raw, str) else raw
    except Exception as e:
        return f"VLM_ERROR: {e}"

# ─── Parse VLM output ─────────────────────────────────────────────────────────
def parse_vlm_output(raw):
    result = {"space_type": None, "confidence": None,
              "evidence": None, "parse_ok": False, "raw_response": raw}
    lower = raw.lower()

    m = re.search(r"space_type\s*[:\-]\s*([a-z_]+)", lower)
    if m and m.group(1).strip() in VALID_TYPES:
        result["space_type"] = m.group(1).strip()

    if result["space_type"] is None:
        for t in VALID_TYPES:
            if re.search(rf"\b{t}\b", lower):
                result["space_type"] = t
                break

    m = re.search(r"confidence\s*[:\-]\s*(high|medium|low)", lower)
    if m:
        result["confidence"] = m.group(1).strip()
    else:
        for level in ("high", "medium", "low"):
            if level in lower:
                result["confidence"] = level
                break

    m = re.search(r"evidence\s*[:\-]\s*(.+)", raw, re.IGNORECASE)
    if m:
        result["evidence"] = m.group(1).strip()

    result["parse_ok"] = result["space_type"] is not None
    return result

# ─── Photometric dark-zone guard ─────────────────────────────────────────────
def is_dark_zone(img_path, threshold=40.0):
    try:
        arr = np.array(Image.open(img_path).convert("L"), dtype=np.float32)
        return float(arr.mean()) < threshold
    except Exception:
        return False

# ─── Image path resolution ────────────────────────────────────────────────────
def find_image_path(record):
    for field in ("rgb_path", "image_path", "frame_path"):
        p = record.get(field)
        if p and Path(p).exists():
            return str(p)
    env   = record.get("environment") or record.get("env")
    traj  = record.get("trajectory")  or record.get("traj") or record.get("sequence")
    frame = record.get("frame_idx")   or record.get("frame_id") or record.get("frame")
    if env and traj and frame is not None:
        frame_str = str(frame).zfill(6)
        candidate = (DATA_DIR / "tartanair" / env / "Easy" / "image_left"
                     / env / "Easy" / str(traj) / "image_left"
                     / f"{frame_str}_left.png")
        if candidate.exists():
            return str(candidate)
    return None

# ─── SAFE MERGE SAVE ─────────────────────────────────────────────────────────
def safe_merge_save(new_env_records, current_env):
    """
    Merge new_env_records into the existing output file.
    Records from ALL other environments are preserved untouched.
    Only the current_env records are replaced.
    """
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    # Load existing file
    existing_all = []
    if OUT_JSON.exists():
        try:
            with open(OUT_JSON, "r") as f:
                existing_all = json.load(f)
        except Exception:
            existing_all = []

    # Keep everything except current env
    other_env_records = [
        r for r in existing_all
        if (r.get("environment") or r.get("env")) != current_env
    ]

    # Merge
    merged = other_env_records + new_env_records

    with open(OUT_JSON, "w") as f:
        json.dump(merged, f, indent=2)

    return len(other_env_records), len(new_env_records), len(merged)

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit",       type=int,   default=None)
    parser.add_argument("--resume",      action="store_true")
    parser.add_argument("--env",         type=str,   default=None)
    parser.add_argument("--dark_thresh", type=float, default=40.0)
    parser.add_argument("--mock",        action="store_true",
                        help="Force mock classifier (no VLM)")
    args = parser.parse_args()

    log("=" * 62)
    log("SemGraph-Route — Phase 1: Space Type Labeling")
    log(f"Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Env     : {args.env or 'ALL'}")
    log("SAFE MERGE: Other environments will NOT be overwritten")
    log("=" * 62)

    if not INPUT_JSON.exists():
        log(f"ERROR: {INPUT_JSON} not found."); sys.exit(1)

    with open(INPUT_JSON, "r") as f:
        records = json.load(f)
    log(f"Loaded {len(records)} keyframes.")

    if args.env:
        records = [r for r in records
                   if (r.get("environment") or r.get("env")) == args.env]
        log(f"Filtered to env='{args.env}': {len(records)} records.")

    if args.limit:
        records = records[:args.limit]
        log(f"Limited to first {args.limit} records.")

    # ── Resume: only within current environment ──────────────────────────────
    done_ids         = set()
    env_done_records = []

    if args.resume and OUT_JSON.exists():
        try:
            with open(OUT_JSON, "r") as f:
                existing_all = json.load(f)
            for i, r in enumerate(existing_all):
                r_env = r.get("environment") or r.get("env")
                if args.env is None or r_env == args.env:
                    fid = r.get("keyframe_idx") or r.get("frame_id") or i
                    done_ids.add(fid)
                    env_done_records.append(r)
            log(f"Resume: {len(done_ids)} frames already done for '{args.env}'.")
        except Exception as e:
            log(f"Resume load error: {e} — starting fresh.")

    # ── Load VLM ────────────────────────────────────────────────────────────
    if args.mock:
        model, model_type = None, "mock"
        log("Mock mode forced.")
    else:
        model, model_type = load_vlm()

    log(f"Classifier: {model_type.upper()}")

    # ── Processing loop ─────────────────────────────────────────────────────
    stats           = Counter()
    type_counts     = Counter()
    conf_counts     = Counter()
    total           = len(records)
    save_every      = 50

    # Start with already-done records for this env
    new_env_records = list(env_done_records)

    for idx, record in enumerate(records):
        frame_id = record.get("keyframe_idx") or record.get("frame_id") or idx

        if frame_id in done_ids:
            stats["skipped"] += 1
            continue

        if idx % 10 == 0:
            pct = 100.0 * idx / max(total, 1)
            log(f"  [{idx:>5}/{total}  {pct:4.1f}%]  "
                f"ok={stats['parse_ok']}  dark={stats['dark_override']}  "
                f"fail={stats['parse_fail']}  no_img={stats['image_not_found']}")

        img_path = find_image_path(record)
        if img_path is None:
            annotated = dict(record)
            annotated.update({
                "space_type": None, "space_confidence": None,
                "space_evidence": None, "space_parse_ok": False,
                "space_unambiguous": False,
                "space_raw_response": "IMAGE_NOT_FOUND",
                "space_vlm_time_s": 0.0, "space_source": model_type,
            })
            new_env_records.append(annotated)
            done_ids.add(frame_id)
            stats["image_not_found"] += 1
            continue

        forced_dark = is_dark_zone(img_path, args.dark_thresh)

        t0 = time.time()

        if forced_dark:
            space_type = "dark_zone"
            confidence = "medium"
            evidence   = "Mean luminance below threshold."
            parse_ok   = True
            raw_resp   = "PHOTOMETRIC_DARK_ZONE"
            stats["dark_override"] += 1

        elif model_type == "moondream":
            pil_img  = Image.open(img_path).convert("RGB")
            raw_resp = query_vlm(model, pil_img)
            parsed   = parse_vlm_output(raw_resp)
            space_type = parsed["space_type"]
            confidence = parsed["confidence"]
            evidence   = parsed["evidence"]
            parse_ok   = parsed["parse_ok"]

        else:
            space_type = mock_classify(record, img_path)
            confidence = "medium"
            evidence   = "Depth/texture heuristic (mock)."
            parse_ok   = True
            raw_resp   = f"MOCK:{space_type}"

        elapsed = time.time() - t0

        if parse_ok:
            stats["parse_ok"] += 1
        else:
            stats["parse_fail"] += 1

        unambiguous = parse_ok and confidence in ("high", "medium")
        if unambiguous:
            stats["unambiguous"] += 1

        if space_type:
            type_counts[space_type] += 1
        conf_counts[confidence or "unknown"] += 1

        annotated = dict(record)
        annotated.update({
            "space_type":         space_type,
            "space_confidence":   confidence,
            "space_evidence":     evidence,
            "space_parse_ok":     parse_ok,
            "space_unambiguous":  unambiguous,
            "space_raw_response": raw_resp if model_type == "moondream" else f"MOCK:{space_type}",
            "space_vlm_time_s":   round(elapsed, 3),
            "space_source":       model_type,
        })
        new_env_records.append(annotated)
        done_ids.add(frame_id)
        stats["processed"] += 1

        # Checkpoint — safe merge every save_every frames
        if stats["processed"] % save_every == 0:
            kept, new, total_saved = safe_merge_save(new_env_records, args.env)
            log(f"    ✓ checkpoint: {new} this-env + {kept} other-envs = {total_saved} total")

    # Final save
    kept, new, total_saved = safe_merge_save(new_env_records, args.env)
    log(f"\n✓ Saved: {new} this-env + {kept} other-envs = {total_saved} total → {OUT_JSON}")

    # Coverage report
    n_proc  = stats["processed"]
    n_ok    = stats["parse_ok"]
    n_unamb = stats["unambiguous"]

    coverage = {
        "environment":           args.env,
        "classifier_mode":       model_type,
        "total_processed":       n_proc,
        "total_skipped_resume":  stats["skipped"],
        "image_not_found":       stats["image_not_found"],
        "parse_ok_pct":          round(100 * n_ok    / max(n_proc, 1), 2),
        "unambiguous_pct":       round(100 * n_unamb / max(n_proc, 1), 2),
        "dark_zone_overrides":   stats["dark_override"],
        "type_distribution":     dict(type_counts),
    }
    with open(COVER_JSON, "w") as f:
        json.dump(coverage, f, indent=2)

    log("")
    log("=" * 62)
    log("PHASE 1 COMPLETE")
    log("=" * 62)
    log(f"  Classifier        : {model_type.upper()}")
    log(f"  Environment       : {args.env}")
    log(f"  Processed         : {n_proc}")
    log(f"  Skipped (resume)  : {stats['skipped']}")
    log(f"  Parse OK          : {n_ok}  ({coverage['parse_ok_pct']}%)")
    log(f"  Dark-zone         : {stats['dark_override']}")
    log(f"  Total file records: {total_saved} (all environments combined)")
    log("")
    log("  Type distribution:")
    for t in sorted(VALID_TYPES):
        n   = type_counts.get(t, 0)
        pct = 100 * n / max(n_proc, 1)
        bar = "█" * int(pct / 2)
        log(f"    {t:<18} {n:>5}  ({pct:5.1f}%)  {bar}")
    log("=" * 62)
    log("NEXT STEP: python learn_space_priors.py")
    save_log()


if __name__ == "__main__":
    main()
    if sys.platform == "win32":
        input("\nPress Enter to close...")
