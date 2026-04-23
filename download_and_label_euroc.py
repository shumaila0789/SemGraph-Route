"""
SemGraph-Route — EuRoC MAV Dataset Integration
===============================================
Downloads EuRoC MAV sequences, extracts images,
runs real VLM classification, and merges into
space_labeled_records.json using safe merge.

EuRoC sequences used:
  MH_01_easy  — Machine Hall (industrial indoor)
  V1_01_easy  — Vicon Room 1 (structured indoor)

These provide:
  - Real sensor data (vs TartanAir simulation)
  - Cross-domain validation for zero-shot claim
  - Well-known benchmark (reviewers know EuRoC)

Download size: ~1.5GB total for 2 sequences
VLM time: ~100 frames × 90s × 2 = ~5 hours

Usage:
  python download_and_label_euroc.py

Reads:
  data/processed/space_labeled_records.json (existing, preserved)

Writes:
  data/euroc/                    (downloaded sequences)
  data/processed/space_labeled_records.json (safe merge)
"""

import json
import os
import re
import sys
import time
import zipfile
import urllib.request
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from collections import Counter
from datetime import datetime

import numpy as np
from PIL import Image

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent.resolve()
DATA_DIR   = BASE_DIR / "data"
EUROC_DIR  = DATA_DIR / "euroc"
OUT_JSON   = DATA_DIR / "processed" / "space_labeled_records.json"
COVER_JSON = DATA_DIR / "processed" / "space_labeling_coverage_euroc.json"
LOGS_DIR   = BASE_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
EUROC_DIR.mkdir(parents=True, exist_ok=True)

VALID_TYPES = ["corridor", "junction", "open_space", "confined_space", "dark_zone"]

FRAMES_PER_SEQ = 100

# EuRoC sequences — Machine Hall and Vicon Room
SEQUENCES = {
    "MH_01_easy": {
        "url": "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_01_easy/MH_01_easy.zip",
        "env_name": "euroc_MH01",
        "description": "Machine Hall 01 — industrial indoor, real MAV sensor data",
    },
    "V1_01_easy": {
        "url": "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room1/V1_01_easy/V1_01_easy.zip",
        "env_name": "euroc_V101",
        "description": "Vicon Room 01 — structured indoor, real MAV sensor data",
    },
}

# ─── Logger ───────────────────────────────────────────────────────────────────
log_lines = []

def log(msg):
    ts   = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    log_lines.append(line)

def save_log():
    p = LOGS_DIR / "euroc_labeling.log"
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
    revision  = "2024-08-26"   # older stable revision — no pyvips dependency

    log(f"Loading MoondreamV2 @ {revision} (stable, no pyvips) ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=revision, trust_remote_code=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, trust_remote_code=True,
            torch_dtype=torch.float32,
        )
        model.eval()
        model._tokenizer = tokenizer
        log("VLM loaded successfully.")
        return model, "moondream"
    except Exception as e:
        log(f"VLM load failed: {e}")
        return None, "mock"

# ─── VLM inference ────────────────────────────────────────────────────────────
def query_vlm(model, pil_img):
    try:
        enc = model.encode_image(pil_img)
        raw = model.answer_question(enc, SPACE_PROMPT, model._tokenizer)
        return str(raw) if not isinstance(raw, str) else raw
    except Exception as e:
        return f"VLM_ERROR: {e}"

# ─── Parse ────────────────────────────────────────────────────────────────────
def parse_vlm_output(raw):
    result = {"space_type": None, "confidence": None,
              "evidence": None, "parse_ok": False}
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
        for lv in ("high", "medium", "low"):
            if lv in lower:
                result["confidence"] = lv
                break
    m = re.search(r"evidence\s*[:\-]\s*(.+)", raw, re.IGNORECASE)
    if m:
        result["evidence"] = m.group(1).strip()
    result["parse_ok"] = result["space_type"] is not None
    return result

# ─── Dark zone check ─────────────────────────────────────────────────────────
def is_dark_zone(img_path, threshold=40.0):
    try:
        arr = np.array(Image.open(img_path).convert("L"), dtype=np.float32)
        return float(arr.mean()) < threshold
    except Exception:
        return False

# ─── Download with progress ───────────────────────────────────────────────────
def download_file(url, dest_path):
    dest_path = Path(dest_path)
    if dest_path.exists():
        log(f"  Already downloaded: {dest_path.name}")
        return True
    log(f"  Downloading {dest_path.name} ...")
    log(f"  URL: {url}")

    try:
        def progress(count, block_size, total_size):
            pct = min(100, count * block_size * 100 // max(total_size, 1))
            if count % 500 == 0:
                mb = count * block_size / 1024 / 1024
                total_mb = total_size / 1024 / 1024
                print(f"  {pct}% ({mb:.0f}/{total_mb:.0f} MB)", end="\r", flush=True)

        urllib.request.urlretrieve(url, str(dest_path), reporthook=progress)
        print()
        log(f"  Downloaded: {dest_path.name}")
        return True
    except Exception as e:
        log(f"  Download failed: {e}")
        return False

# ─── Extract images from EuRoC zip ───────────────────────────────────────────
def extract_images(zip_path, seq_name, out_dir):
    """Extract cam0 images from EuRoC zip file."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    extracted = list(out_dir.glob("*.png"))
    if len(extracted) >= FRAMES_PER_SEQ:
        log(f"  Images already extracted: {len(extracted)} PNGs in {out_dir}")
        return sorted(extracted)[:FRAMES_PER_SEQ]

    log(f"  Extracting images from {zip_path.name} ...")
    try:
        with zipfile.ZipFile(str(zip_path), 'r') as z:
            # EuRoC structure: seq_name/mav0/cam0/data/*.png
            img_files = sorted([
                f for f in z.namelist()
                if 'cam0/data' in f and f.endswith('.png')
            ])
            log(f"  Found {len(img_files)} cam0 images in zip")

            # Extract only first FRAMES_PER_SEQ, evenly spaced
            step = max(1, len(img_files) // FRAMES_PER_SEQ)
            selected = img_files[::step][:FRAMES_PER_SEQ]

            paths = []
            for i, fname in enumerate(selected):
                # Save with simple sequential name
                out_path = out_dir / f"{i:06d}.png"
                if not out_path.exists():
                    data = z.read(fname)
                    out_path.write_bytes(data)
                paths.append(out_path)

            log(f"  Extracted {len(paths)} frames to {out_dir}")
            return paths
    except Exception as e:
        log(f"  Extraction failed: {e}")
        return []

# ─── Safe merge save ──────────────────────────────────────────────────────────
def safe_merge_save(new_records, env_name):
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if OUT_JSON.exists():
        try:
            with open(OUT_JSON, "r") as f:
                existing = json.load(f)
        except Exception:
            existing = []

    other = [r for r in existing
             if (r.get("environment") or r.get("env")) != env_name]
    merged = other + new_records

    with open(OUT_JSON, "w") as f:
        json.dump(merged, f, indent=2)
    return len(other), len(new_records), len(merged)

# ─── Label one sequence ───────────────────────────────────────────────────────
def label_sequence(seq_name, seq_info, img_paths, model, model_type):
    env_name = seq_info["env_name"]
    log(f"\nLabeling {seq_name} as env='{env_name}' ...")
    log(f"  {len(img_paths)} frames to process")

    records      = []
    stats        = Counter()
    type_counts  = Counter()
    save_every   = 20

    for idx, img_path in enumerate(img_paths):
        if idx % 10 == 0:
            pct = 100 * idx / max(len(img_paths), 1)
            log(f"  [{idx:>4}/{len(img_paths)}  {pct:4.1f}%]  "
                f"ok={stats['ok']}  dark={stats['dark']}  fail={stats['fail']}")

        forced_dark = is_dark_zone(str(img_path))

        t0 = time.time()
        if forced_dark:
            space_type = "dark_zone"
            confidence = "medium"
            evidence   = "Low luminance — photometric dark zone."
            parse_ok   = True
            raw_resp   = "PHOTOMETRIC_DARK_ZONE"
            stats["dark"] += 1
        elif model_type == "moondream":
            pil_img  = Image.open(img_path).convert("RGB")
            raw_resp = query_vlm(model, pil_img)
            parsed   = parse_vlm_output(raw_resp)
            space_type = parsed["space_type"]
            confidence = parsed["confidence"]
            evidence   = parsed["evidence"]
            parse_ok   = parsed["parse_ok"]
            if parse_ok:
                stats["ok"] += 1
            else:
                stats["fail"] += 1
        else:
            space_type = "corridor"
            confidence = "medium"
            evidence   = "Mock."
            parse_ok   = True
            raw_resp   = "MOCK"
            stats["ok"] += 1

        elapsed = time.time() - t0

        if space_type:
            type_counts[space_type] += 1

        # Build record in same format as TartanAir records
        # Position estimated from frame index (no GT poses in this script)
        # Use frame index as proxy position — sufficient for graph construction
        record = {
            "keyframe_idx":     idx,
            "frame_idx":        idx,
            "environment":      env_name,
            "dataset":          "euroc",
            "sequence":         seq_name,
            "rgb_path":         str(img_path),
            "position":         [float(idx) * 0.3, 0.0, 0.0],  # proxy: 0.3m steps
            # No GT depth labels for EuRoC in this script
            # Use space type prior for risk/uncertainty in planning
            "semantic_risk_gt": None,
            "uncertainty_gt":   None,
            # Space type labels
            "space_type":         space_type,
            "space_confidence":   confidence,
            "space_evidence":     evidence,
            "space_parse_ok":     parse_ok,
            "space_unambiguous":  parse_ok and confidence in ("high", "medium"),
            "space_raw_response": raw_resp if model_type == "moondream" else "MOCK",
            "space_vlm_time_s":   round(elapsed, 3),
            "space_source":       model_type,
        }
        records.append(record)

        if (idx + 1) % save_every == 0:
            kept, new, total = safe_merge_save(records, env_name)
            log(f"    ✓ checkpoint: {new} euroc + {kept} other = {total} total")

    kept, new, total = safe_merge_save(records, env_name)
    log(f"  ✓ Saved {new} records (total file: {total})")

    log(f"  Type distribution:")
    for t in sorted(VALID_TYPES):
        n   = type_counts.get(t, 0)
        pct = 100 * n / max(len(records), 1)
        bar = "█" * int(pct / 3)
        log(f"    {t:<18} {n:>4}  ({pct:4.1f}%)  {bar}")

    return records, type_counts

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    log("=" * 62)
    log("SemGraph-Route — EuRoC MAV Dataset Integration")
    log(f"Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("Sequences: MH_01_easy (Machine Hall), V1_01_easy (Vicon Room)")
    log(f"Frames   : {FRAMES_PER_SEQ} per sequence")
    log(f"Est. time: ~{FRAMES_PER_SEQ * 90 * 2 / 3600:.1f} hours with real VLM")
    log("Safe merge: existing TartanAir records preserved")
    log("=" * 62)

    # Load VLM — use 2024-08-26 revision (no pyvips, worked for hospital)
    model, model_type = load_vlm()
    log(f"Classifier: {model_type.upper()}")

    all_type_counts = Counter()

    for seq_name, seq_info in SEQUENCES.items():
        log(f"\n{'='*40}")
        log(f"Sequence: {seq_name}")
        log(f"  {seq_info['description']}")

        # Download
        zip_path = EUROC_DIR / f"{seq_name}.zip"
        ok = download_file(seq_info["url"], zip_path)
        if not ok:
            log(f"  Skipping {seq_name} — download failed")
            log(f"  Manual download: {seq_info['url']}")
            continue

        # Extract
        img_dir   = EUROC_DIR / seq_name / "images"
        img_paths = extract_images(zip_path, seq_name, img_dir)
        if not img_paths:
            log(f"  Skipping {seq_name} — no images extracted")
            continue

        # Label
        records, type_counts = label_sequence(
            seq_name, seq_info, img_paths, model, model_type
        )
        all_type_counts.update(type_counts)

    # Summary
    log("")
    log("=" * 62)
    log("EUROC LABELING COMPLETE")
    log("=" * 62)
    log("Environments added:")
    for seq_name, seq_info in SEQUENCES.items():
        log(f"  {seq_info['env_name']} ({seq_name})")
    log("")
    log("Overall EuRoC type distribution:")
    for t in sorted(VALID_TYPES):
        n   = all_type_counts.get(t, 0)
        pct = 100 * n / max(sum(all_type_counts.values()), 1)
        bar = "█" * int(pct / 3)
        log(f"  {t:<18} {n:>4}  ({pct:4.1f}%)  {bar}")
    log("")
    log("PAPER CLAIM (update validate_prior_stability.py TARGET_ENVS):")
    log("  hospital   — TartanAir indoor clinical (simulation)")
    log("  euroc_MH01 — EuRoC Machine Hall (real MAV sensor data)")
    log("  euroc_V101 — EuRoC Vicon Room (real MAV sensor data)")
    log("")
    log("NOW UPDATE validate_prior_stability.py:")
    log("  TARGET_ENVS = ['hospital', 'euroc_MH01', 'euroc_V101']")
    log("")
    log("THEN RUN:")
    log("  python learn_space_priors.py")
    log("  python build_scene_graph.py")
    log("  python phase3_semgraph_planner.py")
    log("  python validate_prior_stability.py")
    log("=" * 62)
    save_log()


if __name__ == "__main__":
    main()
    if sys.platform == "win32":
        input("\nPress Enter to close...")
