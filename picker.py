#!/usr/bin/env python3
"""
Photo Picker — main entry point.

Phase 1 (--preprocess-only):
  Scans images, generates thumbnails, groups bursts,
  detects blur, detects faces, writes report.md + report.json.

Phase 2 (default / --dry-run):
  Runs Phase 1 preprocessing, then calls Ollama bakllava for AI evaluation,
  selects the best photo from each burst group, copies files, and writes report.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from datetime import datetime

SKILL_DIR = Path(__file__).parent
DEFAULT_CONFIG = SKILL_DIR / "config.json"


def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        print(f"[config] {config_path} not found — using defaults")
        return {}
    try:
        with open(config_path, encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"[config] JSON parse error in {config_path}: {e}")
        sys.exit(1)


def make_output_dir(input_dir: Path) -> Path:
    """Create a dated sibling output dir; append _1, _2 if it already exists."""
    date_str = datetime.now().strftime("%Y%m%d")
    base = input_dir.parent / f"{input_dir.name}_by_ai_{date_str}"
    candidate = base
    suffix = 0
    while candidate.exists():
        suffix += 1
        candidate = Path(f"{base}_{suffix}")
    candidate.mkdir(parents=True, exist_ok=True)
    return candidate


def run_preprocess(input_dir: Path, config: dict, output_dir: Path):
    """Phase 1: scan, thumbnail, blur, face detection. Returns list[ImageInfo]."""
    from modules.preprocessor import (
        scan_images,
        generate_thumbnails,
        group_bursts,
        filter_blurry,
        detect_faces,
    )

    pre_cfg = config.get("preprocessing", {})
    blur_threshold = pre_cfg.get("blur_threshold", 80)
    burst_window = pre_cfg.get("burst_window_seconds", 2.0)
    thumb_max = pre_cfg.get("thumbnail_max_size", 1024)

    thumb_dir = output_dir / "thumbs"

    images = scan_images(input_dir)
    if not images:
        print("[error] No supported images found.")
        sys.exit(1)

    generate_thumbnails(images, thumb_dir, max_size=thumb_max)
    group_bursts(images, window_seconds=burst_window)
    filter_blurry(images, threshold=blur_threshold)
    detect_faces(images, config)

    return images


def copy_files(images, dest_dir: Path, label: str) -> int:
    """Copy image files to dest_dir. Returns number of files copied."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for img in images:
        dst = dest_dir / img.path.name
        try:
            shutil.copy2(img.path, dst)
            count += 1
        except OSError as e:
            print(f"  [warn] Cannot copy {img.path.name}: {e}")
    if count:
        print(f"[copy] {count} file(s) → {dest_dir} ({label})")
    return count


def run_full_pipeline(
    input_dir: Path,
    config: dict,
    output_dir: Path,
    dry_run: bool,
) -> None:
    from modules.evaluator import Evaluator, AIEvaluation
    from modules.preprocessor import filter_by_quality
    from modules.selector import Selector
    from modules.reporter import generate_report

    print(f"\n=== Photo Picker — Phase 2 Full Pipeline ===")
    print(f"Input    : {input_dir}")
    print(f"Output   : {output_dir}")
    print(f"Dry-run  : {dry_run}")
    print()

    # ── Phase 1: preprocess ───────────────────────────────────────────────
    print("--- Phase 1: Preprocessing ---")
    images = run_preprocess(input_dir, config, output_dir)

    # ── Phase 2: AI evaluation (non-blurry only) ──────────────────────────
    print("\n--- Phase 2: AI Evaluation ---")
    evaluator = Evaluator(config)
    non_blurry = filter_by_quality(images)
    non_blurry_evals = evaluator.evaluate_batch(non_blurry)

    # Merge: blurry images get a poor-quality default so they fail filter_by_criteria
    _blurry_default = AIEvaluation(
        eyes_open=True, expression="natural", eye_contact="moderate",
        quality="poor", lighting="good", composition="centered",
        reasoning="skipped (blurry)", skipped=True,
    )
    eval_by_path = {img.path: ev for img, ev in zip(non_blurry, non_blurry_evals)}
    evaluations = [eval_by_path.get(img.path, _blurry_default) for img in images]

    # ── Phase 2: selection ────────────────────────────────────────────────
    print("\n--- Phase 2: Selecting Best Photos ---")
    selector = Selector(config)
    selected, not_selected = selector.select_from_all(images, evaluations)
    print(f"[select] {len(selected)} selected, {len(not_selected)} not selected")

    # ── Copy files ────────────────────────────────────────────────────────
    if not dry_run:
        print("\n--- Copying Files ---")
        copy_files(selected, output_dir / "selected", "selected")
        if config.get("output", {}).get("copy_not_selected", True):
            copy_files(not_selected, output_dir / "not_selected", "not selected")
    else:
        print("\n[dry-run] Skipping file copy.")

    # ── Report ────────────────────────────────────────────────────────────
    print("\n--- Writing Report ---")
    generate_report(images, output_dir, input_dir, config, evaluations, selected)

    print(f"\n=== Done. Output: {output_dir} ===\n")
    print(f"  Selected   : {len(selected)} photos")
    print(f"  Not selected: {len(not_selected)} photos")
    if dry_run:
        print("  (dry-run: no files were copied)")


def run_preprocess_only(input_dir: Path, config: dict, output_dir: Path) -> None:
    from modules.reporter import generate_report

    print(f"\n=== Photo Picker — Phase 1 Preprocessing ===")
    print(f"Input : {input_dir}")
    print(f"Output: {output_dir}")
    print()

    images = run_preprocess(input_dir, config, output_dir)
    generate_report(images, output_dir, input_dir, config)

    print(f"\n=== Done. Output: {output_dir} ===\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Photo Picker — AI-assisted photo culling tool"
    )
    parser.add_argument("input_dir", help="Directory containing photos to process")
    parser.add_argument(
        "--preprocess-only",
        action="store_true",
        help="Phase 1 only: scan, thumbnail, blur/face analysis, report — no AI, no file copy",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run full AI pipeline but skip file copying",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help=f"Path to config.json (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output directory (default: auto-generated sibling dir)",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.is_dir():
        print(f"[error] Input directory not found: {input_dir}")
        sys.exit(1)

    config = load_config(Path(args.config))

    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = make_output_dir(input_dir)

    if args.preprocess_only:
        run_preprocess_only(input_dir, config, output_dir)
    else:
        run_full_pipeline(input_dir, config, output_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
