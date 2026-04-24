"""
Reporter: generate report.md and report.json from preprocessed ImageInfo list.
Optionally includes AI evaluation results and selection decisions (Phase 2).
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Optional

from .preprocessor import ImageInfo


def generate_report(
    images: list[ImageInfo],
    output_dir: Path,
    input_dir: Path,
    config: dict,
    evaluations: Optional[list] = None,
    selected: Optional[list[ImageInfo]] = None,
) -> tuple[Path, Path]:
    """
    Write report.md and report.json to output_dir.
    evaluations: list[AIEvaluation] aligned with images (Phase 2 only).
    selected: list of ImageInfo that were selected (Phase 2 only).
    Returns (md_path, json_path).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    md_path = output_dir / "report.md"
    json_path = output_dir / "report.json"

    selected_paths = {img.path for img in selected} if selected else None
    data = _build_data(images, input_dir, config, evaluations, selected_paths)

    _write_markdown(data, md_path)
    _write_json(data, json_path)

    print(f"[report] Wrote {md_path}")
    print(f"[report] Wrote {json_path}")
    return md_path, json_path


def _build_data(
    images: list[ImageInfo],
    input_dir: Path,
    config: dict,
    evaluations,
    selected_paths,
) -> dict:
    blur_threshold = config.get("preprocessing", {}).get("blur_threshold", 80)
    burst_window = config.get("preprocessing", {}).get("burst_window_seconds", 2.0)
    has_ai = evaluations is not None and len(evaluations) == len(images)

    burst_groups: dict[int, list[str]] = {}
    for img in images:
        if img.burst_group >= 0:
            burst_groups.setdefault(img.burst_group, []).append(img.path.name)

    selected_count = len(selected_paths) if selected_paths else None

    summary = {
        "input_dir": str(input_dir),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "total_images": len(images),
        "blurry_count": sum(1 for img in images if img.is_blurry),
        "burst_groups": len(burst_groups),
        "burst_images": sum(1 for img in images if img.burst_group >= 0),
        "portrait_count": sum(1 for img in images if img.portrait_class == "portrait"),
        "group_portrait_count": sum(1 for img in images if img.portrait_class == "group"),
        "non_portrait_count": sum(1 for img in images if img.portrait_class == "non-portrait"),
        "blur_threshold": blur_threshold,
        "burst_window_seconds": burst_window,
        "ai_evaluation": has_ai,
    }
    if selected_count is not None:
        summary["selected_count"] = selected_count
        summary["not_selected_count"] = len(images) - selected_count

    records = []
    for i, img in enumerate(images):
        ev = evaluations[i] if has_ai else None
        rec = {
            "filename": img.path.name,
            "mtime": img.mtime,
            "burst_group": img.burst_group,
            "blur_score": round(img.blur_score, 2),
            "is_blurry": img.is_blurry,
            "portrait_class": img.portrait_class,
            "face_count": img.face_count,
            "face_boxes": img.face_boxes,
            "thumb": str(img.thumb_path) if img.thumb_path else None,
        }
        if selected_paths is not None:
            rec["selected"] = img.path in selected_paths
        if ev is not None:
            rec["ai"] = {
                "eyes_open": ev.eyes_open,
                "expression": ev.expression,
                "eye_contact": ev.eye_contact,
                "quality": ev.quality,
                "lighting": ev.lighting,
                "composition": ev.composition,
                "reasoning": ev.reasoning,
                "skipped": ev.skipped,
            }
        records.append(rec)

    return {
        "summary": summary,
        "images": records,
        "burst_groups": burst_groups,
    }


def _write_markdown(data: dict, path: Path) -> None:
    s = data["summary"]
    has_ai = s.get("ai_evaluation", False)
    has_selection = "selected_count" in s

    lines = []
    phase = "Phase 2" if has_ai else "Phase 1"
    lines.append(f"# Photo Picker — {phase} Report\n")
    lines.append(f"**Input:** `{s['input_dir']}`  ")
    lines.append(f"**Generated:** {s['generated_at']}  ")
    lines.append("")
    lines.append("## Summary\n")
    lines.append("| Item | Value |")
    lines.append("|------|-------|")
    lines.append(f"| Total images | {s['total_images']} |")
    lines.append(f"| Blurry (score < {s['blur_threshold']}) | {s['blurry_count']} |")
    lines.append(f"| Burst groups | {s['burst_groups']} |")
    lines.append(f"| Burst images | {s['burst_images']} |")
    lines.append(f"| Portrait | {s['portrait_count']} |")
    lines.append(f"| Group portrait | {s['group_portrait_count']} |")
    lines.append(f"| Non-portrait | {s['non_portrait_count']} |")
    if has_selection:
        lines.append(f"| **Selected** | **{s['selected_count']}** |")
        lines.append(f"| Not selected | {s['not_selected_count']} |")
    lines.append("")

    if data["burst_groups"]:
        lines.append("## Burst Groups\n")
        lines.append("| Group | Images |")
        lines.append("|-------|--------|")
        for gid, names in sorted(data["burst_groups"].items()):
            lines.append(f"| {gid} | {', '.join(names)} |")
        lines.append("")

    lines.append("## Image Details\n")
    if has_ai:
        lines.append("| Filename | Burst | Blur | Faces | Class | Eyes | Expression | Quality | Selected |")
        lines.append("|----------|-------|------|-------|-------|------|------------|---------|----------|")
        for rec in data["images"]:
            burst = str(rec["burst_group"]) if rec["burst_group"] >= 0 else "—"
            blurry = "blur" if rec["is_blurry"] else f"{rec['blur_score']}"
            ai = rec.get("ai", {})
            eyes = "open" if ai.get("eyes_open", True) else "**CLOSED**"
            expr = ai.get("expression", "—")
            qual = ai.get("quality", "—")
            sel = "**yes**" if rec.get("selected") else "no"
            lines.append(
                f"| {rec['filename']} | {burst} | {blurry} | {rec['face_count']}"
                f" | {rec['portrait_class']} | {eyes} | {expr} | {qual} | {sel} |"
            )
    else:
        lines.append("| Filename | Burst | Blur Score | Blurry | Faces | Class |")
        lines.append("|----------|-------|------------|--------|-------|-------|")
        for rec in data["images"]:
            burst = str(rec["burst_group"]) if rec["burst_group"] >= 0 else "—"
            blurry = "yes" if rec["is_blurry"] else "no"
            lines.append(
                f"| {rec['filename']} | {burst} | {rec['blur_score']} | {blurry}"
                f" | {rec['face_count']} | {rec['portrait_class']} |"
            )
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def _write_json(data: dict, path: Path) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
