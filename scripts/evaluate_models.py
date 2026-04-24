#!/usr/bin/env python3
"""Photo Picker 自动化评测 — 对比多个模型在同一目录的评分相关性"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

SKILL_DIR = Path(__file__).parent.parent
DEFAULT_MODELS = "moondream,bakllava,llava:13b"


def run_picker(test_dir: Path, model_name: str, skill_dir: Path) -> dict:
    """运行一次 picker，返回 {filename: quality_score} 字典"""
    config_path = skill_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["ai_evaluation"]["model"] = model_name

    tmp_config = skill_dir / f"_tmp_config_{model_name.replace(':', '_')}.json"
    tmp_config.write_text(json.dumps(config, indent=2), encoding="utf-8")

    try:
        result = subprocess.run(
            [sys.executable, str(skill_dir / "picker.py"), str(test_dir), "--config", str(tmp_config)],
            capture_output=True, text=True, timeout=600, cwd=str(skill_dir),
        )
        if result.returncode != 0:
            print(f"  ❌ {model_name} 失败:\n{result.stderr[-500:]}")
            return {}
    finally:
        tmp_config.unlink(missing_ok=True)

    output_dirs = sorted(test_dir.parent.glob(f"{test_dir.name}_by_ai_*"), key=lambda p: p.stat().st_mtime)
    if not output_dirs:
        print(f"  ❌ 未找到输出目录")
        return {}

    report = output_dirs[-1] / "report.json"
    if not report.exists():
        print(f"  ❌ report.json 不存在")
        return {}

    data = json.loads(report.read_text(encoding="utf-8"))
    quality_map = {"excellent": 5, "good": 4, "average": 3, "poor": 2, "unknown": 1}
    scores = {}
    for img in data.get("images", []):
        fname = img["filename"]
        quality = img.get("ai", {}).get("quality", "unknown")
        scores[fname] = quality_map.get(quality, 0)

    print(f"  ✅ {model_name}: {len(scores)} 张评分完成")
    return scores


def spearman_correlation(scores1: dict, scores2: dict) -> float:
    common = set(scores1.keys()) & set(scores2.keys())
    if len(common) < 2:
        return 0.0
    names = list(common)
    rank1 = {n: i for i, n in enumerate(sorted(names, key=lambda x: scores1[x], reverse=True))}
    rank2 = {n: i for i, n in enumerate(sorted(names, key=lambda x: scores2[x], reverse=True))}
    n = len(common)
    d2 = sum((rank1[n] - rank2[n]) ** 2 for n in common)
    return 1 - (6 * d2) / (n * (n**2 - 1)) if n > 1 else 0.0


def main():
    parser = argparse.ArgumentParser(description="Photo Picker 自动化评测")
    parser.add_argument("test_dir", help="照片目录路径")
    parser.add_argument("--models", default=DEFAULT_MODELS, help=f"逗号分隔的模型列表 (默认: {DEFAULT_MODELS})")
    parser.add_argument("--skill-dir", default=str(SKILL_DIR), help="photo-picker 根目录")
    args = parser.parse_args()

    test_dir = Path(args.test_dir).expanduser().resolve()
    skill_dir = Path(args.skill_dir).expanduser().resolve()
    models = [m.strip() for m in args.models.split(",")]

    if not test_dir.is_dir():
        print(f"错误: 目录不存在: {test_dir}")
        sys.exit(1)

    print("=" * 50)
    print("🎯 Photo Picker 自动化评测")
    print(f"   目录: {test_dir}")
    print(f"   模型: {', '.join(models)}")
    print("=" * 50)

    results = {}
    for model in models:
        print(f"\n▶️  运行 {model}...")
        results[model] = run_picker(test_dir, model, skill_dir)

    print("\n📊 相关系数矩阵 (Spearman):")
    col_w = 14
    print(" " * col_w, end="")
    for m in models:
        print(f"{m[:col_w-2]:>{col_w}}", end="")
    print()
    for m1 in models:
        print(f"{m1[:col_w-2]:<{col_w}}", end="")
        for m2 in models:
            if m1 == m2:
                val = "1.00"
            else:
                val = f"{spearman_correlation(results.get(m1, {}), results.get(m2, {})):.2f}"
            print(f"{val:>{col_w}}", end="")
        print()

    print("\n解读: 1.0 = 完全一致，0 = 无关，-1 = 完全相反")
    print("=" * 50)


if __name__ == "__main__":
    main()
