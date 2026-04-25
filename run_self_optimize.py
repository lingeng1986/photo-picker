#!/usr/bin/env python3
"""
自优化流程：模糊检测 + 人脸检测阈值调优
"""

import sys
from pathlib import Path

from modules.test_suite import TestSuite, HyperParamTuner

TEST_CASE = "portrait_test_11060218"
MODEL = "llava:13b"

def main():
    from pathlib import Path
    suite_dir = Path.home() / ".photo-picker-tests"
    suite = TestSuite(suite_dir)
    case = suite.load_test_case(TEST_CASE)

    print(f"\n{'='*60}")
    print(f"🎯 自优化: {case.name}")
    print(f"   照片数: {len(case.ground_truth)} | 人工选中: {sum(case.ground_truth.values())}张")
    print(f"{'='*60}\n")

    # 1. 模糊阈值调优（较窄范围，避免超时）
    print("\n📊 步骤1: 模糊阈值调优")
    print("-" * 40)
    tuner = HyperParamTuner(suite, case)

    # 使用更窄的范围（基于之前的结果，40最好）
    blur_thresholds = [30, 40, 50, 60, 80]
    best_blur, blur_result = tuner.tune_blur_threshold(MODEL, blur_thresholds)

    # 保存模糊阈值结果
    blur_config_path = suite.results_dir / f"{TEST_CASE}_best_blur_threshold.txt"
    blur_config_path.write_text(str(best_blur), encoding="utf-8")
    print(f"💾 最佳模糊阈值已保存: {blur_config_path}")

    # 2. 人脸检测 score_threshold 调优
    print("\n📊 步骤2: 人脸检测置信度阈值调优")
    print("-" * 40)

    score_thresholds = [0.3, 0.5, 0.6, 0.7, 0.8]
    best_score_thresh = None
    best_result = None
    best_f1 = 0

    for score_thresh in score_thresholds:
        config = suite._load_default_config()
        config["preprocessing"]["blur_threshold"] = best_blur
        config["detector"]["score_threshold"] = score_thresh

        result = suite.run_evaluation(case, MODEL, config)

        if result.metrics.f1_score > best_f1:
            best_f1 = result.metrics.f1_score
            best_score_thresh = score_thresh
            best_result = result
            print(f"  score_threshold={score_thresh}: 新最佳 F1={best_f1:.2%}")
        else:
            print(f"  score_threshold={score_thresh}: F1={result.metrics.f1_score:.2%}")

    print(f"\n🏆 最佳人脸检测阈值: {best_score_thresh} (F1={best_f1:.2%})")

    # 3. NMS 阈值调优（基于最佳 score_threshold）
    print("\n📊 步骤3: NMS阈值调优")
    print("-" * 40)

    nms_thresholds = [0.2, 0.3, 0.4, 0.5]
    best_nms_thresh = 0.3

    for nms_thresh in nms_thresholds:
        config = suite._load_default_config()
        config["preprocessing"]["blur_threshold"] = best_blur
        config["detector"]["score_threshold"] = best_score_thresh
        config["detector"]["nms_threshold"] = nms_thresh

        result = suite.run_evaluation(case, MODEL, config)

        if result.metrics.f1_score > best_f1:
            best_f1 = result.metrics.f1_score
            best_nms_thresh = nms_thresh
            best_result = result
            print(f"  nms_threshold={nms_thresh}: 新最佳 F1={best_f1:.2%}")
        else:
            print(f"  nms_threshold={nms_thresh}: F1={result.metrics.f1_score:.2%}")

    print(f"\n🏆 最佳NMS阈值: {best_nms_thresh} (F1={best_f1:.2%})")

    # 4. 生成优化报告
    print("\n📊 步骤4: 生成自优化报告")
    print("-" * 40)

    report = f"""# Photo Picker 自优化报告 (模糊+人脸检测)

## 测试集信息
- **名称**: {case.name}
- **描述**: {case.description}
- **照片总数**: {len(case.ground_truth)}
- **人工选中 (Ground Truth)**: {sum(case.ground_truth.values())}张

## 优化参数

### 1. 模糊检测阈值
- **最佳值**: {best_blur}
- **搜索范围**: {blur_thresholds}

### 2. 人脸检测参数
| 参数 | 优化值 | 搜索范围 |
|------|--------|----------|
| score_threshold | {best_score_thresh} | {score_thresholds} |
| nms_threshold | {best_nms_thresh} | {nms_thresholds} |

## 优化结果

**最终F1分数**: {best_f1:.2%}

**推荐配置** (config.json):
```json
{{
  "preprocessing": {{
    "blur_threshold": {best_blur},
    "burst_window_seconds": 2.0,
    "thumbnail_max_size": 1024
  }},
  "detector": {{
    "model": "assets/models/face_detection_yunet_2023mar.onnx",
    "score_threshold": {best_score_thresh},
    "nms_threshold": {best_nms_thresh},
    "top_k": 5000
  }}
}}
```

## 文件位置
- 最佳模糊阈值: `~/.photo-picker-tests/results/{TEST_CASE}_best_blur_threshold.txt`
"""

    report_path = suite.results_dir / f"{TEST_CASE}_detector_optimization_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"💾 优化报告已保存: {report_path}")

    # 保存完整配置
    final_config = {
        "preprocessing": {
            "blur_threshold": best_blur,
            "burst_window_seconds": 2.0,
            "thumbnail_max_size": 1024
        },
        "detector": {
            "model": "assets/models/face_detection_yunet_2023mar.onnx",
            "score_threshold": best_score_thresh,
            "nms_threshold": best_nms_thresh,
            "top_k": 5000
        }
    }

    import json
    config_path = suite.results_dir / f"{TEST_CASE}_best_detector_config.json"
    config_path.write_text(json.dumps(final_config, indent=2), encoding="utf-8")
    print(f"💾 最佳检测配置已保存: {config_path}")

    print(f"\n{'='*60}")
    print(f"✅ 自优化完成! 最终F1: {best_f1:.2%}")
    print(f"{'='*60}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
