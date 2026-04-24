---
name: photo-picker
description: Use local Ollama bakllava to score and filter batches of photos, generate thumbnails and a markdown report, and copy selected plus non-selected images into organized output folders. Use when the user wants AI-assisted photo picking, culling, quality filtering, burst review, or synchronized JPG/RAW selection from a local photo directory.
---

# Photo Picker

AI-assisted photo culling using local Ollama `bakllava`.  
Scans a folder, detects faces/blur, evaluates each portrait with a vision model, picks the best shot from every burst group, and copies results into `selected/` and `not_selected/`.

## Quick start

```bash
# Full pipeline (preprocess + AI evaluation + select + copy)
python picker.py /path/to/photos

# Dry-run: evaluate without copying files
python picker.py /path/to/photos --dry-run

# Phase 1 only: preprocess, blur/face detection, report (no AI, no copy)
python picker.py /path/to/photos --preprocess-only
```

Or via the shell wrapper:

```bash
scripts/picker.sh /path/to/photos
```

## Output layout

```text
<input>_by_ai_YYYYMMDD/
├── selected/          # best photos chosen by AI
├── not_selected/      # rejected photos
├── thumbs/            # resized thumbnails (used for AI input)
├── report.md          # human-readable table with AI scores
└── report.json        # machine-readable full data
```

If the output directory for today already exists, `_1`, `_2`, etc. are appended.

## How it works

1. **Scan** — find all `.jpg`, `.jpeg`, `.png`, `.heic` files sorted by modification time.
2. **Thumbnails** — generate resized copies with macOS `sips` (max 1024 px by default).
3. **Burst grouping** — cluster images taken within 2 s of each other into burst groups.
4. **Blur detection** — compute Laplacian variance with OpenCV; mark images below threshold.
5. **Face detection** — run YuNet to classify each image as `portrait`, `group`, or `non-portrait`.
6. **AI evaluation** — send each thumbnail to Ollama `bakllava` and parse:
   - `eyes_open` (true/false)
   - `expression` (natural / smile / stiff / forced)
   - `eye_contact` (good / moderate / poor / none)
   - `quality` (excellent / good / average / poor)
   - `lighting` (good / over / under)
   - `composition` (good / centered / off)
7. **Selection** — score every image with configurable weights; pick the best from each burst group; pass singletons and non-portraits through.
8. **Copy** — write selected → `selected/`, rejected → `not_selected/`.
9. **Report** — write `report.md` and `report.json` with per-image details.

## AI evaluation scoring weights

| Criterion | Default weight | Notes |
|-----------|---------------|-------|
| `eyes_open` | 10 | Closed eyes = –20 penalty |
| `expression_natural` | 8 | smile ≈ natural > stiff > forced |
| `eye_contact` | 6 | Reduced 40% for group shots |
| `quality` | 5 | Overall sharpness/clarity |
| `lighting` | 3 | Over/under-exposed gets low score |
| `composition` | 2 | good > centered > off |

Blurry images (blur score < threshold) always score 0 and are excluded.

## Graceful degradation

If Ollama is not running or `bakllava` is not installed, the AI step is skipped automatically. The pipeline continues using only preprocessor results (blur + face detection), and all portrait singletons are accepted.

## Dependencies

- Python 3.11+
- `opencv-python` (blur detection + YuNet face detection)
- Ollama with `bakllava` model
- macOS `sips` (thumbnail generation)

Install the model:

```bash
ollama pull bakllava
```

## Configuration

Edit `config.json` to tune behaviour:

```json
{
  "preprocessing": {
    "blur_threshold": 80,
    "burst_window_seconds": 2.0,
    "thumbnail_max_size": 1024
  },
  "ai_evaluation": {
    "model": "bakllava",
    "timeout_seconds": 60,
    "batch_size": 4
  },
  "selection": {
    "weights": {
      "eyes_open": 10,
      "expression_natural": 8,
      "eye_contact": 6,
      "quality": 5,
      "lighting": 3,
      "composition": 2
    }
  },
  "output": {
    "copy_not_selected": true
  }
}
```

Set `output.copy_not_selected` to `false` to skip copying rejected photos.

## Current limitations

- RAW sidecar sync is not implemented in the Python pipeline (`.ARW` pairing was a shell-script feature).
- `bakllava` is a 7B-parameter model; evaluation speed depends on hardware (~5–30 s per image without GPU).
- Thumbnail generation requires macOS `sips`; not available on Linux/Windows.
- Only `.ARW` RAW pairing existed in the legacy shell script; the Python pipeline does not yet copy RAW sidecars.

---

## 自评体系 (Evaluation Framework)

使用自评体系来测试不同模型、调优参数、验证筛选逻辑。

### 核心概念

- **测试集 (Test Case)**: 一组照片 + 人工标注的正确答案
- **Ground Truth**: 人工选中的照片列表（"正确答案"）
- **评估指标**: Accuracy / Precision / Recall / F1

### 快速开始

#### 1. 创建测试集

```bash
python evaluate.py create-case my_portrait_test /path/to/photos \
  --selected IMG_001.jpg,IMG_003.jpg,IMG_005.jpg \
  --description "人像照片测试集"
```

#### 2. 单模型评估

```bash
python evaluate.py eval my_portrait_test --model bakllava
```

#### 3. 多模型对比

```bash
python evaluate.py compare my_portrait_test \
  --models bakllava,llava:13b,moondream
```

输出示例:
```
📊 模型对比结果
======================================================================
模型                 Accuracy  Precision    Recall        F1    耗时(s)
----------------------------------------------------------------------
bakllava               85.71%     90.00%    81.82%    85.71%      245.3
llava:13b              78.57%     83.33%    71.43%    76.92%      312.5
moondream              71.43%     75.00%    68.18%    71.43%      198.7
======================================================================
🏆 最佳模型 (按 F1): bakllava (85.71%)
```

#### 4. 权重调优

自动搜索最佳权重配置:

```bash
python evaluate.py tune-weights my_portrait_test --model bakllava
```

自定义搜索范围:
```bash
python evaluate.py tune-weights my_portrait_test --model bakllava \
  --eyes-open 8 10 12 \
  --expression 6 8 10 \
  --quality 3 5 7
```

#### 5. 模糊阈值调优

```bash
python evaluate.py tune-blur my_portrait_test --model bakllava \
  --thresholds 60,80,100,120,150
```

#### 6. 错误分析

查看模型选错的照片:

```bash
python evaluate.py analyze my_portrait_test --model bakllava
```

输出:
```
📊 错误分析: bakllava
   False Positives (3): 模型选中但人工未选
     • IMG_010.jpg (score: 12.50)
     • IMG_015.jpg (score: 11.80)
     • IMG_022.jpg (score: 10.20)

   False Negatives (2): 模型未选但人工选了
     • IMG_008.jpg (score: 8.50)
     • IMG_019.jpg (score: 7.20)
```

### 测试集管理

```bash
# 列出所有测试集
python evaluate.py list-cases

# 测试集默认存储在 ~/.photo-picker-tests/
```

### 高级用法: 程序化调用

```python
from modules.test_suite import TestSuite, HyperParamTuner

# 加载测试集
suite = TestSuite("~/.photo-picker-tests")
case = suite.load_test_case("my_portrait_test")

# 评估单个模型
result = suite.run_evaluation(case, "bakllava", verbose=True)

# 多模型对比
results = suite.compare_models(case, ["bakllava", "llava:13b", "moondream"])

# 权重调优
tuner = HyperParamTuner(suite, case)
best_weights, best_result = tuner.grid_search_weights(
    "bakllava",
    weight_ranges={
        "eyes_open": [8, 10, 12],
        "expression_natural": [6, 8, 10],
        "eye_contact": [4, 6, 8],
        "quality": [3, 5, 7],
        "lighting": [2, 3, 4],
        "composition": [1, 2, 3],
    }
)

# 模糊阈值调优
best_threshold, best_result = tuner.tune_blur_threshold(
    "bakllava", thresholds=[60, 80, 100, 120]
)
```
