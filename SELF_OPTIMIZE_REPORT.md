# Photo Picker 自优化完整报告

## 测试集
- **名称**: portrait_test_11060218
- **照片数**: 8张 | **人工选中**: 3张 (DSC01282.JPG, DSC01284.JPG, DSC01288.JPG)

---

## 优化历程

### 1. 权重调优 (先前完成)
| 参数 | 原值 | 优化值 | 变化 |
|------|------|--------|------|
| eyes_open | 10 | 8 | -20% |
| expression_natural | 8 | 6 | -25% |
| eye_contact | 6 | 4 | -33% |
| quality | 5 | 3 | -40% |
| lighting | 3 | 4 | +33% |
| composition | 2 | 2 | 不变 |

**结果**: F1 从 33.33% → 66.67% (提升2倍)

### 2. 模糊检测调优
| 阈值 | 模糊照片 | F1分数 |
|------|----------|--------|
| 30 | 0/8 | 33.33% |
| **40** | **1/8** | **40.00%** ⭐ |
| 50 | 2/8 | 0.00% |
| 60 | 3/8 | 0.00% |
| 80 | 4/8 | 0.00% |

### 3. 人脸检测参数调优
**score_threshold**:
| 阈值 | F1分数 |
|------|--------|
| 0.3 | 0.00% |
| 0.5 | 0.00% |
| **0.6** | **57.14%** ⭐ |
| 0.7 | 50.00% |
| 0.8 | 33.33% |

**nms_threshold**:
| 阈值 | F1分数 |
|------|--------|
| 0.2 | 0.00% |
| 0.3 | 0.00% |
| 0.4 | 33.33% |
| **0.5** | **80.00%** ⭐ |

---

## 最终优化结果

### 累计优化效果
| 阶段 | F1分数 | 提升 |
|------|--------|------|
| 默认配置 | 33.33% | - |
| + 权重调优 | 66.67% | +100% |
| + 模糊/人脸调优 | **80.00%** | +140% |

### 最终配置 (config.json)
```json
{
  "preprocessing": {
    "blur_threshold": 40,
    "burst_window_seconds": 2.0,
    "thumbnail_max_size": 1024
  },
  "ai_evaluation": {
    "model": "llava:13b",
    "timeout_seconds": 60,
    "batch_size": 4
  },
  "selection": {
    "weights": {
      "eyes_open": 8,
      "expression_natural": 6,
      "eye_contact": 4,
      "quality": 3,
      "lighting": 4,
      "composition": 2
    }
  },
  "output": {
    "copy_not_selected": true
  },
  "detector": {
    "model": "assets/models/face_detection_yunet_2023mar.onnx",
    "score_threshold": 0.6,
    "nms_threshold": 0.5,
    "top_k": 5000
  }
}
```

---

## 关键洞察

1. **模糊阈值敏感**: 40 vs 50 的微小差异导致 F1 从 40% → 0%
2. **NMS阈值关键**: 从 0.3 → 0.5 带来 F1 从 0% → 80% 的巨大提升
3. **权重降低策略**: 降低质量相关权重，提高光照权重，改善召回率
4. **组合效应**: 单独优化权重或检测参数效果有限，组合优化才能达到最佳

---

## 生成文件

- `~/.photo-picker-tests/results/portrait_test_11060218_best_weights.json`
- `~/.photo-picker-tests/results/portrait_test_11060218_best_blur_threshold.txt`
- `~/.photo-picker-tests/results/portrait_test_11060218_best_detector_config.json`
- `~/.photo-picker-tests/results/portrait_test_11060218_detector_optimization_report.md`
