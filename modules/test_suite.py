"""
Test Suite: 自评体系核心模块
管理测试集、运行评估、生成报告
"""

import json
import tempfile
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    # 相对导入（作为包的一部分）
    from .preprocessor import ImageInfo, scan_images
    from .evaluator import Evaluator
    from .selector import Selector
except ImportError:
    # 绝对导入（直接运行）
    from preprocessor import ImageInfo, scan_images
    from evaluator import Evaluator
    from selector import Selector


@dataclass
class TestCase:
    """单个测试用例"""
    name: str
    description: str
    input_dir: Path
    # 人工标注的正确答案: {filename: should_select (True/False)}
    ground_truth: dict[str, bool]
    # 可选：照片分类标签，用于分组统计
    tags: Optional[dict[str, list[str]]] = None


@dataclass
class EvaluationMetrics:
    """评估指标"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    total: int


@dataclass
class ModelResult:
    """单个模型的评估结果"""
    model_name: str
    config: dict
    metrics: EvaluationMetrics
    per_image_scores: dict[str, float]
    selected_files: list[str]
    duration_seconds: float


class TestSuite:
    """测试套件管理器"""

    def __init__(self, suite_dir: Path):
        self.suite_dir = Path(suite_dir)
        self.cases_dir = self.suite_dir / "cases"
        self.results_dir = self.suite_dir / "results"
        self.cases_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def create_test_case(
        self,
        name: str,
        description: str,
        input_dir: Path,
        selected_files: list[str],
        tags: Optional[dict] = None,
    ) -> TestCase:
        """
        创建测试用例
        selected_files: 人工选中的照片文件名列表
        """
        input_dir = Path(input_dir)
        all_images = scan_images(input_dir)
        all_names = {img.path.name for img in all_images}

        # 验证 selected_files 都存在
        for fname in selected_files:
            if fname not in all_names:
                raise ValueError(f"选中文件不存在: {fname}")

        ground_truth = {name: (name in selected_files) for name in all_names}

        case = TestCase(
            name=name,
            description=description,
            input_dir=input_dir,
            ground_truth=ground_truth,
            tags=tags,
        )

        # 保存测试用例
        case_file = self.cases_dir / f"{name}.json"
        self._save_test_case(case, case_file)
        print(f"[test-suite] 创建测试用例: {name} ({len(all_names)} 张, 选中 {len(selected_files)} 张)")

        return case

    def _save_test_case(self, case: TestCase, path: Path):
        """序列化测试用例"""
        data = {
            "name": case.name,
            "description": case.description,
            "input_dir": str(case.input_dir),
            "ground_truth": case.ground_truth,
            "tags": case.tags,
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def load_test_case(self, name: str) -> TestCase:
        """加载测试用例"""
        case_file = self.cases_dir / f"{name}.json"
        if not case_file.exists():
            raise FileNotFoundError(f"测试用例不存在: {name}")

        data = json.loads(case_file.read_text(encoding="utf-8"))
        return TestCase(
            name=data["name"],
            description=data["description"],
            input_dir=Path(data["input_dir"]),
            ground_truth=data["ground_truth"],
            tags=data.get("tags"),
        )

    def list_test_cases(self) -> list[str]:
        """列出所有测试用例"""
        return [f.stem for f in self.cases_dir.glob("*.json")]

    def run_evaluation(
        self,
        test_case: TestCase,
        model_name: str,
        config: Optional[dict] = None,
        verbose: bool = False,
    ) -> ModelResult:
        """
        对指定模型和配置运行评估
        """
        import time

        start_time = time.time()

        # 使用临时配置
        if config is None:
            config = self._load_default_config()

        config["ai_evaluation"]["model"] = model_name

        with tempfile.TemporaryDirectory() as tmpdir:
            # 运行预处理
            try:
                from .preprocessor import scan_images, generate_thumbnails, group_bursts, filter_blurry, detect_faces, filter_by_quality
            except ImportError:
                from preprocessor import scan_images, generate_thumbnails, group_bursts, filter_blurry, detect_faces, filter_by_quality

            pre_cfg = config.get("preprocessing", {})
            blur_threshold = pre_cfg.get("blur_threshold", 80)
            burst_window = pre_cfg.get("burst_window_seconds", 2.0)
            thumb_max = pre_cfg.get("thumbnail_max_size", 1024)

            images = scan_images(test_case.input_dir)
            thumb_dir = Path(tmpdir) / "thumbs"
            generate_thumbnails(images, thumb_dir, max_size=thumb_max)
            group_bursts(images, window_seconds=burst_window)
            filter_blurry(images, threshold=blur_threshold)
            detect_faces(images, config)

            # AI 评估
            evaluator = Evaluator(config)
            non_blurry = filter_by_quality(images)
            evaluations = evaluator.evaluate_batch(non_blurry)

            # 补齐 blurry 图片的默认评估
            try:
                from .evaluator import AIEvaluation
            except ImportError:
                from evaluator import AIEvaluation
            _blurry_default = AIEvaluation(
                eyes_open=True, expression="natural", eye_contact="moderate",
                quality="poor", lighting="good", composition="centered",
                reasoning="skipped (blurry)", skipped=True,
            )
            eval_by_path = {img.path: ev for img, ev in zip(non_blurry, evaluations)}
            all_evaluations = [eval_by_path.get(img.path, _blurry_default) for img in images]

            # 选片
            selector = Selector(config)
            selected, _ = selector.select_from_all(images, all_evaluations)

            selected_files = [img.path.name for img in selected]

            # 计算每张图的分数（用于分析）
            scores = selector.score_all(images, all_evaluations)
            per_image_scores = {img.path.name: score for img, score in zip(images, scores)}

        duration = time.time() - start_time

        # 计算指标
        metrics = self._compute_metrics(test_case.ground_truth, selected_files)

        result = ModelResult(
            model_name=model_name,
            config=config,
            metrics=metrics,
            per_image_scores=per_image_scores,
            selected_files=selected_files,
            duration_seconds=duration,
        )

        if verbose:
            self._print_metrics(result)

        return result

    def _load_default_config(self) -> dict:
        """加载默认配置"""
        skill_dir = Path(__file__).parent.parent
        config_path = skill_dir / "config.json"
        return json.loads(config_path.read_text(encoding="utf-8"))

    def _compute_metrics(
        self,
        ground_truth: dict[str, bool],
        predicted: list[str],
    ) -> EvaluationMetrics:
        """计算评估指标"""
        predicted_set = set(predicted)

        tp = sum(1 for fname, should_select in ground_truth.items()
                 if should_select and fname in predicted_set)
        fp = sum(1 for fname in predicted_set if not ground_truth.get(fname, False))
        tn = sum(1 for fname, should_select in ground_truth.items()
                 if not should_select and fname not in predicted_set)
        fn = sum(1 for fname, should_select in ground_truth.items()
                 if should_select and fname not in predicted_set)

        total = len(ground_truth)
        accuracy = (tp + tn) / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return EvaluationMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            true_positives=tp,
            false_positives=fp,
            true_negatives=tn,
            false_negatives=fn,
            total=total,
        )

    def _print_metrics(self, result: ModelResult):
        """打印评估结果"""
        m = result.metrics
        print(f"\n📊 {result.model_name} 评估结果:")
        print(f"  准确率 (Accuracy): {m.accuracy:.2%}")
        print(f"  精确率 (Precision): {m.precision:.2%}")
        print(f"  召回率 (Recall): {m.recall:.2%}")
        print(f"  F1 分数: {m.f1_score:.2%}")
        print(f"  TP/FP/TN/FN: {m.true_positives}/{m.false_positives}/{m.true_negatives}/{m.false_negatives}")
        print(f"  耗时: {result.duration_seconds:.1f}s")

    def save_result(self, result: ModelResult, test_case_name: str):
        """保存评估结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_model = result.model_name.replace(":", "-")
        result_file = self.results_dir / f"{test_case_name}_{safe_model}_{timestamp}.json"

        data = {
            "test_case": test_case_name,
            "timestamp": timestamp,
            "model_name": result.model_name,
            "config": result.config,
            "metrics": asdict(result.metrics),
            "selected_files": result.selected_files,
            "duration_seconds": result.duration_seconds,
        }

        result_file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[test-suite] 结果已保存: {result_file}")
        return result_file

    def compare_models(
        self,
        test_case: TestCase,
        models: list[str],
        config: Optional[dict] = None,
    ) -> list[ModelResult]:
        """多模型对比评估"""
        results = []
        print(f"\n🎯 开始多模型对比: {test_case.name}")
        print(f"   模型: {', '.join(models)}")
        print("=" * 50)

        for model in models:
            print(f"\n▶️ 评估 {model}...")
            result = self.run_evaluation(test_case, model, config, verbose=True)
            results.append(result)
            self.save_result(result, test_case.name)

        # 打印对比表
        self._print_comparison(results)
        return results

    def _print_comparison(self, results: list[ModelResult]):
        """打印模型对比表"""
        print("\n" + "=" * 70)
        print("📊 模型对比结果")
        print("=" * 70)
        print(f"{'模型':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'耗时(s)':>10}")
        print("-" * 70)

        for r in results:
            m = r.metrics
            print(f"{r.model_name:<20} {m.accuracy:>10.2%} {m.precision:>10.2%} {m.recall:>10.2%} {m.f1_score:>10.2%} {r.duration_seconds:>10.1f}")

        print("=" * 70)

        # 找出最佳模型
        best = max(results, key=lambda r: r.metrics.f1_score)
        print(f"🏆 最佳模型 (按 F1): {best.model_name} ({best.metrics.f1_score:.2%})")


class HyperParamTuner:
    """超参数调优器"""

    def __init__(self, test_suite: TestSuite, test_case: TestCase):
        self.test_suite = test_suite
        self.test_case = test_case

    def grid_search_weights(
        self,
        model_name: str,
        weight_ranges: dict[str, list[int]],
        verbose: bool = False,
    ) -> tuple[dict, ModelResult]:
        """
        网格搜索最佳权重配置

        weight_ranges: {"eyes_open": [8, 10, 12], "expression_natural": [6, 8, 10], ...}
        """
        import itertools

        keys = list(weight_ranges.keys())
        values = [weight_ranges[k] for k in keys]

        best_config = None
        best_result = None
        best_f1 = 0

        total_combinations = 1
        for v in values:
            total_combinations *= len(v)

        print(f"🔍 网格搜索: {total_combinations} 种组合")

        for i, combo in enumerate(itertools.product(*values)):
            weights = dict(zip(keys, combo))
            config = self.test_suite._load_default_config()
            config["selection"]["weights"] = weights

            result = self.test_suite.run_evaluation(self.test_case, model_name, config)

            if result.metrics.f1_score > best_f1:
                best_f1 = result.metrics.f1_score
                best_config = weights.copy()
                best_result = result

                if verbose:
                    print(f"  [{i+1}/{total_combinations}] 新最佳 F1={best_f1:.2%}: {weights}")
            elif verbose:
                print(f"  [{i+1}/{total_combinations}] F1={result.metrics.f1_score:.2%}: {weights}")

        print(f"\n🏆 最佳权重配置 (F1={best_f1:.2%}):")
        for k, v in best_config.items():
            print(f"  {k}: {v}")

        return best_config, best_result

    def tune_blur_threshold(
        self,
        model_name: str,
        thresholds: list[float] = [60, 80, 100, 120],
    ) -> tuple[float, ModelResult]:
        """调优模糊检测阈值"""
        best_threshold = None
        best_result = None
        best_f1 = 0

        print(f"🔍 模糊阈值调优: {thresholds}")

        for threshold in thresholds:
            config = self.test_suite._load_default_config()
            config["preprocessing"]["blur_threshold"] = threshold

            result = self.test_suite.run_evaluation(self.test_case, model_name, config)

            if result.metrics.f1_score > best_f1:
                best_f1 = result.metrics.f1_score
                best_threshold = threshold
                best_result = result
                print(f"  threshold={threshold}: 新最佳 F1={best_f1:.2%}")
            else:
                print(f"  threshold={threshold}: F1={result.metrics.f1_score:.2%}")

        print(f"\n🏆 最佳模糊阈值: {best_threshold} (F1={best_f1:.2%})")
        return best_threshold, best_result