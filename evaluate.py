#!/usr/bin/env python3
"""
Photo Picker 自评体系 CLI

用法:
    # 创建测试集
    python evaluate.py create-case <name> <input_dir> --selected file1.jpg,file2.jpg

    # 单模型评估
    python evaluate.py eval <case_name> --model bakllava

    # 多模型对比
    python evaluate.py compare <case_name> --models bakllava,llava:13b,moondream

    # 权重调优
    python evaluate.py tune-weights <case_name> --model bakllava

    # 模糊阈值调优
    python evaluate.py tune-blur <case_name> --model bakllava

    # 列出所有测试集
    python evaluate.py list-cases
"""

import argparse
import sys
from pathlib import Path

from modules.test_suite import TestSuite, HyperParamTuner


def cmd_create_case(args):
    """创建测试集"""
    suite = TestSuite(args.suite_dir)

    selected_files = [f.strip() for f in args.selected.split(",")]

    tags = None
    if args.tags:
        tags = {}
        for tag_spec in args.tags:
            tag_name, files = tag_spec.split("=")
            tags[tag_name] = [f.strip() for f in files.split(",")]

    case = suite.create_test_case(
        name=args.name,
        description=args.description or f"Test case: {args.name}",
        input_dir=args.input_dir,
        selected_files=selected_files,
        tags=tags,
    )
    print(f"✅ 测试集创建成功: {args.name}")
    print(f"   路径: {suite.cases_dir / args.name}.json")


def cmd_eval(args):
    """单模型评估"""
    suite = TestSuite(args.suite_dir)
    case = suite.load_test_case(args.case_name)

    result = suite.run_evaluation(case, args.model, verbose=True)
    suite.save_result(result, args.case_name)


def cmd_compare(args):
    """多模型对比"""
    suite = TestSuite(args.suite_dir)
    case = suite.load_test_case(args.case_name)

    models = [m.strip() for m in args.models.split(",")]
    suite.compare_models(case, models)


def cmd_tune_weights(args):
    """权重调优"""
    suite = TestSuite(args.suite_dir)
    case = suite.load_test_case(args.case_name)
    tuner = HyperParamTuner(suite, case)

    # 默认搜索范围
    weight_ranges = {
        "eyes_open": args.eyes_open or [8, 10, 12],
        "expression_natural": args.expression or [6, 8, 10],
        "eye_contact": args.eye_contact or [4, 6, 8],
        "quality": args.quality or [3, 5, 7],
        "lighting": args.lighting or [2, 3, 4],
        "composition": args.composition or [1, 2, 3],
    }

    best_config, best_result = tuner.grid_search_weights(
        args.model, weight_ranges, verbose=True
    )

    # 保存最佳配置
    config_path = suite.results_dir / f"{args.case_name}_best_weights.json"
    import json
    config_path.write_text(json.dumps(best_config, indent=2), encoding="utf-8")
    print(f"\n💾 最佳配置已保存: {config_path}")


def cmd_tune_blur(args):
    """模糊阈值调优"""
    suite = TestSuite(args.suite_dir)
    case = suite.load_test_case(args.case_name)
    tuner = HyperParamTuner(suite, case)

    thresholds = [int(t) for t in args.thresholds.split(",")]
    best_threshold, best_result = tuner.tune_blur_threshold(args.model, thresholds)

    # 保存结果
    config_path = suite.results_dir / f"{args.case_name}_best_blur_threshold.txt"
    config_path.write_text(str(best_threshold), encoding="utf-8")
    print(f"\n💾 最佳阈值已保存: {config_path}")


def cmd_list_cases(args):
    """列出测试集"""
    suite = TestSuite(args.suite_dir)
    cases = suite.list_test_cases()

    if not cases:
        print("暂无测试集")
        return

    print(f"\n📁 测试集列表 ({len(cases)} 个):")
    for name in cases:
        case = suite.load_test_case(name)
        total = len(case.ground_truth)
        selected = sum(case.ground_truth.values())
        print(f"  • {name}: {total} 张 (选中 {selected} 张)")
        if case.description:
            print(f"    {case.description}")


def cmd_analyze(args):
    """分析错误案例"""
    suite = TestSuite(args.suite_dir)
    case = suite.load_test_case(args.case_name)

    # 运行评估
    result = suite.run_evaluation(case, args.model)

    ground_truth = case.ground_truth
    predicted_set = set(result.selected_files)

    # 分类错误
    false_positives = []  # 预测选中但实际不该选
    false_negatives = []  # 预测未选但实际该选

    for fname, should_select in ground_truth.items():
        is_selected = fname in predicted_set
        if is_selected and not should_select:
            false_positives.append(fname)
        elif not is_selected and should_select:
            false_negatives.append(fname)

    print(f"\n📊 错误分析: {args.model}")
    print(f"   False Positives ({len(false_positives)}): 模型选中但人工未选")
    for fname in false_positives[:10]:
        score = result.per_image_scores.get(fname, 0)
        print(f"     • {fname} (score: {score:.2f})")
    if len(false_positives) > 10:
        print(f"     ... 还有 {len(false_positives) - 10} 个")

    print(f"\n   False Negatives ({len(false_negatives)}): 模型未选但人工选了")
    for fname in false_negatives[:10]:
        score = result.per_image_scores.get(fname, 0)
        print(f"     • {fname} (score: {score:.2f})")
    if len(false_negatives) > 10:
        print(f"     ... 还有 {len(false_negatives) - 10} 个")


def main():
    parser = argparse.ArgumentParser(
        description="Photo Picker 自评体系",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 创建测试集（人工选中 5 张）
  python evaluate.py create-case my_test /path/to/photos --selected IMG_001.jpg,IMG_003.jpg,IMG_005.jpg,IMG_007.jpg,IMG_009.jpg

  # 对比 3 个模型
  python evaluate.py compare my_test --models bakllava,llava:13b,moondream

  # 调优权重
  python evaluate.py tune-weights my_test --model bakllava
        """,
    )

    parser.add_argument(
        "--suite-dir",
        default=str(Path.home() / ".photo-picker-tests"),
        help="测试集存储目录 (默认: ~/.photo-picker-tests)",
    )

    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # create-case
    p_create = subparsers.add_parser("create-case", help="创建测试集")
    p_create.add_argument("name", help="测试集名称")
    p_create.add_argument("input_dir", help="照片目录")
    p_create.add_argument("--selected", required=True, help="选中的文件名,逗号分隔")
    p_create.add_argument("--description", help="测试集描述")
    p_create.add_argument("--tags", action="append", help="标签,格式: tagname=file1.jpg,file2.jpg")
    p_create.set_defaults(func=cmd_create_case)

    # eval
    p_eval = subparsers.add_parser("eval", help="单模型评估")
    p_eval.add_argument("case_name", help="测试集名称")
    p_eval.add_argument("--model", default="bakllava", help="模型名称")
    p_eval.set_defaults(func=cmd_eval)

    # compare
    p_compare = subparsers.add_parser("compare", help="多模型对比")
    p_compare.add_argument("case_name", help="测试集名称")
    p_compare.add_argument("--models", default="bakllava,llava:13b", help="逗号分隔的模型列表")
    p_compare.set_defaults(func=cmd_compare)

    # tune-weights
    p_tune_w = subparsers.add_parser("tune-weights", help="权重调优")
    p_tune_w.add_argument("case_name", help="测试集名称")
    p_tune_w.add_argument("--model", default="bakllava", help="模型名称")
    p_tune_w.add_argument("--eyes-open", type=int, nargs="+", help="搜索范围")
    p_tune_w.add_argument("--expression", type=int, nargs="+", help="搜索范围")
    p_tune_w.add_argument("--eye-contact", type=int, nargs="+", help="搜索范围")
    p_tune_w.add_argument("--quality", type=int, nargs="+", help="搜索范围")
    p_tune_w.add_argument("--lighting", type=int, nargs="+", help="搜索范围")
    p_tune_w.add_argument("--composition", type=int, nargs="+", help="搜索范围")
    p_tune_w.set_defaults(func=cmd_tune_weights)

    # tune-blur
    p_tune_b = subparsers.add_parser("tune-blur", help="模糊阈值调优")
    p_tune_b.add_argument("case_name", help="测试集名称")
    p_tune_b.add_argument("--model", default="bakllava", help="模型名称")
    p_tune_b.add_argument("--thresholds", default="60,80,100,120", help="逗号分隔的阈值列表")
    p_tune_b.set_defaults(func=cmd_tune_blur)

    # list-cases
    p_list = subparsers.add_parser("list-cases", help="列出所有测试集")
    p_list.set_defaults(func=cmd_list_cases)

    # analyze
    p_analyze = subparsers.add_parser("analyze", help="分析错误案例")
    p_analyze.add_argument("case_name", help="测试集名称")
    p_analyze.add_argument("--model", default="bakllava", help="模型名称")
    p_analyze.set_defaults(func=cmd_analyze)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()