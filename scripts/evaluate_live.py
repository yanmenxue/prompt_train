from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from intent_router import DEFAULT_MODEL, IntentRouter  # noqa: E402


DEFAULT_CASES = PROJECT_ROOT / "eval" / "benchmark_v1.jsonl"
SYSTEM_OUTPUT_BY_CANDIDATE_ID = {
    "stock_market_information": "SearchStockQuotes",
    "ecommerce_product_recommendation": "RecommendProduct",
}
VALID_SYSTEM_OUTPUTS = frozenset({None, "SearchStockQuotes", "RecommendProduct"})
OPERATIONAL_ERROR_STATUSES = frozenset({"api_error", "invalid_model_output"})


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * percentile)))
    return ordered[index]


def _load_cases(path: Path) -> tuple[list[dict[str, Any]], str]:
    raw_bytes = path.read_bytes()
    dataset_hash = hashlib.sha256(raw_bytes).hexdigest()
    if path.suffix.lower() == ".jsonl":
        payload: Any = [
            json.loads(line)
            for line in raw_bytes.decode("utf-8").splitlines()
            if line.strip()
        ]
    else:
        payload = json.loads(raw_bytes)
    if not isinstance(payload, list) or not payload:
        raise ValueError("评测文件必须是非空 JSON 数组或 JSONL")

    cases: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, raw_case in enumerate(payload):
        if not isinstance(raw_case, Mapping):
            raise ValueError(f"case[{index}] 必须是对象")
        case = dict(raw_case)
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError(f"case[{index}].id 必须是非空字符串")
        if case_id in seen_ids:
            raise ValueError(f"case id 重复: {case_id}")
        seen_ids.add(case_id)
        if not isinstance(case.get("messages"), list) or not case["messages"]:
            raise ValueError(f"case {case_id} 缺少非空 messages")

        expected_candidate = case.get("expected_candidate_id")
        derived_output = SYSTEM_OUTPUT_BY_CANDIDATE_ID.get(expected_candidate)
        if "expected_system_output" in case:
            expected_output = case["expected_system_output"]
            if expected_output not in VALID_SYSTEM_OUTPUTS:
                raise ValueError(f"case {case_id} 的 expected_system_output 非法")
            if expected_candidate in SYSTEM_OUTPUT_BY_CANDIDATE_ID and expected_output != derived_output:
                raise ValueError(f"case {case_id} 的候选标注与系统输出标注冲突")
        elif isinstance(expected_candidate, str):
            expected_output = derived_output
        else:
            raise ValueError(
                f"case {case_id} 必须包含 expected_system_output 或 expected_candidate_id"
            )
        case["expected_system_output"] = expected_output
        case.setdefault("bucket", "unspecified")
        case.setdefault("split", "unspecified")
        cases.append(case)
    return cases, dataset_hash


def _load_case_ids(path: Path) -> list[str]:
    case_ids = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not case_ids:
        raise ValueError("case id 文件必须至少包含一个 id")
    duplicates = [case_id for case_id, count in Counter(case_ids).items() if count > 1]
    if duplicates:
        raise ValueError(f"case id 文件存在重复 id: {duplicates[:5]}")
    return case_ids


def _merge_result_rows(
    cases: Sequence[Mapping[str, Any]],
    result_paths: Sequence[Path],
    *,
    seeds: Sequence[int | None] = (None,),
    repeats: int = 1,
) -> list[dict[str, Any]]:
    """Merge result files by run key; later files deliberately replace earlier rows."""

    expected_by_id = {str(case["id"]): case for case in cases}
    expected_keys = {
        (case_id, seed, repeat)
        for case_id in expected_by_id
        for seed in seeds
        for repeat in range(repeats)
    }
    merged: dict[tuple[str, int | None, int], dict[str, Any]] = {}
    for path in result_paths:
        seen_in_file: set[tuple[str, int | None, int]] = set()
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            raw_row = json.loads(line)
            if not isinstance(raw_row, Mapping):
                raise ValueError(f"{path}:{line_number} 结果行必须是对象")
            row = dict(raw_row)
            case_id = row.get("id")
            if not isinstance(case_id, str) or case_id not in expected_by_id:
                continue
            row_seed = row.get("seed")
            if row_seed is not None and not isinstance(row_seed, int):
                raise ValueError(f"{path}:{line_number} 的 {case_id} seed 非法")
            row_repeat = row.get("repeat", 0)
            if not isinstance(row_repeat, int) or row_repeat < 0:
                raise ValueError(f"{path}:{line_number} 的 {case_id} repeat 非法")
            run_key = (case_id, row_seed, row_repeat)
            if run_key not in expected_keys:
                continue
            if run_key in seen_in_file:
                raise ValueError(f"{path} 内结果运行键重复: {run_key}")
            seen_in_file.add(run_key)
            expected_case = expected_by_id[case_id]
            expected_output = expected_case["expected_system_output"]
            if row.get("expected_system_output") != expected_output:
                raise ValueError(f"{path}:{line_number} 的 {case_id} 期望输出与数据集不一致")
            for field in ("split", "bucket"):
                if row.get(field) != expected_case[field]:
                    raise ValueError(f"{path}:{line_number} 的 {case_id} {field} 与数据集不一致")
            recomputed_error = _error_kind(
                expected_output,
                row.get("actual_system_output"),
                str(row.get("status")),
            )
            if row.get("error_kind") != recomputed_error or row.get("system_correct") != (
                recomputed_error is None
            ):
                raise ValueError(f"{path}:{line_number} 的 {case_id} 正确性字段不一致")
            merged[run_key] = row

    missing = [run_key for run_key in expected_keys if run_key not in merged]
    if missing:
        raise ValueError(f"合并结果缺少 {len(missing)} 次调用: {missing[:5]}")
    rows = [
        merged[(str(case["id"]), seed, repeat)]
        for case in cases
        for seed in seeds
        for repeat in range(repeats)
    ]
    models = {str(row.get("model")) for row in rows}
    if len(models) != 1:
        raise ValueError(f"合并结果包含多个模型: {sorted(models)}")
    prompt_hashes_by_seed: dict[int | None, set[str | None]] = defaultdict(set)
    for row in rows:
        prompt_hashes_by_seed[row.get("candidate_order_seed")].add(row.get("prompt_sha256"))
    inconsistent_prompt_seeds = {
        seed: hashes
        for seed, hashes in prompt_hashes_by_seed.items()
        if len(hashes) > 1
    }
    if inconsistent_prompt_seeds:
        raise ValueError(f"合并结果包含不同提示词版本: {inconsistent_prompt_seeds}")
    return rows


def _error_kind(
    expected_output: str | None,
    actual_output: str | None,
    status: str,
) -> str | None:
    if status in OPERATIONAL_ERROR_STATUSES:
        return "operational_error"
    if actual_output == expected_output:
        return None
    if expected_output is None:
        return "unsafe_false_route"
    if actual_output is None:
        return "false_rejection"
    return "wrong_route"


def _evaluate_one(
    router: IntentRouter,
    case: Mapping[str, Any],
    seed: int | None,
    repeat: int = 0,
) -> dict[str, Any]:
    decision = router.route(case["messages"], order_seed=seed)
    expected_output = case["expected_system_output"]
    status = decision.status.value
    error_kind = _error_kind(expected_output, decision.intent_label, status)
    expected_candidate = case.get("expected_candidate_id")
    return {
        "id": case["id"],
        "split": case["split"],
        "bucket": case["bucket"],
        "seed": seed,
        "repeat": repeat,
        "model": decision.model,
        "prompt_sha256": decision.prompt_sha256,
        "candidate_order_seed": decision.candidate_order_seed,
        "expected_system_output": expected_output,
        "actual_system_output": decision.intent_label,
        "system_correct": error_kind is None,
        "expected_candidate_id": expected_candidate,
        "selected_candidate_id": decision.selected_candidate_id,
        "candidate_exact": (
            decision.selected_candidate_id == expected_candidate
            if isinstance(expected_candidate, str)
            else None
        ),
        "model_output": decision.output_name,
        "raw_model_output": decision.raw_model_output,
        "status": status,
        "error_kind": error_kind,
        "error_type": decision.error_type,
        "completion_tokens": decision.completion_tokens,
        "latency_ms": decision.latency_ms,
    }


def _label_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float | int]]:
    metrics: dict[str, dict[str, float | int]] = {}
    for label in ("SearchStockQuotes", "RecommendProduct", None):
        expected_count = sum(row["expected_system_output"] == label for row in rows)
        predicted_count = sum(
            row["actual_system_output"] == label
            and row["status"] not in OPERATIONAL_ERROR_STATUSES
            for row in rows
        )
        true_positive = sum(
            row["error_kind"] is None and row["expected_system_output"] == label
            for row in rows
        )
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / expected_count if expected_count else 0.0
        metrics["null" if label is None else label] = {
            "support": expected_count,
            "predicted": predicted_count,
            "correct": true_positive,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
        }
    return metrics


def _group_accuracy(
    rows: Sequence[Mapping[str, Any]], key: str
) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    return {
        name: {
            "correct": sum(item["error_kind"] is None for item in items),
            "total": len(items),
            "accuracy": round(
                sum(item["error_kind"] is None for item in items) / len(items), 6
            ),
        }
        for name, items in sorted(grouped.items())
    }


def _sample_accuracy(
    rows: Sequence[Mapping[str, Any]],
    *,
    positive: bool,
) -> dict[str, float | int | None]:
    selected = [
        row
        for row in rows
        if (row["expected_system_output"] is not None) is positive
    ]
    correct = sum(row["error_kind"] is None for row in selected)
    return {
        "correct": correct,
        "total": len(selected),
        "accuracy": round(correct / len(selected), 6) if selected else None,
    }


def _fails_acceptance_thresholds(
    summary: Mapping[str, Any],
    *,
    min_system_accuracy: float | None = None,
    min_positive_accuracy: float | None = None,
    min_negative_accuracy: float | None = None,
) -> bool:
    checks = (
        (min_system_accuracy, summary["system_accuracy"]),
        (min_positive_accuracy, summary["route_metrics"]["positive"]["accuracy"]),
        (min_negative_accuracy, summary["route_metrics"]["negative"]["accuracy"]),
    )
    return any(
        threshold is not None and (actual is None or actual < threshold)
        for threshold, actual in checks
    )


def _build_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    dataset_path: Path,
    dataset_hash: str,
    case_count: int,
) -> dict[str, Any]:
    correct = sum(row["error_kind"] is None for row in rows)
    candidate_rows = [row for row in rows if row["candidate_exact"] is not None]
    candidate_correct = sum(row["candidate_exact"] is True for row in candidate_rows)
    errors = Counter(row["error_kind"] for row in rows if row["error_kind"] is not None)
    statuses = Counter(str(row["status"]) for row in rows)
    confusion = Counter(
        (
            "null" if row["expected_system_output"] is None else row["expected_system_output"],
            "null" if row["actual_system_output"] is None else row["actual_system_output"],
        )
        for row in rows
        if row["status"] not in OPERATIONAL_ERROR_STATUSES
    )
    latencies = [float(row["latency_ms"]) for row in rows]
    completion_tokens = sorted(
        {
            int(row["completion_tokens"])
            for row in rows
            if row["completion_tokens"] is not None
        }
    )
    prompt_sha256s = sorted(
        {
            str(row["prompt_sha256"])
            for row in rows
            if row.get("prompt_sha256") is not None
        }
    )
    candidate_order_seeds = sorted(
        {
            int(row["candidate_order_seed"])
            for row in rows
            if row.get("candidate_order_seed") is not None
        }
    )
    return {
        "model": next(iter({str(row["model"]) for row in rows})),
        "prompt_sha256s": prompt_sha256s,
        "candidate_order_seeds": candidate_order_seeds,
        "dataset": str(dataset_path),
        "dataset_sha256": dataset_hash,
        "cases": case_count,
        "runs": len(rows),
        "system_correct": correct,
        "system_accuracy": round(correct / len(rows), 6),
        "candidate_exact_correct": candidate_correct,
        "candidate_exact_total": len(candidate_rows),
        "candidate_exact_accuracy": round(candidate_correct / len(candidate_rows), 6)
        if candidate_rows
        else None,
        "error_counts": dict(sorted(errors.items())),
        "status_counts": dict(sorted(statuses.items())),
        "label_metrics": _label_metrics(rows),
        "route_metrics": {
            "positive": _sample_accuracy(rows, positive=True),
            "negative": _sample_accuracy(rows, positive=False),
        },
        "confusion": {
            f"{expected}->{actual}": count
            for (expected, actual), count in sorted(confusion.items())
        },
        "split_metrics": _group_accuracy(rows, "split"),
        "bucket_metrics": _group_accuracy(rows, "bucket"),
        "completion_tokens": completion_tokens,
        "latency_ms": {
            "median": round(statistics.median(latencies), 3),
            "p95": round(_percentile(latencies, 0.95), 3),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="调用真实 Qwen 接口，严格评测最终系统 Intent 输出"
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="模型名称")
    parser.add_argument("--timeout", type=float, default=10.0, help="单次 API 超时秒数")
    parser.add_argument(
        "--seeds",
        type=int,
        default=1,
        help="每个样例使用多少组候选顺序；1 表示生产默认顺序",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="每个样例和候选顺序重复调用次数；用于观测服务端非确定性",
    )
    parser.add_argument("--limit", type=int, help="过滤后只运行前 N 个样例")
    parser.add_argument(
        "--split",
        action="append",
        help="只评测指定 split；可重复传入",
    )
    parser.add_argument(
        "--bucket",
        action="append",
        help="只评测指定 bucket；可重复传入",
    )
    parser.add_argument(
        "--case-id-file",
        type=Path,
        help="仅评测文本文件中列出的 case id；每行一个 id，支持 # 注释",
    )
    parser.add_argument("--workers", type=int, default=1, help="并发请求数")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="可选的真实候选放行概率阈值；默认关闭",
    )
    parser.add_argument(
        "--show",
        choices=("all", "errors", "summary"),
        default="errors",
        help="控制标准输出明细，默认只显示错误",
    )
    parser.add_argument("--output", type=Path, help="保存每次调用结果的 JSONL")
    parser.add_argument("--summary-output", type=Path, help="保存汇总 JSON")
    parser.add_argument(
        "--min-system-accuracy",
        type=float,
        help="验收阈值，例如 0.99；低于阈值时返回非零",
    )
    parser.add_argument(
        "--min-positive-accuracy",
        type=float,
        help="正样本正确率阈值；正样本必须召回到标注的具体 Intent",
    )
    parser.add_argument(
        "--min-negative-accuracy",
        type=float,
        help="负样本正确率阈值；负样本必须不召回",
    )
    return parser


def _print_row(row: Mapping[str, Any]) -> None:
    expected = row["expected_system_output"] or "null"
    actual = row["actual_system_output"] or "null"
    seed = "auto" if row["seed"] is None else str(row["seed"])
    repeat = int(row.get("repeat", 0))
    print(
        f"{row['id']:<42} seed={seed:<4} repeat={repeat:<2} "
        f"system={'Y' if row['system_correct'] else 'N'} "
        f"expected={expected:<20} actual={actual:<20} "
        f"candidate={str(row['model_output']):<18} "
        f"error={str(row['error_kind']):<20} "
        f"tokens={str(row['completion_tokens']):<3} "
        f"latency={float(row['latency_ms']):.1f}ms"
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.seeds <= 0:
        raise SystemExit("--seeds 必须大于 0")
    if args.repeats <= 0:
        raise SystemExit("--repeats 必须大于 0")
    if args.workers <= 0:
        raise SystemExit("--workers 必须大于 0")
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit 必须大于 0")
    for name, threshold in (
        ("--min-system-accuracy", args.min_system_accuracy),
        ("--min-positive-accuracy", args.min_positive_accuracy),
        ("--min-negative-accuracy", args.min_negative_accuracy),
    ):
        if threshold is not None and not 0 <= threshold <= 1:
            raise SystemExit(f"{name} 必须位于 [0, 1]")

    cases, dataset_hash = _load_cases(args.cases)
    if args.case_id_file:
        selected_case_ids = _load_case_ids(args.case_id_file)
        available_case_ids = {str(case["id"]) for case in cases}
        unknown_case_ids = [
            case_id
            for case_id in selected_case_ids
            if case_id not in available_case_ids
        ]
        if unknown_case_ids:
            raise SystemExit(f"case id 文件包含未知 id: {unknown_case_ids[:5]}")
        selected_case_id_set = set(selected_case_ids)
        cases = [case for case in cases if str(case["id"]) in selected_case_id_set]
    if args.split:
        selected_splits = set(args.split)
        cases = [case for case in cases if case["split"] in selected_splits]
        if not cases:
            raise SystemExit("split 过滤后没有评测样例")
    if args.bucket:
        selected_buckets = set(args.bucket)
        cases = [case for case in cases if case["bucket"] in selected_buckets]
        if not cases:
            raise SystemExit("bucket 过滤后没有评测样例")
    if args.limit is not None:
        cases = cases[: args.limit]

    seeds: Sequence[int | None] = (None,) if args.seeds == 1 else tuple(range(args.seeds))
    work = [
        (case, seed, repeat)
        for case in cases
        for seed in seeds
        for repeat in range(args.repeats)
    ]
    router = IntentRouter(
        model=args.model,
        timeout_seconds=args.timeout,
        min_route_probability=args.threshold,
    )

    def run(item: tuple[dict[str, Any], int | None, int]) -> dict[str, Any]:
        return _evaluate_one(router, item[0], item[1], item[2])

    if args.workers == 1:
        rows = [run(item) for item in work]
    else:
        with ThreadPoolExecutor(
            max_workers=args.workers, thread_name_prefix="intent-eval"
        ) as executor:
            rows = list(executor.map(run, work))

    for row in rows:
        if args.show == "all" or (args.show == "errors" and row["error_kind"] is not None):
            _print_row(row)

    summary = _build_summary(
        rows,
        dataset_path=args.cases,
        dataset_hash=dataset_hash,
        case_count=len(cases),
    )
    print(json.dumps(summary, ensure_ascii=False, separators=(",", ":")))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                for row in rows
            ),
            encoding="utf-8",
        )
    if args.summary_output:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    operational_errors = summary["error_counts"].get("operational_error", 0)
    if operational_errors:
        return 1
    if any(
        threshold is not None
        for threshold in (
            args.min_system_accuracy,
            args.min_positive_accuracy,
            args.min_negative_accuracy,
        )
    ):
        return int(
            _fails_acceptance_thresholds(
                summary,
                min_system_accuracy=args.min_system_accuracy,
                min_positive_accuracy=args.min_positive_accuracy,
                min_negative_accuracy=args.min_negative_accuracy,
            )
        )
    return int(summary["system_correct"] != summary["runs"])


if __name__ == "__main__":
    raise SystemExit(main())
