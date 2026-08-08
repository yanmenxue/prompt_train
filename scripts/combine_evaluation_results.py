from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from evaluate_live import (
    DEFAULT_CASES,
    OPERATIONAL_ERROR_STATUSES,
    _build_summary,
    _fails_acceptance_thresholds,
    _load_case_ids,
    _load_cases,
    _merge_result_rows,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="按 case id 合并同模型、同提示词版本的分片评测结果"
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--split", action="append", help="只合并指定 split；可重复")
    parser.add_argument("--case-id-file", type=Path, help="仅合并清单中的 case id")
    parser.add_argument("--seeds", type=int, default=1, help="每条样例的候选顺序数量")
    parser.add_argument("--repeats", type=int, default=1, help="每个候选顺序的重复调用次数")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--min-positive-accuracy", type=float)
    parser.add_argument("--min-negative-accuracy", type=float)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for name, threshold in (
        ("--min-positive-accuracy", args.min_positive_accuracy),
        ("--min-negative-accuracy", args.min_negative_accuracy),
    ):
        if threshold is not None and not 0 <= threshold <= 1:
            raise SystemExit(f"{name} 必须位于 [0, 1]")
    if args.seeds <= 0 or args.repeats <= 0:
        raise SystemExit("--seeds 和 --repeats 必须大于 0")
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

    seeds = (None,) if args.seeds == 1 else tuple(range(args.seeds))
    rows = _merge_result_rows(cases, args.input, seeds=seeds, repeats=args.repeats)
    summary = _build_summary(
        rows,
        dataset_path=args.cases,
        dataset_hash=dataset_hash,
        case_count=len(cases),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, separators=(",", ":")))
    if any(str(row["status"]) in OPERATIONAL_ERROR_STATUSES for row in rows):
        return 1
    if args.min_positive_accuracy is not None or args.min_negative_accuracy is not None:
        return int(
            _fails_acceptance_thresholds(
                summary,
                min_positive_accuracy=args.min_positive_accuracy,
                min_negative_accuracy=args.min_negative_accuracy,
            )
        )
    return int(summary["system_correct"] != summary["runs"])


if __name__ == "__main__":
    raise SystemExit(main())
