from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_live import (
    _build_summary,
    _error_kind,
    _fails_acceptance_thresholds,
    _load_case_ids,
    _load_cases,
    _merge_result_rows,
    build_parser,
)


class EvaluateLiveTests(unittest.TestCase):
    def test_parser_supports_repeatable_split_and_bucket_filters(self) -> None:
        args = build_parser().parse_args(
            [
                "--split",
                "dev",
                "--bucket",
                "stock_current_quote",
                "--bucket",
                "stock_history_chart",
                "--repeats",
                "3",
                "--min-positive-accuracy",
                "0.95",
                "--min-negative-accuracy",
                "0.98",
            ]
        )
        self.assertEqual(args.split, ["dev"])
        self.assertEqual(args.bucket, ["stock_current_quote", "stock_history_chart"])
        self.assertEqual(args.repeats, 3)
        self.assertEqual(args.min_positive_accuracy, 0.95)
        self.assertEqual(args.min_negative_accuracy, 0.98)

    def test_load_case_ids_ignores_comments_and_rejects_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cohort.txt"
            path.write_text("# regression\ncase-1\n\ncase-2\n", encoding="utf-8")
            self.assertEqual(_load_case_ids(path), ["case-1", "case-2"])
            path.write_text("case-1\ncase-1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "重复"):
                _load_case_ids(path)

    def test_load_cases_derives_specific_system_output(self) -> None:
        payload = [
            {
                "id": "stock",
                "messages": [{"role": "user", "content": "查股价"}],
                "expected_candidate_id": "stock_market_information",
            },
            {
                "id": "other",
                "messages": [{"role": "user", "content": "查天气"}],
                "expected_candidate_id": "no_route_no_available",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            cases, dataset_hash = _load_cases(path)
        self.assertEqual(cases[0]["expected_system_output"], "SearchStockQuotes")
        self.assertIsNone(cases[1]["expected_system_output"])
        self.assertEqual(cases[0]["bucket"], "unspecified")
        self.assertEqual(len(dataset_hash), 64)

    def test_load_jsonl_and_reject_conflicting_labels(self) -> None:
        record = {
            "id": "conflict",
            "messages": [{"role": "user", "content": "查股价"}],
            "expected_candidate_id": "stock_market_information",
            "expected_system_output": "RecommendProduct",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "冲突"):
                _load_cases(path)

    def test_error_kind_distinguishes_cross_route_and_rejection(self) -> None:
        self.assertIsNone(_error_kind("SearchStockQuotes", "SearchStockQuotes", "routed"))
        self.assertEqual(
            _error_kind("SearchStockQuotes", "RecommendProduct", "routed"),
            "wrong_route",
        )
        self.assertEqual(
            _error_kind("RecommendProduct", None, "no_route"),
            "false_rejection",
        )
        self.assertEqual(
            _error_kind(None, "SearchStockQuotes", "routed"),
            "unsafe_false_route",
        )
        self.assertEqual(
            _error_kind(None, None, "invalid_model_output"),
            "operational_error",
        )

    def test_merge_result_rows_uses_later_retry_for_same_case(self) -> None:
        cases = [
            {
                "id": "case-1",
                "split": "dev",
                "bucket": "retry",
                "expected_system_output": None,
            }
        ]
        failed = {
            "id": "case-1",
            "model": "model-a",
            "split": "dev",
            "bucket": "retry",
            "expected_system_output": None,
            "actual_system_output": None,
            "status": "api_error",
            "error_kind": "operational_error",
            "system_correct": False,
        }
        retried = {
            "id": "case-1",
            "model": "model-a",
            "split": "dev",
            "bucket": "retry",
            "expected_system_output": None,
            "actual_system_output": None,
            "status": "no_route",
            "error_kind": None,
            "system_correct": True,
        }
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.jsonl"
            second = Path(directory) / "second.jsonl"
            first.write_text(json.dumps(failed) + "\n", encoding="utf-8")
            second.write_text(json.dumps(retried) + "\n", encoding="utf-8")
            rows = _merge_result_rows(cases, [first, second])
        self.assertEqual(rows, [retried])

    def test_merge_result_rows_distinguishes_seed_and_repeat(self) -> None:
        cases = [
            {
                "id": "case-1",
                "split": "dev",
                "bucket": "stability",
                "expected_system_output": None,
            }
        ]
        rows = []
        for seed in (0, 1):
            for repeat in (0, 1):
                rows.append(
                    {
                        "id": "case-1",
                        "model": "model-a",
                        "split": "dev",
                        "bucket": "stability",
                        "seed": seed,
                        "repeat": repeat,
                        "expected_system_output": None,
                        "actual_system_output": None,
                        "status": "no_route",
                        "error_kind": None,
                        "system_correct": True,
                    }
                )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.jsonl"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in reversed(rows)),
                encoding="utf-8",
            )
            merged = _merge_result_rows(cases, [path], seeds=(0, 1), repeats=2)
        self.assertEqual(
            [(row["seed"], row["repeat"]) for row in merged],
            [(0, 0), (0, 1), (1, 0), (1, 1)],
        )

    def test_merge_result_rows_rejects_mixed_prompt_versions(self) -> None:
        cases = [
            {
                "id": "case-1",
                "split": "dev",
                "bucket": "prompt-version",
                "expected_system_output": None,
            },
            {
                "id": "case-2",
                "split": "dev",
                "bucket": "prompt-version",
                "expected_system_output": None,
            },
        ]
        rows = []
        for index, case in enumerate(cases, start=1):
            rows.append(
                {
                    "id": case["id"],
                    "model": "model-a",
                    "prompt_sha256": f"prompt-{index}",
                    "candidate_order_seed": 0,
                    "split": "dev",
                    "bucket": "prompt-version",
                    "seed": None,
                    "repeat": 0,
                    "expected_system_output": None,
                    "actual_system_output": None,
                    "status": "no_route",
                    "error_kind": None,
                    "system_correct": True,
                }
            )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mixed.jsonl"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "不同提示词版本"):
                _merge_result_rows(cases, [path])

    def test_summary_uses_system_output_not_route_boolean(self) -> None:
        base = {
            "model": "fake-model",
            "split": "test",
            "bucket": "boundary",
            "candidate_exact": False,
            "completion_tokens": 2,
            "latency_ms": 10.0,
            "status": "routed",
        }
        rows = [
            {
                **base,
                "expected_system_output": "SearchStockQuotes",
                "actual_system_output": "SearchStockQuotes",
                "error_kind": None,
                "candidate_exact": True,
            },
            {
                **base,
                "expected_system_output": "SearchStockQuotes",
                "actual_system_output": "RecommendProduct",
                "error_kind": "wrong_route",
            },
        ]
        summary = _build_summary(
            rows,
            dataset_path=Path("cases.jsonl"),
            dataset_hash="abc",
            case_count=2,
        )
        self.assertEqual(summary["system_correct"], 1)
        self.assertEqual(summary["model"], "fake-model")
        self.assertEqual(summary["prompt_sha256s"], [])
        self.assertEqual(summary["candidate_order_seeds"], [])
        self.assertEqual(summary["system_accuracy"], 0.5)
        self.assertEqual(
            summary["route_metrics"],
            {
                "positive": {"correct": 1, "total": 2, "accuracy": 0.5},
                "negative": {"correct": 0, "total": 0, "accuracy": None},
            },
        )
        self.assertTrue(
            _fails_acceptance_thresholds(summary, min_positive_accuracy=0.95)
        )
        self.assertTrue(
            _fails_acceptance_thresholds(summary, min_negative_accuracy=0.98)
        )
        self.assertEqual(summary["error_counts"], {"wrong_route": 1})
        self.assertEqual(
            summary["confusion"],
            {
                "SearchStockQuotes->RecommendProduct": 1,
                "SearchStockQuotes->SearchStockQuotes": 1,
            },
        )

    def test_route_metrics_count_operational_failures_as_incorrect(self) -> None:
        common = {
            "model": "fake-model",
            "prompt_sha256": "prompt-hash",
            "candidate_order_seed": 0,
            "split": "test",
            "bucket": "acceptance",
            "candidate_exact": None,
            "completion_tokens": 2,
            "latency_ms": 10.0,
        }
        rows = [
            {
                **common,
                "expected_system_output": "SearchStockQuotes",
                "actual_system_output": "SearchStockQuotes",
                "status": "routed",
                "error_kind": None,
            },
            {
                **common,
                "expected_system_output": "RecommendProduct",
                "actual_system_output": "SearchStockQuotes",
                "status": "routed",
                "error_kind": "wrong_route",
            },
            {
                **common,
                "expected_system_output": None,
                "actual_system_output": None,
                "status": "no_route",
                "error_kind": None,
            },
            {
                **common,
                "expected_system_output": None,
                "actual_system_output": None,
                "status": "api_error",
                "error_kind": "operational_error",
                "completion_tokens": None,
            },
        ]
        summary = _build_summary(
            rows,
            dataset_path=Path("cases.jsonl"),
            dataset_hash="abc",
            case_count=4,
        )
        self.assertEqual(
            summary["route_metrics"],
            {
                "positive": {"correct": 1, "total": 2, "accuracy": 0.5},
                "negative": {"correct": 1, "total": 2, "accuracy": 0.5},
            },
        )


if __name__ == "__main__":
    unittest.main()
