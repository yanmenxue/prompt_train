from __future__ import annotations

import json
import unittest
from collections import Counter
from pathlib import Path

from scripts.build_benchmark_v1 import CASES, DEFAULT_OUTPUT, build_jsonl


ROUND3_COHORT_PATH = (
    DEFAULT_OUTPUT.parent
    / "cohorts"
    / "production_misclassifications_20260807_round3.txt"
)


class BenchmarkV1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.disk_text = DEFAULT_OUTPUT.read_text(encoding="utf-8")
        cls.rows = [json.loads(line) for line in cls.disk_text.splitlines() if line]

    def test_frozen_file_matches_reviewed_generator(self) -> None:
        self.assertEqual(self.disk_text, build_jsonl())
        self.assertEqual(len(CASES), 672)
        self.assertEqual(
            Counter(row["review_status"] for row in self.rows),
            {
                "reviewed_2026-08-06": 552,
                "reconciled_with_user_ground_truth_2026-08-07": 8,
                "user_ground_truth_2026-08-07": 94,
                "user_ground_truth_2026-08-07_round2": 18,
            },
        )

    def test_balanced_final_outputs_and_stratified_splits(self) -> None:
        self.assertEqual(
            Counter(row["expected_system_output"] for row in self.rows),
            {"SearchStockQuotes": 163, "RecommendProduct": 164, None: 345},
        )
        self.assertEqual(
            Counter(row["split"] for row in self.rows),
            {"dev": 420, "test": 140, "regression": 112},
        )
        by_bucket: dict[str, list[dict[str, object]]] = {}
        for row in self.rows:
            by_bucket.setdefault(row["bucket"], []).append(row)
        self.assertEqual(len(by_bucket), 33)
        legacy_buckets = {
            name: items
            for name, items in by_bucket.items()
            if not name.startswith("production_regression_")
        }
        self.assertEqual(len(legacy_buckets), 28)
        for items in legacy_buckets.values():
            self.assertEqual(len(items), 20)
            self.assertEqual(Counter(item["split"] for item in items), {"dev": 15, "test": 5})
        self.assertEqual(
            {
                name: len(items)
                for name, items in by_bucket.items()
                if name.startswith("production_regression_")
            },
            {
                "production_regression_NoAvailable": 61,
                "production_regression_RecommendProduct": 15,
                "production_regression_SearchStockQuotes": 18,
                "production_regression_round2_NoAvailable": 12,
                "production_regression_round2_RecommendProduct": 6,
            },
        )

    def test_dataset_has_unique_ids_conversations_and_multiturn_coverage(self) -> None:
        ids = [row["id"] for row in self.rows]
        conversations = [
            json.dumps(row["messages"], ensure_ascii=False, separators=(",", ":"))
            for row in self.rows
        ]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(conversations), len(set(conversations)))
        self.assertEqual(sum(len(row["messages"]) > 1 for row in self.rows), 110)
        self.assertTrue(all(row["messages"][-1]["role"] == "user" for row in self.rows))
        self.assertTrue(
            all(
                message["role"] != "system"
                for row in self.rows
                for message in row["messages"]
            )
        )

    def test_real_candidate_ids_match_specific_system_outputs(self) -> None:
        expected_by_candidate = {
            "stock_market_information": "SearchStockQuotes",
            "ecommerce_product_recommendation": "RecommendProduct",
        }
        for row in self.rows:
            candidate = row["expected_candidate_id"]
            if candidate in expected_by_candidate:
                self.assertEqual(
                    row["expected_system_output"], expected_by_candidate[candidate], row["id"]
                )
            else:
                self.assertIsNone(row["expected_system_output"], row["id"])

    def test_required_regressions_and_boundaries_are_present(self) -> None:
        last_user_texts = {
            row["messages"][-1]["content"]: row["expected_system_output"] for row in self.rows
        }
        self.assertEqual(
            last_user_texts["新版本的儿童手表推荐给我"], "RecommendProduct"
        )
        self.assertEqual(
            last_user_texts["给我一个s后即可的购买链接"], "RecommendProduct"
        )
        self.assertEqual(last_user_texts["预测贵州茅台明天涨还是跌"], None)
        self.assertEqual(last_user_texts["怎么开一个证券账户"], None)
        self.assertEqual(last_user_texts["预算20万推荐一辆家用SUV"], None)
        self.assertEqual(
            last_user_texts["看上证指数行情并帮我写请假邮件"],
            "SearchStockQuotes",
        )
        self.assertEqual(
            last_user_texts["推荐扫地机器人并教我写Python爬虫"],
            "RecommendProduct",
        )
        self.assertIsNone(last_user_texts["查一下茅台股价，再推荐一台5000元笔记本"])
        self.assertEqual(
            last_user_texts["去淘宝再来一单娇兰精华液"],
            "RecommendProduct",
        )
        self.assertEqual(
            last_user_texts[
                "把智能分辨率给我看一下接着然后一拿起来就亮屏的功能开了没？"
                "还有今日大盘情况接着再看查看一下小艺字幕的字幕是左对齐、右对齐还是居中？"
            ],
            "SearchStockQuotes",
        )
        self.assertIsNone(last_user_texts["红米K90Pro max多少钱"])
        self.assertIsNone(last_user_texts["德业股份属于储能龙头吗？属于细分的哪个赛道"])
        self.assertIsNone(last_user_texts["我主要用来运行虚拟机"])
        self.assertIsNone(last_user_texts["你买包"])
        self.assertIsNone(last_user_texts["帮我打开京东，在京东上买夫西地酸乳膏"])
        self.assertEqual(last_user_texts["看下p40电池"], "RecommendProduct")
        self.assertEqual(last_user_texts["搜同款"], "RecommendProduct")
        self.assertIsNone(last_user_texts["如果大盘出现5%下跌，哪些行业板块可能先行下挫？"])

    def test_round3_misclassification_cohort_reuses_unique_benchmark_cases(self) -> None:
        cohort_ids = [
            line.strip()
            for line in ROUND3_COHORT_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        self.assertEqual(len(cohort_ids), 37)
        self.assertEqual(len(set(cohort_ids)), 37)
        rows_by_id = {row["id"]: row for row in self.rows}
        self.assertTrue(set(cohort_ids) <= set(rows_by_id))
        self.assertEqual(
            Counter(rows_by_id[case_id]["expected_system_output"] for case_id in cohort_ids),
            {None: 27, "RecommendProduct": 6, "SearchStockQuotes": 4},
        )


if __name__ == "__main__":
    unittest.main()
