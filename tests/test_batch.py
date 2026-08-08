from __future__ import annotations

import io
import json
import os
import unittest
from unittest.mock import patch

from intent_router import IntentDecision, RoutingStatus
from intent_router.batch import (
    build_parser,
    conversation_without_system_messages,
    credentials_from_args,
    process_jsonl,
)


class _StubRouter:
    def __init__(self) -> None:
        self.conversations: list[list[dict[str, str]]] = []

    def route(self, messages):
        conversation = list(messages)
        self.conversations.append(conversation)
        text = conversation[-1]["content"]
        if "非法输出" in text:
            return IntentDecision(
                status=RoutingStatus.INVALID_MODEL_OUTPUT,
                intent_label=None,
                selected_candidate_id=None,
                selected_candidate_name=None,
                output_name=None,
                latency_ms=1.0,
                model="fake",
                prompt_sha256="prompt-hash",
                candidate_order_seed=0,
                raw_model_output="Ecommerce.",
                prompt_tokens=100,
                completion_tokens=3,
                error_type="unrecognized_candidate_name",
            )
        if "股价" in text:
            return IntentDecision(
                status=RoutingStatus.ROUTED,
                intent_label="SearchStockQuotes",
                selected_candidate_id="stock_market_information",
                selected_candidate_name="StockInfo",
                output_name="StockInfo",
                latency_ms=1.0,
                model="fake",
                prompt_sha256="prompt-hash",
                candidate_order_seed=0,
                raw_model_output="StockInfo",
                prompt_tokens=100,
                completion_tokens=2,
                selected_probability=0.99,
            )
        return IntentDecision(
            status=RoutingStatus.NO_ROUTE,
            intent_label=None,
            selected_candidate_id="no_route_chitchat",
            selected_candidate_name="ChitChat",
            output_name="ChitChat",
            latency_ms=1.0,
            model="fake",
            prompt_sha256="prompt-hash",
            candidate_order_seed=0,
            raw_model_output="ChitChat",
            prompt_tokens=100,
            completion_tokens=3,
            selected_probability=0.98,
        )


class BatchTests(unittest.TestCase):
    def test_confidence_gate_is_disabled_by_default(self) -> None:
        args = build_parser().parse_args(["input.jsonl", "output.jsonl"])
        self.assertIsNone(args.threshold)

    def test_explicit_llm_api_can_be_read_from_named_environment_variable(self) -> None:
        args = build_parser().parse_args(
            [
                "input.jsonl",
                "output.jsonl",
                "--base-url",
                "https://llm.example.test/v1",
                "--api-key-env",
                "ROUTER_TEST_API_KEY",
                "--model",
                "custom-qwen",
            ]
        )
        with patch.dict(os.environ, {"ROUTER_TEST_API_KEY": "test-key-value"}, clear=False):
            credentials = credentials_from_args(args)
        self.assertEqual(credentials.base_url, "https://llm.example.test/v1")
        self.assertEqual(credentials.api_key, "test-key-value")
        self.assertEqual(args.model, "custom-qwen")

    def test_direct_api_key_and_api_base_alias_are_supported(self) -> None:
        args = build_parser().parse_args(
            [
                "input.jsonl",
                "output.jsonl",
                "--api-base",
                "https://another.example.test/v1/",
                "--api-key",
                "temporary-test-key",
            ]
        )
        credentials = credentials_from_args(args)
        self.assertEqual(credentials.base_url, "https://another.example.test/v1")
        self.assertEqual(credentials.api_key, "temporary-test-key")

    def test_missing_named_api_key_environment_variable_is_rejected(self) -> None:
        args = build_parser().parse_args(
            ["input.jsonl", "output.jsonl", "--api-key-env", "MISSING_ROUTER_TEST_KEY"]
        )
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MISSING_ROUTER_TEST_KEY", None)
            with self.assertRaisesRegex(ValueError, "未设置或为空"):
                credentials_from_args(args)

    def test_system_messages_are_removed_wherever_they_appear(self) -> None:
        conversation = conversation_without_system_messages(
            {
                "messages": [
                    {"role": "system", "content": "主Agent系统提示"},
                    {"role": "user", "content": "你好"},
                    {"role": "system", "content": "另一个系统提示"},
                    {"role": "assistant", "content": "你好呀"},
                    {"role": "user", "content": "谢谢"},
                ]
            }
        )
        self.assertEqual([message["role"] for message in conversation], ["user", "assistant", "user"])

    def test_output_contains_model_and_mapped_system_output(self) -> None:
        source = io.StringIO(
            json.dumps(
                {
                    "messages": [
                        {"role": "system", "content": "不能传给路由器"},
                        {"role": "user", "content": "查一下茅台股价"},
                    ]
                },
                ensure_ascii=False,
            )
            + "\n"
            + json.dumps(
                {"messages": [{"role": "user", "content": "你好呀"}]},
                ensure_ascii=False,
            )
            + "\n"
        )
        target = io.StringIO()
        router = _StubRouter()
        stats = process_jsonl(source, target, router)
        rows = [json.loads(line) for line in target.getvalue().splitlines()]

        self.assertEqual(stats.total, 2)
        self.assertEqual(rows[0]["model_output"], "StockInfo")
        self.assertEqual(rows[0]["model"], "fake")
        self.assertEqual(rows[0]["prompt_sha256"], "prompt-hash")
        self.assertEqual(rows[0]["candidate_order_seed"], 0)
        self.assertEqual(rows[0]["raw_model_output"], "StockInfo")
        self.assertEqual(rows[0]["system_output"], "SearchStockQuotes")
        self.assertTrue(rows[0]["should_route"])
        self.assertEqual(rows[1]["model_output"], "ChitChat")
        self.assertEqual(rows[1]["raw_model_output"], "ChitChat")
        self.assertIsNone(rows[1]["system_output"])
        self.assertFalse(rows[1]["should_route"])
        self.assertEqual(router.conversations[0], [{"role": "user", "content": "查一下茅台股价"}])

    def test_invalid_line_still_produces_one_corresponding_output(self) -> None:
        source = io.StringIO("not-json\n" + json.dumps({"messages": []}) + "\n")
        target = io.StringIO()
        stats = process_jsonl(source, target, _StubRouter())
        rows = [json.loads(line) for line in target.getvalue().splitlines()]

        self.assertEqual(len(rows), 2)
        self.assertEqual([row["line_number"] for row in rows], [1, 2])
        self.assertTrue(all(row["status"] == "input_error" for row in rows))
        self.assertEqual(stats.error_count, 2)

    def test_unrecognized_output_preserves_raw_model_response(self) -> None:
        source = io.StringIO(
            json.dumps({"messages": [{"role": "user", "content": "触发非法输出"}]}) + "\n"
        )
        target = io.StringIO()
        process_jsonl(source, target, _StubRouter())
        row = json.loads(target.getvalue())

        self.assertEqual(row["status"], "invalid_model_output")
        self.assertEqual(row["error_type"], "unrecognized_candidate_name")
        self.assertIsNone(row["model_output"])
        self.assertEqual(row["raw_model_output"], "Ecommerce.")
        self.assertIsNone(row["system_output"])

    def test_system_only_record_is_an_input_error(self) -> None:
        source = io.StringIO(
            json.dumps({"messages": [{"role": "system", "content": "主Agent提示"}]}) + "\n"
        )
        target = io.StringIO()
        process_jsonl(source, target, _StubRouter())
        row = json.loads(target.getvalue())
        self.assertEqual(row["status"], "input_error")
        self.assertIsNone(row["model_output"])
        self.assertIsNone(row["raw_model_output"])
        self.assertIsNone(row["system_output"])


if __name__ == "__main__":
    unittest.main()
