from __future__ import annotations

import json
import math
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from intent_router import (
    IntentRouter,
    RouterCredentials,
    RoutingStatus,
    default_candidates,
    load_bailian_credentials,
)


class _FakeCompletions:
    def __init__(
        self,
        outputs: list[str] | None = None,
        error: Exception | None = None,
        logprob: float = 0.0,
        completion_tokens: int | None = None,
    ) -> None:
        self.outputs = outputs or ["StockRoute", "NoProduct"]
        self.error = error
        self.logprob = logprob
        self.completion_tokens = completion_tokens
        self.calls: list[dict] = []

    @property
    def last_kwargs(self):
        return self.calls[-1] if self.calls else None

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        output = self.outputs[min(len(self.calls) - 1, len(self.outputs) - 1)]
        tokens = re.findall(r"[A-Z]+(?=[A-Z][a-z]|$)|[A-Z][a-z]*|[a-z]+|\d+", output)
        if not tokens:
            tokens = [output]
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=output),
                    logprobs=SimpleNamespace(
                        content=[
                            SimpleNamespace(token=token, logprob=self.logprob)
                            for token in tokens
                        ]
                    ),
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=123,
                completion_tokens=(
                    self.completion_tokens
                    if self.completion_tokens is not None
                    else len(tokens)
                ),
            ),
        )


class _FakeClient:
    def __init__(
        self,
        outputs: list[str] | None = None,
        error: Exception | None = None,
        logprob: float = 0.0,
        completion_tokens: int | None = None,
    ) -> None:
        self.completions = _FakeCompletions(
            outputs=outputs,
            error=error,
            logprob=logprob,
            completion_tokens=completion_tokens,
        )
        self.chat = SimpleNamespace(completions=self.completions)


def _conversation_payload(prompt: str) -> dict:
    serialized = prompt.split("<conversation_json>", 1)[1].split(
        "</conversation_json>", 1
    )[0]
    return json.loads(serialized)


class IntentRouterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = _FakeClient()
        self.router = IntentRouter(client=self.client)

    def test_default_candidates_have_two_real_and_generic_virtuals(self) -> None:
        candidates = default_candidates()
        real = [candidate for candidate in candidates if not candidate.is_virtual]
        virtual = [candidate for candidate in candidates if candidate.is_virtual]
        self.assertEqual(
            [item.intent_label for item in real],
            ["SearchStockQuotes", "RecommendProduct"],
        )
        self.assertEqual(len(virtual), 11)
        self.assertEqual(
            {candidate.name for candidate in virtual},
            {
                "StockAdvice",
                "StockResearch",
                "StockOther",
                "FinanceOther",
                "ProductInfo",
                "ProductOther",
                "NonRetail",
                "MultiProduct",
                "ChitChat",
                "NoRequest",
                "NoAvailable",
            },
        )

    def test_two_prompts_are_short_stable_and_candidate_names_are_short(self) -> None:
        first = self.router.prepare([{"role": "user", "content": "推荐一个键盘"}])
        second = self.router.prepare([{"role": "user", "content": "查股票价格"}])
        self.assertEqual(
            first.stock_api_messages[0]["content"],
            second.stock_api_messages[0]["content"],
        )
        self.assertEqual(
            first.product_api_messages[0]["content"],
            second.product_api_messages[0]["content"],
        )
        self.assertLess(len(first.stock_api_messages[0]["content"]), 500)
        self.assertLess(len(first.product_api_messages[0]["content"]), 600)
        self.assertNotEqual(
            first.stock_api_messages[1]["content"],
            second.stock_api_messages[1]["content"],
        )
        names = [item.candidate.name for item in first.presented_candidates]
        self.assertEqual(len(names), 13)
        self.assertTrue(
            all(re.fullmatch(r"[A-Za-z][A-Za-z0-9]{1,23}", name) for name in names)
        )

    def test_prompts_are_generic_and_do_not_expose_backend_labels_or_examples(self) -> None:
        prepared = self.router.prepare([{"role": "user", "content": "占位请求"}])
        prompts = (
            prepared.stock_api_messages[0]["content"],
            prepared.product_api_messages[0]["content"],
        )
        for prompt in prompts:
            self.assertNotIn("SearchStockQuotes", prompt)
            self.assertNotIn("RecommendProduct", prompt)
            self.assertNotIn("示例", prompt)
        self.assertTrue(all(not candidate.examples for candidate in default_candidates()))

        benchmark_path = Path(__file__).resolve().parents[1] / "eval" / "benchmark_v1.jsonl"
        for line in benchmark_path.read_text(encoding="utf-8").splitlines():
            query = json.loads(line)["messages"][-1]["content"].strip()
            if len(query) >= 12:
                self.assertTrue(all(query not in prompt for prompt in prompts))

    def test_multiturn_payload_separates_current_user_history_and_reference(self) -> None:
        prepared = self.router.prepare(
            [
                {"role": "user", "content": "推荐三个型号"},
                {"role": "assistant", "content": "型号甲、型号乙、型号丙"},
                {"role": "user", "content": "第二款，看看它的替换电池"},
            ]
        )
        payload = _conversation_payload(prepared.stock_api_messages[1]["content"])
        self.assertEqual(payload["current_user_request"], "第二款，看看它的替换电池")
        self.assertEqual(
            payload["prior_user_turns"],
            [{"role": "user", "content": "推荐三个型号"}],
        )
        self.assertEqual(
            payload["assistant_tool_reference"],
            [{"role": "assistant", "content": "型号甲、型号乙、型号丙"}],
        )
        self.assertTrue(prepared.stock_api_messages[1]["content"].endswith("/no_think"))
        self.assertEqual(
            prepared.stock_api_messages[1]["content"],
            prepared.product_api_messages[1]["content"],
        )

    def test_self_contained_current_request_omits_assistant_and_tool_history(self) -> None:
        prepared = self.router.prepare(
            [
                {"role": "user", "content": "苹果"},
                {"role": "assistant", "content": "是水果、手机还是股票？"},
                {"role": "tool", "content": "候选对象很多"},
                {"role": "user", "content": "水果，想买一箱脆甜的"},
            ]
        )
        prompt = prepared.stock_api_messages[1]["content"]
        payload = _conversation_payload(prompt)
        self.assertEqual(
            payload["prior_user_turns"],
            [{"role": "user", "content": "苹果"}],
        )
        self.assertNotIn("assistant_tool_reference", payload)
        self.assertNotIn("是水果、手机还是股票", prompt)
        self.assertNotIn("候选对象很多", prompt)

    def test_long_assistant_history_is_compacted_without_losing_current_user(self) -> None:
        router = IntentRouter(
            client=_FakeClient(),
            max_history_chars=2_000,
            max_assistant_history_chars=120,
        )
        assistant = "回答开头" + "历史内容。" * 100 + "回答结尾"
        prepared = router.prepare(
            [
                {"role": "user", "content": "先介绍三个型号"},
                {"role": "assistant", "content": assistant},
                {"role": "user", "content": "我想对比这三款"},
            ]
        )
        payload = _conversation_payload(prepared.stock_api_messages[1]["content"])
        compacted = payload["assistant_tool_reference"][0]["content"]
        self.assertLessEqual(len(compacted), 120)
        self.assertIn("回答开头", compacted)
        self.assertIn("回答结尾", compacted)
        self.assertEqual(payload["current_user_request"], "我想对比这三款")

    def test_stock_boundary_maps_to_private_backend_intent(self) -> None:
        router = IntentRouter(client=_FakeClient(["StockRoute", "NoProduct"]))
        decision = router.route_text("查一下股票价格")
        self.assertEqual(decision.status, RoutingStatus.ROUTED)
        self.assertEqual(decision.intent_label, "SearchStockQuotes")
        self.assertEqual(decision.output_name, "StockInfo")
        self.assertEqual(decision.decision_reason, "stock_boundary")
        self.assertEqual(
            json.loads(decision.raw_model_output or "{}"),
            {"stock": "StockRoute", "product": "NoProduct"},
        )

    def test_product_boundary_maps_to_private_backend_intent(self) -> None:
        router = IntentRouter(client=_FakeClient(["NoStock", "ProductRoute"]))
        decision = router.route_text("推荐一个键盘")
        self.assertEqual(decision.status, RoutingStatus.ROUTED)
        self.assertEqual(decision.intent_label, "RecommendProduct")
        self.assertEqual(decision.output_name, "Ecommerce")

    def test_virtual_boundaries_and_generic_safety_rules_fail_closed(self) -> None:
        cases = [
            (["StockAdvice", "NoProduct"], "预测下周股票走势", "StockAdvice"),
            (["StockRoute", "NoProduct"], "假设大盘下跌哪些板块先跌", "StockAdvice"),
            (["NoStock", "ProductRoute"], "推荐一个酒店", "NonRetail"),
            (["NoStock", "ProductRoute"], "某品牌 X100", "NoRequest"),
        ]
        for outputs, query, expected_name in cases:
            with self.subTest(query=query):
                decision = IntentRouter(client=_FakeClient(outputs)).route_text(query)
                self.assertEqual(decision.status, RoutingStatus.NO_ROUTE)
                self.assertIsNone(decision.intent_label)
                self.assertEqual(decision.output_name, expected_name)

    def test_chitchat_is_preserved_as_diagnostic_virtual_intent(self) -> None:
        router = IntentRouter(client=_FakeClient(["NoStock", "NoProduct"]))
        decision = router.route_text("你好呀")
        self.assertEqual(decision.status, RoutingStatus.NO_ROUTE)
        self.assertEqual(decision.output_name, "ChitChat")
        self.assertEqual(decision.selected_candidate_id, "no_route_chitchat")

    def test_api_request_disables_thinking_without_output_limit(self) -> None:
        router = IntentRouter(client=_FakeClient(["NoStock", "ProductRoute"]))
        router.route_text("推荐耳机")
        self.assertEqual(len(router.client.completions.calls), 2)
        for kwargs in router.client.completions.calls:
            self.assertEqual(kwargs["model"], "qwen3-32b")
            self.assertNotIn("max_tokens", kwargs)
            self.assertEqual(kwargs["temperature"], 0)
            self.assertNotIn("logprobs", kwargs)
            self.assertEqual(
                kwargs["extra_body"],
                {"chat_template_kwargs": {"enable_thinking": False}},
            )
            self.assertTrue(kwargs["messages"][-1]["content"].endswith("/no_think"))

    def test_confidence_gate_is_disabled_by_default(self) -> None:
        client = _FakeClient(["NoStock", "ProductRoute"])
        decision = IntentRouter(client=client).route_text("推荐耳机")
        self.assertEqual(decision.status, RoutingStatus.ROUTED)
        self.assertIsNone(decision.selected_probability)
        self.assertTrue(all("logprobs" not in call for call in client.completions.calls))

    def test_optional_confidence_gate_fails_closed(self) -> None:
        client = _FakeClient(
            ["NoStock", "ProductRoute"],
            logprob=math.log(0.60),
        )
        decision = IntentRouter(
            client=client,
            min_route_probability=0.80,
        ).route_text("推荐耳机")
        self.assertEqual(decision.status, RoutingStatus.LOW_CONFIDENCE)
        self.assertIsNone(decision.intent_label)
        self.assertEqual(decision.selected_probability, 0.6)
        self.assertEqual(decision.error_type, "below_route_threshold")

    def test_invalid_boundary_output_preserves_both_raw_outputs(self) -> None:
        router = IntentRouter(
            client=_FakeClient(["解释文字", "NoProduct", "仍然解释"])
        )
        decision = router.route_text("推荐耳机")
        self.assertEqual(decision.status, RoutingStatus.INVALID_MODEL_OUTPUT)
        self.assertEqual(
            json.loads(decision.raw_model_output or "{}"),
            {"stock": ["解释文字", "仍然解释"], "product": "NoProduct"},
        )
        self.assertIsNone(decision.output_name)
        self.assertIsNone(decision.intent_label)

    def test_completion_token_count_sums_both_calls(self) -> None:
        client = _FakeClient(
            ["StockRoute", "NoProduct"],
            completion_tokens=100,
        )
        decision = IntentRouter(client=client).route_text("查股票价格")
        self.assertEqual(decision.status, RoutingStatus.ROUTED)
        self.assertEqual(decision.completion_tokens, 200)
        self.assertEqual(decision.prompt_tokens, 246)

    def test_api_error_fails_closed_without_error_message(self) -> None:
        client = _FakeClient(error=RuntimeError("secret-bearing upstream message"))
        decision = IntentRouter(client=client).route_text("推荐耳机")
        self.assertEqual(decision.status, RoutingStatus.API_ERROR)
        self.assertEqual(decision.error_type, "RuntimeError")
        self.assertNotIn("secret", str(decision.to_dict()))

    def test_direct_router_ignores_main_agent_system_message(self) -> None:
        prepared = self.router.prepare(
            [
                {"role": "system", "content": "主Agent系统提示词"},
                {"role": "user", "content": "查一下股票价格"},
            ]
        )
        prompt = prepared.stock_api_messages[1]["content"]
        self.assertNotIn("主Agent系统提示词", prompt)
        self.assertEqual(
            _conversation_payload(prompt),
            {"current_user_request": "查一下股票价格"},
        )

    def test_long_latest_message_is_trimmed_to_configured_limit(self) -> None:
        router = IntentRouter(client=_FakeClient(), max_history_chars=40)
        normalized = router._normalize_and_trim_messages(  # noqa: SLF001
            [{"role": "user", "content": "开头" + "中" * 100 + "结尾"}]
        )
        self.assertLessEqual(len(normalized[0]["content"]), 40)
        self.assertIn("开头", normalized[0]["content"])
        self.assertIn("结尾", normalized[0]["content"])

    def test_requires_latest_user_message_and_string_content(self) -> None:
        with self.assertRaisesRegex(ValueError, "最后一条消息"):
            self.router.prepare([{"role": "assistant", "content": "请继续"}])
        with self.assertRaisesRegex(ValueError, "必须是字符串"):
            self.router.prepare(  # type: ignore[list-item]
                [{"role": "user", "content": None}]
            )

    def test_credentials_support_two_line_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "api.txt"
            config.write_text(
                "https://example.test/compatible-mode/v1\nsk-test-value\n",
                encoding="utf-8",
            )
            credentials = load_bailian_credentials(config)
        self.assertEqual(credentials.base_url, "https://example.test/compatible-mode/v1")
        self.assertEqual(credentials.api_key, "sk-test-value")

    def test_custom_client_and_explicit_credentials_are_accepted(self) -> None:
        self.assertIsNotNone(IntentRouter(client=_FakeClient()))
        credentials = RouterCredentials(api_key="test", base_url="https://example.test/v1")
        self.assertIsNotNone(IntentRouter(credentials=credentials).client)


if __name__ == "__main__":
    unittest.main()
