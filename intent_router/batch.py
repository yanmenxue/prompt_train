from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, TextIO

from .router import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_MODEL,
    IntentDecision,
    IntentRouter,
    RouterCredentials,
    load_bailian_credentials,
)


@dataclass(frozen=True)
class BatchStats:
    total: int
    status_counts: dict[str, int]

    @property
    def error_count(self) -> int:
        return sum(
            self.status_counts.get(status, 0)
            for status in ("input_error", "api_error", "invalid_model_output")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "error_count": self.error_count,
            "status_counts": self.status_counts,
        }


def conversation_without_system_messages(record: Any) -> list[dict[str, Any]]:
    """Extract OpenAI messages while dropping the main Agent's system prompt."""

    if not isinstance(record, Mapping):
        raise ValueError("每行顶层必须是 JSON 对象")
    messages = record.get("messages")
    if not isinstance(messages, list):
        raise ValueError("每行必须包含 messages 数组")

    conversation: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise ValueError(f"messages[{index}] 必须是对象")
        if message.get("role") == "system":
            continue
        conversation.append(
            {
                "role": message.get("role"),
                "content": message.get("content"),
            }
        )
    if not conversation:
        raise ValueError("忽略 system message 后没有剩余对话")
    return conversation


def _error_result(line_number: int, error_type: str, error_message: str) -> dict[str, Any]:
    return {
        "line_number": line_number,
        "model": None,
        "prompt_sha256": None,
        "candidate_order_seed": None,
        "model_output": None,
        "raw_model_output": None,
        "system_output": None,
        "status": "input_error",
        "should_route": False,
        "selected_candidate_id": None,
        "selected_probability": None,
        "decision_reason": None,
        "completion_tokens": None,
        "prompt_tokens": None,
        "error_type": error_type,
        "error_message": error_message,
    }


def _decision_result(line_number: int, decision: IntentDecision) -> dict[str, Any]:
    return {
        "line_number": line_number,
        "model": decision.model,
        "prompt_sha256": decision.prompt_sha256,
        "candidate_order_seed": decision.candidate_order_seed,
        "model_output": decision.output_name,
        "raw_model_output": decision.raw_model_output,
        "system_output": decision.intent_label,
        "status": decision.status.value,
        "should_route": decision.should_route,
        "selected_candidate_id": decision.selected_candidate_id,
        "selected_probability": decision.selected_probability,
        "decision_reason": decision.decision_reason,
        "completion_tokens": decision.completion_tokens,
        "prompt_tokens": decision.prompt_tokens,
        "error_type": decision.error_type,
        "error_message": None,
    }


def process_jsonl_line(raw_line: str, line_number: int, router: IntentRouter) -> dict[str, Any]:
    """Process one physical JSONL line and always return one output object."""

    try:
        record = json.loads(raw_line)
    except json.JSONDecodeError:
        return _error_result(line_number, "invalid_json", "该行不是合法 JSON")

    try:
        conversation = conversation_without_system_messages(record)
        decision = router.route(conversation)
    except (TypeError, ValueError) as exc:
        return _error_result(line_number, "invalid_input", str(exc))
    except Exception as exc:  # Defensive for a custom router configured not to fail closed.
        return _error_result(line_number, type(exc).__name__, "处理该行时发生未捕获异常")
    return _decision_result(line_number, decision)


def iter_jsonl_results(
    lines: Iterable[str],
    router: IntentRouter,
    *,
    workers: int = 1,
) -> Iterable[dict[str, Any]]:
    """Yield results in input order with bounded optional concurrency."""

    if workers <= 0:
        raise ValueError("workers 必须大于 0")
    numbered_lines = enumerate(lines, start=1)
    if workers == 1:
        for line_number, raw_line in numbered_lines:
            yield process_jsonl_line(raw_line, line_number, router)
        return

    max_pending = workers * 2
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="intent-router") as executor:
        pending: deque[Future[dict[str, Any]]] = deque()
        for line_number, raw_line in numbered_lines:
            pending.append(executor.submit(process_jsonl_line, raw_line, line_number, router))
            if len(pending) >= max_pending:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()


def process_jsonl(
    input_stream: TextIO,
    output_stream: TextIO,
    router: IntentRouter,
    *,
    workers: int = 1,
) -> BatchStats:
    """Route a JSONL stream, emitting exactly one JSON object per input line."""

    counts: Counter[str] = Counter()
    total = 0
    for result in iter_jsonl_results(input_stream, router, workers=workers):
        output_stream.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
        output_stream.flush()
        counts[result["status"]] += 1
        total += 1
    return BatchStats(total=total, status_counts=dict(sorted(counts.items())))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="批量调用 Qwen 意图选择器")
    parser.add_argument("input", help="输入 JSONL 文件；使用 - 从 stdin 读取")
    parser.add_argument("output", help="输出 JSONL 文件；使用 - 写入 stdout")
    parser.add_argument("--workers", type=int, default=1, help="并发请求数，默认 1")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="百炼配置文件")
    parser.add_argument(
        "--base-url",
        "--api-base",
        dest="base_url",
        help="OpenAI-compatible API base URL；优先于配置文件",
    )
    key_group = parser.add_mutually_exclusive_group()
    key_group.add_argument(
        "--api-key-env",
        metavar="ENV_NAME",
        help="从指定环境变量读取 API Key（推荐）",
    )
    key_group.add_argument(
        "--api-key",
        help="直接指定 API Key；可能留在 shell 历史中，仅建议临时使用",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="模型名称")
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="可选的真实候选放行概率阈值；默认关闭置信度门控",
    )
    return parser


def credentials_from_args(args: argparse.Namespace) -> RouterCredentials:
    """Resolve explicit API arguments before falling back to config/env defaults."""

    api_key = args.api_key
    if args.api_key_env:
        api_key = os.getenv(args.api_key_env)
        if not api_key:
            raise ValueError(f"环境变量 {args.api_key_env} 未设置或为空")
    if api_key is not None:
        api_key = api_key.strip()
        if not api_key:
            raise ValueError("--api-key 不能为空")
    return load_bailian_credentials(
        args.config,
        api_key=api_key,
        base_url=args.base_url,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.workers <= 0:
        raise SystemExit("--workers 必须大于 0")

    input_path = None if args.input == "-" else Path(args.input).expanduser()
    output_path = None if args.output == "-" else Path(args.output).expanduser()
    if input_path is not None and output_path is not None:
        if input_path.resolve() == output_path.resolve():
            raise SystemExit("输入和输出不能是同一个文件")
        output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        credentials = credentials_from_args(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    router = IntentRouter(
        credentials=credentials,
        model=args.model,
        timeout_seconds=args.timeout,
        min_route_probability=args.threshold,
    )

    input_context = (
        nullcontext(sys.stdin)
        if input_path is None
        else input_path.open("r", encoding="utf-8")
    )
    output_context = (
        nullcontext(sys.stdout)
        if output_path is None
        else output_path.open("w", encoding="utf-8", newline="\n")
    )
    with input_context as input_stream, output_context as output_stream:
        stats = process_jsonl(input_stream, output_stream, router, workers=args.workers)

    print(json.dumps(stats.to_dict(), ensure_ascii=False), file=sys.stderr)
    return 2 if stats.error_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
