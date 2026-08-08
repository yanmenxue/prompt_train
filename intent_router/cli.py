from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .router import DEFAULT_CONFIG_PATH, DEFAULT_MODEL, IntentRouter, RoutingStatus


def _load_history(path: Path) -> list[dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("history 文件顶层必须是 JSON 数组")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qwen3-32B 意图选择器")
    parser.add_argument("message", nargs="?", help="当前用户消息")
    parser.add_argument("--history-file", type=Path, help="包含多轮消息数组的 JSON 文件")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="百炼配置文件")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=float, default=10.0, help="API 超时秒数")
    parser.add_argument("--seed", type=int, help="记录评测 seed；当前候选顺序固定")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="可选的真实候选放行概率阈值；默认关闭置信度门控",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    messages: list[dict[str, str]] = []
    if args.history_file:
        messages.extend(_load_history(args.history_file))
    if args.message:
        messages.append({"role": "user", "content": args.message})
    if not messages:
        raise SystemExit("请提供 message 或 --history-file")

    router = IntentRouter(
        config_path=args.config,
        model=args.model,
        timeout_seconds=args.timeout,
        min_route_probability=args.threshold,
    )
    decision = router.route(messages, order_seed=args.seed)
    print(json.dumps(decision.to_dict(), ensure_ascii=False, indent=2))
    return 2 if decision.status in {RoutingStatus.API_ERROR, RoutingStatus.INVALID_MODEL_OUTPUT} else 0


if __name__ == "__main__":
    raise SystemExit(main())
