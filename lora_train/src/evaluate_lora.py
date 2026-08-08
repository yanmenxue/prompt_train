"""Evaluate a LoRA-adapted Qwen3-32B on the 672-case benchmark.

Reuses the project's `scripts/evaluate_live.py` UNCHANGED by pointing the
IntentRouter's credential resolution at the local vLLM endpoint through
environment variables. This means we exercise the exact same fusion pipeline
the on-line 32B goes through — the LoRA is judged identically.

Prereq: start the local server first (see serve_vllm.sh):

    bash lora_train/src/serve_vllm.sh    # serves at http://localhost:8000/v1

Then:

    python lora_train/src/evaluate_lora.py --workers 8

It will:
  - set DASHSCOPE_BASE_URL=http://localhost:8000/v1 and a dummy API key
  - run the full benchmark_v1.jsonl (672 cases) via evaluate_live.py
  - write per-case rows + a summary JSON
  - return non-zero if positive < 0.95 or negative < 0.98

32B baseline (eval/results/benchmark_v1_qwen3_32b_compact_boundary_summary.json):
  663/672 = 98.66%, positive 97.25%, negative 100%. Beat or match it.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EVAL_LIVE = REPO / "scripts" / "evaluate_live.py"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    # In merge mode the served model name is whatever serve_vllm.sh set
    # (qwen3-32b-lora). The IntentRouter passes --model through to the API.
    p.add_argument("--model", default="qwen3-32b-lora")
    p.add_argument("--cases", default=str(REPO / "eval" / "benchmark_v1.jsonl"))
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--timeout", type=float, default=30.0)
    p.add_argument("--output", default="lora_train/eval/results.jsonl")
    p.add_argument("--summary-output", default="lora_train/eval/summary.json")
    p.add_argument("--min-positive-accuracy", type=float, default=0.95)
    p.add_argument("--min-negative-accuracy", type=float, default=0.98)
    p.add_argument("--min-system-accuracy", type=float, default=None)
    p.add_argument("--show", default="errors", choices=("all", "errors", "summary"))
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # Route IntentRouter's credential loader at the local vLLM endpoint.
    # load_bailian_credentials() honors DASHSCOPE_BASE_URL / DASHSCOPE_API_KEY
    # and never logs them, so this is the cleanest zero-code path.
    os.environ["DASHSCOPE_BASE_URL"] = args.base_url
    os.environ["DASHSCOPE_API_KEY"] = os.getenv("DASHSCOPE_API_KEY") or "local-vllm-no-key"

    out_dir = Path(args.output).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(EVAL_LIVE),
        "--cases", args.cases,
        "--model", args.model,
        "--timeout", str(args.timeout),
        "--workers", str(args.workers),
        "--repeats", str(args.repeats),
        "--output", args.output,
        "--summary-output", args.summary_output,
        "--show", args.show,
    ]
    if args.min_system_accuracy is not None:
        cmd += ["--min-system-accuracy", str(args.min_system_accuracy)]
    cmd += ["--min-positive-accuracy", str(args.min_positive_accuracy)]
    cmd += ["--min-negative-accuracy", str(args.min_negative_accuracy)]

    print("==>", " ".join(cmd), flush=True)
    print(f"    (DASHSCOPE_BASE_URL={args.base_url})", flush=True)
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
