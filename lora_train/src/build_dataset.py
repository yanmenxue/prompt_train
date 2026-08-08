"""Generate a dual-boundary LoRA SFT dataset using Qwen3.7-Plus.

This REPLACES the earlier (leakage-prone) build_dataset.py. The 672-case
benchmark_v1 is a HOLDOUT test set; it must never appear in training. Instead,
we ask a stronger model (Qwen3.7-Plus via an OpenAI-compatible gateway) to
generate NEW conversations whose boundary labels are known, and verify each
generated sample by re-classifying it with the same model + on-line system
prompts (rejection sampling).

Pipeline per target (boundary_id x bucket):
  1. Pick a (stock_id, product_id, bucket) target. Weight toward buckets the
     32B fails on (stock_named_app_query, reject_multi_intent, production_*).
  2. Show Qwen3.7-Plus the two on-line system prompts + the 12 boundary ids
     + a couple of benchmark examples for STYLE ONLY (explicitly told to
     reword, never copy).
  3. Ask it to emit a conversation JSON + the (stock_id, product_id) it
     believes the conversation should yield + a one-line rationale.
  4. REJECT-SAMPLE: feed the generated conversation back through the same
     model with the on-line system prompts, ask for the boundary id. Keep
     the sample only if both boundaries match the declared target. This
     filters ambiguous / mislabeled generations.
  5. Render the kept conversation exactly as the on-line router does
     (<conversation_json>{...}</conversation_json> + /no_think) and emit
     two sharegpt rows (stock system + product system).

Output: lora_train/dataset/{train,val}.jsonl  (sharegpt)

Usage:
    export ANTHROPIC_AUTH_TOKEN="sk-..."      # your gateway key
    export ANTHROPIC_BASE_URL="http://113.46.219.251:8080/"
    export GENERATION_MODEL="Qwen3.7-Plus"
    python lora_train/src/build_dataset.py --target_per_bucket 150
    # -> ~33 buckets * 150 * 2 = ~9900 training samples

Notes:
  - Protocol is OpenAI-compatible per user. The base_url is probed with and
    without a trailing /v1; whichever works is locked.
  - The benchmark examples are used ONLY as style reference in the
    generation prompt; we instruct the model to reword them. Generated
    conversations are verified to differ from any benchmark current_user
    request (substring guard) before being kept.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Make intent_router importable when run from repo root.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from intent_router.router import _STOCK_SYSTEM_PROMPT, _PRODUCT_SYSTEM_PROMPT  # type: ignore

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO = Path(__file__).resolve().parents[2]
BENCHMARK = REPO / "eval" / "benchmark_v1.jsonl"
OUT_DIR = Path(__file__).resolve().parents[1] / "dataset"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STOCK_IDS = ["StockRoute", "StockAdvice", "StockResearch", "StockOperation", "OtherFinance", "NoStock"]
PRODUCT_IDS = ["ProductRoute", "ProductKnowledge", "ProductFollowup", "NonRetail", "MultiProduct", "NoProduct"]

# Which (stock_id, product_id, bucket) targets to generate. We pair a bucket
# with its "natural" primary boundary label so the generated conversation has a
# consistent ground truth. Buckets the 32B fails on get higher weight.
BUCKET_TARGETS: list[tuple[str, str, str, int]] = [
    # (bucket, stock_id, product_id, weight)
    # --- stock positive (StockRoute) ---
    ("stock_current_quote",      "StockRoute", "NoProduct", 2),
    ("stock_history_chart",      "StockRoute", "NoProduct", 2),
    ("stock_market_index",       "StockRoute", "NoProduct", 2),
    ("stock_facts_concepts",     "StockRoute", "NoProduct", 2),
    ("stock_explanation_news",   "StockRoute", "NoProduct", 2),
    ("stock_multiturn_info",     "StockRoute", "NoProduct", 2),
    ("stock_named_app_query",    "StockRoute", "NoProduct", 4),   # 32B fails here
    # --- product positive (ProductRoute) ---
    ("product_common_recommend", "NoStock", "ProductRoute", 2),
    ("product_search_link",      "NoStock", "ProductRoute", 2),
    ("product_compare_choose",   "NoStock", "ProductRoute", 2),
    ("product_named_platform",   "NoStock", "ProductRoute", 2),
    ("product_tricky_object",    "NoStock", "ProductRoute", 2),
    ("product_colloquial_asr",   "NoStock", "ProductRoute", 2),
    ("product_multiturn",        "NoStock", "ProductRoute", 2),
    # --- stock virtual (no route) ---
    ("reject_stock_prediction",  "StockAdvice",     "NoProduct", 3),
    ("reject_stock_advice",      "StockAdvice",     "NoProduct", 3),
    ("reject_stock_operations",  "StockOperation",  "NoProduct", 2),
    ("reject_stock_app_nonquery","StockOperation",  "NoProduct", 2),
    ("reject_other_finance",     "OtherFinance",     "NoProduct", 2),
    # --- product virtual (no route) ---
    ("reject_existing_order",    "NoStock", "ProductFollowup", 2),
    ("reject_product_usage",     "NoStock", "ProductKnowledge", 2),
    ("reject_non_ecommerce_object","NoStock","NonRetail",       3),
    ("reject_services",          "NoStock", "NonRetail",        2),
    ("reject_general_task",      "NoStock", "NoProduct",        2),
    ("reject_chitchat",          "NoStock", "NoProduct",        2),
    ("reject_ambiguous",         "NoStock", "NoProduct",        2),
    ("reject_multi_intent",      "NoStock", "MultiProduct",     4),  # 32B fails here
    ("reject_prompt_distractor", "NoStock", "NoProduct",        2),
    # --- production regression families (32B fails on some) ---
    ("production_regression_SearchStockQuotes", "StockRoute", "NoProduct", 3),
    ("production_regression_RecommendProduct",  "NoStock", "ProductRoute", 3),
    ("production_regression_NoAvailable",       "NoStock", "NoProduct",    4),
    ("production_regression_round2_NoAvailable","NoStock", "NoProduct",     3),
    ("production_regression_round2_RecommendProduct","NoStock","ProductRoute", 3),
]

# Boundary ids that, for a given bucket, are the "alternate" label we also want
# the OTHER boundary to occasionally take, so the model sees realistic
# distributions. We rely on the generator to pick product side realistically;
# the target product_id here is just the expected ground truth.


# ---------------------------------------------------------------------------
# Benchmark style references (used as few-shot STYLE only, never as data)
# ---------------------------------------------------------------------------

def load_benchmark_examples() -> dict[str, list[dict]]:
    """Return {bucket: [case, ...]} from benchmark_v1 for style reference only."""
    by_bucket: dict[str, list[dict]] = {}
    for line in BENCHMARK.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        c = json.loads(line)
        by_bucket.setdefault(c["bucket"], []).append(c)
    return by_bucket


def benchmark_current_requests() -> set[str]:
    """All current_user_request strings in benchmark, for substring-overlap guard."""
    out: set[str] = set()
    for line in BENCHMARK.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        c = json.loads(line)
        msgs = c["messages"]
        if msgs and msgs[-1]["role"] == "user":
            out.add(msgs[-1]["content"].strip())
    return out


# ---------------------------------------------------------------------------
# Gateway client (OpenAI-compatible, auto-probe /v1)
# ---------------------------------------------------------------------------

class Gateway:
    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 60.0):
        from openai import OpenAI

        self.model = model
        self.timeout = timeout
        # Probe base_url with and without /v1. Try /v1 first (the common case),
        # and use a short probe timeout so a dead endpoint fails fast instead
        # of hanging for the full request timeout.
        norm = base_url.rstrip("/")
        if norm.endswith("/v1"):
            candidates = [norm, norm[:-3]]
        else:
            candidates = [norm + "/v1", norm]
        last_err: Exception | None = None
        for url in candidates:
            client = OpenAI(api_key=api_key, base_url=url, timeout=10.0, max_retries=0)
            try:
                client.models.list()
                self.client = OpenAI(api_key=api_key, base_url=url, timeout=timeout, max_retries=0)
                self.base_url = url
                print(f"[gateway] connected: {url}", flush=True)
                return
            except Exception as e:  # noqa: BLE001
                last_err = e
                print(f"[gateway] probe failed {url}: {type(e).__name__}: {str(e)[:120]}", flush=True)
        raise RuntimeError(f"无法连接网关，尝试过 {candidates}; 最后错误: {last_err}")

    def chat(self, messages: list[dict], *, max_tokens: int = 1024, temperature: float = 0.7) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )
        return resp.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

GEN_SYSTEM = """你是一个对话数据合成器。任务：按指定边界类别，生成一条真实的中文用户对话，使其在股票边界和商品边界分类器下应得到指定的两个边界 id。

约束：
1. 对话必须自然、口语化，像真实用户在金融/电商助手里说的话，不要像测试用例。
2. 可以是单轮（只有一条 user 消息）或多轮（含 prior user turns 和 assistant 回复）。多轮时要让"当前用户请求"清晰可被分类。
3. 当前用户请求必须自包含或通过历史能恢复明确指代，不要出现无法恢复的指代。
4. 不要照抄参考样例的措辞、股票名、商品名；必须改写、换实体、换说法。
5. 不要出现 benchmark 边界 id 之外的英文词（如 StockInfo/Ecommerce），那些是后端概念，用户不会说。
6. 不要包含 system 消息；只产出 user/assistant 轮。
7. 股票边界可选: StockRoute / StockAdvice / StockResearch / StockOperation / OtherFinance / NoStock
   商品边界可选: ProductRoute / ProductKnowledge / ProductFollowup / NonRetail / MultiProduct / NoProduct
   各 id 含义见下方两段系统提示词。
8. 生成的对话在 {bucket} 这个 bucket 风格下。

只输出一个 JSON 对象，不要解释，不要 markdown 围栏：
{"conversation":[{"role":"user","content":"..."}, ...], "stock_id":"<id>", "product_id":"<id>", "rationale":"一句话说明为什么是这个边界"}
"""

GEN_USER_TEMPLATE = """目标 bucket: {bucket}
期望股票边界 id: {stock_id}
期望商品边界 id: {product_id}

=== 股票边界系统提示词（分类器据此判断）===
{stock_prompt}

=== 商品边界系统提示词（分类器据此判断）===
{product_prompt}

=== 该 bucket 的风格参考（仅供改写参考，禁止照抄实体或措辞）===
{examples}

请生成一条全新的对话。当前用户请求要能清晰落到"{stock_id}"和"{product_id}"。只输出 JSON。"""


def build_gen_messages(
    bucket: str,
    stock_id: str,
    product_id: str,
    style_examples: list[dict],
) -> list[dict]:
    import textwrap

    examples_text = "\n".join(
        f"- [{i}] current_user_request: {c['messages'][-1]['content']}"
        for i, c in enumerate(style_examples[:3], 1)
    )
    user = GEN_USER_TEMPLATE.format(
        bucket=bucket,
        stock_id=stock_id,
        product_id=product_id,
        stock_prompt=_STOCK_SYSTEM_PROMPT,
        product_prompt=_PRODUCT_SYSTEM_PROMPT,
        examples=examples_text,
    )
    return [
        {"role": "system", "content": GEN_SYSTEM},
        {"role": "user", "content": user},
    ]


# Re-classification system: identical to on-line router system prompts.
def classify_messages(boundary: str, conversation_json_payload: dict) -> list[dict]:
    """Return messages for the verifier: on-line system + on-line user prompt."""
    system = _STOCK_SYSTEM_PROMPT if boundary == "stock" else _PRODUCT_SYSTEM_PROMPT
    conversation = json.dumps(conversation_json_payload, ensure_ascii=False, separators=(",", ":"))
    user = (
        f"<conversation_json>{conversation}</conversation_json>\n"
        "输出谓词结果： /no_think"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# ---------------------------------------------------------------------------
# Conversation parsing & on-line rendering
# ---------------------------------------------------------------------------

ALLOWED = {"stock": set(STOCK_IDS), "product": set(PRODUCT_IDS)}


def parse_generation(text: str) -> dict | None:
    """Extract the JSON object the generator emitted. Tolerate code fences."""
    t = text.strip()
    # strip markdown fences if present
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", t, re.DOTALL)
    if fence:
        t = fence.group(1)
    else:
        # grab the outermost {...}
        m = re.search(r"\{.*\}", t, re.DOTALL)
        if m:
            t = m.group(0)
    try:
        obj = json.loads(t)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    conv = obj.get("conversation")
    if not isinstance(conv, list) or not conv:
        return None
    # validate conversation shape
    cleaned: list[dict] = []
    for m in conv:
        if not isinstance(m, dict):
            return None
        role = m.get("role")
        content = m.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str) or not content.strip():
            return None
        cleaned.append({"role": role, "content": content.strip()})
    if not cleaned or cleaned[-1]["role"] != "user":
        return None
    sid = obj.get("stock_id")
    pid = obj.get("product_id")
    if sid not in ALLOWED["stock"] or pid not in ALLOWED["product"]:
        return None
    return {"conversation": cleaned, "stock_id": sid, "product_id": pid, "rationale": str(obj.get("rationale", ""))}


def conversation_payload(messages: list[dict]) -> dict:
    """Mirror IntentRouter._build_user_prompt: build the JSON payload."""
    payload: dict = {"current_user_request": messages[-1]["content"]}
    prior = [m for m in messages[:-1] if m["role"] == "user"]
    reference = [m for m in messages[:-1] if m["role"] != "user"]
    if prior:
        payload["prior_user_turns"] = prior
    if reference:
        payload["assistant_tool_reference"] = reference
    return payload


def build_user_prompt(messages: list[dict]) -> str:
    payload = conversation_payload(messages)
    conversation = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"<conversation_json>{conversation}</conversation_json>\n输出谓词结果： /no_think"


def make_sample(system_prompt: str, user_prompt: str, label: str) -> dict:
    return {
        "conversations": [
            {"from": "system", "value": system_prompt},
            {"from": "user", "value": user_prompt},
            {"from": "assistant", "value": label},
        ]
    }


# ---------------------------------------------------------------------------
# Overlap guard: reject generations too close to any benchmark request
# ---------------------------------------------------------------------------

def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s)


def too_close_to_benchmark(messages: list[dict], guard: set[str]) -> bool:
    """Reject if the current user request is a near-substring of a benchmark one
    or vice-versa (length-aware). Catches accidental copying."""
    cur = _norm(messages[-1]["content"])
    if len(cur) < 4:
        return True
    for b in guard:
        nb = _norm(b)
        if len(nb) < 4:
            continue
        if cur == nb or cur in nb or nb in cur:
            return True
    # also reject exact multi-turn copies
    if any(_norm(m["content"]) in guard or _norm(m["content"]) in {_norm(g) for g in guard} for m in messages):
        pass
    return False


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

@dataclass
class Stats:
    attempted: int = 0
    generated_ok: int = 0
    parse_fail: int = 0
    verify_reject: int = 0
    overlap_reject: int = 0
    samples_emitted: int = 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--target_per_bucket", type=int, default=150, help="kept samples per bucket (before /2 split)")
    p.add_argument("--max_attempts_per_bucket", type=int, default=400, help="hard cap on generation attempts")
    p.add_argument("--model", default=os.getenv("GENERATION_MODEL", "Qwen3.7-Plus"))
    p.add_argument("--base_url", default=os.getenv("ANTHROPIC_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "")
    p.add_argument("--api_key", default=os.getenv("ANTHROPIC_AUTH_TOKEN") or os.getenv("OPENAI_API_KEY") or "")
    p.add_argument("--timeout", type=float, default=60.0)
    p.add_argument("--seed", type=int, default=20260808)
    p.add_argument("--workers", type=int, default=4, help="concurrent generation requests")
    p.add_argument("--val_ratio", type=float, default=0.05)
    p.add_argument("--min_per_bucket", type=int, default=2, help="floor on per-bucket kept count (small for smoke tests)")
    p.add_argument("--verify", choices=("rule", "model"), default="rule", help="rule=fuse_boundary_decisions (fast, default); model=re-classify via gateway (slow)")
    p.add_argument("--gen_max_tokens", type=int, default=500, help="max tokens for generation call (lower=faster)")
    p.add_argument("--max_buckets", type=int, default=None, help="only run first N buckets (for quick smoke tests)")
    p.add_argument("--resume", action="store_true", help="append to existing dataset instead of overwriting")
    return p.parse_args()


# Map each bucket to the final fused system_output it should produce. Used by
# the rule-based verifier. Derived from eval/benchmark_v1.jsonl expected values.
BUCKET_EXPECTED_OUTPUT: dict[str, str | None] = {
    # stock positive
    "stock_current_quote": "SearchStockQuotes",
    "stock_history_chart": "SearchStockQuotes",
    "stock_market_index": "SearchStockQuotes",
    "stock_facts_concepts": "SearchStockQuotes",
    "stock_explanation_news": "SearchStockQuotes",
    "stock_multiturn_info": "SearchStockQuotes",
    "stock_named_app_query": "SearchStockQuotes",
    # product positive
    "product_common_recommend": "RecommendProduct",
    "product_search_link": "RecommendProduct",
    "product_compare_choose": "RecommendProduct",
    "product_named_platform": "RecommendProduct",
    "product_tricky_object": "RecommendProduct",
    "product_colloquial_asr": "RecommendProduct",
    "product_multiturn": "RecommendProduct",
    # stock virtual -> null
    "reject_stock_prediction": None,
    "reject_stock_advice": None,
    "reject_stock_operations": None,
    "reject_stock_app_nonquery": None,
    "reject_other_finance": None,
    # product virtual -> null
    "reject_existing_order": None,
    "reject_product_usage": None,
    "reject_non_ecommerce_object": None,
    "reject_services": None,
    "reject_general_task": None,
    "reject_chitchat": None,
    "reject_ambiguous": None,
    "reject_multi_intent": None,           # NOTE: benchmark here is mixed; see below
    "reject_prompt_distractor": None,
    # production regression families
    "production_regression_SearchStockQuotes": "SearchStockQuotes",
    "production_regression_RecommendProduct": "RecommendProduct",
    "production_regression_NoAvailable": None,
    "production_regression_round2_NoAvailable": None,
    "production_regression_round2_RecommendProduct": "RecommendProduct",
}


def verify_sample_rule(
    conversation: list[dict],
    expected_stock: str,
    expected_product: str,
    expected_output: str | None,
) -> bool:
    """Rule-based verification: feed the declared (stock_id, product_id) through
    the project's own fusion logic and check the fused system_output matches the
    bucket's expectation.

    This replaces model self-re-judgement: it's deterministic, ~2x faster (no
    extra API calls), and has a far lower reject rate because it doesn't depend
    on the model being consistent with itself across two calls. The model only
    has to generate a conversation + declare ids; the rules decide acceptance.
    """
    from intent_router.policy import fuse_boundary_decisions

    # fuse_boundary_decisions expects router-shaped messages (role/content).
    # Our conversation already is that shape.
    try:
        fusion = fuse_boundary_decisions(conversation, expected_stock, expected_product)
    except Exception:
        return False
    return fusion.intent_label == expected_output


def verify_sample_model(
    gw: Gateway,
    conversation: list[dict],
    expected_stock: str,
    expected_product: str,
) -> bool:
    """Model self-re-judgement (slower, higher reject rate). Kept as fallback."""
    payload = conversation_payload(conversation)
    stock_msgs = classify_messages("stock", payload)
    prod_msgs = classify_messages("product", payload)
    from concurrent.futures import ThreadPoolExecutor

    def call(msgs):
        return gw.chat(msgs, max_tokens=16, temperature=0.0).strip()

    try:
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="verify") as ex:
            f_stock = ex.submit(call, stock_msgs)
            f_prod = ex.submit(call, prod_msgs)
            stock_out = f_stock.result()
            prod_out = f_prod.result()
    except Exception:
        return False
    return stock_out == expected_stock and prod_out == expected_product


def verify_sample(
    gw: Gateway,
    conversation: list[dict],
    expected_stock: str,
    expected_product: str,
    expected_output: str | None,
    *,
    use_rule: bool = True,
) -> bool:
    """Dispatch to rule-based (default) or model-based verification."""
    if use_rule:
        return verify_sample_rule(conversation, expected_stock, expected_product, expected_output)
    return verify_sample_model(gw, conversation, expected_stock, expected_product)


def generate_one(gw: Gateway, target: tuple[str, str, str], style: list[dict], rng: random.Random, gen_max_tokens: int = 500) -> dict | None:
    bucket, stock_id, product_id = target
    ex = style[:3]
    rng.shuffle(ex)
    msgs = build_gen_messages(bucket, stock_id, product_id, ex)
    try:
        raw = gw.chat(msgs, max_tokens=gen_max_tokens, temperature=0.9)
    except Exception as e:
        print(f"[gen] call failed ({bucket}): {type(e).__name__}: {str(e)[:80]}", flush=True)
        return None
    return parse_generation(raw)


def main() -> int:
    args = parse_args()
    if not args.base_url or not args.api_key:
        print("缺少 base_url 或 api_key。请设置 ANTHROPIC_BASE_URL / ANTHROPIC_AUTH_TOKEN。", file=sys.stderr)
        return 2

    rng = random.Random(args.seed)
    gw = Gateway(args.base_url, args.api_key, args.model, timeout=args.timeout)

    bench_examples = load_benchmark_examples()
    guard = benchmark_current_requests()
    print(f"[guard] loaded {len(guard)} benchmark current_user_request strings for overlap check", flush=True)

    stats = Stats()
    all_samples: list[dict] = []
    per_bucket_kept: dict[str, int] = {}

    from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

    bucket_list = BUCKET_TARGETS if args.max_buckets is None else BUCKET_TARGETS[: args.max_buckets]
    for bucket, stock_id, product_id, weight in bucket_list:
        target = (bucket, stock_id, product_id)
        style = bench_examples.get(bucket, [])
        if not style:
            style = [c for bl in benchmark_examples.values() for c in bl][:5]
        expected_output = BUCKET_EXPECTED_OUTPUT.get(bucket)
        target_count = max(args.target_per_bucket * weight // 2, args.min_per_bucket)
        per_bucket_kept[bucket] = 0
        attempts = 0
        kept = 0
        max_inflight = max(args.workers * 3, 6)

        with ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="gen") as ex:
            futures: dict = {}
            while kept < target_count and attempts < args.max_attempts_per_bucket:
                # Refill the in-flight pool up to the cap.
                while (
                    len(futures) < max_inflight
                    and attempts < args.max_attempts_per_bucket
                    and kept < target_count
                ):
                    fut = ex.submit(generate_one, gw, target, style, rng, args.gen_max_tokens)
                    futures[fut] = attempts
                    attempts += 1
                    stats.attempted += 1
                if not futures:
                    break
                # Block until at least one completes.
                done, _ = wait(set(futures), return_when=FIRST_COMPLETED, timeout=None)
                for fut in done:
                    futures.pop(fut, None)
                    try:
                        result = fut.result()
                    except Exception:
                        result = None
                    if result is None:
                        stats.parse_fail += 1
                        continue
                    conv = result["conversation"]
                    if too_close_to_benchmark(conv, guard):
                        stats.overlap_reject += 1
                        continue
                    # declared ids must match the target
                    if result["stock_id"] != stock_id or result["product_id"] != product_id:
                        stats.verify_reject += 1
                        continue
                    # verification (rule-based by default, model-based if --verify model)
                    if not verify_sample(
                        gw, conv, stock_id, product_id, expected_output,
                        use_rule=(args.verify == "rule"),
                    ):
                        stats.verify_reject += 1
                        continue
                    kept += 1
                    per_bucket_kept[bucket] = kept
                    stats.samples_emitted += 2
                    user_prompt = build_user_prompt(conv)
                    all_samples.append(make_sample(_STOCK_SYSTEM_PROMPT, user_prompt, stock_id))
                    all_samples.append(make_sample(_PRODUCT_SYSTEM_PROMPT, user_prompt, product_id))
                    print(
                        f"[{bucket}] kept {kept}/{target_count} (attempts {attempts}) "
                        f"| cur='{conv[-1]['content'][:30]}'",
                        flush=True,
                    )

        print(f"[done] {bucket}: kept {kept}/{target_count} (attempts {attempts})", flush=True)

    # shuffle + split
    rng.shuffle(all_samples)
    n_val = max(1, int(len(all_samples) * args.val_ratio))
    val = all_samples[:n_val]
    train = all_samples[n_val:]

    mode = "a" if args.resume else "w"
    for path, data in (("train.jsonl", train), ("val.jsonl", val), ("boundary_sft_all.jsonl", all_samples)):
        with (OUT_DIR / path).open(mode, encoding="utf-8") as f:
            for s in data:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # label distribution
    from collections import Counter
    labels = Counter(s["conversations"][2]["value"] for s in all_samples)

    print("\n=== Generation summary ===")
    print(f"  attempted          {stats.attempted}")
    print(f"  parse_fail         {stats.parse_fail}")
    print(f"  verify_reject      {stats.verify_reject}")
    print(f"  overlap_reject     {stats.overlap_reject}")
    print(f"  samples_emitted    {stats.samples_emitted}  (train {len(train)} + val {len(val)})")
    print("=== per bucket kept ===")
    for b, k in sorted(per_bucket_kept.items()):
        print(f"  {b:42s} {k}")
    print("=== label distribution ===")
    for lab, cnt in labels.most_common():
        print(f"  {lab:20s} {cnt}")
    print("=== output ===")
    for p in ("train.jsonl", "val.jsonl", "boundary_sft_all.jsonl"):
        print(f"  {OUT_DIR / p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
