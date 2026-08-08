"""Smoke-check the tokenization BEFORE running training.

Run on the training machine (where the model/tokenizer actually lives):

    python lora_train/src/check_tokenize.py --model_name ../../models/Qwen3_32B/

It prints, for a few training samples:
  - total token count (so you can confirm max_length is enough)
  - whether prompt_ids is an exact prefix of full_ids (the completion-only
    correctness check)
  - which token positions carry loss (should be only 1-3 assistant tokens)
  - the decoded assistant tokens (should be the label id like "StockRoute")

If anything looks wrong here, don't bother running train_qlora.py — fix this
first. Training will waste time/GPU otherwise.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer

from lora_train.src.train_qlora import build_tokenize_fn, VALID_LABELS  # type: ignore


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="Qwen/Qwen3-32B")
    p.add_argument("--train_file", default="lora_train/dataset/train.jsonl")
    p.add_argument("--n", type=int, default=3, help="how many samples to inspect")
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenize = build_tokenize_fn(tokenizer, max_length=512)

    import json

    rows = [json.loads(line) for line in Path(args.train_file).read_text(encoding="utf-8").splitlines() if line.strip()][: args.n]

    for i, row in enumerate(rows):
        convs = row["conversations"]
        label = convs[2]["value"]
        print(f"\n===== sample {i} (label={label}) =====")
        print(f"system 长度(字): {len(convs[0]['value'])}")
        print(f"user 长度(字):   {len(convs[1]['value'])}")

        # Reproduce the two tokenizations inside build_tokenize_fn
        messages_full = [
            {"role": "system", "content": convs[0]["value"]},
            {"role": "user", "content": convs[1]["value"]},
            {"role": "assistant", "content": label},
        ]
        messages_prompt = messages_full[:-1]
        full = tokenizer.apply_chat_template(messages_full, tokenize=True, add_generation_prompt=False, enable_thinking=False, return_dict=False)
        prompt = tokenizer.apply_chat_template(messages_prompt, tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False)
        full_ids = list(full.input_ids) if hasattr(full, "input_ids") else list(full)
        prompt_ids = list(prompt.input_ids) if hasattr(prompt, "input_ids") else list(prompt)

        print(f"full token 数:    {len(full_ids)}")
        print(f"prompt token 数:  {len(prompt_ids)}")
        is_prefix = full_ids[: len(prompt_ids)] == prompt_ids
        print(f"prompt 是 full 精确前缀: {is_prefix}")
        if not is_prefix:
            print("  ⚠️ 前缀不对齐! completion-only mask 会错位, 别跑训练")
            # show where they diverge
            for j in range(min(len(prompt_ids), len(full_ids))):
                if prompt_ids[j] != full_ids[j]:
                    print(f"  首个分歧在位置 {j}: prompt={prompt_ids[j]} full={full_ids[j]}")
                    break
        else:
            # assistant tokens = full_ids[len(prompt):]
            asst = full_ids[len(prompt_ids):]
            decoded = tokenizer.decode(asst)
            print(f"assistant token 数: {len(asst)}")
            print(f"assistant 解码: {decoded!r}  (应≈ {label})")
            n_loss = len([t for t in asst if True])  # all assistant tokens carry loss
            print(f"算 loss 的 token 数: ~{n_loss} (训练时 shift 后会少1)")

        # also run the actual tokenize fn to see labels
        out = tokenize(row)
        labels = out["labels"]
        n_valid = sum(1 for x in labels if x != -100)
        print(f"build_tokenize_fn 输出: input_ids 长度={len(out['input_ids'])}, 有效 label 数={n_valid}")
        if n_valid == 0:
            print("  ⚠️ 有效 label=0, 训练 loss 会恒0, 别跑!")

    print("\n如果上面都正常 (前缀对齐 + 有效 label 1-3 + 解码≈label), 就可以跑训练了。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
