"""Self-contained Qwen3-32B QLoRA SFT trainer for the dual-boundary classifier.

This is a transparent, dependency-light alternative to LLaMA-Factory so you
can see exactly how tokenization, completion-only loss, and QLoRA fit together
for this single-id-output classification task. Use the LLaMA-Factory YAML
(configs/qwen3_32b_qlora.yaml) for production; use this for understanding and
debugging.

Launch with DeepSpeed ZeRO-3 across 8x3090:

    deepspeed --num_gpus=8 lora_train/src/train_qlora.py \\
        --model_name Qwen/Qwen3-32B \\
        --train_file lora_train/dataset/train.jsonl \\
        --val_file   lora_train/dataset/val.jsonl \\
        --output_dir lora_train/runs/qwen3_32b_boundary_qlora \\
        --deepspeed lora_train/configs/ds_zero3_offload.json

Single-GPU smoke test (drop the deepspeed launcher):

    python lora_train/src/train_qlora.py --max_steps 20 --max_samples 64

Key design choices, explained inline:
  - nf4 double-quant base, bf16 compute, LoRA in bf16  -> fits 24GB
  - mask user/system tokens, compute loss only on the assistant id token
  - /no_think is part of the user turn (matches on-line distribution)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)

# System prompts must match intent_router/router.py byte-for-byte.
ROOT = Path(__file__).resolve().parents[2]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from intent_router.router import _STOCK_SYSTEM_PROMPT, _PRODUCT_SYSTEM_PROMPT  # type: ignore

SYSTEM_PROMPTS = (_STOCK_SYSTEM_PROMPT, _PRODUCT_SYSTEM_PROMPT)
VALID_LABELS = {
    "StockRoute", "StockAdvice", "StockResearch", "StockOperation", "OtherFinance", "NoStock",
    "ProductRoute", "ProductKnowledge", "ProductFollowup", "NonRetail", "MultiProduct", "NoProduct",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="Qwen/Qwen3-32B")
    p.add_argument("--train_file", default="lora_train/dataset/train.jsonl")
    p.add_argument("--val_file", default="lora_train/dataset/val.jsonl")
    p.add_argument("--output_dir", default="lora_train/runs/qwen3_32b_boundary_qlora")
    p.add_argument("--deepspeed", default=None, help="path to ds config; set None to disable")
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--max_samples", type=int, default=None, help="cap dataset size for smoke tests")
    p.add_argument("--per_device_batch", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--lora_rank", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--eval_steps", type=int, default=100)
    p.add_argument("--logging_steps", type=int, default=5)
    p.add_argument("--load_in_4bit", action="store_true", default=True)
    p.add_argument("--bf16", action="store_true", default=True)
    return p.parse_args()


def load_sharegpt(path: Path, max_samples: int | None = None) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if max_samples:
        rows = rows[:max_samples]
    return rows


@dataclass
class TokenizerCache:
    tokenizer: Any


def build_tokenize_fn(tokenizer: Any, max_length: int):
    """Produce input_ids + labels where only the assistant id token has a
    non-ignored label. We achieve completion-only loss by:
      1. tokenizing system+user (the prompt) without the assistant,
      2. tokenizing the assistant id alone,
      3. concatenating, with labels = [-100]*len(prompt) + assistant_tokens.
    No chat template override needed: we still render via apply_chat_template
    so special tokens / /no_think placement match the on-line format.
    """
    def tokenize(example: dict) -> dict[str, torch.Tensor]:
        convs = example["conversations"]
        assert len(convs) == 3 and convs[0]["from"] == "system" and convs[1]["from"] == "user" and convs[2]["from"] == "assistant"
        label = convs[2]["value"].strip()
        assert label in VALID_LABELS, f"unexpected label: {label!r}"

        messages_full = [
            {"role": "system", "content": convs[0]["value"]},
            {"role": "user", "content": convs[1]["value"]},
            {"role": "assistant", "content": label},
        ]
        messages_prompt = messages_full[:-1]

        full_ids = tokenizer.apply_chat_template(
            messages_full,
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        prompt_ids = tokenizer.apply_chat_template(
            messages_prompt,
            tokenize=True,
            add_generation_prompt=True,   # ends right before assistant turn
            enable_thinking=False,
        )
        # Loss mask: ignore everything except the assistant response tokens.
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
        input_ids = full_ids

        # Truncate from the left if too long (keep the assistant tail).
        if len(input_ids) > max_length:
            input_ids = input_ids[-max_length:]
            labels = labels[-max_length:]

        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        }

    return tokenize


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    # ---- tokenizer -------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- model (QLoRA: nf4 base, bf16 compute) ---------------------------
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
        "attn_implementation": "flash_attention_2",
    }
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_storage=torch.uint8,
        )
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # ---- LoRA ------------------------------------------------------------
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    if args.load_in_4bit:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ---- data ------------------------------------------------------------
    train_rows = load_sharegpt(Path(args.train_file), args.max_samples)
    val_rows = load_sharegpt(Path(args.val_file), args.max_samples)
    tokenize = build_tokenize_fn(tokenizer, args.max_length)
    train_ds = Dataset.from_list(train_rows).map(tokenize, remove_columns=["conversations"])
    val_ds = Dataset.from_list(val_rows).map(tokenize, remove_columns=["conversations"])

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=None,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt",
    )

    # ---- training args ---------------------------------------------------
    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_batch,
        per_device_eval_batch_size=args.per_device_batch * 2,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.0,
        adam_beta1=0.9,
        adam_beta2=0.95,
        max_grad_norm=1.0,
        bf16=args.bf16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="tensorboard",
        seed=args.seed,
        ddp_find_unused_parameters=False,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        deepspeed=args.deepspeed,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Training complete. Adapter saved to", args.output_dir)


if __name__ == "__main__":
    main()
