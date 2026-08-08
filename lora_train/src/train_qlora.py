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
import os
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
from torch.nn import CrossEntropyLoss


class BoundaryTrainer(Trainer):
    """Trainer that computes loss ONLY on the assistant label positions, and
    bypasses accelerate's @convert_to_fp32 wrapper that OOMs on big-vocab
    models.

    Problem: accelerate wraps PeftModel.forward with convert_to_fp32, which
    casts the FULL logits tensor [B, T, V=151552] to fp32 the instant
    forward returns — BEFORE our slicing can cut it down. On a 24GB card with
    a 32B 4bit base already ~20GB, that ~300MB fp32 cast peaks over the top.

    Fix: call the underlying model forward directly (skipping the accelerate
    wrapper), keep logits in bf16, slice to the 1-2 assistant positions, then
    cast only that tiny slice to fp32 for the cross-entropy.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = CrossEntropyLoss(ignore_index=-100)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        # Call the base model forward directly, bypassing accelerate's
        # convert_to_fp32 wrapper that casts the full logits to fp32.
        # model is a PeftModel; model.base_model.model is the underlying Qwen3.
        base = model.base_model.model if hasattr(model, "base_model") else model
        outputs = base(**inputs)
        logits = outputs.logits  # [B, T, V], bf16, NOT auto-cast to fp32

        # Shift for causal LM: predict token t from position t-1.
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Find positions that actually carry loss (label != -100).
        # For our task this is typically 1-2 tokens per sample.
        valid = (shift_labels != -100).any(dim=0)  # [T]
        keep = valid.nonzero(as_tuple=False).squeeze(-1)  # [K], K = 1..3
        if keep.numel() == 0:
            loss = shift_logits.sum() * 0.0
        else:
            shift_logits = shift_logits[:, keep, :].contiguous()  # [B, K, V]
            shift_labels = shift_labels[:, keep].contiguous()      # [B, K]
            # fp32 cast only on [B, K, V] with K~1-2 -> ~1MB, not ~300MB.
            loss = self.loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)).to(torch.float32),
                shift_labels.view(-1).to(torch.long),
            )
        return (loss, outputs) if return_outputs else loss

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
    p.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="seq len cap; boundary samples tokenize to ~330-380 tokens (system ~270 + user ~80 + assistant ~3 + special). 512 covers them with margin; custom compute_loss keeps the fp32-logits OOM away regardless of length.",
    )
    p.add_argument("--max_samples", type=int, default=None, help="cap dataset size for smoke tests")
    p.add_argument("--per_device_batch", type=int, default=1)
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
    p.add_argument(
        "--load_in_4bit",
        action="store_true",
        default=False,
        help="QLoRA 4bit base. Default False = bf16 full-precision base sharded "
             "across GPUs via ZeRO-3 (recommended for 4x3090, avoids the 4bit "
             "dequant OOM peak in backward). Set True only for single-GPU smoke "
             "on smaller models.",
    )
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument(
        "--attn_impl",
        choices=("flash_attention_2", "sdpa", "eager"),
        default="flash_attention_2",
        help="attention impl; falls back to sdpa if flash-attn not installed",
    )
    # deepspeed/torchrun launcher passes --local_rank (or --local-rank) to
    # every spawned process; argparse must accept it or the process exits
    # with "unrecognized arguments". The value is read by transformers/HF
    # internally via env (LOCAL_RANK), we just need to not choke on it.
    p.add_argument("--local_rank", type=int, default=-1)
    p.add_argument("--local-rank", dest="local_rank_dash", type=int, default=-1)
    args = p.parse_args()
    # argparse turns `--deepspeed None` into the STRING "None", not Python None.
    # transformers would then try to load a config file named "None" and fail.
    # Normalize: "none"/"None"/"" -> None.
    if isinstance(args.deepspeed, str) and args.deepspeed.strip().lower() in {"none", ""}:
        args.deepspeed = None
    return args


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
    non-ignored label (completion-only loss).

    Qwen3's chat template does NOT support return_assistant_tokens_mask (it
    lacks the `{% generation %}` keyword), so we use the prompt-prefix slice
    method WITH a strict prefix check: tokenize the full conversation and the
    prompt-only (add_generation_prompt=True), then assert the prompt ids are
    an exact prefix of the full ids. If the check fails (rare boundary
    quirk), we drop the sample (emit all -100 labels) rather than risk
    silently mis-aligned labels.
    """
    def to_id_list(out: Any) -> list[int]:
        """Normalize apply_chat_template output to a plain list[int].

        Some transformers versions return a tokenizers.Encoding object (with
        .ids, .type_ids, ...) instead of a list[int]. Slicing/len/== on an
        Encoding behave differently from a list and silently break the prefix
        check. Always coerce to list[int].
        """
        if isinstance(out, list):
            return [int(x) for x in out]
        # tokenizers.Encoding -> has .ids
        ids = getattr(out, "ids", None)
        if ids is not None:
            return [int(x) for x in ids]
        # dict-like (return_dict=True) -> "input_ids"
        if isinstance(out, dict) and "input_ids" in out:
            return [int(x) for x in out["input_ids"]]
        # last resort: try iteration
        try:
            return [int(x) for x in out]
        except Exception:
            raise RuntimeError(f"无法从 apply_chat_template 输出提取 token ids: {type(out)}")

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

        full_ids = to_id_list(tokenizer.apply_chat_template(
            messages_full,
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=False,
            return_dict=False,
        ))
        prompt_ids = to_id_list(tokenizer.apply_chat_template(
            messages_prompt,
            tokenize=True,
            add_generation_prompt=True,   # ends right before assistant turn
            enable_thinking=False,
            return_dict=False,
        ))

        # STRICT prefix check: prompt_ids must be an exact prefix of full_ids.
        # If not, the chat template has a boundary quirk and a hand-rolled
        # slice would silently mis-align labels. Drop the sample instead.
        n_prompt = len(prompt_ids)
        if n_prompt >= len(full_ids) or full_ids[:n_prompt] != prompt_ids:
            print(
                f"[warn] prompt is not an exact prefix of full; dropping sample "
                f"(prompt={n_prompt}, full={len(full_ids)})",
                flush=True,
            )
            input_ids = full_ids[-max_length:]
            return {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": [-100] * len(input_ids),
            }

        # completion-only loss: mask prompt, keep assistant tail.
        labels = [-100] * n_prompt + full_ids[n_prompt:]
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

    # DeepSpeed ZeRO-3: initialize the process group BEFORE loading the model,
    # so from_pretrained can run inside deepspeed.zero.Init() (see the model
    # block below) and partition each parameter across ranks at creation time.
    # NCCL requires torch.cuda.set_device first; the deepspeed launcher sets
    # LOCAL_RANK in the environment.
    if args.deepspeed:
        import torch.distributed as dist

        local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")

    # ---- tokenizer -------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- model (QLoRA: nf4 base, bf16 compute) ---------------------------
    # Resolve attention impl: flash_attention_2 if available, else sdpa.
    # flash-attn install is fiddly (needs matching CUDA/torch); sdpa is
    # built into torch and is plenty fast for the short sequences here.
    attn_impl = args.attn_impl
    if attn_impl == "flash_attention_2":
        try:
            import flash_attn  # noqa: F401
        except Exception:
            print("[warn] flash-attn not installed; falling back to sdpa", flush=True)
            attn_impl = "sdpa"
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "dtype": torch.bfloat16,
        "attn_implementation": attn_impl,
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
    # ZeRO-3: load under deepspeed.zero.Init() so each parameter is partitioned
    # across ranks AT CREATION TIME. The two naive paths both die in DS 0.19.4's
    # _configure_distributed_model, which calls module.to(device) on whatever
    # we hand it:
    #   - full bf16 base loaded to CPU per rank -> .to() fills the 24GB GPU
    #     and OOMs (64GB model, 24GB card).
    #   - init_empty_weights (meta) -> .to() hits "Cannot copy out of meta
    #     tensor; no data".
    # zero.Init patches parameter creation so each rank only ever holds its
    # 1/world_size partition (~11GB for 32B across 6 GPUs); the later .to() is
    # then a no-op on already-placed shards. Requires the process group, which
    # we initialized at the top of main(). low_cpu_mem_usage=True makes
    # from_pretrained stream tensor-by-tensor from disk (meta init), so each
    # rank's CPU peak is ~one tensor, not the full 64GB state_dict; without it
    # every rank buffers the whole 64GB on CPU before partitioning (6x = 384GB
    # transient). OS page cache dedupes the 6 disk reads to ~64GB of real I/O.
    if args.deepspeed:
        from deepspeed import zero

        # zero.Init parses the DS config immediately, before HF Trainer has a
        # chance to fill the "auto" placeholders. Resolve them now from the
        # actual CLI args + world size so the batch assertion passes.
        # zero.Init's config_dict_or_path accepts a dict OR a file path
        # (verified against DS 0.19.4 signature); pass the resolved dict.
        world_size = (
            torch.distributed.get_world_size()
            if torch.distributed.is_initialized()
            else 1
        )
        ds_config_dict = json.loads(
            Path(args.deepspeed).read_text(encoding="utf-8")
        )
        micro = args.per_device_batch
        accum = args.grad_accum
        ds_config_dict["train_micro_batch_size_per_gpu"] = micro
        ds_config_dict["gradient_accumulation_steps"] = accum
        ds_config_dict["train_batch_size"] = micro * accum * world_size

        model_kwargs["low_cpu_mem_usage"] = True
        with zero.Init(config_dict_or_path=ds_config_dict):
            model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    else:
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
    # NOTE: do NOT call prepare_model_for_kbit_training() here. On 24GB cards
    # it casts LayerNorm weights/inputs to fp32, which pushes a 32B 4bit base
    # (already ~20GB) over the edge and OOMs. Modern peft (>=0.11) handles
    # the kbit prep automatically inside get_peft_model for QLoRA nf4 + bf16
    # compute, which is sufficient for RMSNorm stability on Qwen3.
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ---- data ------------------------------------------------------------
    train_rows = load_sharegpt(Path(args.train_file), args.max_samples)
    val_rows = load_sharegpt(Path(args.val_file), args.max_samples)
    tokenize = build_tokenize_fn(tokenizer, args.max_length)
    # num_proc=1 + load_from_cache_file=False: the tokenize fn returns plain
    # lists which Arrow's multiprocess writer can choke on (OverflowError);
    # single-process avoids that and keeps cache off for smoke runs.
    train_ds = Dataset.from_list(train_rows).map(
        tokenize, remove_columns=["conversations"], num_proc=1, load_from_cache_file=False
    )
    val_ds = Dataset.from_list(val_rows).map(
        tokenize, remove_columns=["conversations"], num_proc=1, load_from_cache_file=False
    )

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
    # bf16 base + ZeRO-3: use the standard Trainer (default loss). ZeRO-3
    #   must hook the model forward to all-gather sharded params; our custom
    #   compute_loss bypasses accelerate via model.base_model.model, which
    #   would skip that gather and break under ZeRO-3. bf16 logits are ~155MB
    #   so the default loss's fp32 cast (~310MB) fits with sharded params.
    # 4bit QLoRA single-GPU: use BoundaryTrainer to skip the convert_to_fp32
    #   peak that OOMs a 24GB card holding the full 4bit base.
    if args.load_in_4bit:
        trainer = BoundaryTrainer(
            model=model, args=targs, train_dataset=train_ds,
            eval_dataset=val_ds, data_collator=collator,
        )
    else:
        trainer = Trainer(
            model=model, args=targs, train_dataset=train_ds,
            eval_dataset=val_ds, data_collator=collator,
        )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Training complete. Adapter saved to", args.output_dir)


if __name__ == "__main__":
    main()
