#!/usr/bin/env bash
# Serve the QLoRA adapter on a local OpenAI-compatible endpoint that the
# existing IntentRouter / evaluate_live.py can hit unchanged.
#
# Two modes:
#   (a) Merge LoRA into the base once, serve the merged model.  Simpler, faster.
#   (b) Hot-swap LoRA via vllm --enable-lora.  Multiple adapters at once.
#
# This script uses (a) by default because the router sends a fixed system+
# user prompt and we only have one adapter. Use --lora for (b).
#
# Requirements:
#   pip install vllm bitsandbytes
#   (optional) flash-attn2 for speed
#
# After it boots, point the evaluator at http://localhost:8000/v1 :
#   python scripts/evaluate_live.py \
#       --model qwen3-32b-lora \
#       --base-url http://localhost:8000/v1 \
#       --api-key-env ROUTER_LLM_API_KEY \
#       --cases eval/benchmark_v1.jsonl \
#       --workers 8 \
#       --output lora_train/eval/results.jsonl \
#       --summary-output lora_train/eval/summary.json \
#       --min-positive-accuracy 0.95 \
#       --min-negative-accuracy 0.98
set -euo pipefail

# Default to the local model dir (matches train_qlora.py's --model_name default
# path style). Override with MODEL=... if you use a different base.
MODEL="${MODEL:-../../models/Qwen3_32B/}"
ADAPTER_DIR="${ADAPTER_DIR:-lora_train/runs/qwen3_32b_boundary_qlora}"
MERGED_DIR="${MERGED_DIR:-lora_train/runs/qwen3_32b_boundary_merged}"
PORT="${PORT:-8000}"
MODE="${MODE:-merge}"   # merge | lora
# vLLM tensor-parallel size: 32B bf16 needs ~64GB VRAM -> 3x3090 minimum.
# Defaults to all visible GPUs; override with TP_SIZE=N.
TP_SIZE="${TP_SIZE:-$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1)}"

if [[ "$MODE" == "merge" ]]; then
    if [[ ! -d "$MERGED_DIR" ]]; then
        echo ">> merging LoRA adapter $ADAPTER_DIR into $MERGED_DIR"
        python - <<PY
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, shutil
# device_map="cpu": merge loads the full 64GB bf16 base into CPU RAM (the host
# has 469GB), NOT GPU. Naive load (default device_map) tries to put the model
# on a single 24GB GPU and OOMs before merge can run. Merge itself is a pure
# CPU matmul (LoRA A@B added to base weight), no GPU needed.
base = AutoModelForCausalLM.from_pretrained("$MODEL", torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="cpu")
tok = AutoTokenizer.from_pretrained("$MODEL", trust_remote_code=True)
model = PeftModel.from_pretrained(base, "$ADAPTER_DIR")
model = model.merge_and_unload()
model.save_pretrained("$MERGED_DIR", safe_serialization=True, max_shard_size="4GB")
tok.save_pretrained("$MERGED_DIR")
print("merged to", "$MERGED_DIR")
PY
    else
        echo ">> merged model exists, skipping merge: $MERGED_DIR"
    fi
    SERVE_PATH="$MERGED_DIR"
    vllm serve "$SERVE_PATH" \
        --host 0.0.0.0 --port "$PORT" \
        --served-model-name qwen3-32b-lora \
        --dtype bfloat16 \
        --max-model-len 2048 \
        --gpu-memory-utilization 0.90 \
        --tensor-parallel-size "$TP_SIZE" \
        --enforce-eager
else
    # Hot-swap mode: keep base in nf4-ish fp16 and load adapter via --enable-lora.
    # Note: vllm LoRA currently expects fp16/bf16 base; for a true 4bit serving
    # base use merge mode above.
    vllm serve "$MODEL" \
        --host 0.0.0.0 --port "$PORT" \
        --served-model-name qwen3-32b \
        --dtype bfloat16 \
        --max-model-len 2048 \
        --gpu-memory-utilization 0.90 \
        --tensor-parallel-size "$TP_SIZE" \
        --enable-lora \
        --lora-modules "boundary=$ADAPTER_DIR" \
        --enforce-eager
    # In lora mode, pass --model boundary when calling evaluate_live.py so vllm
    # routes to the adapter named "boundary".
fi
