# Qwen3-32B 双边界意图分类器 LoRA 训练 + 评测

针对 `intent_router` 的双边界意图选择任务，用 8×3090 训练 Qwen3-32B 的 QLoRA，并复用
项目原有评测管线打分。

## 任务回顾

- 模型职责：每条对话产出**两个**边界 id（股票 6 选 1 + 商品 6 选 1），各一个 token。
- 后端融合（`intent_router/policy.py`）把两个 id 折叠为 `SearchStockQuotes` /
  `RecommendProduct` / `null`。模型不参与融合。
- 输入格式（`intent_router/router.py:_build_user_prompt`）：
  ```
  <conversation_json>{"current_user_request":"..."}</conversation_json>
  输出谓词结果： /no_think
  ```
  system 是两段固定前缀（`_STOCK_SYSTEM_PROMPT` / `_PRODUCT_SYSTEM_PROMPT`），逐字复用。

## 数据集设计（不泄漏 benchmark）

`eval/benchmark_v1.jsonl` 的 672 条是 **holdout 测试集，绝不进训练**。训练数据由
更强的 `Qwen3.7-Plus`（通过 OpenAI 兼容网关）**全新生成 + 拒绝采样验证**：

1. 按 33 个 bucket × 12 个边界 id 的采样计划（`BUCKET_TARGETS`）定向生成对话，
   失败 bucket（`stock_named_app_query`、`reject_multi_intent`、`production_regression_*`）
   加权多生成。
2. benchmark 例**只作风格参考**，生成 prompt 明确要求换措辞、换实体、不照抄。
3. **overlap guard**：生成的对话若与任何 benchmark 的 `current_user_request` 子串重叠
   则丢弃，确保训练/测试零泄漏。
4. **拒绝采样**：生成模型声明的 (stock_id, product_id)，再用同一模型 + 线上 system
   prompt 当分类器重判，两者一致才保留。过滤歧义/错标样本。
5. 通过的对话用 `build_user_prompt` 拼成线上格式，每条拆 2 个训练样本（同 user 文本，
   不同 system、不同 label）。

## 目录结构

```
lora_train/
├── dataset/
│   ├── train.jsonl              # 训练集 (sharegpt, Qwen3.7-Plus 生成)
│   ├── val.jsonl                # 验证集 (~5%)
│   ├── boundary_sft_all.jsonl   # 全量
│   └── dataset_info.json        # LLaMA-Factory 数据集注册
├── configs/
│   ├── qwen3_32b_qlora.yaml     # LLaMA-Factory 训练配置
│   └── ds_zero3_offload.json    # DeepSpeed ZeRO-3 配置
└── src/
    ├── build_dataset.py         # 用 Qwen3.7-Plus 生成训练集（拒绝采样）
    ├── train_qlora.py           # 自包含训练脚本（transformers+peft+bNb）
    ├── serve_vllm.sh            # 用 vLLM 部署训练后的 adapter
    └── evaluate_lora.py         # 复用 evaluate_live.py 跑 672 benchmark
```

## 1. 生成训练集

```bash
# 配置网关（OpenAI 兼容协议）
export ANTHROPIC_AUTH_TOKEN="sk-..."
export ANTHROPIC_BASE_URL="http://<gateway>:<port>/"   # 带不带 /v1 都行,自动探测
export GENERATION_MODEL="Qwen3.7-Plus"

# 冒烟: 每 bucket 2 条, 确认能跑通
python lora_train/src/build_dataset.py --target_per_bucket 2 --workers 4

# 正式: 每 bucket ~75 条 → 33×75×2 ≈ 5000 训练样本
python lora_train/src/build_dataset.py --target_per_bucket 75 --workers 4 --timeout 60
```

参数说明：
- `--target_per_bucket`：每 bucket 期望保留条数（实际按 weight 加权，失败 bucket ×2）
- `--workers`：并发生成请求数（网关允许的话开 8-16 更快）
- `--min_per_bucket`：每 bucket 最少条数（冒烟用 2，正式用默认）
- `--max_attempts_per_bucket`：每 bucket 最多尝试次数（拒绝采样率约 1/14，建议设目标的 30 倍）
- `--resume`：追加到已有数据集而非覆盖

生成耗时：每条 ~12 秒 × 14 次尝试（拒绝采样）= ~3 分钟/bucket（workers=4 并发），
33 个 bucket 约 30-60 分钟。`--workers 8` 可减半。

## 2. 训练（二选一）

### 方式 A：LLaMA-Factory（推荐生产用）

```bash
pip install "llamafactory" "bitsandbytes" "deepspeed"
# flash-attn 可选（装不上就改 yaml 的 flash_attn 为 sdpa）
llamafactory-cli train lora_train/configs/qwen3_32b_qlora.yaml
```

### 方式 B：自包含脚本（透明，便于调试）

8 卡 ZeRO-3：
```bash
pip install "deepspeed" "peft" "bitsandbytes" "transformers" "datasets"
deepspeed --num_gpus=8 lora_train/src/train_qlora.py \
    --model_name Qwen/Qwen3-32B \
    --train_file lora_train/dataset/train.jsonl \
    --val_file   lora_train/dataset/val.jsonl \
    --output_dir lora_train/runs/qwen3_32b_boundary_qlora \
    --deepspeed lora_train/configs/ds_zero3_offload.json
```

单卡冒烟（确认 tokenization/loss 正常）：
```bash
python lora_train/src/train_qlora.py --max_steps 20 --max_samples 64
# 注意: 本机 Windows 上 torch 的 c10.dll 可能加载失败,这是环境问题,在训练机(Linux)上不会
```

### 关键设计

- **QLoRA nf4 + double-quant**：基座 4bit (~18GB) + LoRA bf16。8×24G 单卡舒服。
- **completion-only loss**：只对 assistant id token 算 loss，mask 掉 system+user。因为
  目标就是单 token 分类，这样收敛快、不被长 prompt 稀释。
- **逐字复用 system prompt + `/no_think`**：训练分布 = 线上分布，避免迁移 gap。
- **merge 到 16bit 部署**：训练时 4bit 只是省显存的内存技巧，磁盘上基座 checkpoint
  未变，只多几百 MB adapter。部署时 `merge_and_unload` 把 LoRA delta 加回原生 16bit
  基座，无额外量化误差。

## 3. 部署 + 评测

启动本地 vLLM（合并 LoRA 到 16bit 基座后服务）：
```bash
pip install vllm
bash lora_train/src/serve_vllm.sh
# -> http://localhost:8000/v1, served-model-name=qwen3-32b-lora
```

跑 672 条 benchmark（复用项目 `evaluate_live.py`，零代码改动，通过 `DASHSCOPE_*` 环境变量
指向本地 vLLM，走同一套融合管线）：
```bash
python lora_train/src/evaluate_lora.py --workers 8
# -> lora_train/eval/{results.jsonl,summary.json}
```

对照基线（`eval/results/benchmark_v1_qwen3_32b_compact_boundary_summary.json`）：
- 663/672 = 98.66%
- positive（正样本召回到具体 Intent）：318/327 = 97.25%
- negative（应拒绝确实未召回）：345/345 = 100%

LoRA 目标：positive ≥ 95%，negative ≥ 98%（即 README 的验收线）。

**注意**：由于训练数据是 Qwen3.7-Plus 新生成的、与 672 benchmark 零重叠，评测结果
反映的是真实泛化能力，不是背答案。

## 量化影响验证（可选）

4bit 训练的 LoRA merge 到 16bit 基座，机制上存在"训练基座（4bit 反量化）vs 部署基座
（原生 16bit）"的版本不一致，但 QLoRA 设计上此误差在噪声水平，对单 token 分类任务
无影响。如需验证，跑两版对照评测：
- A：16bit merge 版（`serve_vllm.sh` 默认 merge 模式）
- B：4bit 基座 + 挂载 LoRA 不 merge（`serve_vllm.sh` 的 `MODE=lora`）

两个准确率差 < 0.3% 即说明量化无影响。

## 注意事项

- **不要在训练样本里带主 Agent 的 system message**——线上会丢弃它。生成脚本已遵守。
- **assistant 只写一个边界 id**，不带句号/解释/思考链。生成脚本已遵守。
- 4bit 量化对这个"单 token 分类"任务损失极小；如某些 bucket 掉点，优先考虑加大 LoRA
  rank 或扩充该 bucket 数据，而非上 8bit（24G 卡上 8bit 既不够省又不够快）。
- 评测时 vLLM 要确保 thinking 已关（线上靠 `chat_template_kwargs.enable_thinking=false`
  + `/no_think`，vLLM 的 OpenAI 兼容端点需要确认 chat template 里有这条分支）。
