# PromptGen 首版意图选择器

这是一个面向 Qwen3-32B 的双边界意图选择器。每次请求分别判断股票信息边界和普通零售
商品发现边界，再由后端做保守融合；对外只可能返回两个真实 Intent Label：

- `SearchStockQuotes`：股票行情、K线、价格、走势及股市事实；明确排除预测、荐股和投资建议。
- `RecommendProduct`：普通电商商品的检索、选购与推荐。

其他请求不会被硬塞给最相似的后端 Intent。两个模型调用各自在 6 个简短候选中选择一个，
后端再把虚拟边界和冲突统一折叠为 `intent_label: null`。模型看不到后端 Label、真实/虚拟
映射或 Agent 编排信息。

| 融合后的候选名 | 后端 Intent Label | 含义 |
|---|---|---|
| `StockInfo` | `SearchStockQuotes` | 股票行情与股市事实 |
| `Ecommerce` | `RecommendProduct` | 电商商品检索、选购、购买链接与推荐 |
| `StockAdvice` | `null` | 预测、荐股或投资决策 |
| `StockResearch` | `null` | 公司龙头、赛道、持股与业绩/股价关系等深度研究 |
| `StockOther` | `null` | 其它股票相关目标 |
| `FinanceOther` | `null` | 非股票金融产品自身 |
| `ProductInfo` | `null` | 商品参数、价格、适用性与已有设备兼容建议 |
| `ProductOther` | `null` | 已有商品/订单操作以及商品使用与故障 |
| `NonRetail` | `null` | 整车、房产、保险、旅行及专业服务 |
| `MultiProduct` | `null` | 分别发现多个无关商品 |
| `ChitChat` | `null` | 纯闲聊 |
| `NoRequest` | `null` | 纯名称、陈述、残句或无法恢复的指代 |
| `NoAvailable` | `null` | 其它任务、信息不足或多意图 |

模型侧拆成两个互不影响的闭集候选表：

```text
Stock boundary
├── StockRoute
├── StockResearch
├── StockAdvice
├── StockOperation
├── OtherFinance
└── NoStock

Product boundary
├── ProductRoute
├── ProductKnowledge
├── ProductFollowup
├── NonRetail
├── MultiProduct
└── NoProduct
```

## 快速运行

环境要求为 Python 3.10+ 与 `openai` 2.x：

```bash
python -m pip install -e .
python -m intent_router "贵州茅台现在多少钱？"
python -m intent_router "推荐一辆20万的SUV"
```

默认读取 `~/bailian_api.txt`。支持以下两种格式：

```text
https://<workspace-id>.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
sk-xxxxxxxx
```

或：

```text
DASHSCOPE_BASE_URL=https://<workspace-id>.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
DASHSCOPE_API_KEY=sk-xxxxxxxx
```

也可以通过同名环境变量提供配置。密钥不会进入提示词、结果或错误信息。

多轮输入使用 JSON 文件：

```json
[
  {"role": "user", "content": "预算五千"},
  {"role": "assistant", "content": "想买什么？"},
  {"role": "user", "content": "笔记本，主要写代码"}
]
```

```bash
python -m intent_router --history-file conversation.json
```

代码调用：

```python
from intent_router import IntentRouter

router = IntentRouter()  # 应在服务进程内复用，以复用 HTTP 连接
decision = router.route([
    {"role": "user", "content": "预算五千"},
    {"role": "assistant", "content": "想买什么？"},
    {"role": "user", "content": "笔记本，主要写代码"},
])

if decision.should_route:
    dispatch(decision.intent_label)
else:
    answer_with_main_agent()
```

如果实际下游 Label 不同，可以用 `dataclasses.replace` 只修改 `default_candidates()` 中两个
真实候选的 `intent_label`，再通过 `IntentRouter(candidates=...)` 注入。边界候选名是提示词协议，
不应改名；模型不依赖也看不到后端 Label 字面值。

## 算法设计

一次请求的处理如下：

1. 批处理入口和 `IntentRouter` 都忽略主 Agent 的 system message；最后一条有效消息必须来自
   用户。当前用户消息优先保留，全部内容受最近 16 条、12000 字符的总上限约束。
2. 对话被序列化为 `current_user_request`、可选的 `prior_user_turns` 和
   `assistant_tool_reference`。当前请求是唯一决策入口；历史用户轮只恢复明确承接，
   assistant/tool 只提供实体或事实。自包含请求不携带 assistant/tool；出现明确指代时才保留，
   每条最多 1200 字符并采用头尾截取。
3. 同一份对话数据分别送入两个固定前缀：股票边界 prompt 396 字符，商品边界 prompt
   488 字符。二者都不含 examples、品牌、型号、benchmark 原句、后端 Label 或分发信息。
4. 股票表区分公开信息、投资决策、公司研究、证券操作、其它金融和无股票请求；商品表区分
   商品发现、商品知识、履约、非零售对象、多商品和无商品请求。两个调用互不覆盖对方判断。
5. 后端先做闭集融合：仅股票正向时得到 `StockInfo`，仅商品正向时得到 `Ecommerce`；两者
   同时成立或都不成立时默认不召回。随后仅使用与能力契约有关的通用结构门控，例如未来预测、
   证券软件本身、非零售对象、多个无关商品、残句和纯名称；没有品牌/型号或完整 query 特判。
6. 对少量模型格式错误，只追加“仅输出一个 id”重试一次；仍不可解析或 API 失败则 fail-closed。
   `raw_model_output` 会保留两个边界的原始输出以及可能的重试输出。
7. Qwen3-32B 同时通过 `chat_template_kwargs.enable_thinking=false` 和 `/no_think` 关闭思考，
   `temperature=0`，不设置生成 token 上限。两个 system 前缀对所有请求保持稳定。
8. 默认关闭置信度门控且不请求 `logprobs`。需要对照时可显式设置
   `min_route_probability` 或 CLI 的 `--threshold`；生成概率不是业务校准置信度。
9. 最终只有 `StockInfo/Ecommerce` 进入私有后端映射，其余候选均为 `null`。`decision_reason`
   记录融合或门控原因，只用于调试，不是下游业务协议。

输出状态：

- `routed`：返回一个真实 `intent_label`，可以分发。
- `no_route`：模型有效选择虚拟候选，由主 Agent 回复。
- `low_confidence`：仅在显式设置阈值时出现；模型选了真实候选但未达到阈值，不分发。
- `invalid_model_output`：输出不是候选英文名，不分发并记录告警。
- `api_error`：模型调用失败，不分发并记录告警。

多个虚拟候选最终都折叠为相同的 `NO_ROUTE`；它们的 `selected_candidate_id` 和
`decision_reason` 只适合诊断和误差分析，不应成为下游业务契约。

扩展到千级 Intent 时，不应把全部候选塞入本提示词；建议先做向量/小模型 Top-K 召回，再把 Top-K 与若干动态负原型交给本选择器精排。

服务进程应复用同一个 `IntentRouter`。两个边界候选表及英文名称分别位于稳定的 system
前缀中；每次请求只改变独立的对话数据区。代码关闭 SDK 自动重试和 Qwen 思考模式，本版本
按用户要求不把双调用时延作为选择算法约束。

## 测试

```bash
python -m unittest discover -s tests -v
python scripts/evaluate_live.py
# 显式复现当前实验及双业务门槛：
python scripts/evaluate_live.py \
  --model qwen3-32b \
  --workers 2 \
  --min-positive-accuracy 0.95 \
  --min-negative-accuracy 0.98
# 固定错例 cohort，同时检查候选位置和同请求重复采样：
python scripts/evaluate_live.py \
  --case-id-file eval/cohorts/production_misclassifications_20260807_round3.txt \
  --repeats 3 \
  --min-positive-accuracy 0.95 \
  --min-negative-accuracy 0.98
```

`eval/cases.json` 是快速 smoke 集。严格 benchmark 位于
`eval/benchmark_v1.jsonl`，包含 560 条原始审校样例和两批共 112 条用户提供的生产回归样例，
共 672 条。第一批 94 条源数据保存在 `eval/production_regression_20260807.jsonl`；
第二批原始输入去掉批内重复及已经由第一批覆盖的 query 后，18 条增量保存在
`eval/production_regression_20260807_round2.jsonl`。标注边界见 `eval/LABELING.md`。
生成器会去重、生成稳定 id，并保证冻结文件完全一致：

```bash
python scripts/build_benchmark_v1.py --check
python scripts/evaluate_live.py \
  --cases eval/benchmark_v1.jsonl \
  --split dev \
  --workers 2 \
  --output artifacts/dev-results.jsonl \
  --summary-output artifacts/dev-summary.json \
  --min-system-accuracy 0.99

# 可按 bucket 定向复测；多次 --bucket 取并集
python scripts/evaluate_live.py --bucket reject_multi_intent --workers 1

# 限流后用同一模型/提示词版本的成功重试覆盖失败行，再严格校验完整性
python scripts/combine_evaluation_results.py \
  --input artifacts/dev-results.jsonl \
  --input artifacts/retry-results.jsonl \
  --output artifacts/dev-merged.jsonl \
  --summary-output artifacts/dev-merged-summary.json
```

严格指标直接比较最终系统输出 `SearchStockQuotes`、`RecommendProduct` 与 `null`，
因此股票误分到商品 Agent 不会被“都需要分发”掩盖。报告同时给出具体 Label 的
precision/recall、错误类型、混淆矩阵、分桶准确率、虚拟候选 exact match、运行错误
和 completion token 数。`route_metrics.positive` 表示所有应召回样本正确召回到具体
Intent 的比例；`route_metrics.negative` 表示所有应拒绝样本确实未召回的比例，API
错误在两者中都不计为正确。`--repeats` 用于观测服务端非确定性；候选顺序在当前双边界
prompt 中固定，不再把线上请求按 seed 换序。重复评测分片合并时必须传入相同的运行维度，
合并器会按 `(case id, seed, repeat)` 用后续成功结果覆盖运行错误。时延仅作观测。当前默认
关闭置信度门控；如使用
`--threshold 0.90` 做对照实验，仍需在真实流量标注集上按所需 precision/coverage
校准，因为生成概率不是天然校准后的业务置信度。

当前 Qwen3-32B 双边界提示词（组合 SHA-256
`1f0cf88e963f93f4a2da76276a3d22f2973ee9cef4da6defc8490d1d4aa78ab7`）在 672 条
benchmark 上为 663/672；正样本 318/327（97.25%），负样本 345/345（100%），无
operational error，均高于 95%/98% 验收线。两类召回 precision 都是 100%。机器可读结果位于
`eval/results/benchmark_v1_qwen3_32b_compact_boundary_{rows.jsonl,summary.json}`。

## JSONL 批处理

输入文件每行格式为：

```json
{"messages":[{"role":"system","content":"主Agent提示词"},{"role":"user","content":"查一下茅台股价"}]}
```

脚本会删除该行内所有 `role=system` 的消息，然后将剩余对话交给选择器：

```bash
python scripts/route_jsonl.py input.jsonl output.jsonl
# 可选并发；输出顺序仍与输入一致
python scripts/route_jsonl.py input.jsonl output.jsonl --workers 4
```

可以直接指定任意 OpenAI-compatible LLM API。推荐从命名环境变量读取密钥，避免密钥进入 shell 历史：

```bash
export ROUTER_LLM_API_KEY="sk-..."
python scripts/route_jsonl.py input.jsonl output.jsonl \
  --base-url "https://example.com/compatible-mode/v1" \
  --api-key-env ROUTER_LLM_API_KEY \
  --model qwen3-32b
```

也支持 `--api-base`（`--base-url` 的别名）和 `--api-key`。显式 API 参数优先于 `--config`；未指定时仍按 `DASHSCOPE_API_KEY`、`DASHSCOPE_BASE_URL` 和 `~/bailian_api.txt` 的既有规则加载。

每个物理输入行一定对应一个输出行。正常输出示例：

```json
{"line_number":1,"model":"qwen3-32b","prompt_sha256":"...","candidate_order_seed":0,"model_output":"StockInfo","raw_model_output":"{\"stock\":\"StockRoute\",\"product\":\"NoProduct\"}","system_output":"SearchStockQuotes","status":"routed","should_route":true,"selected_candidate_id":"stock_market_information","selected_probability":null,"decision_reason":"stock_boundary","completion_tokens":12,"error_type":null,"error_message":null}
```

`raw_model_output` 以 JSON 字符串保存股票和商品两个模型原始响应；发生格式重试时对应值会是
数组。`model_output` 是后端融合后的候选名，`system_output` 才是最终业务输出。
原始响应可能回显用户输入，应按对话数据的敏感级别保存和清理。
`prompt_sha256` 与 `candidate_order_seed` 用于核对不同调用端是否实际使用相同提示词版本
和部署版本；当前生产候选顺序固定，它们不会暴露提示词正文。

虚拟候选（包括 `ChitChat`）的 `system_output` 为 `null`，并通过 `status=no_route` 表明由主 Agent 处理。非法 JSON、仅含 system message 等单行错误也会输出 `input_error` 行，不会令后续输入错位。输入或输出使用 `-` 时分别表示 stdin/stdout。
