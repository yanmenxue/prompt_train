# Intent Router 优化实验记录

本文记录使用真实 Qwen 接口完成的基线与优化实验。最终验收以冻结的
大规模数据集及最终系统输出三分类准确率为准：`SearchStockQuotes`、
`RecommendProduct`、`null`。虚拟候选之间的差异仅作为诊断指标，不能替代最终
系统输出准确率。

## 实验约定

- 模型：E001–E005 使用 `qwen3-32b`；E006 起使用各节明确记录的模型
- 接口：OpenAI-compatible `/v1/chat/completions`
- 解码：`temperature=0`、`enable_thinking=false`、不设置输出 token 上限
- 在线请求的模型原始输出和后端映射均参与评测
- `system accuracy`：最终系统输出与标注完全一致；股票误分到商品也计为错误
- `candidate exact`：模型可见候选 id 的细粒度诊断指标
- `unsafe false route`：标注为 `null` 但实际返回真实 Intent Label
- `false rejection`：标注为真实 Intent Label 但实际返回 `null`
- `wrong route`：两个真实 Intent Label 之间互相错分
- `operational error`：API 错误或模型输出无法解析

## E001：补齐泛化购买表达的电商边界

日期：2026-08-06

问题：用户尚未选定具体商品或形成订单时，用“给我买一个”“给我购买链接”等
动作表达商品检索需求，模型容易将其理解为下单操作并选择 `ProductOther`。

优化：

1. 在 `Ecommerce` 描述中明确纳入尚无已选商品/已有订单的普通购买表达。
2. 增加真实失败表达作为正例。
3. 将 `ProductOther` 的下单、支付、退款、物流和售后边界限定为已展示/选定商品
   或已有订单。
4. 强化多目标请求优先选择 `NoAvailable` 及只能输出单一 id 的约束，避免模型
   同时逐行输出两个候选。

评测集：30 条 smoke case × 3 种候选排列，共 90 次在线调用。该集合仅用于快速
回归，不作为最终验收集。

| 版本 | system accuracy | candidate exact | operational errors | completion tokens |
|---|---:|---:|---:|---|
| 提交 `ab5e807` | 85/90（94.44%） | 未作为对照指标 | 0 | 2–3 |
| E001 | 90/90（100.00%） | 88/90（97.78%） | 0 | 2–3 |

提交前版本的 5 个系统输出错误全部来自两个新增回归 case：

- “给我一个s后即可的购买链接”：3/3 被错误拒绝。
- “给我买个鼠标青春版”：2/3 被错误拒绝。

E001 中两个 case 在 3 种候选排列下均映射为 `RecommendProduct`（6/6）。
`candidate exact` 的 2 个差异发生在 `ProductOther` 与 `NoAvailable` 两个虚拟
候选之间，最终系统输出均为 `null`。

## B001：冻结 benchmark 上的初始基线

日期：2026-08-06

数据：`eval/benchmark_v1.jsonl`，SHA-256
`3e0508469f4ff0593835efe936ea578c94352131cb6cdc7360eb4815085e6691`。本阶段只运行
420 条 dev，140 条 test 不参与 prompt 决策。使用生产默认候选顺序，并发数 8。

| 指标 | 结果 |
|---|---:|
| system accuracy | 388/420（92.38%） |
| candidate exact | 375/420（89.29%） |
| false rejection | 20 |
| unsafe false route | 12 |
| wrong route / operational error | 0 / 0 |
| completion tokens | 2–3 |

主要错误簇：

- `RecommendProduct` 召回率仅 82.86%，主题含股票/房产/旅行的实体商品、商品比较、
  库存和多轮目标切换常被 `ProductOther` 拒绝。
- 非股票金融对象被 `StockInfo` 接收 6 次。
- 证券账户/持仓/交易操作被 `StockInfo` 接收 3 次。
- 已发生股票事实有 2 次被错误拒绝。

## E002：以核心对象和最终结果重写候选边界

日期：2026-08-06

假设：候选描述混用了“相关主题”和“Agent 可完成的最终动作”，其中
`ProductOther` 的宽泛开头会吞掉普通购买表达，`StockInfo` 又缺少和其它金融对象
相对照的原型。

优化：

1. `StockInfo` 用具体股票/股票市场对象及当前、历史、事实结果定义，并补充分红、
   公告和已经发生的涨跌解释。
2. `StockOther` 明确承接个人持仓、证券账户、交易和资金操作。
3. `NoAvailable` 增加基金、债券、期货、外汇、黄金、加密货币和银行理财原型。
4. `Ecommerce` 用普通实体商品的搜索、筛选、比较、价格、库存和链接定义；强调
   商品主题、用途中的股票/汽车等词不会改变实体商品购买目标。
5. 删除 `ProductOther` 的宽泛兜底表述，只保留已有商品/订单履约、商品使用处理、
   非普通电商对象和专业服务。
6. 分类规则要求按当前期望的最终结果判断，忽略已经取消、否定或替换的旧目标。

同一冻结 dev 集前后对照：

| 指标 | B001 | E002 | 变化 |
|---|---:|---:|---:|
| system accuracy | 388/420（92.38%） | 405/420（96.43%） | +17 / +4.05pp |
| candidate exact | 375/420（89.29%） | 388/420（92.38%） | +13 / +3.09pp |
| false rejection | 20 | 4 | -16 |
| unsafe false route | 12 | 11 | -1 |
| operational error | 0 | 0 | 0 |

分 Label 结果：`SearchStockQuotes` recall 100%，`RecommendProduct` recall 从
82.86% 升至 96.19%；非股票金融错分由 6 次降至 1 次。剩余错误主要是
`Ecommerce` 过度接收汽车、酒店、服务和已有订单，以及少量未来预测。

## X001（未采用）：将 `Ecommerce` 改名为 `ProductSearch`

日期：2026-08-06

目的：让 Product 下的两个叶子共享 `Product` 前缀，使生成过程先确定大类，再在
`Search/Other` 间确定叶子；其它 prompt、映射和数据保持 E002 不变。

| 指标 | E002 | X001 | 变化 |
|---|---:|---:|---:|
| system accuracy | 405/420（96.43%） | 402/420（95.71%） | -3 / -0.71pp |
| false rejection | 4 | 2 | -2 |
| unsafe false route | 11 | 15 | +4 |
| wrong route | 0 | 1 | +1 |

结论：`ProductSearch` 的强语义使普通商品召回略升，但把更多汽车、酒店和专业服务
吸入真实电商候选，且引入股票/商品交叉错分。风险和总体准确率均恶化，因此不落地，
模型可见名称恢复为 `Ecommerce`。

## E003：增加互斥的层级决策顺序

日期：2026-08-06

假设：只有候选卡时，模型会直接寻找局部相似项；在稳定 system 前缀中显式要求先
恢复当前单一目标、再确定对象大类、最后按期望结果选叶子，可以让一次短输出实际
执行层级分类，同时不向模型暴露后端映射。

优化：

1. 先判定非股票金融对象，再进入 Stock 分支。
2. Stock 分支按“尚未发生的预测/投资决策 → 账户交易或软件本身 → 当前历史事实”
   的互斥顺序选择 `StockAdvice/StockOther/StockInfo`。
3. Product 分支先识别已有商品/订单、使用故障、汽车房产保险旅游和专业服务，再把
   其余普通实体商品发现需求交给 `Ecommerce`。
4. 商品对象缺失和多独立目标继续由 `NoAvailable` 承接。

同一冻结 dev 集前后对照：

| 指标 | E002 | E003 | 变化 |
|---|---:|---:|---:|
| system accuracy | 405/420（96.43%） | 413/420（98.33%） | +8 / +1.90pp |
| candidate exact | 388/420（92.38%） | 403/420（95.95%） | +15 / +3.57pp |
| false rejection | 4 | 1 | -3 |
| unsafe false route | 11 | 6 | -5 |
| wrong route / operational error | 0 / 0 | 0 / 0 | 0 / 0 |

Stock 预测、账户操作、非股票金融、专业服务及所有多轮真实候选分桶均达到 100%。
剩余 7 个系统错误中，5 个属于汽车/酒店/机票被 `Ecommerce` 接收，另有 1 个主题
为比特币图案的 T 恤被拒绝，以及 1 个无上下文指代被当成股票信息。

## E004：拆分 `NonRetail` 虚拟候选

日期：2026-08-06

假设：已有订单/商品使用和汽车房产/专业服务是两个相距较远的拒绝语义，把它们都
放在 `ProductOther` 中会形成模糊的负原型。增加一个更集中的虚拟候选可提升
`Ecommerce` 与非普通电商对象之间的边界。

优化：

1. 新增模型可见 `NonRetail`，专门描述整车/二手车、房产、保险、旅行线路、酒店、
   机票、餐饮及本地/专业服务；后端仍映射为 `null`。
2. `ProductOther` 收窄为已展示/选定商品、购物车、已有订单操作和已有商品使用故障。
3. 金融词只有在期望对象本身是金融产品时才进入 `NoAvailable`；作为 T 恤、教材或
   配件的主题时仍按实体商品判断。
4. 没有可恢复前文的指代和时间词不能凭空提供核心对象。

同一冻结 dev 集前后对照：

| 指标 | E003 | E004 | 变化 |
|---|---:|---:|---:|
| system accuracy | 413/420（98.33%） | 419/420（99.76%） | +6 / +1.43pp |
| false rejection | 1 | 0 | -1 |
| unsafe false route | 6 | 1 | -5 |
| wrong route / operational error | 0 / 0 | 0 / 0 | 0 / 0 |

`SearchStockQuotes` 和 `RecommendProduct` recall 均为 100%，`RecommendProduct`
precision 100%；28 个桶中 27 个达到 100%。唯一错误为“美股今晚会不会暴跌”被
`StockInfo` 接收。

E004 改变了虚拟候选 taxonomy，而冻结数据中的非零售样例仍保留旧的
`no_route_product_other` 诊断标注，因此本实验起不再横向比较 `candidate exact`；
最终系统输出三分类指标不受影响。

### E004 首次 test 结果

E004 完成 dev 调优后首次运行 140 条 test，结果为 138/140（98.57%），未达到独立
test 99% 门槛。两个错误为：

- “推荐一个保险柜放家里”因“保险”子串被 `NonRetail` 拒绝。
- “推荐靠谱的婚纱摄影工作室”被 `Ecommerce` 错误接收。

## X002（未采用）：扩写完整交付物规则

日期：2026-08-06

尝试同时在 `Ecommerce` 和全局 Product 规则中增加保险柜、旅行箱等正例，并把
NonRetail 写成“车辆/合同/预订/服务提供方”。原有两个 test 错误被修复，但新增
“已选第二款耳机后直接付款”和“预算十五万推荐汽车”两个 unsafe false route，test
仍为 138/140（98.57%）。说明扩写真实候选会提高其全局吸引力，造成边界摆动，故
回退，不提交算法代码。

## X003（未采用）：强化多轮目标替换指令

日期：2026-08-06

针对“儿童手表不要了，改推荐亲子旅行线路”增加丢弃旧目标的强指令和候选示例。
该样例被修复，但“鼠标不修了，改推荐无线鼠标”被错误拒绝，无上下文“第二个”被
错误分发；test 再次为 138/140（98.57%）。该自然语言特例同样造成摆动，故回退。

## E005：局部消歧合同、服务和已选商品状态

日期：2026-08-06

优化：

1. 只在 `NonRetail` 中声明“保险”指保险合同而非保险柜实体商品，并增加摄影工作室
   服务原型；不扩写 `Ecommerce`，避免提升真实候选的全局吸引力。
2. 在 `StockAdvice` 增加“今晚是否大跌”一类未发生行情原型。
3. 在 `ProductOther` 中把“刚才那款、第二款、就它”定义为已经完成选品，后续下单
   或付款属于履约操作。

为避免把 API 限流混入算法准确率，最终结果由两次同版本、低并发（workers=2）、
无运行错误的 dev/test 调用合并。一次 workers=8 的全量尝试在第 544–560 个请求
触发 17 次 `RateLimitError`，该次运行不计入语义验收。

| 指标 | dev | test | 合并 |
|---|---:|---:|---:|
| system accuracy | 420/420（100%） | 139/140（99.29%） | 559/560（99.82%） |
| SearchStockQuotes precision / recall | 100% / 100% | 100% / 100% | 100% / 100% |
| RecommendProduct precision / recall | 100% / 100% | 97.22% / 100% | 99.29% / 100% |
| null precision / recall | 100% / 100% | 100% / 98.57% | 100% / 99.64% |
| operational error | 0 | 0 | 0 |
| completion tokens | 2–3 | 2–3 | 2–3 |

最终唯一错误为 test 中“儿童手表先不买了，改成推荐亲子旅游线路”选择了
`Ecommerce`。更强的目标替换特例会造成其它边界回归（见 X003），因此在满足
dev、test 和完整集均 ≥99% 后保留 E005 的更稳定版本。

机器可读汇总：`eval/results/benchmark_v1_e005_summary.json`。

## E006：生产错例回归与细粒度负候选

日期：2026-08-07

模型：`qwen3.5-35b-a3b`。请求同时使用
`chat_template_kwargs.enable_thinking=false` 与 `/no_think`，不设置生成 token 上限，
`temperature=0`，最终输出仍为一个短英文候选名。

数据与契约变更：

1. 将用户提供的错例按最终系统输出原样纳入
   `eval/production_regression_20260807.jsonl`；删除 3 组完全相同的重复 query 后为 94 条，
   包含 18 条 `SearchStockQuotes`、15 条 `RecommendProduct` 和 61 条 `null`。
2. 冻结 benchmark 扩展为 654 条，并增加独立 `regression` split。用户标注优先于旧的
   合成规则；据此修正 8 条“附带任务遮蔽唯一可调用子任务”的旧标签。另一个独立的
   股票预测/荐股、整车/房产重大推荐目标仍标为 `null`，避免把所有混合请求都强行分发。
3. 新增 `StockResearch` 与 `ProductInfo` 两个模型可见负候选，分别隔离公司/赛道深度
   研究和纯商品信息/兼容问答；后端都折叠为 `null`。
4. 补齐复购、纯型号询价、既有设备兼容、ASR 商品名、股票公开事实与深度研究、候选名
   提示注入、残缺指代等边界。候选树和后端 Label 映射仍完全隔离。
5. 评测器增加 `--bucket` 定向复测；新增结果合并脚本，按 case id 让后续成功重试覆盖
   限流行，并验证数据集标签、模型一致性、完整性和最终错误状态。

未优化提示词在新增 94 条回归集上仅为 20/94（21.28%），包含 28 次 false rejection
和 46 次 unsafe false route。E006 最终结果：

| 指标 | dev | legacy test | production regression | 合并 |
|---|---:|---:|---:|---:|
| system accuracy | 420/420 | 140/140 | 94/94 | 654/654（100%） |
| SearchStockQuotes precision / recall | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% |
| RecommendProduct precision / recall | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% |
| null precision / recall | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% |
| operational error | 0 | 0 | 0 | 0 |
| completion tokens | 2–3 | 2–3 | 2–3 | 2–3 |

最终 dev 由同一提示词版本的 405 条成功主运行结果和 15 条低并发成功重试合并；主运行
末段的 13 次 `RateLimitError` 未计作语义错误，也没有被静默当成 `null`。完整结果的
延迟观测为 median 282.855 ms、p95 419.719 ms，本轮不以接口时延作为验收条件。

`candidate exact` 为 532/593（89.71%），原因是旧数据的虚拟候选诊断标签未随新 taxonomy
全部重标；最终三分类输出不受影响。94 条生产回归样例已用于提示词调优，因此 94/94 是
回归通过率，不应解释为独立未见生产流量上的无偏准确率。

机器可读结果：`eval/results/benchmark_v1_qwen3_5_e006_final_rows.jsonl` 与
`eval/results/benchmark_v1_qwen3_5_e006_final_summary.json`。

## E007：第二批生产错例与对象/动作边界泛化

日期：2026-08-07

模型与调用参数沿用 E006：`qwen3.5-35b-a3b`、`temperature=0`、双重关闭 thinking、
不设置生成 token 上限。

数据处理：用户给出 39 行生产标注，其中 37 个 query 唯一；2 行是批内重复，另有
19 个唯一 query 已由第一批生产回归覆盖，因此本轮净新增 18 条（`null` 12 条、
`RecommendProduct` 6 条）。增量源数据保存在
`eval/production_regression_20260807_round2.jsonl`。冻结 benchmark 扩展到 672 条，
SHA-256 为 `bfb45ebc9ae29d500b2780e229e42dbfa2681192adc3c50bf6de592f3c57bc98`。

E006 提示词在新增 18 条上的基线只有 7/18（38.89%），包含 9 次
`unsafe false route` 和 2 次 `false rejection`。本轮没有按 query 字符串写规则，
而是收敛为以下可迁移边界：

1. 对象边界：药品、摩托车等整车即使指定普通电商平台也进入 `NonRetail`；品牌销量、
   平均售价、型号介绍进入 `ProductInfo`。
2. 商品动作边界：明确查看替换配件、搜索同款、可识别的 ASR 截断商品词进入
   `Ecommerce`；指定平台查价优先于无平台的独立价格问答；“不要最新款”是筛选条件，
   不是否定查看动作。
3. 规格边界：已知具体型号问参数或适用性进入 `ProductInfo`；没有具体型号、根据空间
   或人数反推整个品类应选规格，属于 `Ecommerce` 商品选型。
4. 股票边界：条件性市场预测和“价格是否已反映未来业绩”的估值结论进入
   `StockAdvice`；ETF/基金自身选标的或纠正净值/交易价格、裸公司名和无请求的市场陈述
   进入 `NoAvailable`；股息率等一般市场指标解释仍进入 `StockInfo`。
5. 信息完整性优先：没有可恢复前文的“这几款、那些品牌”等指代不能因为同时出现价格、
   国补、配置或搜索词而触发真实候选。

最终提示词版本的严格结果：

| 指标 | dev | legacy test | production regression | 合并 |
|---|---:|---:|---:|---:|
| system accuracy | 420/420 | 140/140 | 112/112 | 672/672（100%） |
| SearchStockQuotes precision / recall | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% |
| RecommendProduct precision / recall | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% |
| null precision / recall | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% |
| operational error | 0 | 0 | 0 | 0 |
| completion tokens | 2–3 | 2–3 | 2–3 | 2–3 |

dev 分片首次调用的末段出现 75 次 `RateLimitError`，随后只重跑受影响 bucket 并按
case id 严格覆盖，所有成功语义调用均正确。生产回归一次完整运行为 111/112：一个
无上下文复数指代样例随机选择了 `Ecommerce`；在不改提示词的情况下，该单例连续
3 次复测以及其 61 条完整 `NoAvailable` bucket 复测均正确，最终合并结果采用完整
bucket 复测。这里的 112/112 是回归通过率，不应被解释成未见流量上的无偏准确率；
首轮 111/112 也表明 `temperature=0` 的服务端输出仍可能存在非确定性。

第二批 18 条在 3 种候选顺序下共 54 次调用全部正确，说明这些新增边界没有依赖生产
默认候选位置；这仍是回归稳定性检查，不替代独立测试集。

合并结果的 `candidate exact` 为 543/599（90.65%）；虚拟候选之间的差异不影响后端
三分类。延迟仅作观测，median 278.053 ms、p95 450.841 ms。机器可读结果：
`eval/results/benchmark_v1_qwen3_5_e007_final_rows.jsonl` 与
`eval/results/benchmark_v1_qwen3_5_e007_final_summary.json`。

## E008：正负样本双门槛与重复采样稳定性

日期：2026-08-07

用户再次报告 39 行错分；去重后为 37 个 query，且都已存在于两批生产回归中：
10 个应召回正样本（4 个 `SearchStockQuotes`、6 个 `RecommendProduct`）和 27 个
应拒绝负样本。为避免复制消息和标签，新增 case-id cohort
`eval/cohorts/production_misclassifications_20260807_round3.txt`，继续以冻结 benchmark
作为唯一标签来源。

本轮业务验收指标改为：

- 正样本正确率：应召回样本必须召回到标注的具体 Intent，要求不低于 95%。
- 负样本正确率：应拒绝样本必须实际不召回，要求不低于 98%。
- API 错误和不可解析输出在两项指标中都计为错误，不能借 fail-closed 的 `null` 虚假通过。

评测器相应增加 `route_metrics.positive/negative`、两个独立门槛、`--repeats` 和
`--case-id-file`。候选换序由 `--seeds` 控制，同一输入和顺序的重复采样由
`--repeats` 控制，二者不再混淆。重复运行结果按 `(case id, seed, repeat)` 严格合并；
不同模型，或同一候选顺序下不同 `prompt_sha256` 的结果禁止合并。批处理输出也记录
模型、提示词哈希和候选顺序 seed，便于定位调用端版本差异。

E007 提示词在生产默认顺序下每条重复 5 次，共 185 次，正样本 50/50、负样本
135/135，没有复现用户侧错误。但在 3 种候选顺序下共 111 次时出现 1 次
false rejection：正样本 29/30（96.67%）、负样本 81/81（100%）。错误发生在“指数
基金所代表的市场点位”请求；固定该候选顺序再调用 5 次只有 1 次正确，说明不是单纯
随机噪声，而是候选位置放大的语义边界不稳。

优化只强化通用边界：若当前最终问题明确询问 ETF/指数基金所代表股票市场的点位、
位置或振幅，选择 `StockInfo`，即使前文评论过收益；基金自身的标的选择、净值、
交易价格和收益仍选择 `NoAvailable`。没有针对基金名称或完整 query 写特判。

优化后，37 条 cohort 在 3 种候选顺序、每种重复 3 次下共 333 次调用，正样本
90/90（100%）、负样本 243/243（100%）。主运行后段有 112 次 `RateLimitError`；
完整运行键重试后合并，所有成功的语义调用均正确。机器可读 cohort 结果为
`eval/results/e008_round3_cohort_after_final_rows.jsonl` 与
`eval/results/e008_round3_cohort_after_final_summary.json`。

最终提示词默认候选顺序的全量结果如下。test 主运行后段的 43 次限流按相同提示词哈希
重跑对应 bucket；最终合并结果没有运行错误。

| 指标 | dev | legacy test | production regression | 合并 |
|---|---:|---:|---:|---:|
| 正样本正确率 | 216/216 | 72/72 | 39/39 | 327/327（100%） |
| 负样本正确率 | 204/204 | 68/68 | 73/73 | 345/345（100%） |
| system accuracy | 420/420 | 140/140 | 112/112 | 672/672（100%） |
| operational error | 0 | 0 | 0 | 0 |
| completion tokens | 2–3 | 2–3 | 2–3 | 2–3 |

默认顺序 system prompt SHA-256 为
`b92d3107bdd1d6bb8bbab548d2616c30494e231588ddd9a11c844aa0ec4c9381`。
全量延迟仅作观测，median 303.226 ms、p95 523.255 ms。机器可读结果为
`eval/results/benchmark_v1_qwen3_5_e008_final_rows.jsonl` 与
`eval/results/benchmark_v1_qwen3_5_e008_final_summary.json`。

37 条均已用于回归调优，333/333 和 672/672 不能当作未见生产流量的无偏准确率；
它们验证的是在当前模型服务、提示词哈希和候选排列范围内满足业务门槛。用户侧若仍
复现不同结果，应首先对比批处理输出中的 `model`、`prompt_sha256`、
`candidate_order_seed` 和 `raw_model_output`。

## E009：当前请求与长历史回答隔离

日期：2026-08-07

问题：旧版把所有 user/assistant/tool 消息平铺进一个 JSON 数组。历史 assistant 回答较长时，
其中的商品、股票和动作词容易被模型误认为当前用户任务；同时长回答可能耗尽 12000 字符
预算，挤掉真正建立指代对象的用户消息。

本轮只调整对话数据组织，不改候选语义：

1. 单轮继续使用旧的消息数组，system prompt 与 user payload 都保持原样；默认 system prompt
   SHA-256 仍为 `b92d3107bdd1d6bb8bbab548d2616c30494e231588ddd9a11c844aa0ec4c9381`。
2. 多轮数据按角色拆成 `prior_user_turns`、`assistant_reference_context`、
   `tool_reference_context` 和最后的 `current_user_request`。当前消息是唯一分类对象，历史用户
   只补全承接关系，assistant/tool 只作为实体、选项或事实参考。
3. 当前消息优先占用字符预算。assistant/tool 只有在当前消息出现明确引用信号时才保留，
   每条最多截取 1200 字符；截取保留开头约三分之二和结尾约三分之一。
4. `IntentRouter` 自身也忽略 system message，使直接调用与批处理入口一致。

真实模型验证使用 `qwen3.5-35b-a3b`、关闭 thinking、`temperature=0`：

- 原有 40 条正向多轮样本为 40/40。
- benchmark 中全部 110 条多轮样本，在每条历史 assistant 后追加数千字符、并混入商品推荐、
  股票行情、预测、基金、汽车、药品和订单等强干扰后，为 110/110；其中正样本 41/41，
  负样本 69/69。
- 用户给出的 23 个当前 query 前置一轮无关长 assistant 后为 21/23，负样本 18/18；两个
  剩余错误在无历史单轮下也会出现，分别是“裸公司名是否应触发股票查询”和“云南白药牙膏
  是否被药品边界误伤”，不属于历史组织问题。

一次 672 条全量调用在前 166 条语义调用全部正确后触发服务端限流，余下 506 条均为
`RateLimitError`，因此该次运行不作为准确率报告。单轮请求保持字节级旧格式；变化覆盖的
全部 110 条多轮样本已单独完成语义验证。上述强干扰数据是结构压力测试，仍需要用户提供
真实失败会话的完整 `messages` 才能确认真实长回答中的指代信息是否被正确保留。

## E010：提示词压缩与规则泛化

日期：2026-08-07

目标是在不改变候选树、后端映射和双门槛契约的前提下，大幅减少 E008 的长篇规则与逐条
校准样例。极限压缩到 3202 字符的首版在 112 条 regression 上出现明显退化，因此最终没有
以最短字符数为目标，而是保留少量能表达通用决策结构的原型。

最终组织方式：

1. 全局规则收敛为目标恢复、按预期结果与最终对象分类、重点节点成立条件、子任务集合合并、
   其它互斥边界和少量抽象对照，不展示后端分发信息。
2. 候选描述删除重复枚举，保留股票信息/研究/预测、商品发现/知识/履约、非普通零售对象之间
   的定义；样例主要表达 ASR 型号、上下文不足、具体选购、平台查价和混合任务等结构。
3. 多轮输入沿用 E009 的角色隔离，但把规则修正为“以当前轮为入口”：当前轮明确要求保留旧
   目标时，旧目标仍进入子任务合并；明确取消或替换时则移除。
4. system prompt 从 11593 字符降到 6129 字符，减少 47.1%。以“查一下贵州茅台今天股价”
   实测，服务端 `prompt_tokens` 从约 6038 降到 3205，减少 46.9%；completion 仍为 2–3 tokens。

真实调用使用 `qwen3.5-35b-a3b`、默认候选顺序、`temperature=0`、双重关闭 thinking，且没有
设置生成 token 上限。最终 system prompt SHA-256 为
`96259421ddbc610b2341b5bd818de3a7b596d429db53a6e0e5c54685718f8235`。

同哈希首轮完整 672 条为 666/672（99.11%）：正样本 321/327（98.17%），负样本
345/345（100%），没有 API 或解析错误，已经通过 95%/98% 门槛。随后只重跑出现偏差的完整
bucket，并按 case id 用后续结果覆盖；合并结果如下：

| 指标 | dev | legacy test | production regression | 合并 |
|---|---:|---:|---:|---:|
| 正样本正确率 | 215/216 | 72/72 | 39/39 | 326/327（99.69%） |
| 负样本正确率 | 204/204 | 68/68 | 73/73 | 345/345（100%） |
| system accuracy | 419/420 | 140/140 | 112/112 | 671/672（99.85%） |
| operational error | 0 | 0 | 0 | 0 |
| completion tokens | 2–3 | 2–3 | 2–3 | 2–3 |

唯一剩余错误是“某股票自 2022 年以来每年涨跌幅”在两次完整 bucket 调用中都选择了
`StockResearch` 而非 `StockInfo`。继续为单例增加提示词会重新走向逐 query 过拟合；当前版本
保留这一已知误差，因为核心正样本指标仍高出门槛 4.69 个百分点，负样本为 100%。这意味着
严格 system accuracy 相比 E008 的回归合并结果少 1 条，但业务双门槛没有退化；上线仍应监控
该历史序列与深度研究边界。

合并结果延迟仅作观测：median 310.568 ms、p95 596.735 ms。机器可读文件为
`eval/results/benchmark_v1_qwen3_5_e010_compact_final_rows.jsonl` 与
`eval/results/benchmark_v1_qwen3_5_e010_compact_final_summary.json`。这些 benchmark 样例已参与
调优，结果只证明回归兼容性，不代表未见流量上的无偏泛化准确率。

## E011：Qwen3-32B 双边界短提示词

日期：2026-08-07

目标是改用 `qwen3-32b`，同时彻底删除候选 examples、实体枚举和逐样例校准文字。紧凑的
单次多类 prompt 在完整 benchmark 上无法同时达到两个门槛：已验证版本最好约为正样本
301/327、负样本 315/345。进一步增加拒绝描述会提高负样本但明显损失正样本，说明问题不只是
字符数，而是一个生成步骤同时承担领域识别、动作边界、对象边界和多任务合并，决策相互干扰。

最终算法把问题分解为两个独立闭集：

1. 股票边界只区分公开股票信息、投资决策、公司研究、证券操作、其它金融和无股票请求。
2. 商品边界只区分普通商品发现、商品知识、履约、非零售对象、多商品和无商品请求。
3. 后端仅在一个边界正向时提出召回；双正向、双负向和虚拟分支默认不召回。
4. 再使用不含品牌、型号和完整 benchmark 文本的通用结构策略做安全校验，例如未来/条件预测、
   证券软件本身、非零售中心对象、已有商品知识、多商品、纯名称、残句和无先行对象的指代。
5. 若模型没有严格输出单个 id，只追加格式约束重试一次；重试不增加语义样例，仍失败则
   fail-closed。

两个稳定 system prompt 分别为 396 和 488 字符，合计 884 字符；相对 E008 的 11593 字符
减少 92.4%，相对 E010 的 6129 字符减少 85.6%。两者均为固定短候选表，不包含 examples、
后端 Label、真实/虚拟标记、Agent 名称或编排信息。对 Qwen3-32B 的回归对照中，相同描述的
紧凑 JSON 形式比逐项短表更容易混淆边界，因此生产版本采用测得更稳的短表；两者语义定义
完全一致。

对话数据也统一为固定 JSON envelope：`current_user_request` 始终单独放置，历史 user 轮进入
`prior_user_turns`；assistant/tool 只在当前轮含明确指代时进入 `assistant_tool_reference`。
长 assistant 不能覆盖用户动作，也不能挤掉当前请求。单轮和多轮使用同一 envelope，避免两种
prompt 模板漂移。

正式真实调用使用 `qwen3-32b`、`temperature=0`、双重关闭 thinking、不设置生成上限。
最终 672 条完整运行中，格式异常由同一请求内的一次性格式重试恢复；评测脚本一次完成，
没有 API 或解析错误。结果为：

| 指标 | 结果 |
|---|---:|
| 正样本正确率 | 318/327（97.25%） |
| 负样本正确率 | 345/345（100%） |
| system accuracy | 663/672（98.66%） |
| `SearchStockQuotes` precision / recall | 100% / 97.55% |
| `RecommendProduct` precision / recall | 100% / 96.95% |
| operational error | 0 |
| 总 completion tokens | 通常 12–13；格式重试样本 21 |
| latency | median 642.571 ms；p95 782.008 ms |

组合 prompt SHA-256 为
`1f0cf88e963f93f4a2da76276a3d22f2973ee9cef4da6defc8490d1d4aa78ab7`。机器可读文件为
`eval/results/benchmark_v1_qwen3_32b_compact_boundary_rows.jsonl` 与
`eval/results/benchmark_v1_qwen3_32b_compact_boundary_summary.json`。

最终没有 false positive，9 个错误全部是保守拒绝。这满足正样本不低于 95%、负样本不低于
98% 的业务门槛，但 benchmark 已参与架构和通用边界选择，不能视为未见流量上的无偏估计；
上线应继续分别监控两个业务指标，而不是只看整体 accuracy。
