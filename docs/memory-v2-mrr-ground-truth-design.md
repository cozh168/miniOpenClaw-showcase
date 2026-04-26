# memory_module_v2 MRR评估：ground truth自动生成与计算方案（方案A）

## 目标
对 `backend/memory_module_v2` 的记忆检索效果做 **MRR（Mean Reciprocal Rank）** 评估，评估对象为：
- 检索入口：`memory_module_v2.service.api.search_memory()`
- 候选排序：返回的 `MemorySearchResponse.hits`（按 `MemoryHit.rank` 从 1 开始）

本设计同时覆盖两部分：
1. 自动生成 ground truth（labels），从 `backend/memory_module_v1/sessions/*.json` 的对话文本推导“正样本 `exchange_id`”
2. 基于 ground truth 计算 MRR@K（可选：分别评估 dense/keyword/hybrid 等检索模式）

## 范围与不做的事
范围：
- 只评估 `memory_module_v2` 的检索排序质量（MRR）
- ground truth 的生成只依赖 v1 sessions 文本与 v2 的 exchange 切分规则

不做：
- 不需要集成 `langfuse`（可作为可选“附加上报”能力）
- 不做 LLM grader / 人工评注（本方案为规则自动生成）

## 关键定义
### Query（评估查询）
对每条正样本 label，都定义一个 query 文本：
- **query = 该 exchange 的 `ply_start` 位置对应的 user content**

该规则要求对同一个 session：
- 先将 `messages[]` 归一化为 `NormalizedMessage(msg_index=idx, role, content, tool_calls)`
- 再切分得到 `Exchange(ply_start, ply_end, ...)`
- 用 `messages[ply_start].content` 作为 query

### 正样本（Relevant Exchange IDs）
本方案 A 强制：
- **只用 `exchange.has_substantive_assistant == true` 的 exchange 作为正样本**
- 每个 label 的 `relevant_exchange_ids` 为单元素：
  - `relevant_exchange_ids = [exchange.exchange_id]`

### Reciprocal Rank 与 MRR
对每条 query label：
- 从检索器返回的 `hits` 中，按顺序寻找第一个命中的正样本：
  - 命中条件：`hit.exchange_id in relevant_exchange_ids`
- 设命中位置为 `rank_hit`（1-based）
- 若 topK 内没有任何命中，则 RR = 0
- RR 定义为：`RR = 1 / rank_hit`

整体 MRR：
- `MRR@K = 平均_{labels}(RR)`，其中检索调用时 `top_k=K`

## ground truth 生成：labels 生成器（规则）
### 输入
- session 数据目录：`backend/memory_module_v1/sessions/*.json`

### 生成步骤（每个 session）
1. 读取 session 原始 JSON：`load_session_raw(session_id)`
2. 归一化 messages：为每条消息附加 `msg_index`
3. 调用切分器得到 exchanges：
   - `segment_exchanges(session_id, messages, min_exchange_chars, max_ply_len, min_assistant_chars)`
   - 参数建议默认使用 `memory_module_v2.service.config.get_memory_v2_config()` 中的值
4. 遍历每个 `exchange`：
   - 若 `exchange.has_substantive_assistant != true`：跳过
   - 查找 query：
     - query 来源：`messages[exchange.ply_start].content`
   - 输出一条 label（JSONL 一行）：
     - `session_id`
     - `ply_start`, `ply_end`
     - `query`
     - `relevant_exchange_ids: [exchange.exchange_id]`
     - `substantive: true`

### 输出格式
建议输出到：
- `backend/memory_module_v2/eval/ground_truth_v1_sessions.jsonl`

每行 JSON：
```json
{
  "session_id": "4231f405-0b25-4542-ba6c-1eb4570f2139",
  "ply_start": 3,
  "ply_end": 7,
  "query": "（ply_start处user的content）",
  "relevant_exchange_ids": ["<exchange_id>"],
  "substantive": true
}
```

额外建议（可选）：
- 单独写一个 `ground_truth.meta.json` 记录生成参数（切分阈值、时间、样本数等），便于可复现。

## MRR评估：评测脚本
### 输入
- ground truth JSONL（上一步输出）
- 检索模式（mode）：建议评估
  - `dense_distilled`
  - `keyword_verbatim`
  - `hybrid_cross`
- topK：建议评估 `K in {1, 3, 5, 10}`（或至少 10）

### 依赖/前置条件（非常重要）
要保证检索结果非空，需要满足：
1. exchanges 已被蒸馏写入 storage（至少需要 `memory_module_v2` 的 objects）
   - 建议执行：`backend/memory_module_v2/script/distill_all_sessions.py`
2. BM25 keyword 索引已构建（对 `keyword_verbatim/hybrid_cross`）
   - 依赖你们当前 BM25 的缓存构建与 rebuild 策略
   - 建议在评测前触发一次 health check/增量 rebuild（或直接用你们已有的启动流程）

### 评测流程（每条 label）
1. 调用：
   - `search_memory(query=label.query, mode=SearchMode.<...>, top_k=K, filters=可选)`
2. 读回：
   - `response.hits`（列表已按 rank 顺序排列）
3. 计算 RR：
   - 找第一个命中的相关 exchange_id
4. 汇总：
   - 输出整体 `MRR@K`、以及（可选）命中率 `HitRate@K`

### 输出
建议输出到：
- `backend/memory_module_v2/eval/mrr_results.json`

包含字段：
- mode, K
- label_count
- mrr
- hit_rate（可选）
- per_query（可选；只在 debug/需要分析时输出）

## 关于 langfuse 的关系
本评测不需要 `langfuse` 才能算 MRR。
如果你希望：
- 在 `langfuse` 里查看每条 query 的命中列表、分数、以及具体 exchange_id
也可以在评测脚本中“可选”上报检索结果（不影响离线计算部分）。

## 建议实现产出（代码文件清单，供实现时对齐）
建议创建以下脚本（具体命名可在实现时微调）：
1. `backend/memory_module_v2/eval/generate_ground_truth.py`
   - 读取 v1 sessions
   - 调用 `segment_exchanges`
   - 生成 ground truth JSONL
2. `backend/memory_module_v2/eval/evaluate_mrr.py`
   - 读取 ground truth
   - 调用 `search_memory`
   - 计算 MRR@K并输出结果 JSON
3. 可选：`backend/memory_module_v2/eval/run_mrr_eval.py`
   - 批量跑不同 mode 和不同 K

## 运行方式（草案）
1. 生成 ground truth：
   - `python -m memory_module_v2.eval.generate_ground_truth --output ground_truth.jsonl`
2. 确保已蒸馏 & keyword 索引可用
   - 运行 `distill_all_sessions.py`
3. 计算 MRR：
   - `python -m memory_module_v2.eval.evaluate_mrr --labels ground_truth.jsonl --mode hybrid_cross --top_k 10`

## 风险与校验点
1. `exchange.ply_start` 对应的 messages 角色必须是 `user`
   - 若由于切分器逻辑导致 ply_start 指向了非 user，可能会得到空 query 或噪声
   - 实现时建议对 query 进行空值跳过与计数统计
2. 样本覆盖率
   - `substantive=true` 过滤会减少样本数，这是预期的一部分
3. 评测可复现性
   - 需要把 `segment_exchanges` 的阈值参数写入 meta

## 需要你确认的两个点（本设计已先按你的选择落地）
- 已确认 query 取 `exchange.ply_start` 处的 `user content`
- 已确认正样本只用 `substantive=true` exchange

