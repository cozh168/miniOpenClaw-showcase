# memory_module_v2（Structured Distillation + Postgres Hybrid Retrieval）开发说明

本文档把论文 **Structured Distillation for Personalized Agent Memory** 的核心思想，工程化落地到本项目 `backend`，实现一个可嵌入的 **`memory_module_v2`**：

- **离线/增量蒸馏（distill）**：从 `memory_module_v1/sessions/*.json` 读取对话，按 exchange（ply 区间）切分，并蒸馏成结构化对象（compound object）。
- **两层架构**（论文核心）：  
  - **索引层**：distilled object（短文本 + metadata）用于检索与排序  
  - **证据层**：命中后回跳原始 exchange（verbatim）给 Agent/用户阅读
- **检索实现**：
  - **dense**：Postgres `pgvector`
  - **keyword**：应用层 `rank-bm25`（`rank_bm25.BM25Okapi`），对 **verbatim exchanges** 做 BM25
  - **融合**：应用层做融合（RRF/weighted sum）

> 说明：本方案不使用 Postgres 内置 FTS 作为 keyword 召回（中文分词较弱）。Keyword 侧使用 `rank-bm25`，并通过可插拔 `preprocess_func` 支持中文分词/自定义 tokenizer。Dense 侧仍用 pgvector。

---

## 目标与边界

### 目标（MVP 到可用）

- **MVP-1**：Distill + Dense 检索 + 回跳证据
  - 从 sessions 生成 exchanges
  - 对 exchanges 生成 distilled objects
  - `pgvector` 相似度检索 distilled objects
  - 返回命中对象的 verbatim snippet（证据层）
- **MVP-2**：加 Keyword(BM25) + 混合融合（论文推荐的互补信号）
  - 对 verbatim exchanges 构建 BM25 索引（rank-bm25）
  - 同时跑 dense（distilled）+ keyword（verbatim）
  - 结果融合（RRF / weighted sum）
- **MVP-3**：Rooms / Files facets + 过滤 + 运维与治理
  - room_assignments 与 files_touched 进入索引与过滤条件
  - 增量重建、幂等、去重、健康检查与统计

### 非目标（先不做）

- 完整复刻论文 107 配置离线评测与 grader pipeline
- UI（memory palace 导航界面）
- pg_bm25 直接接入（后续扩展点预留）

---

## 推荐项目结构

建议将 v2 作为 `backend` 内可独立测试/迁移的模块，边界清晰，避免与 agent/middleware 交织成一团。

```text
backend/
  memory_module_v2/
    dev.md
    README.md                      # 简短使用说明（可后补）
    __init__.py

    domain/
      models.py                    # dataclasses / pydantic：Exchange、DistilledObject、SearchHit…
      enums.py                     # SearchMode、RoomType…

    ingest/
      session_reader.py            # 读取 memory_module_v1/sessions/*.json -> NormalizedMessage
      exchange_segmenter.py        # messages -> exchanges（ply 切分）
      text_cleaner.py              # 去除明显 terminal 噪声（可配置）
      file_path_extractor.py       # regex 提取 files_touched（不要用 LLM）

    distill/
      prompts.py                   # distillation prompt（batch prompt / per-turn prompt）
      distiller.py                 # LLM 调用：Exchange -> DistilledObject

    storage/
      pg.py                        # SQLAlchemy/asyncpg 连接与迁移入口
      schema.sql                   # 建表 SQL（本文档给出）
      repos.py                     # ExchangesRepo / ObjectsRepo（CRUD、upsert、查询）

    retrieval/
      dense.py                     # pgvector 查询 distilled objects
      keyword.py                   # BM25(rank-bm25) 查询 verbatim exchanges
      fusion.py                    # RRF / weighted sum / CombMNZ（先实现 RRF+weighted）
      service.py                   # search_memory() 主入口

    service/
      api.py                       # distill_session / distill_all / search_memory / get_exchange…
      config.py                    # v2 配置（top_k、阈值、长度限制…）

    integrations/
      tools.py                     # LangChain tools：search_memory / distill_session / get_exchange
      middleware.py                # 每轮自动检索注入（可选开关）
```

---

## 与现有 sessions 的对齐（消息结构与 backref）

### sessions 文件结构（当前事实）

`backend/service/session_manager.py` 与样例 `memory_module_v1/sessions/*.json` 显示：

- `SessionRecord.messages` 是一个列表，每项至少 `{role, content}`，可选 `tool_calls`
- **没有 sequence 字段**，因此 v2 中的 ply 以 **`msg_index`（列表下标）** 作为回跳定位基础

### backref 约定（关键）

所有 exchange / distilled object 必须携带：

- `session_id`
- `ply_start`（包含）
- `ply_end`（包含）

并且约定：

- `ply_start`/`ply_end` 是 `messages[]` 的 **0-based index**
- 证据层回跳时，直接从 session 文件加载 messages，按 `[ply_start:ply_end+1]` 截取并格式化为 verbatim snippet

---

## Exchange 切分契约（messages -> exchanges）

### 名词

- **ply**：一条 message 的索引位置（`msg_index`）
- **exchange**：一个用户请求到一个“实质 assistant 回复”结束的区间

### 切分规则（对齐论文 + 适配本项目日志噪声）

1. 维护一个当前区间 `[start, end]` 与标志 `has_substantive_assistant`
2. 遍历 messages（按 msg_index 递增）：
   - 遇到 `role=user`
     - 若当前区间存在且 `has_substantive_assistant==True`：关闭当前 exchange，新开 exchange 从该 user ply 开始
     - 否则扩展当前区间（或创建）
   - 遇到 `role=assistant`
     - 扩展当前区间
     - 若该 assistant 文本被判定为 “实质回复”：`has_substantive_assistant=True`
3. 遍历结束后若存在未关闭区间：关闭为 exchange
4. 过滤：
   - `min_exchange_chars`（默认 100）以下丢弃
   - `max_ply_len`（默认 20）以上按固定长度切片为多个 exchange

### “实质 assistant 回复”判定（建议可配置）

因为 sessions 中常混入 terminal 工具输出/报错堆栈（见样例），建议判定函数为：

- 输入：assistant 的 `content` + 可选 tool_calls
- 输出：bool

默认 heuristic（可先实现最简单版本）：

- `len(content.strip()) >= min_assistant_chars`（建议 80～120）
- 且 content 不满足明显“纯工具输出”模板（可维护一个 `prefix_blacklist`）

> 备注：MVP 阶段可以先简化为 “assistant content 长度阈值”，后续再做 cleaner。

---

## Distilled Object schema（compound object）

对齐论文 Table 1：

### RoomAssignment

- `room_type`: `"file" | "concept" | "workflow"`
- `room_key`: `str`（短标识，尽量 snake_case）
- `room_label`: `str`（短标签，可读）
- `relevance`: `float`（0～1）

### DistilledObject（核心）

必须字段：

- `object_id: str`（uuid 或 hash）
- `exchange_id: str`（建议 deterministic：`sha1(session_id:ply_start:ply_end)`）
- `session_id: str`
- `ply_start: int`
- `ply_end: int`
- `exchange_core: str`（1–2 句，强调复用原词汇，不要编造）
- `specific_context: str`（一个具体细节，尽量原样拷贝）
- `room_assignments: RoomAssignment[]`（1–3 个）
- `files_touched: str[]`（regex 提取；不要 LLM 生成）
- `distill_text: str`（固定：`exchange_core + "\n" + specific_context`）
- `distilled_at: timestamptz`
- `distill_provider: str`
- `distill_model: str`

可选字段（后续扩展）：

- `tags: str[]`（per-turn distill 可用）
- `language: str`（zh/en，用于 tokenizer 配置）
- `project_id: str`（如果要做多项目隔离）

---

## 工程落地 API 形态（服务层）

建议把 v2 的核心能力做成一个服务类（例如 `MemoryServiceV2`），提供最小而稳定的 API。

### 1) distill_session

**用途**：对一个 session 的新 exchanges 做蒸馏并入库（幂等）。

**签名（建议）**

```python
distill_session(session_id: str, *, force: bool = False) -> DistillSessionResult
```

**入参**

- `session_id`: sessions 文件名对应的 id
- `force`: 是否强制重新蒸馏（默认 False；True 则重算并覆盖对象表/向量）

**出参：DistillSessionResult**

- `session_id: str`
- `exchanges_total: int`
- `exchanges_new: int`（本次新增/需处理）
- `objects_created: int`
- `objects_updated: int`
- `objects_skipped: int`（过滤/重复）
- `started_at: timestamptz`
- `finished_at: timestamptz`
- `errors: [{exchange_id, error}]`

### 2) search_memory

**用途**：按论文两层架构进行记忆检索。默认返回的是 **verbatim 证据**（exchange 原文片段），distilled 仅用于路由与排序解释。

**签名（建议）**

```python
search_memory(
  query: str,
  *,
  mode: SearchMode = "hybrid_cross",
  top_k: int = 10,
  filters: MemorySearchFilters | None = None,
  debug: bool = False,
) -> MemorySearchResponse
```

**SearchMode（建议枚举）**

- `dense_distilled`：仅 pgvector 搜 distilled objects（MVP-1）
- `keyword_verbatim`：仅 BM25(rank-bm25) 搜 verbatim exchanges（MVP-2，默认**窗口化 + 分片加载**）
- `hybrid_cross`：dense(distilled) + keyword(verbatim) 融合（推荐默认）

**filters：MemorySearchFilters（建议）**

- `session_ids?: str[]`：只搜某些 session
- `time_range?: {from?: timestamptz, to?: timestamptz}`：按 exchange/object 时间过滤
- `room_keys?: str[]`：命中 `room_key` 之一
- `files?: str[]`：命中 `files_touched` 之一（支持 basename）
- `min_fused_score?: float`：融合后阈值（可选）

**出参：MemorySearchResponse**

- `query: str`
- `mode: SearchMode`
- `top_k: int`
- `hits: MemoryHit[]`
- `debug?: MemorySearchDebug`

**MemoryHit（返回证据层 + 可解释索引层）**

- `rank: int`
- `session_id: str`
- `exchange_id: str`
- `object_id?: str`（dense 命中时）
- `ply_start: int`
- `ply_end: int`
- `verbatim_snippet: str`（用户/assistant 原文片段，供注入/展示）
- `rooms?: RoomAssignment[]`
- `files_touched?: str[]`
- `scores: { dense?: float, keyword?: float, fused: float }`

**MemorySearchDebug（建议）**

- `dense_candidates: [{object_id, exchange_id, score}]`
- `keyword_candidates: [{exchange_id, score}]`
- `fusion: {method: "rrf"|"weighted_sum", params: {...}}`

### 3) get_exchange（证据层回跳）

**用途**：给 UI/tool/middleware 取某个 exchange 的原文证据。

```python
get_exchange(session_id: str, ply_start: int, ply_end: int) -> ExchangeEvidence
```

**ExchangeEvidence**

- `session_id, ply_start, ply_end`
- `messages: [{msg_index, role, content, tool_calls?}]`
- `verbatim_snippet: str`

---

## 数据库存储契约（可直接建表的 SQL）

这里给出 **核心两表**（`memory_exchanges`、`memory_objects`）的建表方案，覆盖：

- exchange（verbatim，供 BM25 索引构建与证据回跳）
- object（distilled，带 pgvector）
- msg_index 回跳字段（ply_start/ply_end）
- 幂等去重的唯一约束
- 推荐索引

> 约定：schema 用 `memory_v2`（可改）。pgvector 维度请替换为你的 embedding 模型维度（例如 384/768/1024/1536）。

### 前置：扩展

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### Schema

```sql
CREATE SCHEMA IF NOT EXISTS memory_v2;
```

### 1) memory_exchanges（证据层 + keyword 语料层）

```sql
CREATE TABLE IF NOT EXISTS memory_v2.memory_exchanges (
  exchange_id      text PRIMARY KEY,

  -- backref to sessions
  session_id       text NOT NULL,
  ply_start        integer NOT NULL,
  ply_end          integer NOT NULL,

  -- evidence
  verbatim_text    text NOT NULL,
  verbatim_snippet text NOT NULL,

  -- timestamps
  created_at       timestamptz NOT NULL DEFAULT now(),
  updated_at       timestamptz NOT NULL DEFAULT now(),

  -- lightweight signals
  message_count    integer NOT NULL DEFAULT 0,
  has_substantive_assistant boolean NOT NULL DEFAULT false,
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_memory_exchanges_session_ply
  ON memory_v2.memory_exchanges(session_id, ply_start, ply_end);

CREATE INDEX IF NOT EXISTS ix_memory_exchanges_session_id
  ON memory_v2.memory_exchanges(session_id);

CREATE INDEX IF NOT EXISTS ix_memory_exchanges_created_at
  ON memory_v2.memory_exchanges(created_at);
```

> 备注：不再维护 `tsvector`/GIN（FTS），keyword 侧改用 `rank-bm25`。如未来需要做 Postgres 内 keyword 过滤（非召回），可在不影响 BM25 的情况下额外加 FTS。

### 2) memory_objects（索引层 + dense 检索层）

```sql
-- 将 vector(1024) 替换为你的 embedding 维度
CREATE TABLE IF NOT EXISTS memory_v2.memory_objects (
  object_id        text PRIMARY KEY,
  exchange_id      text NOT NULL REFERENCES memory_v2.memory_exchanges(exchange_id) ON DELETE CASCADE,

  -- backref (冗余一份，便于过滤/回跳)
  session_id       text NOT NULL,
  ply_start        integer NOT NULL,
  ply_end          integer NOT NULL,

  -- distilled fields
  exchange_core    text NOT NULL,
  specific_context text NOT NULL,
  distill_text     text NOT NULL,

  -- facets
  room_assignments jsonb NOT NULL DEFAULT '[]'::jsonb,
  files_touched    jsonb NOT NULL DEFAULT '[]'::jsonb,

  -- model lineage
  distill_provider text NOT NULL,
  distill_model    text NOT NULL,
  distilled_at     timestamptz NOT NULL DEFAULT now(),

  -- dense embedding
  embedding        vector(1536)
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_memory_objects_exchange
  ON memory_v2.memory_objects(exchange_id);

CREATE INDEX IF NOT EXISTS ix_memory_objects_session_id
  ON memory_v2.memory_objects(session_id);

CREATE INDEX IF NOT EXISTS ix_memory_objects_distilled_at
  ON memory_v2.memory_objects(distilled_at);

CREATE INDEX IF NOT EXISTS ix_memory_objects_room_assignments_gin
  ON memory_v2.memory_objects USING GIN (room_assignments);

CREATE INDEX IF NOT EXISTS ix_memory_objects_files_touched_gin
  ON memory_v2.memory_objects USING GIN (files_touched);

-- 向量索引（数据量上来再启用；先验证维度/距离策略）
-- CREATE INDEX IF NOT EXISTS ix_memory_objects_embedding_ivfflat
--   ON memory_v2.memory_objects USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
```

### 主键与去重策略（建议）

- `exchange_id`：deterministic，避免重复蒸馏写入  
  - `sha1(f"{session_id}:{ply_start}:{ply_end}")`
- `object_id`：推荐 **直接等于 exchange_id**（MVP 最省事）  
  - 好处：融合与回跳无需映射表

---

## 检索实现建议（dense + keyword + 融合）

### Dense（distilled）查询（pgvector）

- 输入：`query_embedding`
- 输出：`(object_id/exchange_id, dense_score)`
- 距离建议：cosine distance；融合时可将 distance 映射为 similarity，例如 \(1/(1+d)\)

### Keyword（verbatim）查询（rank-bm25）

- 输入：`query`
- 语料：`memory_exchanges.verbatim_text`（或更推荐：`verbatim_snippet` + 关键 metadata 拼接）
- 输出：`(exchange_id, keyword_score)`

#### LangChain 内置用法（来自 venv）

你的环境里已有 `langchain_community.retrievers.bm25.BM25Retriever`，底层使用 `rank_bm25.BM25Okapi`，并允许传入自定义 `preprocess_func`：

- `BM25Retriever.from_texts(texts, metadatas, ids, preprocess_func=...)`
- 召回：`vectorizer.get_top_n(processed_query, docs, n=k)`

> 关键点：默认 `preprocess_func` 只是 `text.split()`，中文必须替换为自定义分词（比如 jieba/自己写 tokenizer）。

#### 建议的 BM25 索引策略（工程化：默认窗口化 + 分片加载）

rank-bm25 是**内存索引**，为了避免全量 verbatim 导致内存不可控，`keyword_verbatim` 默认不做“全量常驻”，而是采用：

- **窗口化（Windowed corpus）**
  - 仅纳入最近一段时间或最近 N 条 exchanges（例如最近 30 天或最近 20k 条），由配置控制
  - 原则：keyword 信号主要覆盖“近期记忆”，远期记忆交给 dense(distilled) 兜底
- **分片加载（Shard by scope）**
  - 优先按 `session_id` 或 `project_id` 分片构建 BM25（每个 shard 一个 BM25）
  - 查询时基于 `filters.session_ids/project_id` 只加载相关 shard，避免全量常驻
- **批量重建（Batch rebuild, not per-message）**
  - 新 exchange 入库后只标记 shard “dirty”
  - 后台按阈值/间隔重建（例如累计新增 ≥ 500 条或每 10 分钟），避免每条新增都重建

> 说明：rank-bm25 不擅长真正的在线增量更新；工程上通常采用“批量重建 + 窗口化 + 分片”来控制成本。

#### BM25 语料字段（建议）

为了贴近论文“facets 提升 exact-term”效果，建议 BM25 语料不是纯 verbatim，而是：

```text
bm25_text = verbatim_snippet
         + "\nFILES: " + " ".join(files_touched)
         + "\nROOMS: " + " ".join(room_key + " " + room_label)
```

注意：这会让 keyword 信号更强，但也可能引入噪声；可通过配置开关控制是否拼 facets。

### 融合（RRF / weighted_sum）

建议先实现两种（与 `langchain_postgres/v2` 的思路对齐）：

- **RRF(k=60)**：对分数尺度不敏感，稳定
- **weighted_sum**：需要归一化（0-1），可调权重

默认建议：

- `dense_top_k = 200`
- `keyword_top_k = 200`（BM25）
- `final_top_k = top_k`（默认 10）
- fusion 默认 `RRF(k=60)`

---

## 与 Agent 集成方式（tools + middleware）

### Tool：`search_memory`

- 入参：`query, mode, top_k, filters`
- 返回：`hits[]`（含 `verbatim_snippet` + backref）

### Middleware：每轮自动注入（可选开关）

- 在 `AgentManager.astream()` 处理当前 user message 前：
  - 调用 `search_memory(query=message, mode=hybrid_cross, top_k=3)`
  - 将返回的 **verbatim_snippet** 注入一条 assistant 消息（注明 `session_id` + ply range）
- 不建议默认展示 `distill_text`（对齐论文：索引≠展示）

---

## 详细重构计划清单（按里程碑）

### 里程碑 A：数据契约与最小入库（MVP-1）

- [ ] 定义 domain models：Exchange、DistilledObject、SearchHit、Filters
- [ ] session_reader：读取 `memory_module_v1/sessions/*.json` -> NormalizedMessage[]
- [ ] exchange_segmenter：messages -> exchanges（含过滤、max_ply_len 切片）
- [ ] file_path_extractor：regex 提取 `files_touched`
- [ ] distiller（LLM）：Exchange -> {exchange_core, specific_context, room_assignments}
- [ ] 建表：按本文 SQL 创建 `memory_exchanges`、`memory_objects`
- [ ] distill_session API：幂等 upsert（按 exchange_id 去重）
- [ ] dense 检索：pgvector similarity 查询（先不做索引优化）

**验收标准**

- `distill_session(session_id)` 可重复执行不产生重复数据
- `search_memory(mode=dense_distilled)` 返回可回跳的 verbatim 证据片段

### 里程碑 B：Keyword + Hybrid（MVP-2）

- [ ] keyword 检索：BM25(rank-bm25) topK exchanges
- [ ] BM25 索引构建：从 `memory_exchanges` 拉语料并缓存（支持按 filters 重建/子索引）
- [ ] 融合器：RRF（优先）/ weighted_sum
- [ ] `search_memory(mode=hybrid_cross)`：dense(distilled) + keyword(verbatim) 融合输出

**验收标准**

- hybrid 结果存在 dense-only 与 keyword-only 的互补候选
- 命中均可回跳到原始 session 的 `[ply_start, ply_end]`

### 里程碑 C：集成 Agent（tools + middleware）

- [ ] `tools.search_memory` 暴露给 agent
- [ ] middleware 自动注入（加环境变量开关）
- [ ] 注入文本格式最小规范（source、ply range、snippet）

### 里程碑 D：增量与治理（MVP-3）

- [ ] 变更检测：session 的更新时间/消息数变化触发增量
- [ ] force 重蒸馏策略：重建某 session
- [ ] 统计与健康检查：对象数、exchange 数、索引状态
- [ ] 噪声清洗：terminal 输出/报错堆栈识别与剔除（提升 exchange 质量与 BM25）

---

## 实施注意事项（踩坑提示）

- **中文 keyword**：本方案使用 `rank-bm25`，效果高度依赖 tokenizer。务必实现 `preprocess_func`：
  - 最小可用：按空白/标点切分（对中文较差）
  - 推荐：引入中文分词（例如 jieba）并保留英文/路径 token（对代码/文件名类查询很重要）
  - 需要兼顾：中文词、英文标识符、snake_case、路径、报错关键字
- **向量索引**：数据量小时先别急着建 ivfflat/hnsw；先跑通与验证，再做性能优化（并配 `ANALYZE`）。
- **幂等主键**：exchange_id 必须 deterministic，否则增量与融合会很痛苦。
- **证据层优先**：对 agent 注入时优先 verbatim_snippet，distill_text 只做路由/解释（对齐论文）。

---

## 依赖与配置

### Python 依赖（新增）

keyword 侧使用 `rank-bm25`（LangChain 的 `BM25Retriever` 依赖它），中文分词建议用 `jieba`：

- `rank-bm25`
- `jieba`

建议把它们加入 `backend/requirements.txt`（或你的依赖管理文件）中。

### 环境变量（建议）

以下变量建议统一由 `backend/config/.env` 管理（名称可按你的习惯调整）：

- `BM25_INDEX_DIR`
  - **用途**：BM25 索引的落盘目录（缓存）。
  - **默认建议**：`./storage/memory_v2/bm25`（相对 backend 工作目录）。
  - **内容建议**：
    - `bm25.pkl`（序列化的 BM25Okapi / retriever.vectorizer）
    - `corpus.jsonl`（exchange_id + bm25_text + backref 元数据快照）
    - `index_meta.json`（built_at、corpus_size、source_max_updated_at 等）

- `BM25_REBUILD_ON_START`
  - **用途**：启动时是否强制重建 BM25（忽略缓存）。
  - **建议默认**：`false`

- `BM25_SHARDING`
  - **用途**：BM25 分片维度。
  - **可选值**：`none | session | project`
  - **建议默认**：`session`（你当前数据天然有 session_id，最容易落地）

- `BM25_MAX_DOCS`（可选）
  - **用途**：BM25 语料最大文档数限制（防止误把海量数据塞进内存）。
  - **建议默认**：空（不限制），或根据机器内存设置（例如 `50000`）。

- `BM25_WINDOW_DAYS`（推荐）
  - **用途**：窗口化语料的天数（只纳入最近 N 天的 exchanges）。
  - **建议默认**：`30`（或 `90`，看你对“长期 keyword recall”的需求）

- `BM25_REBUILD_MIN_NEW_DOCS`（推荐）
  - **用途**：触发一次批量重建所需的最小新增 exchanges 数（按 shard 统计）。
  - **建议默认**：`500`

- `BM25_REBUILD_MIN_SECONDS`（推荐）
  - **用途**：两次重建的最小时间间隔（防抖）。
  - **建议默认**：`600`（10 分钟）

- `BM25_USE_FACETS_IN_CORPUS`
  - **用途**：BM25 语料是否拼接 `files_touched/rooms`（提升 exact-term 命中，但可能引入噪声）。
  - **建议默认**：`true`

- `BM25_K` / `BM25_B`（可选）
  - **用途**：BM25Okapi 参数（k1、b）。
  - **建议默认**：使用库默认值；后续再调参。

> 说明：`rank-bm25` 的 pickle 缓存不保证跨 Python/依赖版本稳定，因此即使落盘也必须能随时重建；`index_meta.json` 用于判断是否需要重建。

### tokenizer（preprocess_func）推荐约定

LangChain `BM25Retriever` 需要 `preprocess_func(text) -> list[str]`。为了适配工程对话检索（中文 + 代码/路径），推荐 tokenizer 同时满足：

#### 必须保留的 token 类型

- **文件路径**：如 `backend/service/session_manager.py`、`C:\Users\...`、`./foo/bar.ts`
- **文件名**：如 `agent_factory.py`、`dev.md`
- **标识符**：snake_case、kebab-case、camelCase、PascalCase
- **报错/关键字**：如 `ImportError`、`KeyError`、`HTTP 401`、`trace_id`
- **版本/数字**：如 `0.4.0`、`2026-03-17`、`384`、`1536`

#### 推荐处理流程（实现约定）

1. **先抽取高价值 token（regex）**：路径/文件名/标识符/版本号（加入 token 列表，避免被中文分词吞掉）。
2. **对剩余文本做中文分词**：`jieba.lcut(text)`。
3. **英文/标识符再拆分**（可选但推荐）：
   - `snake_case` 按 `_` 拆分
   - `kebab-case` 按 `-` 拆分
   - `camelCase/PascalCase` 做大小写边界拆分（同时保留原 token）
4. **规范化**：
   - 全部 `lower()`（路径可保留原样 + lower 版各一份，二选一）
   - 去除空 token
   - 不要做激进停用词过滤（尤其别过滤 `.py`、`/`、`_` 等会影响路径/标识符的信号）

#### 推荐正则（用于“先抽取”）

- 路径/文件：匹配类似 `foo/bar/baz.py`、`baz.tsx`、`C:\a\b\c.go`
- 标识符：匹配 `[_A-Za-z][_A-Za-z0-9]*`
- 版本号：匹配 `\d+(\.\d+){1,3}`

> 目标不是 NLP 完美分词，而是保证“工程 recall”里用户最常搜的线索（文件/标识符/报错）稳定可检索。


