# memory_module_v2

基于论文 **Structured Distillation for Personalized Agent Memory** 的两层记忆架构。

## 快速开始

### 1. 初始化数据库

确保 PostgreSQL 运行且已安装 pgvector 扩展：

```bash
# 在 Postgres 中执行
CREATE EXTENSION IF NOT EXISTS vector;
```

然后在 Python 中初始化 schema：

```python
from memory_module_v2.storage.pg import ensure_schema
ensure_schema()
```

或直接执行 SQL：

```bash
psql -f backend/memory_module_v2/storage/schema.sql
```

### 2. 蒸馏 Session

```python
from memory_module_v2.service.api import distill_session

result = distill_session("your_session_id")
print(f"Exchanges: {result.exchanges_total}, Objects: {result.objects_created}")
```

批量蒸馏所有 sessions：

```python
from memory_module_v2.ingest.session_reader import list_session_ids
from memory_module_v2.service.api import distill_session

for sid in list_session_ids():
    result = distill_session(sid)
    print(f"{sid}: {result.objects_created} objects created")
```

### 3. 检索记忆

```python
from memory_module_v2.service.api import search_memory
from memory_module_v2.domain.enums import SearchMode

# Hybrid 检索（推荐）
response = search_memory("如何配置 LLM provider", mode=SearchMode.HYBRID_CROSS, top_k=5)

for hit in response.hits:
    print(f"[{hit.rank}] score={hit.scores['fused']:.3f}")
    print(f"  session={hit.session_id[:12]}... ply={hit.ply_start}-{hit.ply_end}")
    print(f"  {hit.verbatim_snippet[:200]}")
    print()
```

### 4. 证据回跳

```python
from memory_module_v2.service.api import get_exchange

evidence = get_exchange("session_id", ply_start=0, ply_end=5)
print(evidence.verbatim_snippet)
```

### 5. Agent 集成

在 `.env` 中启用：

```
MEMORY_V2_ENABLED=true
MEMORY_V2_AUTO_INJECT=true
MEMORY_V2_INJECT_TOP_K=3
```

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MEMORY_V2_ENABLED` | `false` | 启用 v2 记忆模块 |
| `MEMORY_V2_AUTO_INJECT` | `false` | 每轮自动检索注入 |
| `MEMORY_V2_INJECT_TOP_K` | `3` | 自动注入的 top_k |
| `BM25_INDEX_DIR` | `./storage/memory_v2/bm25` | BM25 索引缓存目录 |
| `BM25_WINDOW_DAYS` | `30` | BM25 语料窗口天数 |
| `BM25_MAX_DOCS` | `50000` | BM25 最大文档数 |
| `BM25_SHARDING` | `session` | BM25 分片策略 |
| `BM25_REBUILD_ON_START` | `false` | 启动时强制重建 BM25 |
| `BM25_REBUILD_MIN_NEW_DOCS` | `500` | 触发重建的最小新增数 |
| `BM25_REBUILD_MIN_SECONDS` | `600` | 两次重建最小间隔(秒) |
| `BM25_USE_FACETS_IN_CORPUS` | `true` | BM25 语料拼接 facets |

## 常见故障排查

### pgvector 扩展不可用

```
Error: extension "vector" is not available
```

需要在 PostgreSQL 中安装 pgvector。参考 [pgvector 安装文档](https://github.com/pgvector/pgvector)。

### BM25 索引为空

检查是否已蒸馏 session 并确保 `memory_v2.memory_exchanges` 表中有数据：

```sql
SELECT count(*) FROM memory_v2.memory_exchanges;
```

### Embedding 维度不匹配

schema.sql 中默认使用 `vector(1024)`，如果你的 embedding 模型维度不同，修改 schema.sql 中的维度并重建表。

## 架构

```
索引层 (distilled objects) ──pgvector──> dense 检索
                                           │
                                           ├─ RRF 融合 ──> search_memory
                                           │
证据层 (verbatim exchanges) ──BM25──> keyword 检索
```

- **索引层**：蒸馏后的结构化对象，用于路由与排序
- **证据层**：原始对话片段，用于展示与注入
- 检索命中后返回 verbatim evidence，不展示蒸馏文本
