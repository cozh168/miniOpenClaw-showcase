# Postgres Checkpointer 使用说明

当 `CHECKPOINTER=postgres` 时，短期记忆使用 `langgraph-checkpoint-postgres` 的 `AsyncPostgresSaver`。

## 环境变量

- `CHECKPOINTER=postgres`：启用 Postgres
- `POSTGRES_DSN`：完整连接串，例如 `postgresql://user:password@localhost:5432/dbname`
- 或分别设置：`POSTGRES_HOST`、`POSTGRES_PORT`、`POSTGRES_USER`、`POSTGRES_PASSWORD`、`POSTGRES_DB`

## 建表

首次使用前，应用启动时会自动调用 `AsyncPostgresSaver.setup()` 创建以下表（由 langgraph-checkpoint-postgres 管理）：

- `checkpoints`：图状态快照（thread_id、checkpoint_id、checkpoint JSONB）
- `checkpoint_blobs`：大对象二进制
- `checkpoint_writes`：写入日志
- `checkpoint_migrations`：schema 版本

无需手动建表。若需在其它环境提前建表，可单独运行一次 `init_checkpointer_async()` 或参考 [langgraph-checkpoint-postgres](https://pypi.org/project/langgraph-checkpoint-postgres/)。
