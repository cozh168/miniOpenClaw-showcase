# Mini-OpenClaw

Mini-OpenClaw 是一个本地运行、文件优先、可审计的 AI Agent 工作台。项目目标不是再做一个聊天壳子，而是把 Agent 落地时常见的工程问题放到一个可运行的系统里：长期记忆、工具调用边界、知识库检索、安全前置检查、可观测性和评测。

它适合用来学习和展示 Agent Runtime 的工程化设计，也适合作为 Agent 开发工程师面试项目。

## 功能亮点

- **Agent 编排**：基于 LangChain 1.x `create_agent` 构建 Agent Graph，支持工具调用、系统提示词拼接、可选 checkpointer 和 summarization middleware。
- **长期记忆**：支持 v1 Chroma/MEMORY.md 检索，以及 v2 结构化蒸馏记忆。v2 将原始对话证据和可检索记忆对象分层存储，支持 `pgvector + BM25 + RRF` 混合检索。
- **文档级 RAG**：可索引 `backend/knowledge/` 下的 Markdown、TXT、JSON、CSV、PDF、XLSX 文件，并在前端展示检索命中来源。
- **工具安全层**：对 `terminal`、`python_repl`、`fetch_url`、`read_file` 等工具提供启停配置、命令白名单、路径限制、SSRF 防护和 JSONL 审计日志。
- **Guardian 前置安全检查**：在 Agent 执行前检测 prompt injection、越权指令、系统提示词/密钥窃取等风险，支持 fail-open / fail-closed 策略。
- **可观测性**：可接入 Langfuse 追踪 prompt、工具轨迹、记忆检索、RAG 命中和 token usage。
- **评测体系**：提供 RAG 对比评测脚本，统计成功率、关键词命中率、检索命中率、平均延迟和平均检索数量。
- **前后端工作台**：FastAPI + SSE 流式输出，Next.js 三栏界面，支持会话列表、聊天区、文件/记忆/技能检查器。

## 效果展示

### 工作台页面

![Mini-OpenClaw 工作台](./page_show1.png)

![工作台细节](./Snipaste_2026-03-18_21-42-43.png)

### Guardian 拦截与 Langfuse 追踪

![Guardian prompt injection 测试](./guardian_test.png)

![Langfuse 中的 Guardian 追踪](./guardian_test_langfuse.png)

### 结构化记忆示意

![数据库结构](./database_schema.png)

## 技术栈

### 后端

- Python 3.10+
- FastAPI
- LangChain 1.x / LangGraph
- OpenAI-compatible Chat / Embedding API
- Postgres / pgvector
- BM25 / RRF hybrid retrieval
- Langfuse

### 前端

- Next.js 14 App Router
- React 18
- TypeScript
- Tailwind CSS
- Monaco Editor

## 快速开始

### 1. 启动后端

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy config\.env.example config\.env
uvicorn app:app --host 0.0.0.0 --port 8002 --reload
```

按需在 `backend/config/.env` 中配置模型、Embedding、Guardian、Langfuse、Postgres 等参数。

### 2. 启动前端

```bash
cd frontend
npm install
npm run dev
```

访问 [http://localhost:7788](http://localhost:7788)。

### 3. Docker Compose

```bash
docker compose up --build
```

默认服务：

- Frontend: [http://127.0.0.1:7788](http://127.0.0.1:7788)
- Backend: [http://127.0.0.1:8002](http://127.0.0.1:8002)

## 核心配置

### 模型配置

```env
LLM_PROVIDER=zhipu
LLM_MODEL=glm-5
LLM_API_KEY=
LLM_BASE_URL=

EMBEDDING_PROVIDER=bailian
EMBEDDING_MODEL=text-embedding-v4
EMBEDDING_API_KEY=
EMBEDDING_BASE_URL=
```

已支持的对话模型接入方式包括：智谱 `zhipu`、百炼 `bailian`、DeepSeek `deepseek`、OpenAI-compatible `openai`。

### 长期记忆

```env
MEMORY_BACKEND=v2
MEMORY_V2_INJECT=tool
MEMORY_V2_INJECT_TOP_K=3
```

`MEMORY_V2_INJECT` 可选值：

| 值 | 含义 |
| --- | --- |
| `tool` | 注册 `search_memory` 工具，由 Agent 自主决定何时检索 |
| `always` | 每轮自动注入记忆检索上下文 |
| `off` | 启用 v2 API 能力，但不自动注入 Agent |

### Guardian

```env
GUARDIAN_ENABLED=true
GUARDIAN_PROVIDER=openai
GUARDIAN_MODEL=gpt-4.1-mini
GUARDIAN_API_KEY=
GUARDIAN_BASE_URL=https://api.openai.com/v1
GUARDIAN_TIMEOUT_MS=1500
GUARDIAN_FAIL_MODE=closed
```

`GUARDIAN_FAIL_MODE=closed` 表示 Guardian 调用失败时拦截请求；`open` 表示失败时放行。

### 文档 RAG

```env
KNOWLEDGE_RAG_TOP_K=3
KNOWLEDGE_RAG_CHUNK_SIZE=1200
KNOWLEDGE_RAG_CHUNK_OVERLAP=200
KNOWLEDGE_RAG_MAX_CHUNKS_PER_FILE=48
KNOWLEDGE_RAG_DENSE_ENABLED=true
```

把本地文档放到 `backend/knowledge/`，前端打开文档 RAG 开关后即可检索。

### 工具权限与审计

```env
TOOL_AUDIT_ENABLED=true
TOOL_AUDIT_MAX_ENTRIES=500
TERMINAL_TOOL_ENABLED=true
PYTHON_REPL_TOOL_ENABLED=true
FETCH_URL_TOOL_ENABLED=true
READ_FILE_TOOL_ENABLED=true
TERMINAL_ALLOWED_COMMANDS=Get-ChildItem,Get-Content,Select-String,python,pytest,uvicorn,npm,node,git
READ_FILE_ALLOWED_PREFIXES=backend/,frontend/,docs/,skills/,workspace/,knowledge/
READ_FILE_BLOCKED_PREFIXES=backend/config/.env,backend/storage/,frontend/node_modules/
FETCH_URL_BLOCKED_HOSTS=localhost,127.0.0.1
FETCH_URL_ALLOW_PRIVATE_HOSTS=false
```

相关接口：

- `GET /api/tool-security`：查看当前工具策略
- `GET /api/tool-audit?limit=20`：查看最近工具运行记录

## 评测

项目内置文档 RAG 对比评测：

```bash
cd backend
python evals/run_chat_eval.py --base-url http://127.0.0.1:8002 --mode compare
```

评测会对比：

- `rag_mode = false`
- `rag_mode = true`

报告输出到：

```text
backend/evals/reports/chat_eval_report.json
```

当前示例报告中，RAG 开启后检索命中率从 `0.0` 提升到 `1.0`。

## 测试

```bash
cd backend
python -m pytest tests -q
```

当前后端测试覆盖 Guardian、工具安全、Prompt Builder、知识库检索等模块。

## 项目结构

```text
miniOpenClaw/
├── backend/
│   ├── api/                  # Chat、会话、文件、配置、工具安全等接口
│   ├── config/               # 环境变量与运行时配置
│   ├── graph/                # Agent factory、Guardian、LLM、checkpointer
│   ├── memory_module_v1/     # 旧版长期记忆
│   ├── memory_module_v2/     # 结构化蒸馏、Postgres、pgvector、BM25、融合检索
│   ├── service/              # RAG、prompt、session、tool security 等服务
│   ├── tools/                # terminal、python_repl、fetch_url、read_file 等工具
│   ├── skills/               # Skill 目录与 SKILL.md
│   ├── knowledge/            # 本地知识库文档
│   ├── evals/                # RAG 评测脚本、数据集与报告
│   └── app.py                # FastAPI 入口
├── frontend/
│   └── src/                  # Next.js 前端
├── docs/                     # 设计文档
└── docker-compose.yml
```

## 面试讲法

这个项目可以概括为：

> 我做了一个本地可审计的 Agent 工作台，重点解决 Agent 落地中的四类问题：记忆如何持久化和回溯、工具调用如何加边界、安全检查如何前置、以及 RAG/Agent 行为如何评测和观测。

推荐重点讲：

- `backend/graph/agent.py`：SSE 流式对话、RAG/记忆注入、工具事件输出
- `backend/graph/agent_factory.py`：Agent middleware、checkpointer、summarization 组合
- `backend/graph/guardian.py`：before_agent 安全拦截
- `backend/service/tool_security.py`：工具权限与审计
- `backend/service/knowledge_base.py`：本地文档 RAG
- `backend/evals/run_chat_eval.py`：RAG 对比评测

## Roadmap

- 会话压缩策略进一步产品化
- 增加 sub-agent 分工与任务队列
- 引入更完整的工具权限 UI
- 扩展 Agent eval 数据集和失败样例分析
- 将 memory v2 的重建、增量更新和检索指标做成可视化面板

## 致谢

- 初版项目思路参考：[lyxhnu/langchain-miniopenclaw](https://github.com/lyxhnu/langchain-miniopenclaw)
- 结构化长期记忆方案参考项目内 `docs/Structured Distillation for Personalized Agent Memory.pdf`
