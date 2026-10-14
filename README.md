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
- **前后端工作台**：FastAPI + SSE 流式输出，Next.js 三栏界面，支持会话搜索、文件引用、工作区编辑、工具代码保存、运行日志、回收站和移动端布局。

## 效果展示

### 工作台页面

![新版 Mini-OpenClaw 三栏工作台](./docs/screenshots/workbench.png)

新版截图使用示例会话与文件数据展示布局，不包含私人会话或密钥。

![工作台细节](./Snipaste_2026-03-18_21-42-43.png)

### Guardian 拦截与 Langfuse 追踪

![Guardian prompt injection 测试](./guardian_test.png)

![Langfuse 中的 Guardian 追踪](./guardian_test_langfuse.png)

### 结构化记忆示意

![数据库结构](./database_schema.png)

## 技术栈

### 后端

- Python 3.11（本地与 CI 使用的版本）
- FastAPI
- LangChain 1.x / LangGraph
- OpenAI-compatible Chat / Embedding API
- Postgres / pgvector
- BM25 / RRF hybrid retrieval
- Langfuse

### 前端

- Node.js 20 LTS / npm
- Next.js 15 App Router
- React 18
- TypeScript
- Tailwind CSS 4
- Monaco Editor

## 快速开始

从仓库根目录开始操作。后端和前端需要分别在两个终端运行。默认地址仅供本机使用：前端 `127.0.0.1:7788`，后端 `127.0.0.1:8002`。

### 1. 安装后端并配置模型

macOS / Linux：

```bash
cd backend
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
if [ ! -f config/.env ]; then cp config/.env.example config/.env; fi
```

Windows PowerShell：

```powershell
cd backend
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
if (-not (Test-Path config\.env)) { Copy-Item config\.env.example config\.env }
```

如果已经有 `config/.env`，保留现有文件，不要再次复制覆盖。使用 DeepSeek 的最简配置如下，请在 `.env` 中修改相应项：

```env
LLM_PROVIDER=deepseek
LLM_MODEL=deepseek-chat
LLM_API_KEY=
DEEPSEEK_API_KEY=填写你的DeepSeek密钥
GUARDIAN_ENABLED=false
MEMORY_BACKEND=off
KNOWLEDGE_RAG_DENSE_ENABLED=false
EMBEDDING_API_KEY=
BAILIAN_API_KEY=
DASHSCOPE_API_KEY=
```

模板中的 `BAILIAN_API_KEY` / `DASHSCOPE_API_KEY` 是占位符；最简配置需要清空，避免启动时尝试无效的 Embedding 请求。此配置使用 DeepSeek 进行聊天，暂不启用需要独立模型密钥的 Guardian、长期记忆或稠密检索。文档 RAG 的开关保存在 `backend/config/config.json`，默认关闭，也可在界面中切换。需要这些功能时，再按下文配置对应服务和密钥。

在已激活虚拟环境的后端终端启动：

```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8002
```

打开 [健康检查](http://127.0.0.1:8002/health)，应看到 `{"status":"ok"}`。健康检查确认进程运行；模型密钥是否有效需发送一条实际消息验证。开发时可给启动命令加 `--reload`。

### 2. 启动前端

另开终端，从仓库根目录执行：

```bash
cd frontend
npm ci
npm run build
npm run start -- --hostname 127.0.0.1
```

访问 [http://127.0.0.1:7788](http://127.0.0.1:7788)。开发界面时使用 `npm run dev -- --hostname 127.0.0.1`。重新执行 `npm run build` 前，先按 `Ctrl+C` 停止正在运行的前端服务，构建完成后再启动，避免旧进程读取新的 `.next` 资源造成页面或样式异常。

前端 API 地址使用浏览器当前主机名和固定端口 `8002`，因此前后端应使用同一个主机名。这里没有单独的前端密钥配置；模型密钥只放在后端 `.env`。当前后端没有用户登录，Python 和终端工具以本地后端进程权限执行，请保持默认本机绑定，不要直接暴露到公网。

### 3. Docker Compose

先完成 `.env` 配置。Docker 镜像构建不包含 `.env`；Compose 在运行时读取它。

```bash
docker compose up --build
```

默认服务：

- Frontend: [http://127.0.0.1:7788](http://127.0.0.1:7788)
- Backend: [http://127.0.0.1:8002](http://127.0.0.1:8002)

Compose 同样只向本机发布端口。工作区、知识库、会话与索引通过目录挂载保存在宿主机。当前 Compose 不自动提供 PostgreSQL；开启 memory v2 或 Postgres checkpointer 时需要自行配置可访问的数据库。

### 常见启动问题

- **页面只有文字或样式丢失**：停止前端，在 `frontend/` 重新运行 `npm run build`，然后 `npm run start -- --hostname 127.0.0.1`；浏览器强制刷新。
- **聊天被 Guardian 拦截或提示鉴权失败**：仅配置 DeepSeek 时，检查最简配置中的 `GUARDIAN_ENABLED=false`；启用 Guardian 时必须填写它自己的 `GUARDIAN_API_KEY`。
- **端口被占用**：停止已有的项目进程后再启动。前端 API 固定使用 `8002`，不要只修改后端端口。
- **跨域请求返回 403**：默认仅允许 `localhost` 和 `127.0.0.1` 的项目端口。如确需其他前端地址，在后端 `.env` 配置 `BACKEND_ALLOWED_ORIGINS=http://192.168.1.10:7788` 后重启后端；多个来源用逗号分隔，不支持通配符或路径。额外来源配置不会添加登录鉴权。

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
TERMINAL_ALLOWED_COMMANDS=ls,cat,pwd,Get-ChildItem,Get-Content,Select-String,python,pytest,uvicorn,npm,node,git
READ_FILE_ALLOWED_PREFIXES=backend/,frontend/,docs/,skills/,workspace/,knowledge/
READ_FILE_BLOCKED_PREFIXES=backend/config/.env,backend/storage/,frontend/node_modules/
FETCH_URL_BLOCKED_HOSTS=localhost,127.0.0.1
FETCH_URL_ALLOW_PRIVATE_HOSTS=false
```

上面的命令白名单是可选示例。`TERMINAL_ALLOWED_COMMANDS` 留空时不限制命令名；填写后只允许列表内的命令。此策略不提供操作系统沙箱，Python 工具仍以运行后端的本地用户权限执行。

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

GitHub CI 会检查后端测试、前端类型/Lint/SSE 回归、依赖审计，并构建和启动 Docker 服务验证健康端点。

后端测试使用临时目录和模拟模型，不需要真实 API 密钥或 PostgreSQL：

```bash
cd backend
python -m pytest tests -q
```

前端检查：

```bash
cd frontend
npm ci
npm run lint
npx tsc --noEmit
npm run build
```

GitHub Actions 对 pull request 和 `main` 自动执行后端依赖安装/测试及前端类型、Lint 和构建检查。后端覆盖 Guardian、工具策略、Prompt Builder、知识库、文件路径与内容校验、回收站、并发会话与流式响应。实际模型接入和浏览器交互仍需在本地使用真实配置验证。

`.gitignore` 排除密钥、会话、索引、知识库、本地构建产物和工具生成的 `backend/workspace/generated/`。`workspace/` 中的身份/用户/系统提示文件及长期记忆模板属于源文件，编辑成个人资料后，在提交前检查 diff；不要把个人信息或密钥写入准备发布的模板。

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


