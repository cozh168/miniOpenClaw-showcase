# Agent Engineering Upgrades

This document summarizes the three resume-facing upgrades added on top of the original project.

## 1. Document-level RAG

What changed:

- added a reusable knowledge index service at `backend/service/knowledge_base.py`
- added a `search_knowledge_base` tool
- added runtime `rag_mode` integration to the chat loop
- added frontend toggle and retrieval cards for document hits

How to use:

1. Put local documents under `backend/knowledge/`
2. Start backend and frontend
3. Turn on `文档 RAG 已开` in the navbar
4. Ask questions about local docs such as:
   - `什么是 XSS？`
   - `未付款订单多久会自动取消？`

Optional config in `backend/config/.env`:

```env
KNOWLEDGE_RAG_TOP_K=3
KNOWLEDGE_RAG_CHUNK_SIZE=1200
KNOWLEDGE_RAG_CHUNK_OVERLAP=200
KNOWLEDGE_RAG_MAX_CHUNKS_PER_FILE=48
KNOWLEDGE_RAG_DENSE_ENABLED=true
```

## 2. Evaluation system

What changed:

- added `backend/evals/run_chat_eval.py`
- added dataset `backend/evals/datasets/knowledge_rag_cases.jsonl`
- added report output under `backend/evals/reports/`

Run:

```bash
cd backend
python evals/run_chat_eval.py --base-url http://127.0.0.1:8002 --mode compare
```

The script compares:

- `rag_mode = false`
- `rag_mode = true`

Metrics:

- success rate
- answer keyword hit rate
- retrieval hit rate
- average latency
- average retrieval count

## 3. Containerized deployment

What changed:

- added `backend/Dockerfile`
- added `frontend/Dockerfile`
- added root `docker-compose.yml`
- added root `.dockerignore`

Run:

```bash
docker compose up --build
```

Services:

- frontend: `http://127.0.0.1:7788`
- backend: `http://127.0.0.1:8002`

The compose file keeps local project state mounted for:

- `backend/config`
- `backend/knowledge`
- `backend/workspace`
- `backend/skills`
- `backend/memory_module_v1`
- `backend/sessions`
- `backend/storage`

## 4. Tool permissions and audit logging

What changed:

- added a reusable security layer at `backend/service/tool_security.py`
- added configurable enable/disable flags for `terminal`, `python_repl`, `fetch_url`, and `read_file`
- added SSRF-style host blocking for `fetch_url`
- added sensitive path blocking for `read_file`
- added JSONL audit logs under `backend/storage/tool_audit/tool_runs.jsonl`
- added API endpoints `GET /api/tool-security` and `GET /api/tool-audit`

How to use:

1. Start the backend
2. Inspect the current policy at `http://127.0.0.1:8002/api/tool-security`
3. Inspect recent tool runs at `http://127.0.0.1:8002/api/tool-audit?limit=20`

Optional config in `backend/config/.env`:

```env
TOOL_AUDIT_ENABLED=true
TOOL_AUDIT_MAX_ENTRIES=500
TERMINAL_ALLOWED_COMMANDS=Get-ChildItem,Get-Content,Select-String,python,pytest,uvicorn,npm,node,git
READ_FILE_ALLOWED_PREFIXES=backend/,frontend/,docs/,skills/,workspace/,knowledge/
READ_FILE_BLOCKED_PREFIXES=backend/config/.env,backend/storage/,frontend/node_modules/
FETCH_URL_ALLOWED_HOSTS=docs.python.org,platform.openai.com
FETCH_URL_ALLOW_PRIVATE_HOSTS=false
```
