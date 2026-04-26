# Eval Toolkit

This directory turns the project into something you can measure, not only demo.

## 1. Document RAG eval

Run the backend first, then execute:

```bash
cd backend
python evals/run_chat_eval.py --base-url http://127.0.0.1:8002 --mode compare
```

The script will:

- toggle `rag_mode` off and on
- create isolated sessions for each case
- send questions through `/api/chat`
- record latency, retrieval hits, and answer-keyword hits
- write a JSON report to `backend/evals/reports/chat_eval_report.json`

Dataset location:

```text
backend/evals/datasets/knowledge_rag_cases.jsonl
```

Each line is one case:

```json
{"id":"xss-basics","question":"什么是 XSS？","expected_sources":["knowledge/Safety Knowledge/XSS.md"],"answer_keywords":["XSS","脚本"]}
```

## 2. Existing memory v2 eval

The repository already includes memory-focused utilities under:

```text
backend/memory_module_v2/eval/
```

Those scripts are still useful for `memory_module_v2` retrieval quality, while
`run_chat_eval.py` focuses on end-to-end chat behavior and document RAG.
