from __future__ import annotations

import argparse
import asyncio
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import httpx


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    question: str
    expected_sources: list[str]
    answer_keywords: list[str]


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    question: str
    success: bool
    latency_ms: float | None
    answer_keyword_hit: bool
    retrieval_hit: bool
    retrieval_count: int
    final_answer: str
    retrieval_sources: list[str]
    error: str | None


async def iter_sse_events(response: httpx.Response):
    event = "message"
    data_lines: list[str] = []
    async for line in response.aiter_lines():
        if not line:
            if data_lines:
                yield event, json.loads("\n".join(data_lines))
            event = "message"
            data_lines = []
            continue
        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].strip())

    if data_lines:
        yield event, json.loads("\n".join(data_lines))


def load_cases(dataset_path: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    for raw_line in dataset_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        cases.append(
            EvalCase(
                case_id=str(payload["id"]),
                question=str(payload["question"]),
                expected_sources=[str(item) for item in payload.get("expected_sources", [])],
                answer_keywords=[str(item) for item in payload.get("answer_keywords", [])],
            )
        )
    return cases


async def get_rag_mode(client: httpx.AsyncClient) -> bool:
    response = await client.get("/api/config/rag-mode")
    response.raise_for_status()
    return bool(response.json()["enabled"])


async def set_rag_mode(client: httpx.AsyncClient, enabled: bool) -> None:
    response = await client.put("/api/config/rag-mode", json={"enabled": enabled})
    response.raise_for_status()


async def create_session(client: httpx.AsyncClient, title: str) -> str:
    response = await client.post("/api/sessions", json={"title": title})
    response.raise_for_status()
    return str(response.json()["id"])


def answer_keyword_hit(answer: str, keywords: list[str]) -> bool:
    if not keywords:
        return True
    normalized_answer = answer.lower()
    return all(keyword.lower() in normalized_answer for keyword in keywords)


def retrieval_hit(retrieval_sources: list[str], expected_sources: list[str]) -> bool:
    if not expected_sources:
        return True
    return any(
        expected_source in actual_source
        for expected_source in expected_sources
        for actual_source in retrieval_sources
    )


async def run_case(client: httpx.AsyncClient, case: EvalCase) -> CaseResult:
    session_id = await create_session(client, title=f"eval-{case.case_id}")
    final_answer_parts: list[str] = []
    retrieval_sources: list[str] = []
    retrieval_count = 0
    error: str | None = None
    latency_ms: float | None = None
    started_at = perf_counter()

    async with client.stream(
        "POST",
        "/api/chat",
        json={
            "message": case.question,
            "session_id": session_id,
            "stream": True,
        },
    ) as response:
        response.raise_for_status()
        async for event, data in iter_sse_events(response):
            if event == "retrieval":
                results = data.get("results", []) or []
                retrieval_count += len(results)
                retrieval_sources.extend(
                    str(item.get("source", ""))
                    for item in results
                    if isinstance(item, dict)
                )
                continue

            if event == "token":
                final_answer_parts.append(str(data.get("content", "")))
                continue

            if event == "done":
                latency_ms = round((perf_counter() - started_at) * 1000, 2)
                if not final_answer_parts and data.get("content"):
                    final_answer_parts.append(str(data["content"]))
                continue

            if event == "error":
                error = str(data.get("error", "unknown error"))

    final_answer = "".join(final_answer_parts).strip()
    return CaseResult(
        case_id=case.case_id,
        question=case.question,
        success=error is None and bool(final_answer),
        latency_ms=latency_ms,
        answer_keyword_hit=answer_keyword_hit(final_answer, case.answer_keywords),
        retrieval_hit=retrieval_hit(retrieval_sources, case.expected_sources),
        retrieval_count=retrieval_count,
        final_answer=final_answer,
        retrieval_sources=retrieval_sources,
        error=error,
    )


def summarize(results: list[CaseResult]) -> dict[str, Any]:
    latencies = [result.latency_ms for result in results if result.latency_ms is not None]
    return {
        "case_count": len(results),
        "success_rate": round(
            sum(1 for result in results if result.success) / len(results), 4
        )
        if results
        else 0.0,
        "answer_keyword_hit_rate": round(
            sum(1 for result in results if result.answer_keyword_hit) / len(results), 4
        )
        if results
        else 0.0,
        "retrieval_hit_rate": round(
            sum(1 for result in results if result.retrieval_hit) / len(results), 4
        )
        if results
        else 0.0,
        "avg_latency_ms": round(statistics.mean(latencies), 2) if latencies else None,
        "avg_retrieval_count": round(
            statistics.mean(result.retrieval_count for result in results), 2
        )
        if results
        else 0.0,
    }


async def run_mode(
    client: httpx.AsyncClient,
    cases: list[EvalCase],
    *,
    rag_enabled: bool,
) -> dict[str, Any]:
    await set_rag_mode(client, rag_enabled)
    mode_results: list[CaseResult] = []
    for case in cases:
        mode_results.append(await run_case(client, case))

    return {
        "rag_enabled": rag_enabled,
        "summary": summarize(mode_results),
        "results": [asdict(result) for result in mode_results],
    }


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run document-RAG evaluation against a live Mini-OpenClaw backend."
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8002",
        help="Backend base URL, for example http://127.0.0.1:8002",
    )
    parser.add_argument(
        "--dataset",
        default=str(Path(__file__).with_name("datasets") / "knowledge_rag_cases.jsonl"),
        help="Path to a JSONL dataset of chat eval cases.",
    )
    parser.add_argument(
        "--mode",
        choices=["compare", "rag_on", "rag_off"],
        default="compare",
        help="Compare rag on/off, or run a single mode.",
    )
    parser.add_argument(
        "--report",
        default="",
        help="Optional output report path. Defaults to backend/evals/reports/chat_eval_report.json",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset).resolve()
    cases = load_cases(dataset_path)
    if not cases:
        raise SystemExit(f"No eval cases found in {dataset_path}")

    report_path = (
        Path(args.report).resolve()
        if args.report
        else Path(__file__).with_name("reports") / "chat_eval_report.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient(base_url=args.base_url, timeout=120.0) as client:
        try:
            health = await client.get("/health")
            health.raise_for_status()
        except httpx.HTTPError as exc:
            raise SystemExit(
                f"Backend is not reachable at {args.base_url}. Start the API first, then rerun the eval. ({exc})"
            ) from exc
        original_rag_mode = await get_rag_mode(client)

        try:
            if args.mode == "compare":
                modes = [False, True]
            elif args.mode == "rag_on":
                modes = [True]
            else:
                modes = [False]

            mode_reports: list[dict[str, Any]] = []
            for rag_enabled in modes:
                mode_reports.append(await run_mode(client, cases, rag_enabled=rag_enabled))
        finally:
            await set_rag_mode(client, original_rag_mode)

    report = {
        "base_url": args.base_url,
        "dataset": str(dataset_path),
        "modes": mode_reports,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    console_report = json.dumps(report, ensure_ascii=True, indent=2)
    print(console_report)
    print(f"\nSaved report to: {report_path}")


if __name__ == "__main__":
    asyncio.run(main())
