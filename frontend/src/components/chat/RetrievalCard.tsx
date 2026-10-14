"use client";

import { BookOpen, ChevronDown, FileText } from "lucide-react";

import type { RetrievalResult } from "@/lib/api";

function getSummaryLabel(results: RetrievalResult[]) {
  const kinds = new Set(results.map((item) => item.kind ?? "memory"));
  if (kinds.size === 1 && kinds.has("knowledge")) return "文档片段";
  if (kinds.size === 1 && kinds.has("memory")) return "记忆片段";
  return "参考片段";
}

export function RetrievalCard({ results }: { results: RetrievalResult[] }) {
  if (!results.length) return null;

  return (
    <details className="group mb-4 overflow-hidden rounded-xl border border-[var(--color-line)] bg-white">
      <summary className="flex cursor-pointer list-none items-center gap-2 bg-[#f8fcff] px-3.5 py-3 text-xs text-[var(--color-ink-soft)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-[var(--color-ocean)]">
        <BookOpen aria-hidden="true" className="text-[var(--color-ocean)]" size={15} />
        <span className="font-medium text-[var(--color-ink)]">参考来源</span>
        <span className="ml-1 rounded-full bg-[#eaf4fa] px-2 py-0.5 text-[10px] text-[#7091ab]">{results.length} 条{getSummaryLabel(results)}</span>
        <ChevronDown aria-hidden="true" className="ml-auto shrink-0 transition-transform group-open:rotate-180" size={13} />
      </summary>
      <div className="divide-y divide-[var(--color-line)] border-t border-[var(--color-line)] px-3.5">
        {results.map((item, index) => (
          <div className="py-3" key={`${item.source}-${index}`}>
            <div className="mb-2 flex items-start gap-1.5 text-[11px] text-[var(--color-ink-soft)]">
              <FileText aria-hidden="true" className="mt-0.5 shrink-0" size={12} />
              <span className="min-w-0 flex-1 break-all">{item.source}</span>
              {Number.isFinite(item.score) && <span className="mono shrink-0 pl-2 text-[10px]">相关度 {item.score.toFixed(3)}</span>}
            </div>
            <p className="max-h-52 overflow-y-auto whitespace-pre-wrap break-words text-xs leading-6 text-[#5d7993]">{item.text}</p>
          </div>
        ))}
      </div>
    </details>
  );
}
