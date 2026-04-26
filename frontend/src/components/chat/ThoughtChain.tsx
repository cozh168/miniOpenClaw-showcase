"use client";

import { TerminalSquare } from "lucide-react";

import type { ToolCall } from "@/lib/api";

export function ThoughtChain({ toolCalls }: { toolCalls: ToolCall[] }) {
  if (!toolCalls.length) {
    return null;
  }

  const toolNames = Array.from(new Set(toolCalls.map(tc => tc.tool))).join(", ");

  return (
    <details className="mb-4 rounded-3xl border border-[rgba(212,106,74,0.18)] bg-[rgba(212,106,74,0.08)] p-4">
      <summary className="flex cursor-pointer list-none items-center gap-2 text-sm font-medium text-[var(--color-ember)] outline-none">
        <TerminalSquare size={16} />
        <span>工具调用 {toolCalls.length} 次</span>
        {toolNames && <span className="text-[11px] opacity-70 font-normal ml-1">({toolNames})</span>}
      </summary>
      <div className="mt-3 space-y-3">
        {toolCalls.map((toolCall, index) => (
          <div className="rounded-2xl bg-white/70 p-3" key={`${toolCall.tool}-${index}`}>
            <div className="mb-2 text-sm font-medium">
              {toolCall.tool}
            </div>
            <div className="space-y-2 text-xs">
              {toolCall.input && (
                <div className="rounded-2xl bg-[rgba(13,37,48,0.06)] p-3">
                  <div className="mb-1 font-medium text-[var(--color-ink-soft)]">Input</div>
                  <pre className="mono whitespace-pre-wrap break-all">{toolCall.input}</pre>
                </div>
              )}
              {toolCall.output && (
                <div className="rounded-2xl bg-[rgba(13,37,48,0.06)] p-3">
                  <div className="mb-1 font-medium text-[var(--color-ink-soft)]">Output</div>
                  <div className="mono whitespace-pre-wrap break-all max-h-60 overflow-y-auto">{toolCall.output}</div>
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
    </details>
  );
}
