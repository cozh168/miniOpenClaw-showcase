"use client";

import { AlertCircle, CheckCircle2, ChevronDown, Code2, FileDown, LoaderCircle, TerminalSquare } from "lucide-react";
import { useEffect, useState } from "react";

import { CodeBlock, CopyButton } from "@/components/chat/CodeBlock";
import { createWorkspaceFile, type ToolCall } from "@/lib/api";

function getInput(input: string) {
  try {
    const parsed: unknown = JSON.parse(input);
    if (parsed && typeof parsed === "object" && "code" in parsed && typeof parsed.code === "string") return { text: parsed.code, code: true };
    return { text: JSON.stringify(parsed, null, 2), code: false };
  } catch { return { text: input, code: false }; }
}

function ToolTrace({ call, first, isPending }: { call: ToolCall; first: boolean; isPending: boolean }) {
  const running = call.completed !== true && !call.output && isPending;
  const failed = /(?:^|\n)\s*(?:Blocked:|(?:Read|Fetch) failed:|Timed out after|Traceback \(most recent call last\):|(?:[A-Za-z_][A-Za-z0-9_]*(?:Error|Exception)|Error|Exception):)/.test(call.output);
  const [open, setOpen] = useState(first || running);
  const [saving, setSaving] = useState(false);
  const [savedPath, setSavedPath] = useState("");
  const [saveError, setSaveError] = useState("");
  const input = getInput(call.input);
  const python = call.tool === "python_repl" || call.tool === "python";
  const saveable = python && input.code && Boolean(input.text.trim());
  useEffect(() => { if (running) setOpen(true); }, [running]);

  async function saveCode() {
    if (saving || savedPath) return;
    setSaving(true); setSaveError("");
    const path = `workspace/generated/${call.tool}-${Date.now()}-${Math.random().toString(36).slice(2, 6)}.py`;
    try {
      const result = await createWorkspaceFile(path, input.text);
      setSavedPath(result.path);
      window.dispatchEvent(new CustomEvent("workspace-files-changed", { detail: { path: result.path } }));
    } catch { setSaveError("保存失败，请检查工作区连接后重试。"); }
    finally { setSaving(false); }
  }

  return (
    <details className="group mb-4 min-w-0 overflow-hidden rounded-xl border border-[var(--color-line)] bg-white" onToggle={(event) => setOpen(event.currentTarget.open)} open={open}>
      <summary className="flex min-w-0 cursor-pointer list-none items-center gap-2.5 px-3.5 py-3 text-xs focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-[var(--color-ocean)]">
        <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-[var(--color-ocean)] text-white shadow-sm"><Code2 size={17} /></span>
        <span className="shrink-0 font-semibold text-[var(--color-ink)]">工具调用</span>
        <span className="mono min-w-0 truncate text-[11px] text-[#6d88aa]" title={call.tool}>{call.tool}</span>
        <span aria-label={running ? "执行中" : failed ? "执行异常" : "执行完成"} className={`ml-auto flex shrink-0 items-center gap-1 rounded-full px-2 py-1 text-[10px] ${failed ? "bg-[#fff1f0] text-[#b45661]" : "bg-[#e9faf5] text-[#168c87]"}`}>
          {running ? <LoaderCircle size={12} className="animate-spin motion-reduce:animate-none" /> : failed ? <AlertCircle size={12} /> : <CheckCircle2 size={12} />}
          <span className="hidden sm:inline">{running ? "执行中" : failed ? "执行异常" : "执行完成"}</span>
        </span>
        <ChevronDown aria-hidden="true" className={`shrink-0 text-[#6a87a7] transition-transform ${open ? "rotate-180" : ""}`} size={14} />
      </summary>
      <div className="space-y-2.5 border-t border-[var(--color-line)] p-3">
        {call.input && <div className="min-w-0 overflow-hidden rounded-lg bg-[#f0f6fa] p-2">
          <div className="mb-1.5 flex items-center gap-1.5 px-1 text-[11px] text-[#607f9f]"><Code2 className="text-[var(--color-ocean)]" size={12} /><span>输入 <span className="text-[#8fa4b9]">(Input)</span></span><div className="ml-auto"><CopyButton content={input.text} label={input.code ? "复制代码" : "复制输入"} /></div></div>
          <CodeBlock code={input.text} language={python && input.code ? "python" : "text"} numbered={input.code} maxHeight={330} />
          {saveable && <div className="flex flex-wrap items-center gap-2 px-1 pb-0.5 pt-2">
            <button className="flex items-center gap-1.5 rounded-md px-1 py-1 text-[10px] text-[#50758d] hover:text-[var(--color-ocean)] disabled:cursor-default" disabled={saving || Boolean(savedPath) || running} onClick={() => { void saveCode(); }} type="button">{saving ? <LoaderCircle size={12} className="animate-spin" /> : <FileDown size={12} />}{saving ? "正在保存…" : savedPath ? "已保存到工作区" : "保存到工作区"}</button>
            {savedPath && <span className="max-w-full break-all text-[10px] text-[#718a9b]" title={savedPath}>{savedPath.split("/").pop()}</span>}
            {saveError && <span className="text-[10px] text-[#b45661]" role="alert">{saveError}</span>}
          </div>}
        </div>}
        <div className="min-w-0 overflow-hidden rounded-lg bg-[#f0f6fa] p-2">
          <div className="mb-1.5 flex items-center gap-1.5 px-1 text-[11px] text-[#607f9f]"><TerminalSquare className="text-[var(--color-ocean)]" size={12} /><span>输出 <span className="text-[#8fa4b9]">(Output)</span></span>{Boolean(call.output) && <div className="ml-auto"><CopyButton content={call.output} label="复制结果" /></div>}</div>
          {running ? <p className="flex items-center gap-2 px-3 py-4 text-xs text-[#7992ab]" role="status"><LoaderCircle size={13} className="animate-spin" />正在等待工具返回结果…</p> : <CodeBlock code={call.output || "（无文本输出）"} maxHeight={270} />}
        </div>
      </div>
    </details>
  );
}

export function ThoughtChain({ toolCalls, isPending = false }: { toolCalls: ToolCall[]; isPending?: boolean }) {
  if (!toolCalls.length) return null;
  return <div className="min-w-0">{toolCalls.map((call, index) => <ToolTrace call={call} first={index === 0} isPending={isPending} key={`${call.tool}-${index}`} />)}</div>;
}
