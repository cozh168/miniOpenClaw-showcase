"use client";

import { AlertCircle, Check, CheckCircle2, Copy, FileText, LoaderCircle } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { RetrievalCard } from "@/components/chat/RetrievalCard";
import { parseFileReferences } from "@/components/chat/FileReferencePicker";
import { ThoughtChain } from "@/components/chat/ThoughtChain";
import { BrandMark } from "@/components/layout/BrandMark";
import type { RetrievalResult, ToolCall } from "@/lib/api";

function fallbackCopy(content: string) {
  const previousFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null;
  const textarea = document.createElement("textarea");
  textarea.value = content;
  textarea.style.position = "fixed";
  textarea.style.opacity = "0";
  textarea.setAttribute("readonly", "");
  document.body.appendChild(textarea);
  try {
    textarea.select();
    if (!document.execCommand("copy")) throw new Error("Clipboard unavailable");
  } finally {
    textarea.remove();
    previousFocus?.focus({ preventScroll: true });
  }
}

export function ChatMessage({
  role,
  content,
  toolCalls,
  retrievals,
  isPending = false
}: {
  role: "user" | "assistant";
  content: string;
  toolCalls: ToolCall[];
  retrievals: RetrievalResult[];
  isPending?: boolean;
}) {
  const isUser = role === "user";
  const [copyState, setCopyState] = useState<"idle" | "copied" | "error">("idle");
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const showContent = Boolean(content.trim()) && (isUser || !toolCalls.some((call) => call.output.trim() === content.trim()));
  const responseFailed = /(?:^|\n)(?:请求失败[:：]|发送失败[,，]|模型请求失败)/.test(content);

  useEffect(() => () => {
    if (timerRef.current) clearTimeout(timerRef.current);
  }, []);

  async function copyAnswer() {
    try {
      try {
        if (!navigator.clipboard) throw new Error("Clipboard unavailable");
        await navigator.clipboard.writeText(content);
      } catch {
        fallbackCopy(content);
      }
      setCopyState("copied");
    } catch {
      setCopyState("error");
    }
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => setCopyState("idle"), 2500);
  }

  if (isUser) {
    const parsed = parseFileReferences(content);
    return (
      <article aria-label="你的消息" className="ml-auto flex w-fit min-w-0 max-w-[94%] items-start gap-3 md:max-w-[89%]">
        <span className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-[#3483f8] text-[12px] font-medium text-white shadow-[0_2px_7px_#3483f826]" aria-hidden="true">U</span>
        <div className="min-w-0 rounded-xl rounded-tl-md bg-[#eaf3ff] px-4 py-2.5 text-[13px] leading-6 text-[var(--color-ink)]">
          <div className="whitespace-pre-wrap break-words">{parsed.message}</div>
          {parsed.files.length > 0 && <div className="mt-2.5 space-y-1.5 border-t border-[#d4e4f6] pt-2.5">
            {parsed.files.map((file) => <details className="min-w-0" key={file.path}>
              <summary className="flex cursor-pointer list-none items-center gap-1.5 text-[10px] text-[#4e769b]"><FileText className="shrink-0" size={12} /><span className="min-w-0 truncate" title={file.path}>{file.path}</span><span className="ml-auto shrink-0">{file.truncated ? "部分内容" : "已引用"}</span></summary>
              <pre className="mono mt-2 max-h-44 overflow-auto whitespace-pre-wrap break-words rounded-md bg-white/60 p-2.5 text-[10px] leading-5 text-[#57708c]">{file.content || "（空文件）"}</pre>
            </details>)}
          </div>}
        </div>
      </article>
    );
  }

  return (
    <article aria-label="助手的回复" className="flex min-w-0 items-start gap-3">
      <div className="mt-0.5 shrink-0"><BrandMark size={32} /></div>
      <div className="min-w-0 flex-1">
        <div className="mb-3 flex items-center gap-2.5 pt-1"><p className="text-[12px] font-semibold text-[var(--color-ink)]">Mini-OpenClaw</p><span className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[9px] font-medium ${responseFailed ? "bg-[#fff1f0] text-[#b45661]" : "bg-[#e9faf5] text-[#158d87]"}`}>{responseFailed ? <AlertCircle size={10} /> : isPending ? <LoaderCircle size={10} className="animate-spin" /> : <CheckCircle2 size={10} />}{responseFailed ? "回复失败" : isPending ? "正在回复" : "已回复"}</span></div>
        <RetrievalCard results={retrievals} />
        {showContent && (
          <div className="markdown min-w-0 text-[13px] leading-7">
            <ReactMarkdown
              components={{
                table: ({ children }) => <div className="my-4 max-w-full overflow-x-auto rounded-lg border border-[var(--color-line)]"><table>{children}</table></div>,
                pre: ({ children }) => <pre className="max-w-full overflow-x-auto">{children}</pre>,
                a: ({ children, href }) => <a href={href} rel="noopener noreferrer" target="_blank">{children}</a>
              }}
              remarkPlugins={[remarkGfm]}
            >
              {content}
            </ReactMarkdown>
          </div>
        )}
        {!content.trim() && !toolCalls.length && (
          <div className="flex items-center gap-2 text-sm text-[var(--color-ink-soft)]" role="status">
            {isPending && <LoaderCircle className="animate-spin motion-reduce:animate-none" size={14} />}
            {isPending ? "正在思考…" : "回复内容为空"}
          </div>
        )}
        <ThoughtChain toolCalls={toolCalls} isPending={isPending} />
        {showContent && (
          <div className="mt-3 flex items-center gap-2">
            <button
              className="flex items-center gap-1.5 rounded-md px-1.5 py-1 text-[11px] text-[#939bac] transition-colors hover:bg-[var(--color-muted)] hover:text-[var(--color-ink)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--color-ocean)]"
              onClick={() => { void copyAnswer(); }}
              type="button"
            >
              {copyState === "copied" ? <Check size={12} /> : <Copy size={12} />}
              {copyState === "copied" ? "已复制" : "复制回答"}
            </button>
            <span aria-live="polite" className="text-[11px] text-[#b04e50]">{copyState === "error" ? "复制失败，请选择文字复制" : ""}</span>
          </div>
        )}
      </div>
    </article>
  );
}
