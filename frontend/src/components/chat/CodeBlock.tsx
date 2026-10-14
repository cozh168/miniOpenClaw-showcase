"use client";

import { Check, Copy } from "lucide-react";
import { useEffect, useRef, useState } from "react";

export async function copyText(content: string) {
  try {
    if (!navigator.clipboard) throw new Error("Clipboard unavailable");
    await navigator.clipboard.writeText(content);
  } catch {
    const previous = document.activeElement instanceof HTMLElement ? document.activeElement : null;
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
      previous?.focus({ preventScroll: true });
    }
  }
}

export function CopyButton({ content, label = "复制", className = "" }: {
  content: string;
  label?: string;
  className?: string;
}) {
  const [state, setState] = useState<"idle" | "copied" | "error">("idle");
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(() => () => { if (timer.current) clearTimeout(timer.current); }, []);

  async function copy() {
    try { await copyText(content); setState("copied"); }
    catch { setState("error"); }
    if (timer.current) clearTimeout(timer.current);
    timer.current = setTimeout(() => setState("idle"), 2400);
  }

  return (
    <button
      className={`inline-flex shrink-0 items-center gap-1.5 rounded-md border border-[var(--color-line)] bg-white/80 px-2 py-1 text-[10px] font-medium text-[var(--color-ink-soft)] transition-colors hover:border-[#bed4e2] hover:text-[var(--color-ocean)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--color-ocean)] ${className}`}
      onClick={() => { void copy(); }}
      type="button"
      title={state === "error" ? "复制失败，请选择文字复制" : undefined}
      aria-live="polite"
    >
      {state === "copied" ? <Check aria-hidden="true" size={11} /> : <Copy aria-hidden="true" size={11} />}
      {state === "copied" ? "已复制" : state === "error" ? "复制失败" : label}
    </button>
  );
}

const PYTHON_TOKEN = /(#[^\n]*|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|\b(?:False|None|True|and|as|assert|async|await|break|class|continue|def|del|elif|else|except|finally|for|from|global|if|import|in|is|lambda|nonlocal|not|or|pass|raise|return|try|while|with|yield)\b|\b(?:print|range|len|sum|min|max|round|enumerate|zip|list|dict|set|str|float|int)\b|\b\d+(?:\.\d+)?\b)/g;

function pythonLine(line: string) {
  const pieces: Array<{ text: string; color?: string }> = [];
  let cursor = 0;
  for (const match of line.matchAll(PYTHON_TOKEN)) {
    const index = match.index ?? 0;
    if (index > cursor) pieces.push({ text: line.slice(cursor, index) });
    const token = match[0];
    const color = token.startsWith("#") ? "#8299a4" : /^("|')/.test(token) ? "#b54d61" : /^\d/.test(token) ? "#006f9d" : /^(print|range|len|sum|min|max|round|enumerate|zip|list|dict|set|str|float|int)$/.test(token) ? "#2859b9" : "#7b43bc";
    pieces.push({ text: token, color });
    cursor = index + token.length;
  }
  if (cursor < line.length) pieces.push({ text: line.slice(cursor) });
  return pieces.map((piece, index) => <span key={index} style={piece.color ? { color: piece.color } : undefined}>{piece.text}</span>);
}

export function CodeBlock({ code, language = "text", numbered = false, maxHeight = 320 }: {
  code: string;
  language?: string;
  numbered?: boolean;
  maxHeight?: number;
}) {
  const python = /^(python|py)$/i.test(language);
  const lines = code.replace(/\r\n/g, "\n").split("\n");
  return (
    <div className="mono min-w-0 max-w-full overflow-auto rounded-md bg-[#fbfdff] text-[11px] leading-5 text-[#233d5c]" style={{ maxHeight }} tabIndex={0} aria-label={python ? "Python 代码" : "文本内容"}>
      <div className="w-max min-w-full py-2">
        {lines.map((line, index) => (
          <div className="flex whitespace-pre" key={index}>
            {numbered && <span className="sticky left-0 w-10 shrink-0 select-none border-r border-[#e5eef4] bg-[#f0f6f8] pr-2 text-right text-[10px] text-[#91a5b2]" aria-hidden="true">{index + 1}</span>}
            <code className="block px-3">{python ? pythonLine(line) : line || " "}</code>
          </div>
        ))}
      </div>
    </div>
  );
}
