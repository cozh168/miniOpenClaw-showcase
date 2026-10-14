"use client";

import { ArrowDown, ArrowUpRight, Calculator, Code2, FileText, Lightbulb, PanelRightOpen, Pencil, X } from "lucide-react";
import { useEffect, useRef, useState, type FormEvent } from "react";

import { ChatInput } from "@/components/chat/ChatInput";
import { ChatMessage } from "@/components/chat/ChatMessage";
import { parseFileReferences } from "@/components/chat/FileReferencePicker";
import { BrandMark } from "@/components/layout/BrandMark";
import { Modal } from "@/components/layout/Modal";
import { useAppStore } from "@/lib/store";

const SUGGESTIONS = [
  {
    title: "梳理一个想法",
    description: "把模糊的想法拆成可执行的步骤",
    prompt: "我有一个想法，想请你帮我梳理。先问我三个关键问题，再一起制定行动计划。",
    icon: Lightbulb
  },
  {
    title: "总结一段内容",
    description: "提炼重点，整理成清晰的笔记",
    prompt: "请帮我总结下面的内容，提炼关键观点，并列出值得继续思考的问题：\n\n",
    icon: FileText
  },
  {
    title: "复利计算示例",
    description: "调用 Python，逐年核对计算结果",
    prompt: "请调用 Python，计算本金 10000 元、年收益率 5%、连续复利 10 年后的金额，并逐年列出结果。",
    icon: Calculator
  },
  {
    title: "一起写段代码",
    description: "从需求出发，获得可运行的示例",
    prompt: "请帮我写一个 Python 脚本，将多个 CSV 文件合并，并说明如何运行。先向我确认文件格式和合并规则。",
    icon: Code2
  }
];

export function ChatPanel({ onOpenWorkspace, onOpenAgent }: {
  onOpenWorkspace?: () => void;
  onOpenAgent?: () => void;
} = {}) {
  const { messages, sendMessage, isStreaming, isSessionBusy, sessionError, tokenStats, currentSessionId, sessions, renameCurrentSession } = useAppStore();
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const followRef = useRef(true);
  const [showScrollButton, setShowScrollButton] = useState(false);
  const [seed, setSeed] = useState({ value: "", id: 0 });
  const [renameOpen, setRenameOpen] = useState(false);
  const [draftTitle, setDraftTitle] = useState("");
  const [renameBusy, setRenameBusy] = useState(false);
  const [renameError, setRenameError] = useState("");
  const currentTitle = sessions.find((session) => session.id === currentSessionId)?.title ?? "新会话";
  const firstUserMessage = messages.find((message) => message.role === "user");
  const summary = firstUserMessage ? parseFileReferences(firstUserMessage.content).message.replace(/\s+/g, " ") : "从一个问题开始，让每一步思考与工具执行清晰可见。";

  useEffect(() => {
    followRef.current = true;
    setShowScrollButton(false);
    setSeed({ value: "", id: 0 });
    setRenameOpen(false);
    const container = scrollRef.current;
    if (container) container.scrollTop = 0;
  }, [currentSessionId]);

  useEffect(() => {
    const container = scrollRef.current;
    if (container && followRef.current) {
      container.scrollTop = messages.length ? container.scrollHeight : 0;
    }
  }, [messages]);

  function returnToBottom() {
    followRef.current = true;
    setShowScrollButton(false);
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }

  async function rename(event: FormEvent) {
    event.preventDefault();
    if (!draftTitle.trim() || renameBusy) return;
    setRenameBusy(true); setRenameError("");
    try { await renameCurrentSession(draftTitle); setRenameOpen(false); }
    catch { setRenameError("重命名失败，请确认后端连接后重试。"); }
    finally { setRenameBusy(false); }
  }

  return (
    <section aria-label="对话工作台" className="relative flex h-full min-h-0 min-w-0 flex-1 flex-col bg-white">
      <div className="flex shrink-0 items-center gap-3 border-b border-[var(--color-line)] bg-[#fcfdff] px-5 py-4 md:px-7">
        <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg text-[#668aa7]"><FileText size={23} strokeWidth={1.65} /></div>
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <h2 className="min-w-0 truncate text-[15px] font-semibold tracking-[-0.025em] text-[var(--color-ink)]" title={currentTitle}>{currentTitle}</h2>
            <button aria-label="重命名当前会话" className="flex h-6 w-6 shrink-0 items-center justify-center rounded-md text-[#6e8aa7] transition-colors hover:bg-[#eaf3f9] hover:text-[var(--color-ocean)]" disabled={!currentSessionId || isStreaming || isSessionBusy} onClick={() => { setDraftTitle(currentTitle); setRenameError(""); setRenameOpen(true); }} type="button"><Pencil size={13} /></button>
          </div>
          <p className="mt-1 truncate text-[11px] leading-5 text-[#7892b1]" title={summary}>{summary}</p>
        </div>
        <div className="hidden shrink-0 items-center gap-1.5 text-[10px] text-[#8ba1b6] xl:flex" title={tokenStats ? `上下文 ${tokenStats.total_tokens.toLocaleString()} tokens` : undefined} role="status">
          <span className={`h-1.5 w-1.5 rounded-full ${isStreaming ? "animate-pulse bg-[var(--color-ocean)] motion-reduce:animate-none" : sessionError ? "bg-[#c77479]" : "bg-[#75adaf]"}`} />
          <span>{isSessionBusy ? "加载中" : isStreaming ? "正在回复" : sessionError ? "连接失败" : "就绪"}</span>
        </div>
        {onOpenWorkspace && <button aria-label="打开工作区" className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg border border-[var(--color-line)] bg-white text-[#507998] hover:bg-[#f2f8fc]" onClick={onOpenWorkspace} title="打开工作区" type="button"><PanelRightOpen size={16} /></button>}
      </div>
      {sessionError && <p className="mx-5 mt-3 rounded-lg bg-red-50 px-3 py-2 text-xs text-red-600" role="alert">{sessionError}</p>}

      <div
        className="min-h-0 flex-1 overflow-y-auto overscroll-contain"
        onScroll={() => {
          const container = scrollRef.current;
          if (!container) return;
          const nearBottom = container.scrollHeight - container.scrollTop - container.clientHeight < 120;
          followRef.current = nearBottom;
          setShowScrollButton(!nearBottom);
        }}
        ref={scrollRef}
      >
        {!messages.length ? (
          <div className="mx-auto flex min-h-full w-full max-w-[740px] flex-col justify-center px-5 py-6 md:px-8 md:py-10">
            <div className="mb-5 flex items-center gap-3">
              <BrandMark size={42} />
              <div><p className="text-[14px] font-semibold text-[var(--color-ink)]">Mini-OpenClaw</p><p className="mt-0.5 text-[11px] text-[#7a94ae]">你的 AI 任务工作台</p></div>
            </div>
            <h2 className="font-display text-[25px] font-semibold leading-[1.4] tracking-[-0.04em] text-[var(--color-ink)] md:text-[31px]">
              今天，一起完成什么？
            </h2>
            <p className="mt-3 max-w-lg text-[13px] leading-7 text-[var(--color-ink-soft)]">
              提问、分析资料，或让 Python 帮你验证一个结果。<br className="hidden sm:block" />
              文件、对话和执行过程，都在一个工作区里。
            </p>
            <div className="mt-7 grid grid-cols-2 gap-2.5">
              {SUGGESTIONS.map(({ title, description, prompt, icon: Icon }) => (
                <button
                  className="group relative flex flex-col items-start gap-2.5 rounded-xl border border-[var(--color-line)] bg-[#fcfdff] p-3.5 text-left transition-colors hover:border-[#bddbe5] hover:bg-[#f2f9fc] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--color-ocean)] sm:flex-row"
                  aria-label={title}
                  disabled={isSessionBusy || isStreaming}
                  key={title}
                  onClick={() => setSeed((previous) => ({ value: prompt, id: previous.id + 1 }))}
                  type="button"
                >
                  <Icon aria-hidden="true" className="mt-0.5 shrink-0 text-[var(--color-ocean)]" size={18} strokeWidth={1.7} />
                  <div className="min-w-0 flex-1">
                    <p className="text-[13px] font-medium text-[var(--color-ink)]">{title}</p>
                    <p className="mt-1.5 text-xs leading-5 text-[var(--color-ink-soft)]">{description}</p>
                  </div>
                  <ArrowUpRight aria-hidden="true" className="absolute right-4 top-4 mt-0.5 shrink-0 text-[#b1b7c7] transition-colors group-hover:text-[var(--color-ocean)] sm:static" size={15} />
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="mx-auto w-full max-w-[960px] space-y-7 px-4 py-5 md:px-6 md:py-6">
            {messages.map((message, index) => (
              <ChatMessage
                content={message.content}
                isPending={isStreaming && index === messages.length - 1}
                key={message.id}
                retrievals={message.retrievals}
                role={message.role}
                toolCalls={message.toolCalls}
              />
            ))}
          </div>
        )}
      </div>

      <div className="relative shrink-0 bg-gradient-to-t from-[#fafdff] to-white px-3 pb-3 pt-3 md:px-5 md:pb-4">
        {showScrollButton && (
          <button
            aria-label="回到最新消息"
            className="absolute -top-12 left-1/2 z-10 flex h-9 -translate-x-1/2 items-center gap-2 rounded-full border border-[var(--color-line)] bg-white px-3 text-xs text-[var(--color-ink-soft)] shadow-sm transition-colors hover:text-[var(--color-ocean)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--color-ocean)]"
            onClick={returnToBottom}
            type="button"
          >
            <ArrowDown size={14} />最新消息
          </button>
        )}
        <div className="mx-auto max-w-[960px]">
          <ChatInput
            disabled={isStreaming || isSessionBusy}
            sessionId={currentSessionId}
            onSend={async (value) => {
              followRef.current = true;
              setShowScrollButton(false);
              await sendMessage(value);
            }}
            seedId={seed.id}
            seedValue={seed.value}
            onOpenAgent={onOpenAgent}
            onOpenWorkspace={onOpenWorkspace}
          />
        </div>
      </div>
      {renameOpen && <Modal labelledBy="chat-rename-title" onClose={() => { if (!renameBusy) setRenameOpen(false); }} className="compact-modal">
        <form onSubmit={(event) => { void rename(event); }}>
          <div className="modal-heading"><h2 id="chat-rename-title">重命名会话</h2><button aria-label="关闭重命名窗口" className="icon-button" disabled={renameBusy} onClick={() => setRenameOpen(false)} type="button"><X size={18} /></button></div>
          <label className="field-label" htmlFor="chat-session-name">会话名称</label>
          <input className="text-field" data-autofocus id="chat-session-name" maxLength={100} onChange={(event) => setDraftTitle(event.target.value)} required value={draftTitle} />
          {renameError && <p className="form-error" role="alert">{renameError}</p>}
          <div className="modal-actions"><button className="secondary-button" disabled={renameBusy} onClick={() => setRenameOpen(false)} type="button">取消</button><button className="primary-button" disabled={renameBusy || !draftTitle.trim()} type="submit">{renameBusy ? "正在保存…" : "保存名称"}</button></div>
        </form>
      </Modal>}
    </section>
  );
}
