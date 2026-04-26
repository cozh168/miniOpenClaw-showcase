"use client";

import { useEffect, useState } from "react";
import { createPortal } from "react-dom";
import { Database, FileStack, Plus, SlidersHorizontal, Sparkles } from "lucide-react";

import { InspectorPanel } from "@/components/editor/InspectorPanel";
import { useAppStore } from "@/lib/store";

export function Navbar() {
  const [isInspectorOpen, setIsInspectorOpen] = useState(false);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  const {
    createNewSession,
    ragMode,
    toggleRagMode,
    renameCurrentSession,
    sessions,
    currentSessionId
  } = useAppStore();

  const currentTitle =
    sessions.find((session) => session.id === currentSessionId)?.title ?? "新会话";

  return (
    <header className="panel flex items-center justify-between rounded-[30px] px-5 py-4">
      <div className="flex items-center gap-4">
        <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-[rgba(15,139,141,0.14)] text-ocean">
          <Sparkles size={20} />
        </div>
        <div>
          <p className="text-xs uppercase tracking-[0.32em] text-[var(--color-ink-soft)]">
            Mini-OpenClaw
          </p>
          <div className="flex items-center gap-3">
            <h1 className="text-xl font-semibold tracking-[-0.04em]">{currentTitle}</h1>
            <button
              className="rounded-full border border-[var(--color-line)] px-3 py-1 text-xs text-[var(--color-ink-soft)]"
              onClick={() => {
                const next = window.prompt("重命名当前会话", currentTitle);
                if (next) {
                  void renameCurrentSession(next);
                }
              }}
              type="button"
            >
              Rename
            </button>
          </div>
        </div>
      </div>

      <div className="flex items-center gap-3">
        <button
          className="flex items-center gap-2 rounded-full border border-[var(--color-line)] bg-white/60 px-4 py-2 text-sm"
          onClick={() => void createNewSession()}
          type="button"
        >
          <Plus size={16} />
          新会话
        </button>
        <button
          className={`flex items-center gap-2 rounded-full px-4 py-2 text-sm ${
            ragMode
              ? "bg-ocean text-white"
              : "border border-[var(--color-line)] bg-white/60 text-ink"
          }`}
          onClick={() => void toggleRagMode()}
          type="button"
        >
          <Database size={16} />
          {ragMode ? "文档 RAG 已开" : "文档 RAG 已关"}
        </button>
        <div className="hidden items-center gap-2 rounded-full bg-[rgba(212,106,74,0.12)] px-4 py-2 text-sm text-[var(--color-ember)] md:flex">
          <FileStack size={16} />
          File-first Memory
        </div>
        <button
          className="flex h-9 w-9 items-center justify-center rounded-full border border-[var(--color-line)] bg-white/60 text-[var(--color-ink-soft)] hover:bg-white/80 hover:text-ink"
          onClick={() => setIsInspectorOpen(true)}
          title="Open Inspector"
          type="button"
        >
          <SlidersHorizontal size={16} />
        </button>
      </div>

      {isInspectorOpen &&
        mounted &&
        createPortal(
          <div className="fixed inset-0 z-[9999] flex items-center justify-center">
            <div
              className="absolute inset-0 bg-black/20 backdrop-blur-sm"
              onClick={() => setIsInspectorOpen(false)}
            />
            <div className="relative z-10 flex h-[80vh] w-[90vw] max-w-4xl flex-col shadow-2xl">
              <InspectorPanel onClose={() => setIsInspectorOpen(false)} />
            </div>
          </div>,
          document.body
        )}
    </header>
  );
}
