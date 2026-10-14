"use client";

import Editor from "@monaco-editor/react";
import { Brain, Check, Code2, FileText, Loader2, Save, X } from "lucide-react";
import { useState } from "react";

import { useAppStore } from "@/lib/store";

const FILE_LABELS: Record<string, string> = {
  "workspace/SOUL.md": "助手行为",
  "workspace/IDENTITY.md": "角色设定",
  "workspace/USER.md": "用户偏好",
  "workspace/AGENTS.md": "工作方式",
  "memory_module_v1/long_term_memory/MEMORY.md": "长期记忆",
  "SKILLS_SNAPSHOT.md": "技能概览"
};

function fileLabel(path: string) {
  if (FILE_LABELS[path]) return FILE_LABELS[path];
  const parts = path.split("/");
  return parts[parts.length - 1] === "SKILL.md"
    ? parts[parts.length - 2] || "技能文件"
    : parts[parts.length - 1];
}

function fileGroup(path: string) {
  if (path.startsWith("workspace/")) return "assistant";
  if (path.startsWith("memory_module")) return "memory";
  return "skills";
}

export function InspectorPanel({ onClose, onBusyChange }: { onClose?: () => void; onBusyChange?: (busy: boolean) => void }) {
  const {
    editableFiles,
    inspectorPath,
    inspectorContent,
    inspectorDirty,
    loadInspectorFile,
    updateInspectorContent,
    saveInspector
  } = useAppStore();
  const [saving, setSaving] = useState(false);
  const [loading, setLoading] = useState(false);
  const [plainText, setPlainText] = useState(false);
  const [feedback, setFeedback] = useState<{ error: boolean; text: string } | null>(null);
  const busy = saving || loading;
  const groups = [
    { id: "assistant", label: "助手设定", icon: FileText },
    { id: "memory", label: "记忆", icon: Brain },
    { id: "skills", label: "技能", icon: Code2 }
  ].map((group) => ({
    ...group,
    files: editableFiles.filter((path) => fileGroup(path) === group.id)
  }));

  async function selectFile(path: string) {
    if (busy || path === inspectorPath) return;
    if (inspectorDirty && !window.confirm("当前文件有未保存的修改。切换文件将丢弃这些修改，是否继续？")) return;
    const previousPath = inspectorPath;
    const previousContent = inspectorContent;
    const previousDirty = inspectorDirty;
    setLoading(true);
    onBusyChange?.(true);
    setFeedback(null);
    try {
      await loadInspectorFile(path);
    } catch {
      // The store changes the path before fetching; restore it if fetching fails.
      try {
        await loadInspectorFile(previousPath);
      } catch {
        // A failed restoration still resets the path and retains the old content.
      }
      if (previousDirty) updateInspectorContent(previousContent);
      setFeedback({ error: true, text: "文件加载失败，请重新选择文件后重试。" });
    } finally {
      setLoading(false);
      onBusyChange?.(false);
    }
  }

  async function save() {
    if (!inspectorDirty || busy) return;
    setSaving(true);
    onBusyChange?.(true);
    setFeedback(null);
    try {
      await saveInspector();
      setFeedback({ error: false, text: "修改已保存" });
    } catch {
      setFeedback({ error: true, text: "保存失败，修改仍保留在编辑器中，请重试。" });
    } finally {
      setSaving(false);
      onBusyChange?.(false);
    }
  }

  function edit(value: string) {
    if (busy || value === inspectorContent) return;
    setFeedback(null);
    updateInspectorContent(value);
  }

  return (
    <section className="flex h-full min-h-0 flex-col overflow-hidden rounded-xl bg-[var(--color-surface)] text-[var(--color-ink)]">
      <header className="flex shrink-0 items-start justify-between gap-3 border-b border-[var(--color-line)] px-4 py-4 sm:px-6 sm:py-5">
        <div className="min-w-0">
          <h2 className="text-base font-semibold tracking-tight sm:text-lg" id="inspector-title">工作台设置</h2>
          <p className="mt-1 text-xs leading-5 text-[var(--color-ink-soft)] sm:text-sm">调整助手的设定、技能与记忆文件。</p>
        </div>
        <div className="flex shrink-0 items-center gap-2">
          <button
            className="inline-flex h-9 items-center gap-2 rounded-lg bg-[var(--color-ocean)] px-3 text-xs font-medium text-white transition hover:brightness-95 disabled:cursor-not-allowed disabled:opacity-40 sm:text-sm"
            disabled={!inspectorDirty || busy}
            onClick={() => void save()}
            type="button"
          >
            {saving ? <Loader2 className="animate-spin" size={15} /> : <Save size={15} />}
            {saving ? "保存中" : "保存修改"}
          </button>
          {onClose && (
            <button
              aria-label="关闭工作台设置"
              disabled={busy}
              className="flex h-9 w-9 items-center justify-center rounded-lg text-[var(--color-ink-soft)] transition hover:bg-[var(--color-muted)] hover:text-[var(--color-ink)]"
              onClick={onClose}
              type="button"
            >
              <X size={18} />
            </button>
          )}
        </div>
      </header>

      <div className="shrink-0 border-b border-[var(--color-line)] bg-[var(--color-muted)] px-4 py-3 md:hidden">
        <label className="mb-1.5 block text-xs font-medium text-[var(--color-ink-soft)]" htmlFor="inspector-file-select">选择编辑文件</label>
        <select
          className="h-10 w-full min-w-0 rounded-lg border border-[var(--color-line)] bg-[var(--color-surface)] px-3 text-sm"
          disabled={busy}
          id="inspector-file-select"
          onChange={(event) => void selectFile(event.target.value)}
          value={inspectorPath}
        >
          {groups.filter((group) => group.files.length).map((group) => (
            <optgroup key={group.id} label={group.label}>
              {group.files.map((path) => <option key={path} value={path}>{fileLabel(path)}</option>)}
            </optgroup>
          ))}
        </select>
      </div>

      <div className="flex min-h-0 flex-1">
        <nav aria-label="设置文件" className="hidden w-[220px] shrink-0 overflow-y-auto border-r border-[var(--color-line)] bg-[var(--color-muted)] p-3 md:block">
          {groups.filter((group) => group.files.length).map((group) => {
            const Icon = group.icon;
            return (
              <div className="mb-5 last:mb-0" key={group.id}>
                <p className="mb-2 flex items-center gap-2 px-2 text-[11px] font-medium text-[var(--color-ink-soft)]">
                  <Icon size={13} />{group.label}
                </p>
                <div className="space-y-1">
                  {group.files.map((path) => (
                    <button
                      aria-current={path === inspectorPath ? "page" : undefined}
                      className={`w-full truncate rounded-lg px-3 py-2.5 text-left text-sm transition disabled:cursor-wait ${path === inspectorPath ? "bg-[var(--color-surface)] font-medium text-[var(--color-ocean)] shadow-sm" : "text-[var(--color-ink-soft)] hover:bg-[var(--color-surface)] hover:text-[var(--color-ink)]"}`}
                      disabled={busy}
                      key={path}
                      onClick={() => void selectFile(path)}
                      title={path}
                      type="button"
                    >
                      {fileLabel(path)}
                    </button>
                  ))}
                </div>
              </div>
            );
          })}
        </nav>

        <div className="flex min-h-0 min-w-0 flex-1 flex-col">
          <div className="flex shrink-0 flex-wrap items-center justify-between gap-x-3 gap-y-2 border-b border-[var(--color-line)] px-4 py-3">
            <div className="min-w-0 flex-1 basis-40">
              <p className="text-sm font-medium">{fileLabel(inspectorPath)}</p>
              <p className="mono mt-1 truncate text-[10px] text-[var(--color-ink-soft)] sm:text-[11px]" title={inspectorPath}>{inspectorPath}</p>
            </div>
            <button
              aria-pressed={plainText}
              className="shrink-0 rounded-md border border-[var(--color-line)] px-2.5 py-1.5 text-xs text-[var(--color-ink-soft)] transition hover:bg-[var(--color-muted)]"
              onClick={() => setPlainText((current) => !current)}
              type="button"
            >
              {plainText ? "代码编辑器" : "纯文本编辑"}
            </button>
          </div>

          <div className="relative min-h-0 flex-1 overflow-hidden">
            {loading ? (
              <div className="flex h-full items-center justify-center gap-2 text-sm text-[var(--color-ink-soft)]" role="status">
                <Loader2 className="animate-spin" size={17} />正在加载文件…
              </div>
            ) : plainText ? (
              <textarea
                aria-label={`编辑${fileLabel(inspectorPath)}`}
                className="mono h-full w-full resize-none border-0 bg-[var(--color-surface)] p-4 text-[13px] leading-6 text-[var(--color-ink)] outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-[var(--color-ocean)]"
                onChange={(event) => edit(event.target.value)}
                readOnly={saving}
                spellCheck={false}
                value={inspectorContent}
              />
            ) : (
              <Editor
                defaultLanguage="markdown"
                height="100%"
                loading={
                  <div className="flex flex-col items-center gap-3 px-5 text-center text-sm text-[var(--color-ink-soft)]" role="status">
                    <Loader2 className="animate-spin" size={20} />
                    <p>编辑器加载中</p>
                    <p className="text-xs">若加载较慢，可切换上方的「纯文本编辑」继续。</p>
                  </div>
                }
                onChange={(value) => edit(value ?? "")}
                options={{
                  automaticLayout: true,
                  fontFamily: "IBM Plex Mono, Menlo, monospace",
                  fontSize: 13,
                  lineHeight: 24,
                  lineNumbersMinChars: 3,
                  minimap: { enabled: false },
                  padding: { top: 16, bottom: 16 },
                  readOnly: saving,
                  renderLineHighlight: "none",
                  scrollBeyondLastLine: false,
                  wordWrap: "on"
                }}
                path={inspectorPath}
                theme="vs-light"
                value={inspectorContent}
              />
            )}
          </div>

          <footer className="flex shrink-0 items-center gap-2 border-t border-[var(--color-line)] px-4 py-2.5 text-xs text-[var(--color-ink-soft)]">
            {feedback ? (
              <span className={`inline-flex items-center gap-1.5 ${feedback.error ? "text-red-600" : "text-[var(--color-ocean)]"}`} role={feedback.error ? "alert" : "status"}>
                {!feedback.error && <Check size={13} />}{feedback.text}
              </span>
            ) : (
              <span className="inline-flex items-center gap-1.5" role="status">
                <span className={`h-1.5 w-1.5 rounded-full ${inspectorDirty ? "bg-amber-500" : "bg-[var(--color-ocean)]"}`} />
                {saving ? "正在保存修改…" : inspectorDirty ? "有未保存的修改" : "文件已同步"}
              </span>
            )}
          </footer>
        </div>
      </div>
    </section>
  );
}
