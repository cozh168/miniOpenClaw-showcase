"use client";

import { AlertCircle, ChevronRight, Globe2, LoaderCircle, Paperclip, Send, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import { FileReferencePicker, withFileReferences, type FileReference } from "@/components/chat/FileReferencePicker";

export function ChatInput({
  disabled,
  onSend,
  seedValue = "",
  seedId = 0,
  sessionId = null,
  onOpenAgent,
  onOpenWorkspace
}: {
  disabled: boolean;
  onSend: (value: string) => Promise<void>;
  seedValue?: string;
  seedId?: number;
  sessionId?: string | null;
  onOpenAgent?: () => void;
  onOpenWorkspace?: () => void;
}) {
  const [value, setValue] = useState("");
  const [pending, setPending] = useState(false);
  const [error, setError] = useState("");
  const [references, setReferences] = useState<FileReference[]>([]);
  const [pickerOpen, setPickerOpen] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const valueRef = useRef("");
  const compositionRef = useRef(false);
  const sendingRef = useRef(false);
  const previousSessionRef = useRef(sessionId);
  const referencesRef = useRef(references);
  referencesRef.current = references;
  const draftsRef = useRef(new Map<string, { value: string; references: FileReference[] }>());
  const busy = disabled || pending;
  const remainingChars = Math.max(0, 10000 - references.reduce((sum, file) => sum + file.content.length, 0));

  useEffect(() => {
    const previous = previousSessionRef.current;
    previousSessionRef.current = sessionId;
    if (previous === sessionId || (!previous && sessionId && sendingRef.current)) return;
    draftsRef.current.set(previous ?? "", { value: valueRef.current, references: referencesRef.current });
    const draft = draftsRef.current.get(sessionId ?? "");
    const nextValue = draft?.value ?? "";
    const nextReferences = draft?.references ?? [];
    setValue(nextValue); valueRef.current = nextValue;
    setReferences(nextReferences); referencesRef.current = nextReferences;
    setError(""); setPickerOpen(false);
  }, [sessionId]);

  useEffect(() => {
    if (!seedId) return;
    setValue(seedValue);
    valueRef.current = seedValue;
    setError("");
    textareaRef.current?.focus();
  }, [seedId, seedValue]);

  useEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea) return;
    textarea.style.height = "auto";
    textarea.style.height = `${Math.min(textarea.scrollHeight, 200)}px`;
    textarea.style.overflowY = textarea.scrollHeight > 200 ? "auto" : "hidden";
  }, [value]);

  async function submit() {
    const nextValue = valueRef.current.trim();
    if (!nextValue || busy || sendingRef.current || compositionRef.current) return;
    sendingRef.current = true;
    setPending(true);
    setError("");
    const submittedValue = valueRef.current;
    const submittedReferences = references;
    setValue("");
    valueRef.current = "";
    setReferences([]);
    referencesRef.current = [];

    try {
      await onSend(withFileReferences(nextValue, submittedReferences));
    } catch {
      const restoredValue = valueRef.current ? `${submittedValue}\n\n${valueRef.current}` : submittedValue;
      setValue(restoredValue);
      valueRef.current = restoredValue;
      setReferences(submittedReferences);
      referencesRef.current = submittedReferences;
      setError("发送失败。请检查网络和模型配置后重试，你的输入已保留。");
    } finally {
      sendingRef.current = false;
      setPending(false);
    }
  }

  return (
    <div>
      <div className="rounded-xl border border-[#dce7f2] bg-white p-2 shadow-[0_3px_15px_rgba(38,73,117,0.055)] transition-shadow focus-within:border-[#9bbfcd] focus-within:ring-2 focus-within:ring-[#006c80]/5">
        {references.length > 0 && <div className="flex flex-wrap gap-1.5 border-b border-[var(--color-line)] px-2 pb-2 pt-1">
          {references.map((file) => <span className="inline-flex max-w-full items-center gap-1.5 rounded-md border border-[#dcebf1] bg-[#edf7fa] px-2 py-1 text-[10px] text-[#286277]" key={file.path} title={`${file.path}${file.truncated ? "（引用部分内容）" : ""}`}>
            <Paperclip size={11} className="shrink-0" /> <span className="max-w-[180px] truncate">{file.path.split("/").pop()}{file.truncated ? " · 部分内容" : ""}</span>
            <button aria-label={`移除引用 ${file.path}`} className="flex h-4 w-4 shrink-0 items-center justify-center rounded-sm hover:bg-[#d5eaf0]" disabled={busy} onClick={() => setReferences((current) => current.filter((reference) => reference.path !== file.path))} type="button"><X size={11} /></button>
          </span>)}
        </div>}
        <div className="flex items-end gap-1.5">
          <button aria-label="引用工作区文件" className="flex h-10 w-9 shrink-0 items-center justify-center rounded-lg text-[#547795] transition-colors hover:bg-[#edf6fa] hover:text-[var(--color-ocean)]" disabled={busy || references.length >= 3 || remainingChars <= 0} onClick={() => setPickerOpen(true)} title={references.length >= 3 || remainingChars <= 0 ? "已达到引用上限，请先移除一个文件" : "选择工作区文件并附上内容"} type="button"><Paperclip size={19} /></button>
        <textarea
          aria-label="输入消息"
          className="block min-h-[40px] min-w-0 flex-1 resize-none bg-transparent px-1 py-2 text-[13px] leading-6 text-[var(--color-ink)] outline-none placeholder:text-[#8198b5]"
          onChange={(event) => {
            setValue(event.target.value);
            valueRef.current = event.target.value;
            if (error) setError("");
          }}
          onCompositionEnd={() => { compositionRef.current = false; }}
          onCompositionStart={() => { compositionRef.current = true; }}
          onKeyDown={(event) => {
            if (event.nativeEvent.isComposing || compositionRef.current || event.keyCode === 229) return;
            if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
              event.preventDefault();
              void submit();
            }
          }}
          placeholder={busy ? "继续写下你的下一个问题…" : "输入消息，或引用工作区文件…"}
          ref={textareaRef}
          rows={1}
          value={value}
        />
          {onOpenAgent && <button aria-label="查看当前助手设置" className="mb-0.5 hidden h-9 shrink-0 items-center gap-1.5 rounded-lg border border-[var(--color-line)] px-2.5 text-[11px] text-[#5b7797] hover:bg-[#f3f8fc] sm:inline-flex" onClick={onOpenAgent} type="button"><Globe2 size={14} /><span>当前助手</span><ChevronRight size={12} /></button>}
          <button
            aria-label={busy ? "正在回复" : "发送消息"}
            className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-[var(--color-ocean)] text-white shadow-sm transition-colors hover:bg-[#00586b] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--color-ocean)] focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:bg-[#e5edf4] disabled:text-[#91a8bd]"
            disabled={busy || !value.trim()}
            onClick={() => { void submit(); }}
            type="button"
          >
            {busy ? <LoaderCircle aria-hidden="true" className="animate-spin motion-reduce:animate-none" size={17} /> : <Send aria-hidden="true" size={18} strokeWidth={1.8} />}
          </button>
        </div>
      </div>
      {error ? (
        <p className="mt-2 flex items-start gap-1.5 px-1 text-xs leading-5 text-[#b04e50]" role="alert">
          <AlertCircle className="mt-0.5 shrink-0" size={13} />{error}
        </p>
      ) : (
        <div className="mt-2 flex items-center justify-between gap-2 px-1 text-[10px] leading-4 text-[#91a3ba]">
          {onOpenWorkspace ? <button className="hover:text-[var(--color-ocean)]" onClick={onOpenWorkspace} type="button">工作区中的文件可随消息引用</button> : <span>支持工具调用与文件引用</span>}
          <span className="shrink-0">⌘ / Ctrl + Enter 发送</span>
        </div>
      )}
      {pickerOpen && <FileReferencePicker existingPaths={references.map((file) => file.path)} remainingChars={remainingChars} onClose={() => setPickerOpen(false)} onConfirm={(file) => {
        setReferences((current) => [...current, file]);
        setPickerOpen(false);
        textareaRef.current?.focus();
      }} />}
    </div>
  );
}
