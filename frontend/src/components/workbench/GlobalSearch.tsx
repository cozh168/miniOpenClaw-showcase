"use client";

import { useEffect, useRef, useState } from "react";
import { FileText, MessageSquare, Search, Wrench, X } from "lucide-react";
import { Modal } from "@/components/layout/Modal";
import { getRuntimeInfo, getWorkspaceFiles, type WorkspaceFileNode } from "@/lib/api";
import { flattenFiles } from "@/lib/navigation";
import { useAppStore } from "@/lib/store";

export function GlobalSearch({ onClose, onFile, onChat, onTools }: {
  onClose: () => void; onFile: (path: string) => Promise<void>; onChat: () => void; onTools: () => void;
}) {
  const { sessions, selectSession, isStreaming, isSessionBusy } = useAppStore();
  const [query, setQuery] = useState("");
  const [files, setFiles] = useState<WorkspaceFileNode[]>([]);
  const [tools, setTools] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const busyRef = useRef(false);
  const [error, setError] = useState("");
  useEffect(() => {
    let live = true;
    void Promise.allSettled([getWorkspaceFiles(), getRuntimeInfo()]).then(([tree, runtime]) => {
      if (!live) return;
      if (tree.status === "fulfilled") setFiles(flattenFiles(tree.value.files).filter((file) => file.editable));
      if (runtime.status === "fulfilled") setTools(runtime.value.tools.map((tool) => tool.name));
      if (tree.status === "rejected" || runtime.status === "rejected") setError("部分内容加载失败，会话搜索仍可使用。");
      setLoading(false);
    });
    return () => { live = false; };
  }, []);
  const term = query.trim().toLocaleLowerCase();
  const matchingSessions = sessions.filter((item) => item.title.toLocaleLowerCase().includes(term)).slice(0, 6);
  const matchingFiles = files.filter((item) => item.path.toLocaleLowerCase().includes(term)).slice(0, 8);
  const matchingTools = tools.filter((item) => item.toLocaleLowerCase().includes(term)).slice(0, 6);
  async function choose(action: () => Promise<void>) {
    if (busyRef.current) return;
    busyRef.current = true;
    setBusy(true); setError("");
    try { await action(); onClose(); } catch { setError("打开失败，请确认服务连接后重试。"); } finally { busyRef.current = false; setBusy(false); }
  }
  return <Modal labelledBy="search-title" onClose={() => { if (!busyRef.current) onClose(); }} className="search-modal">
    <div className="search-modal-input"><Search size={20} /><label className="sr-only" id="search-title" htmlFor="global-search">搜索工作台</label><input id="global-search" data-autofocus placeholder="搜索文件、会话或工具…" value={query} onChange={(event) => setQuery(event.target.value)} /><button className="icon-button" onClick={onClose} disabled={busy} aria-label="关闭搜索"><X size={18} /></button></div>
    <div className="search-results">
      {error && <p role="alert" className="form-error">{error}</p>}
      <p className="search-group-title">会话</p>{matchingSessions.map((item) => <button key={item.id} disabled={busy || isStreaming || isSessionBusy} onClick={() => void choose(async () => { await selectSession(item.id); onChat(); })}><MessageSquare size={16} /><span>{item.title}</span><small>{item.message_count} 条消息</small></button>)}
      <p className="search-group-title">文件 {loading ? "· 加载中" : ""}</p>{matchingFiles.map((item) => <button key={item.path} disabled={busy} onClick={() => void choose(() => onFile(item.path))}><FileText size={16} /><span>{item.name}</span><small>{item.path}</small></button>)}
      <p className="search-group-title">工具</p>{matchingTools.map((item) => <button key={item} disabled={busy} onClick={() => { onTools(); onClose(); }}><Wrench size={16} /><span>{item}</span><small>查看工具集</small></button>)}
      {!loading && !matchingSessions.length && !matchingFiles.length && !matchingTools.length && <div className="view-empty">没有找到相关内容，试试其他关键词。</div>}
    </div><div className="search-footer">搜索本地工作台中的实际内容 <kbd>Esc</kbd> 关闭</div>
  </Modal>;
}
