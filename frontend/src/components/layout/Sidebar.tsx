"use client";

import { useMemo, useState } from "react";
import { Activity, Bot, ChevronDown, Code2, Database, FolderOpen, Home, MessageSquare, Plus, Search, Trash2, Wrench, X } from "lucide-react";

import { BrandMark } from "@/components/layout/BrandMark";
import { Modal } from "@/components/layout/Modal";
import type { SessionSummary } from "@/lib/api";
import { useAppStore } from "@/lib/store";
import type { WorkbenchView } from "@/lib/navigation";

function groupLabel(timestamp: number) {
  const day = new Date(timestamp * 1000);
  const today = new Date(); today.setHours(0, 0, 0, 0);
  const days = Math.floor((today.getTime() - day.getTime()) / 86400000);
  if (day >= today) return "今天";
  if (days === 0) return "昨天";
  if (days < 7) return "近一周";
  return "更早的会话";
}

export function Sidebar({ onNavigate, onClose, activeView, onViewChange }: { onNavigate: () => void; onClose: () => void; activeView: WorkbenchView; onViewChange: (view: WorkbenchView) => void }) {
  const { sessions, currentSessionId, selectSession, createNewSession, removeSession, messages, isStreaming, isSessionBusy } = useAppStore();
  const [query, setQuery] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [pendingDelete, setPendingDelete] = useState<SessionSummary | null>(null);
  const groups = useMemo(() => {
    const items = sessions.filter((session) => session.title.toLocaleLowerCase().includes(query.toLocaleLowerCase().trim()));
    const grouped: Array<{ label: string; sessions: SessionSummary[] }> = [];
    for (const session of items) {
      const label = groupLabel(session.updated_at);
      const group = grouped.find((entry) => entry.label === label);
      if (group) group.sessions.push(session); else grouped.push({ label, sessions: [session] });
    }
    return grouped;
  }, [sessions, query]);

  async function navigate(action: () => Promise<void>) {
    setBusy(true); setError("");
    try { await action(); onViewChange("chat"); onNavigate(); }
    catch { setError("操作失败，请确认后端连接后重试。"); }
    finally { setBusy(false); }
  }
  async function confirmDelete() {
    if (!pendingDelete) return;
    setBusy(true); setError("");
    try { await removeSession(pendingDelete.id); setPendingDelete(null); }
    catch { setError("删除失败，请稍后重试。"); }
    finally { setBusy(false); }
  }

  return (
    <aside className="session-sidebar" aria-label="会话导航">
      <div className="sidebar-brand"><BrandMark size={35} /><div><span className="brand-name font-display">Mini-OpenClaw</span><span className="brand-caption">本地可审计的 AI Agent 运行时</span></div><button className="icon-button mobile-close" type="button" onClick={onClose} aria-label="关闭侧栏"><X size={18} /></button></div>
      <button className="new-session-button" disabled={busy || isStreaming || isSessionBusy} onClick={() => void navigate(createNewSession)} type="button"><Plus size={17} /><span>新会话</span></button>
      <nav className="app-navigation" aria-label="工作台功能">{([{view:"chat",label:"首页",icon:Home},{view:"agent",label:"智能体",icon:Bot},{view:"knowledge",label:"知识库 (RAG)",icon:Database},{view:"tools",label:"工具集",icon:Wrench},{view:"workspace",label:"工作空间",icon:FolderOpen},{view:"observability",label:"观测分析",icon:Activity}] as const).map(({view,label,icon:Icon}) => <button key={view} className={activeView===view ? "is-active" : ""} aria-current={activeView===view ? "page" : undefined} onClick={()=>{onViewChange(view);onNavigate();}}><Icon size={18} /><span>{label}</span></button>)}</nav>
      <div className="session-search"><Search size={15} /><input aria-label="搜索会话" placeholder="搜索会话…" value={query} onChange={(event) => setQuery(event.target.value)} />{query && <button type="button" aria-label="清除搜索" onClick={() => setQuery("")}><X size={13} /></button>}</div>
      <div className="history-heading"><span>会话历史</span><span className="mono">{sessions.length}</span></div>
      {error && !pendingDelete && <p className="sidebar-error" role="alert">{error}</p>}
      <nav className="session-list" aria-label="历史会话">
        {groups.map((group) => <div className="session-group" key={group.label}><p className="session-group-label">{group.label}</p>{group.sessions.map((session) => (
          <div className={`session-item ${session.id === currentSessionId ? "is-active" : ""}`} key={session.id}>
            <button className="session-select" type="button" aria-current={session.id === currentSessionId ? "page" : undefined} title={session.title} disabled={busy || isStreaming || isSessionBusy} onClick={() => void navigate(() => selectSession(session.id))}>
              <MessageSquare size={15} /><span className="session-text"><span className="session-title">{session.title}</span><span className="session-meta">{session.message_count ? `${session.message_count} 条消息` : "开始一段新的对话"}</span></span>
            </button>
            <button className="session-delete" type="button" aria-label={`删除会话：${session.title}`} title="删除会话" disabled={busy || isStreaming || isSessionBusy} onClick={() => { setError(""); setPendingDelete(session); }}><Trash2 size={14} /></button>
          </div>
        ))}</div>)}
        {!groups.length && <div className="history-empty"><MessageSquare size={24} /><p>{query ? "没有找到相关会话" : "从一段新会话开始"}</p><span>{query ? "试试其他关键词" : "你的聊天记录会显示在这里"}</span></div>}
      </nav>
      <details className="raw-messages"><summary><Code2 size={15} /><span>查看原始消息</span><ChevronDown size={14} className="details-chevron" /></summary><div className="raw-message-list">{messages.length ? messages.map((message) => <div className="raw-message" key={message.id}><div><span>{message.role === "user" ? "你" : "助手"}</span>{message.toolCalls.length > 0 && <span>{message.toolCalls.length} 次工具调用</span>}</div><p>{message.content || "工具执行记录"}</p></div>) : <p className="raw-empty">发送消息后，可在这里查看对话内容。</p>}</div></details>
      <button className={`trash-navigation ${activeView==="trash" ? "is-active" : ""}`} onClick={()=>{onViewChange("trash");onNavigate();}}><Trash2 size={18} /><span>回收站</span></button>
      {pendingDelete && <Modal labelledBy="delete-title" onClose={() => { if (!busy) setPendingDelete(null); }} className="compact-modal"><h2 id="delete-title">移入回收站</h2><p className="delete-description">将「{pendingDelete.title}」移入回收站？聊天记录会保留，之后可以恢复。</p>{error && <p className="form-error" role="alert">{error}</p>}<div className="modal-actions"><button className="secondary-button" type="button" disabled={busy} onClick={() => setPendingDelete(null)}>取消</button><button className="danger-button" type="button" disabled={busy} onClick={() => void confirmDelete()}>{busy ? "正在移动…" : "移入回收站"}</button></div></Modal>}
    </aside>
  );
}
