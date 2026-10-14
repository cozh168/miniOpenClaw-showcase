"use client";

import { useState, type RefObject } from "react";
import { FileText, FolderOpen, Loader2, PanelLeftClose, PanelLeftOpen, PanelRight, Search, Settings2, X } from "lucide-react";
import { VIEW_LABELS, type WorkbenchView } from "@/lib/navigation";
import { useAppStore } from "@/lib/store";

export function Navbar({ sidebarOpen, onToggleSidebar, menuButtonRef, activeView, onViewChange, onSearch, onOpenWorkspace }: {
  sidebarOpen: boolean; onToggleSidebar: () => void; menuButtonRef: RefObject<HTMLButtonElement>;
  activeView: WorkbenchView; onViewChange: (view: WorkbenchView) => void; onSearch: () => void; onOpenWorkspace: () => void;
}) {
  const { ragMode, toggleRagMode, sessions, currentSessionId, isStreaming, isSessionBusy } = useAppStore();
  const [ragBusy, setRagBusy] = useState(false);
  const [error, setError] = useState("");
  const currentTitle = sessions.find((session)=>session.id===currentSessionId)?.title ?? "新会话";
  async function changeRag() {
    setRagBusy(true); setError("");
    try { await toggleRagMode(); } catch { setError("文档检索设置更新失败，请稍后重试。"); } finally { setRagBusy(false); }
  }
  return <>
    <header className="workspace-header">
      <div className="header-left"><button className="icon-button" ref={menuButtonRef} onClick={onToggleSidebar} aria-label={sidebarOpen ? "收起会话侧栏" : "打开会话侧栏"} aria-controls="session-sidebar" aria-expanded={sidebarOpen}>{sidebarOpen ? <PanelLeftClose size={17} /> : <PanelLeftOpen size={17} />}</button><div className="header-breadcrumb"><FolderOpen size={15} /><span>{VIEW_LABELS[activeView]}</span><i>/</i><strong title={activeView==="chat" ? currentTitle : VIEW_LABELS[activeView]}>{activeView==="chat" ? currentTitle : "本地工作台"}</strong></div></div>
      <button className="global-search-button" onClick={onSearch} aria-label="搜索文件、会话或工具"><Search size={16} /><span>搜索文件、会话或工具…</span><kbd>⌘ K</kbd></button>
      <div className="header-actions"><button className={`rag-toggle ${ragMode ? "is-active" : ""}`} aria-pressed={ragMode} aria-label={ragMode ? "关闭文档检索" : "开启文档检索"} disabled={ragBusy || isStreaming || isSessionBusy} onClick={()=>void changeRag()}>{ragBusy ? <Loader2 size={14} className="animate-spin" /> : <span className="rag-dot" />}<span className="rag-label">RAG {ragMode ? "已开启" : "已关闭"}</span><span className="toggle-track"><span /></span></button><button className={`settings-button ${activeView==="logs" ? "is-active" : ""}`} onClick={()=>onViewChange("logs")} aria-label="查看运行日志"><FileText size={16} /><span>运行日志</span></button><button className={`settings-button ${activeView==="settings" ? "is-active" : ""}`} onClick={()=>onViewChange("settings")} aria-label="打开设置"><Settings2 size={17} /><span>设置</span></button><button className="icon-button panel-toggle" onClick={onOpenWorkspace} aria-label="打开文件面板"><PanelRight size={18} /></button><span className="header-avatar" aria-label="本地用户">U</span></div>
    </header>
    {error && <div className="header-error" role="alert">{error}<button onClick={()=>setError("")} aria-label="关闭错误提示"><X size={14} /></button></div>}
  </>;
}
