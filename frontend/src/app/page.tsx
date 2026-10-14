"use client";

import { useEffect, useRef, useState, type CSSProperties } from "react";
import { ChatPanel } from "@/components/chat/ChatPanel";
import { WorkbenchPanel, type WorkbenchTab } from "@/components/editor/WorkbenchPanel";
import { Navbar } from "@/components/layout/Navbar";
import { isModalOpen } from "@/components/layout/Modal";
import { ResizeHandle } from "@/components/layout/ResizeHandle";
import { Sidebar } from "@/components/layout/Sidebar";
import { GlobalSearch } from "@/components/workbench/GlobalSearch";
import { WorkbenchViews } from "@/components/workbench/WorkbenchViews";
import type { WorkbenchView } from "@/lib/navigation";
import { AppProvider, useAppStore } from "@/lib/store";

function Workspace() {
  const { sidebarWidth, setSidebarWidth, inspectorDirty, loadInspectorFile, isSessionBusy } = useAppStore();
  const [activeView, setActiveView] = useState<WorkbenchView>("chat");
  const [activeTab, setActiveTab] = useState<WorkbenchTab>("workspace");
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [panelOpen, setPanelOpen] = useState(true);
  const [smallScreen, setSmallScreen] = useState(false);
  const [searchOpen, setSearchOpen] = useState(false);
  const [externalBusy, setExternalBusy] = useState(false);
  const panelBusyRef = useRef(false);
  const externalBusyRef = useRef(false);
  const sidebarRef = useRef<HTMLDivElement>(null);
  const panelRef = useRef<HTMLElement>(null);
  const menuButtonRef = useRef<HTMLButtonElement>(null);
  const closeSidebar = () => { setSidebarOpen(false); menuButtonRef.current?.focus(); };
  const closePanel = () => { if (!panelBusyRef.current && !externalBusyRef.current) { setPanelOpen(false); document.querySelector<HTMLButtonElement>('.panel-toggle')?.focus(); } };
  const navigate = (view: WorkbenchView) => {
    setActiveView(view);
    if (view === "workspace") { setActiveTab("workspace"); setPanelOpen(true); }
    if (view === "knowledge") setActiveTab("workspace");
    if (window.matchMedia("(max-width: 767px)").matches) setSidebarOpen(false);
  };
  async function openFile(path: string) {
    if (panelBusyRef.current || externalBusyRef.current || isSessionBusy) throw new Error("工作台仍在加载，请稍后重试。");
    if (inspectorDirty && !window.confirm("有尚未保存的修改，确定切换文件并放弃修改吗？")) return;
    externalBusyRef.current = true; setExternalBusy(true);
    try {
      await loadInspectorFile(path);
      setActiveTab(path.startsWith("skills/") || path==="SKILLS_SNAPSHOT.md" ? "skills" : path.startsWith("memory") ? "memory" : ["SOUL.md","IDENTITY.md","USER.md","AGENTS.md"].some((name)=>path===`workspace/${name}`) ? "prompt" : "workspace");
      setPanelOpen(true); setSidebarOpen(!window.matchMedia("(max-width: 767px)").matches);
    } finally { externalBusyRef.current = false; setExternalBusy(false); }
  }
  useEffect(() => {
    const mobile = window.matchMedia("(max-width: 767px)");
    const narrow = window.matchMedia("(max-width: 1199px)");
    const updateMobile = () => setSidebarOpen(!mobile.matches);
    const updateNarrow = () => { setSmallScreen(narrow.matches); setPanelOpen(!narrow.matches); };
    updateMobile(); updateNarrow(); mobile.addEventListener("change",updateMobile); narrow.addEventListener("change",updateNarrow);
    return () => { mobile.removeEventListener("change",updateMobile); narrow.removeEventListener("change",updateNarrow); };
  }, []);
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase()==="k" && !event.isComposing) { event.preventDefault(); if (!isModalOpen()) setSearchOpen(true); }
    };
    document.addEventListener("keydown",onKey); return ()=>document.removeEventListener("keydown",onKey);
  }, []);
  useEffect(() => {
    if (!inspectorDirty) return;
    const protectDraft = (event: BeforeUnloadEvent) => { event.preventDefault(); event.returnValue = ""; };
    window.addEventListener("beforeunload", protectDraft);
    return () => window.removeEventListener("beforeunload", protectDraft);
  }, [inspectorDirty]);
  useEffect(() => {
    const sideDrawer = sidebarOpen && window.matchMedia("(max-width: 767px)").matches;
    const fileDrawer = panelOpen && smallScreen;
    if (!sideDrawer && !fileDrawer) return;
    const panel = fileDrawer ? panelRef.current : sidebarRef.current;
    const previous = document.activeElement as HTMLElement | null;
    panel?.querySelector<HTMLElement>('button:not([disabled])')?.focus();
    const handleKey = (event: KeyboardEvent) => {
      if (isModalOpen()) return;
      if (event.key === "Escape" && !event.isComposing) { if (fileDrawer) closePanel(); else closeSidebar(); }
      if (event.key !== "Tab") return;
      const items = Array.from(panel?.querySelectorAll<HTMLElement>('button:not([disabled]), input:not([disabled]), textarea:not([disabled]), select:not([disabled]), a[href], summary, [tabindex="0"]') ?? []).filter((item)=>item.getClientRects().length>0);
      const first=items[0],last=items[items.length-1];
      if (!first) { event.preventDefault(); panel?.focus(); }
      else if (!panel?.contains(document.activeElement)) { event.preventDefault(); (event.shiftKey ? last : first)?.focus(); }
      else if (event.shiftKey && document.activeElement===first) { event.preventDefault();last?.focus(); }
      else if (!event.shiftKey && document.activeElement===last) { event.preventDefault();first?.focus(); }
    };
    document.addEventListener("keydown",handleKey);
    return ()=>{document.removeEventListener("keydown",handleKey);previous?.focus();};
  }, [sidebarOpen,panelOpen,smallScreen]);
  const showFiles = () => { setPanelOpen(true); if (smallScreen) setSidebarOpen(false); };
  return <main className="workspace-shell" style={{"--sidebar-width":`${Math.min(380,Math.max(224,sidebarWidth))}px`} as CSSProperties}>
    {sidebarOpen && <button className="sidebar-backdrop" onClick={closeSidebar} aria-label="关闭会话侧栏" />}
    <div className="workspace-sidebar" data-open={sidebarOpen} id="session-sidebar" ref={sidebarRef}><Sidebar activeView={activeView} onViewChange={navigate} onNavigate={()=>{if(window.matchMedia("(max-width: 767px)").matches)closeSidebar();}} onClose={closeSidebar} /></div>
    {sidebarOpen && <div className="workspace-resizer"><ResizeHandle width={sidebarWidth} onResize={(delta)=>setSidebarWidth(Math.min(380,Math.max(224,sidebarWidth+delta)))} /></div>}
    <div className="workspace-main"><Navbar sidebarOpen={sidebarOpen} onToggleSidebar={()=>{setSidebarOpen((open)=>!open);if(smallScreen)setPanelOpen(false);}} menuButtonRef={menuButtonRef} activeView={activeView} onViewChange={navigate} onSearch={()=>setSearchOpen(true)} onOpenWorkspace={showFiles} />
      <div className="workspace-columns"><div className="workspace-conversation"><div className="chat-surface" style={{display:activeView==="chat" ? "flex" : "none"}}><ChatPanel onOpenWorkspace={showFiles} onOpenAgent={()=>navigate("agent")} /></div>{activeView!=="chat" && <WorkbenchViews view={activeView} onFile={openFile} onChat={()=>navigate("chat")} />}</div>
        {panelOpen && smallScreen && <button className="file-panel-backdrop" aria-label="关闭文件面板遮罩" onClick={closePanel} />}
        <aside ref={panelRef} className="workspace-files" data-open={panelOpen} role={smallScreen && panelOpen ? "dialog" : undefined} aria-modal={smallScreen && panelOpen ? true : undefined} aria-label="文件工作台" tabIndex={-1}><WorkbenchPanel activeTab={activeTab} onTabChange={setActiveTab} onClose={closePanel} onBusyChange={(busy)=>{panelBusyRef.current=busy;}} externalBusy={externalBusy} /></aside>
      </div>
    </div>
    {searchOpen && <GlobalSearch onClose={()=>setSearchOpen(false)} onFile={openFile} onChat={()=>navigate("chat")} onTools={()=>navigate("tools")} />}
  </main>;
}
export default function Page() { return <AppProvider><Workspace /></AppProvider>; }
