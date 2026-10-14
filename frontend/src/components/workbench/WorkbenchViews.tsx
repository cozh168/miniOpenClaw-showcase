"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Activity, Bot, ChevronRight, Database, FileText, FolderOpen, RefreshCw, RotateCcw, Settings2, Terminal, Wrench } from "lucide-react";
import { getRuntimeInfo, getToolAudit, getWorkspaceFiles, listTrashedSessions, type RuntimeInfo, type SessionSummary, type ToolAuditEntry, type WorkspaceFileNode } from "@/lib/api";
import { flattenFiles, VIEW_LABELS, type WorkbenchView } from "@/lib/navigation";
import { useAppStore } from "@/lib/store";

const TOOL_DESCRIPTIONS: Record<string, string> = {
  python_repl: "运行 Python，完成计算、数据处理和脚本任务。", terminal: "在本地后端工作目录执行终端命令。", read_file: "读取允许访问的本地文件。", fetch_url: "获取网页内容，供助手分析。", search_knowledge_base: "检索本地知识库中的相关文档。", memory_search: "检索长期记忆中的相关信息。"
};
const STATUS_LABELS: Record<string, string> = { success: "成功", error: "失败", blocked: "已阻止", timeout: "超时" };

export function WorkbenchViews({ view, onFile, onChat }: { view: Exclude<WorkbenchView, "chat">; onFile: (path: string) => Promise<void>; onChat: () => void }) {
  const { sessions, tokenStats, restoreDeletedSession, isStreaming, isSessionBusy, ragMode } = useAppStore();
  const [runtime, setRuntime] = useState<RuntimeInfo | null>(null);
  const [files, setFiles] = useState<WorkspaceFileNode[]>([]);
  const [audit, setAudit] = useState<ToolAuditEntry[]>([]);
  const [trash, setTrash] = useState<SessionSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [query, setQuery] = useState("");
  const [status, setStatus] = useState("");
  const loadSequence = useRef(0);
  const invalidateLoad = useCallback(() => { loadSequence.current++; }, []);
  const load = useCallback(async () => {
    const sequence = ++loadSequence.current;
    setLoading(true); setError("");
    try {
      if (view === "trash") { const result = await listTrashedSessions(); if (sequence === loadSequence.current) setTrash(result); }
      else if (view === "logs" || view === "observability") { const result = await getToolAudit(); if (sequence === loadSequence.current) setAudit(result.items); }
      else {
        const results = await Promise.allSettled([getRuntimeInfo(), getWorkspaceFiles()]);
        if (sequence !== loadSequence.current) return;
        if (results[0].status === "fulfilled") setRuntime(results[0].value);
        if (results[1].status === "fulfilled") setFiles(flattenFiles(results[1].value.files));
        if (results.some((result) => result.status === "rejected")) setError("部分内容加载失败，请确认后端连接后刷新。");
      }
    } catch { if (sequence === loadSequence.current) setError("内容加载失败，请确认后端连接后刷新。"); }
    finally { if (sequence === loadSequence.current) setLoading(false); }
  }, [view]);
  useEffect(() => { void load(); setQuery(""); setStatus(""); return invalidateLoad; }, [load, invalidateLoad]);
  async function perform(action: () => Promise<void>) {
    setBusy(true); setError("");
    try { await action(); } catch { setError("操作失败，请稍后重试。未保存的编辑内容已保留。"); } finally { setBusy(false); }
  }
  const title = VIEW_LABELS[view];
  const description: Record<Exclude<WorkbenchView, "chat">, string> = {
    agent: "查看当前助手的模型与配置，编辑文件来调整它的行为。", knowledge: "管理本地知识资料，使用顶部 RAG 开关控制文档检索。", tools: "查看当前工具状态，以及助手可以完成的任务。", workspace: "浏览本地文件，在右侧打开并编辑文本文件。", observability: "根据实际会话与最近工具记录查看运行情况。", logs: "查看工具执行输入、输出与状态，记录保存在本机。", settings: "当前后端的实际配置摘要。", trash: "已删除的会话可以恢复，聊天记录会一并保留。"
  };
  const visibleFiles = files.filter((file) => (view !== "knowledge" || file.path.startsWith("knowledge/")) && file.path.toLocaleLowerCase().includes(query.toLocaleLowerCase()));
  const visibleAudit = audit.filter((entry) => (!status || entry.status === status) && `${entry.tool} ${entry.input_preview} ${entry.output_preview}`.toLocaleLowerCase().includes(query.toLocaleLowerCase()));
  const success = audit.filter((entry) => entry.status === "success").length;
  const durations = audit.map((entry) => entry.duration_ms).filter((duration): duration is number => duration !== null);
  return <section className="workbench-view" aria-label={title}>
    <div className="view-heading"><div><p className="view-eyebrow">MINI-OPENCLAW / {view.toUpperCase()}</p><h1>{title}</h1><p>{description[view]}</p></div><button className="secondary-button" disabled={loading || busy} onClick={() => void load()}><RefreshCw size={14} className={loading ? "animate-spin" : ""} />刷新</button></div>
    {error && <p className="view-alert" role="alert">{error}</p>}
    {loading && <p className="view-loading" role="status">正在加载实际运行数据…</p>}
    {view === "agent" && <><div className="agent-profile view-card"><span className="profile-icon"><Bot size={26} /></span><div><h2>默认智能体 · Mini-OpenClaw</h2><p>当前使用 {runtime?.provider ?? "…"} / {runtime?.model ?? "…"}</p></div><button className="primary-button" onClick={onChat}>开始对话<ChevronRight size={14} /></button></div><div className="view-grid">{[{path:"workspace/SOUL.md",name:"行为与风格",detail:"定义助手的语气、价值观与工作方式。"},{path:"workspace/IDENTITY.md",name:"助手身份",detail:"配置助手名称、角色和身份。"},{path:"workspace/USER.md",name:"用户信息",detail:"让助手了解你的偏好和背景。"},{path:"workspace/AGENTS.md",name:"工作规则",detail:"配置文件与任务的处理规则。"}].map((item) => <button className="view-card config-card" key={item.path} disabled={busy} onClick={() => void perform(() => onFile(item.path))}><FileText size={20} /><h3>{item.name}</h3><p>{item.detail}</p><span>编辑 {item.path.split("/").pop()}<ChevronRight size={14} /></span></button>)}</div></>}
    {(view === "workspace" || view === "knowledge") && <><div className="view-toolbar"><div className="view-search"><input aria-label={`搜索${title}文件`} placeholder="按文件名或路径搜索…" value={query} onChange={(event) => setQuery(event.target.value)} /></div><span>{visibleFiles.length} 个文件{view === "knowledge" ? ` · RAG ${ragMode ? "已开启" : "已关闭"}` : ""}</span></div><div className="file-overview view-card">{visibleFiles.map((file) => <button key={file.path} disabled={!file.editable || busy} onClick={() => void perform(() => onFile(file.path))}><FileText size={18} /><span><strong>{file.name}</strong><small>{file.path}</small></span><small>{file.editable ? "打开编辑" : "二进制文件"}</small><ChevronRight size={14} /></button>)}{!loading && !visibleFiles.length && <div className="view-empty"><FolderOpen size={28} /><p>暂时没有{query ? "匹配的" : ""}文件</p><span>{view === "knowledge" ? "将资料放入 backend/knowledge，再刷新文件列表。" : "可使用右侧文件面板的新建按钮创建文本文件。"}</span></div>}</div></>}
    {view === "tools" && <div className="view-grid">{runtime?.tools.map((tool) => <div className="view-card tool-profile" key={tool.name}><Wrench size={22} /><span className={`status-pill ${tool.enabled ? "status-success" : ""}`}>{tool.enabled ? "已启用" : "已关闭"}</span><h3 className="mono">{tool.name}</h3><p>{TOOL_DESCRIPTIONS[tool.name] ?? "由当前后端提供，助手按任务需要调用。"}</p></div>)}</div>}
    {view === "settings" && <div className="view-card settings-summary"><h2><Settings2 size={18} />运行配置</h2><dl>{[["模型服务",runtime?.provider],["当前模型",runtime?.model],["Python",runtime?.python_version],["运行环境",runtime?.python_environment],["记忆后端",runtime?.memory_backend === "off" ? "未启用记忆检索" : runtime?.memory_backend],["文档检索",ragMode ? "已开启" : "已关闭"]].map(([key,value]) => <div key={key}><dt>{key}</dt><dd>{value ?? "加载中…"}</dd></div>)}</dl><p>模型和工具开关由本地 .env 配置；修改后需要重启后端。右侧记忆、技能和提示词文件可直接编辑保存。</p></div>}
    {view === "observability" && <><div className="metrics-grid">{[{label:"本地会话",value:sessions.length,icon:Bot},{label:"最近工具执行",value:audit.length,icon:Wrench},{label:"工具成功率",value:audit.length ? `${Math.round(success/audit.length*100)}%` : "—",icon:Activity},{label:"平均执行耗时",value:durations.length ? `${Math.round(durations.reduce((a,b)=>a+b,0)/durations.length)} ms` : "—",icon:Terminal}].map(({label,value,icon:Icon}) => <div className="view-card metric-card" key={label}><Icon size={18} /><span>{label}</span><strong>{value}</strong></div>)}</div><div className="view-card"><h2>工具调用分布</h2><p className="view-note">基于最近 {audit.length} 次记录（最多 100 次）。耗时不包含模型生成时间。</p>{Array.from(new Set(audit.map((item)=>item.tool))).map((tool) => { const count=audit.filter((item)=>item.tool===tool).length; return <div className="tool-bar" key={tool}><span className="mono">{tool}</span><div><i style={{width:`${count/Math.max(1,audit.length)*100}%`}} /></div><strong>{count}</strong></div>; })}{!audit.length && <p className="view-empty">调用工具后会在这里显示实际数据。</p>}</div>{tokenStats && <p className="view-note">当前会话上下文估算：系统 {tokenStats.system_tokens.toLocaleString()} tokens，消息 {tokenStats.message_tokens.toLocaleString()} tokens。该数值不代表服务商账单。</p>}</>}
    {view === "logs" && <><div className="view-toolbar"><div className="view-search"><input aria-label="搜索运行日志" placeholder="搜索工具、输入或输出…" value={query} onChange={(event)=>setQuery(event.target.value)} /></div><select aria-label="筛选执行状态" value={status} onChange={(event)=>setStatus(event.target.value)}><option value="">全部状态</option>{Object.entries(STATUS_LABELS).map(([value,label])=><option key={value} value={value}>{label}</option>)}</select></div><div className="audit-list">{visibleAudit.map((entry,index) => <details className="view-card audit-entry" key={`${entry.timestamp}-${index}`}><summary><Terminal size={17} /><strong className="mono">{entry.tool}</strong><span className={`status-pill status-${entry.status}`}>{STATUS_LABELS[entry.status] ?? entry.status}</span><time>{new Date(entry.timestamp).toLocaleString("zh-CN")}</time><ChevronRight size={15} /></summary><div className="audit-detail"><p>输入</p><pre className="mono">{entry.input_preview || "无文本输入"}</pre><p>输出{entry.duration_ms !== null ? ` · ${entry.duration_ms} ms` : ""}</p><pre className="mono">{entry.output_preview || entry.reason || "无文本输出"}</pre><small>日志中的长内容可能已截断；完整工具结果可在对应会话中查看。</small></div></details>)}{!loading && !visibleAudit.length && <div className="view-empty">没有匹配的工具记录。</div>}</div></>}
    {view === "trash" && <div className="view-card trash-list">{trash.map((item) => <div key={item.id}><RotateCcw size={19} /><span><strong>{item.title}</strong><small>{item.message_count} 条消息 · {new Date(item.updated_at*1000).toLocaleDateString("zh-CN")}</small></span><button className="secondary-button" disabled={busy || isStreaming || isSessionBusy} onClick={() => void perform(async () => { await restoreDeletedSession(item.id); setTrash((items)=>items.filter((row)=>row.id!==item.id)); })}>恢复会话</button></div>)}{!loading && !trash.length && <div className="view-empty"><RotateCcw size={28} /><p>回收站是空的</p><span>从侧栏删除的会话会显示在这里。</span></div>}</div>}
  </section>;
}
