"use client";

import Editor from "@monaco-editor/react";
import {
  Check,
  ChevronDown,
  ChevronRight,
  Code2,
  Cpu,
  FileCode2,
  FilePlus2,
  Files,
  FileText,
  Folder,
  FolderOpen,
  Loader2,
  RefreshCw,
  Save,
  X
} from "lucide-react";
import { useEffect, useRef, useState, type FormEvent } from "react";

import { Modal } from "@/components/layout/Modal";
import {
  createWorkspaceFile,
  getRuntimeInfo,
  getWorkspaceFiles,
  type RuntimeInfo,
  type WorkspaceFileNode
} from "@/lib/api";
import { useAppStore } from "@/lib/store";

export type WorkbenchTab = "workspace" | "memory" | "skills" | "prompt";

type WorkbenchPanelProps = {
  activeTab: WorkbenchTab;
  onTabChange?: (tab: WorkbenchTab) => void;
  onClose?: () => void;
  onBusyChange?: (busy: boolean) => void;
  externalBusy?: boolean;
};

type RequestKind = "refresh" | "load" | "save" | "create";
type Feedback = { error: boolean; text: string };

const TABS: Array<{ id: WorkbenchTab; label: string; english?: string }> = [
  { id: "workspace", label: "工作空间" },
  { id: "memory", label: "记忆", english: "Memory" },
  { id: "skills", label: "技能", english: "Skills" },
  { id: "prompt", label: "提示词", english: "Prompt" }
];

const PROMPT_FILES = new Set([
  "workspace/AGENTS.md",
  "workspace/IDENTITY.md",
  "workspace/SOUL.md",
  "workspace/USER.md"
]);

const EMPTY_HINTS: Record<WorkbenchTab, string> = {
  workspace: "工作空间里还没有文件。新建文本文件，或把资料放入后端的 workspace / knowledge 目录后刷新。",
  memory: "当前没有记忆文件。长期记忆需要在后端配置中启用；启用前也可以编辑自己的记忆笔记。",
  skills: "当前没有技能文件。将技能放入后端的 skills 目录后刷新。",
  prompt: "当前没有助手设定文件。可以在 workspace 目录新建 AGENTS.md、IDENTITY.md、SOUL.md 或 USER.md。"
};

function isInTab(path: string, tab: WorkbenchTab) {
  if (tab === "workspace") return path === "workspace" || path.startsWith("workspace/") || path === "knowledge" || path.startsWith("knowledge/");
  if (tab === "memory") return path === "memory" || path.startsWith("memory/") || path === "memory_module_v1/long_term_memory" || path.startsWith("memory_module_v1/long_term_memory/");
  if (tab === "skills") return path === "skills" || path.startsWith("skills/") || path === "SKILLS_SNAPSHOT.md";
  return PROMPT_FILES.has(path);
}

function tabFiles(nodes: WorkspaceFileNode[], tab: WorkbenchTab): WorkspaceFileNode[] {
  return nodes.flatMap((node) => {
    if (node.type === "file") return isInTab(node.path, tab) ? [node] : [];
    const children = tabFiles(node.children ?? [], tab);
    return children.length || isInTab(node.path, tab) ? [{ ...node, children }] : [];
  });
}

function flattenFiles(nodes: WorkspaceFileNode[]): WorkspaceFileNode[] {
  return nodes.flatMap((node) => node.type === "file" ? [node] : flattenFiles(node.children ?? []));
}

function findFile(nodes: WorkspaceFileNode[], path: string): WorkspaceFileNode | undefined {
  for (const node of nodes) {
    if (node.path === path) return node;
    const child = findFile(node.children ?? [], path);
    if (child) return child;
  }
  return undefined;
}

function editorLanguage(path: string) {
  const extension = path.split(".").pop()?.toLowerCase();
  const languages: Record<string, string> = {
    py: "python", json: "json", md: "markdown", js: "javascript", jsx: "javascript",
    ts: "typescript", tsx: "typescript", yaml: "yaml", yml: "yaml", html: "html",
    css: "css", sh: "shell", sql: "sql", xml: "xml", toml: "ini"
  };
  return languages[extension ?? ""] ?? "plaintext";
}

function sizeLabel(size: number) {
  if (size < 1024) return `${size} B`;
  return size < 1024 * 1024 ? `${(size / 1024).toFixed(1)} KB` : `${(size / (1024 * 1024)).toFixed(1)} MB`;
}

function errorMessage(error: unknown, fallback: string) {
  if (!(error instanceof Error)) return fallback;
  try {
    const body = JSON.parse(error.message) as { detail?: unknown };
    if (typeof body.detail === "string") return body.detail;
  } catch {
    // Plain network errors use the local, actionable fallback.
  }
  return fallback;
}

function FileTree({ nodes, expanded, selectedPath, disabled, depth = 0, onToggle, onSelect }: {
  nodes: WorkspaceFileNode[];
  expanded: Set<string>;
  selectedPath: string;
  disabled: boolean;
  depth?: number;
  onToggle: (path: string) => void;
  onSelect: (file: WorkspaceFileNode) => void;
}) {
  return (
    <ul className="space-y-0.5">
      {nodes.map((node) => {
        const directory = node.type === "directory";
        const open = expanded.has(node.path);
        const FileIcon = ["python", "json", "javascript", "typescript", "html", "css"].includes(editorLanguage(node.path)) ? FileCode2 : FileText;
        return (
          <li key={node.path}>
            <button
              aria-current={!directory && node.path === selectedPath ? "true" : undefined}
              aria-expanded={directory ? open : undefined}
              className={`flex h-8 w-full items-center gap-2 rounded-lg pr-2 text-left text-[12px] transition disabled:cursor-wait ${node.path === selectedPath && !directory ? "bg-[#e8f2ff] font-medium text-[#122b49]" : "text-[#294a6f] hover:bg-[#f0f6fc]"}`}
              disabled={disabled}
              onClick={() => directory ? onToggle(node.path) : onSelect(node)}
              style={{ paddingLeft: 7 + depth * 12 }}
              title={directory ? node.path : `${node.path} · ${sizeLabel(node.size)}${node.editable ? "" : " · 不支持文本编辑"}`}
              type="button"
            >
              {directory ? <>{open ? <ChevronDown size={12} className="shrink-0 text-[#7189a7]" /> : <ChevronRight size={12} className="shrink-0 text-[#7189a7]" />}{open ? <FolderOpen size={15} className="shrink-0 text-[#6b85a5]" /> : <Folder size={15} className="shrink-0 text-[#6b85a5]" />}</> : <><span className="w-3 shrink-0" /><FileIcon size={14} className={`shrink-0 ${node.editable ? "text-[#148298]" : "text-[#8a9cb1]"}`} /></>}
              <span className="min-w-0 flex-1 truncate">{node.name}</span>
              {directory && <span className="text-[10px] font-normal text-[#92a4b9]">{flattenFiles(node.children ?? []).length}</span>}
            </button>
            {directory && open && <FileTree nodes={node.children ?? []} expanded={expanded} selectedPath={selectedPath} disabled={disabled} depth={depth + 1} onToggle={onToggle} onSelect={onSelect} />}
          </li>
        );
      })}
    </ul>
  );
}

export function WorkbenchPanel({ activeTab, onTabChange, onClose, onBusyChange, externalBusy = false }: WorkbenchPanelProps) {
  const { inspectorPath, inspectorContent, inspectorDirty, isSessionBusy, ragMode, loadInspectorFile, updateInspectorContent, saveInspector } = useAppStore();
  const [files, setFiles] = useState<WorkspaceFileNode[]>([]);
  const [runtime, setRuntime] = useState<RuntimeInfo | null>(null);
  const [runtimeError, setRuntimeError] = useState(false);
  const [listError, setListError] = useState<string | null>(null);
  const [listLoaded, setListLoaded] = useState(false);
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [viewedPath, setViewedPath] = useState(inspectorPath);
  const [requestKind, setRequestKind] = useState<RequestKind | null>(null);
  const [plainText, setPlainText] = useState(false);
  const [feedback, setFeedback] = useState<Feedback | null>(null);
  const [newFileOpen, setNewFileOpen] = useState(false);
  const [newFilePath, setNewFilePath] = useState("workspace/notes.md");
  const [newFileError, setNewFileError] = useState<string | null>(null);
  const startupCompleteRef = useRef(!isSessionBusy);
  if (!isSessionBusy) startupCompleteRef.current = true;
  const awaitingStartup = !startupCompleteRef.current;
  const busy = requestKind !== null || externalBusy || awaitingStartup;
  const externalBusyRef = useRef(externalBusy);
  const awaitingStartupRef = useRef(awaitingStartup);
  externalBusyRef.current = externalBusy;
  awaitingStartupRef.current = awaitingStartup;
  const busyRef = useRef(false);
  const mountedRef = useRef(true);
  const refreshQueuedRef = useRef(false);
  const refreshRef = useRef<() => Promise<void>>(async () => undefined);
  const onBusyChangeRef = useRef(onBusyChange);
  const automaticSelectionRef = useRef<WorkbenchTab | null>(null);
  const approvedTabRef = useRef<WorkbenchTab | null>(null);
  const approvedCreateRef = useRef<{ path: string; content: string } | null>(null);
  onBusyChangeRef.current = onBusyChange;
  const visibleFiles = tabFiles(files, activeTab);
  const leafFiles = flattenFiles(visibleFiles);
  const currentFile = findFile(files, viewedPath);
  const currentEditable = currentFile?.type === "file" && currentFile.editable && viewedPath === inspectorPath;
  const currentName = viewedPath.split("/").pop() || "选择一个文件";

  function beginRequest(kind: RequestKind) {
    if (busyRef.current || externalBusyRef.current || !mountedRef.current || (kind !== "refresh" && awaitingStartupRef.current)) return false;
    busyRef.current = true;
    setRequestKind(kind);
    onBusyChangeRef.current?.(true);
    return true;
  }

  function finishRequest() {
    busyRef.current = false;
    if (mountedRef.current) {
      setRequestKind(null);
      onBusyChangeRef.current?.(false);
    }
    if (refreshQueuedRef.current && mountedRef.current && !externalBusyRef.current) {
      refreshQueuedRef.current = false;
      void refreshRef.current();
    }
  }

  async function refreshFiles() {
    if (!beginRequest("refresh")) {
      refreshQueuedRef.current = true;
      return;
    }
    try {
      const [fileResult, runtimeResult] = await Promise.allSettled([getWorkspaceFiles(), getRuntimeInfo()]);
      if (!mountedRef.current) return;
      if (fileResult.status === "fulfilled") {
        setFiles(fileResult.value.files);
        setExpanded((previous) => new Set([...previous, ...fileResult.value.files.filter((node) => node.type === "directory").map((node) => node.path)]));
        setListError(null);
        setListLoaded(true);
      } else {
        setListError("文件列表加载失败，请确认后端运行后点击刷新。");
      }
      if (runtimeResult.status === "fulfilled") {
        setRuntime(runtimeResult.value);
        setRuntimeError(false);
      } else {
        setRuntimeError(true);
      }
    } finally {
      finishRequest();
    }
  }
  refreshRef.current = refreshFiles;

  async function selectFile(file: WorkspaceFileNode, discardApproved = false) {
    if (busyRef.current || externalBusyRef.current || awaitingStartupRef.current || file.type !== "file" || file.path === viewedPath) return;
    setFeedback(null);
    // Metadata previews keep the current text draft available for returning to it.
    if (!file.editable) {
      setViewedPath(file.path);
      return;
    }
    if (file.path === inspectorPath) { setViewedPath(file.path); return; }
    if (inspectorDirty && !discardApproved && !window.confirm("当前文件有未保存的修改。切换文件会丢弃这些修改，是否继续？")) return;
    if (!beginRequest("load")) return;
    try {
      await loadInspectorFile(file.path);
      if (mountedRef.current) setViewedPath(file.path);
    } catch {
      if (mountedRef.current) setFeedback({ error: true, text: "文件读取失败，原有内容已保留，请重试。" });
    } finally {
      finishRequest();
    }
  }

  async function save() {
    if (!inspectorDirty || !currentEditable || !beginRequest("save")) return;
    setFeedback(null);
    try {
      await saveInspector();
      if (mountedRef.current) setFeedback({ error: false, text: "修改已保存" });
      window.dispatchEvent(new Event("workspace-files-changed"));
    } catch {
      if (mountedRef.current) setFeedback({ error: true, text: "保存失败，请保留编辑内容并重试。" });
    } finally {
      finishRequest();
    }
  }

  function changeTab(tab: WorkbenchTab) {
    if (busyRef.current || externalBusyRef.current || awaitingStartupRef.current || tab === activeTab || !onTabChange) return;
    if (inspectorDirty && !window.confirm("当前文件有未保存的修改。切换标签会丢弃这些修改，是否继续？")) return;
    approvedTabRef.current = tab;
    setFeedback(null);
    onTabChange(tab);
  }

  function openNewFile() {
    if (busyRef.current || externalBusyRef.current || awaitingStartupRef.current) return;
    if (inspectorDirty && !window.confirm("当前文件有未保存的修改。新建并打开文件会丢弃这些修改，是否继续？")) return;
    approvedCreateRef.current = { path: inspectorPath, content: inspectorContent };
    setNewFilePath("workspace/notes.md");
    setNewFileError(null);
    setNewFileOpen(true);
  }

  async function createFile(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (busyRef.current || externalBusyRef.current || awaitingStartupRef.current) return;
    const path = newFilePath.trim().replace(/\\/g, "/");
    if (!/^(workspace|knowledge|memory|skills|memory_module_v1\/long_term_memory)\//.test(path) || path.split("/").some((part) => !part || part === "." || part === "..") || /[\u0000-\u001f]/.test(path)) {
      setNewFileError("请输入有效的相对文件路径，例如 workspace/notes.md。");
      return;
    }
    if (inspectorDirty && (approvedCreateRef.current?.path !== inspectorPath || approvedCreateRef.current.content !== inspectorContent) && !window.confirm("当前文件有未保存的修改。打开新文件会丢弃这些修改，是否继续？")) return;
    if (!beginRequest("create")) return;
    setNewFileError(null);
    let created = false;
    try {
      const result = await createWorkspaceFile(path, "");
      created = true;
      await loadInspectorFile(result.path);
      if (mountedRef.current) {
        setViewedPath(result.path);
        setNewFileOpen(false);
        setFeedback({ error: false, text: "文件已创建，可以开始编辑" });
      }
      const nextTab: WorkbenchTab = result.path.startsWith("skills/") ? "skills" : result.path.startsWith("memory") ? "memory" : PROMPT_FILES.has(result.path) ? "prompt" : "workspace";
      automaticSelectionRef.current = nextTab;
      if (nextTab !== activeTab) onTabChange?.(nextTab);
      window.dispatchEvent(new Event("workspace-files-changed"));
    } catch (error) {
      if (mountedRef.current) setNewFileError(created ? "文件已经创建，但读取失败。请关闭弹窗并刷新文件列表后打开。" : errorMessage(error, "创建失败，请检查文件名；已有文件不会被覆盖。"));
      if (created) window.dispatchEvent(new Event("workspace-files-changed"));
    } finally {
      finishRequest();
    }
  }

  function edit(value: string) {
    if (busyRef.current || externalBusyRef.current || awaitingStartupRef.current || !currentEditable || value === inspectorContent) return;
    setFeedback(null);
    updateInspectorContent(value);
  }

  useEffect(() => {
    mountedRef.current = true;
    if (busyRef.current) onBusyChangeRef.current?.(true);
    const handleFilesChanged = () => void refreshRef.current();
    window.addEventListener("workspace-files-changed", handleFilesChanged);
    void refreshRef.current();
    return () => {
      mountedRef.current = false;
      window.removeEventListener("workspace-files-changed", handleFilesChanged);
      onBusyChangeRef.current?.(false);
    };
  }, []);

  useEffect(() => setViewedPath(inspectorPath), [inspectorPath]);

  useEffect(() => {
    if (!externalBusy && !busyRef.current && refreshQueuedRef.current) {
      refreshQueuedRef.current = false;
      void refreshRef.current();
    }
  }, [externalBusy]);

  useEffect(() => {
    if (!listLoaded || busy || isSessionBusy || automaticSelectionRef.current === activeTab) return;
    automaticSelectionRef.current = activeTab;
    const approved = approvedTabRef.current === activeTab;
    approvedTabRef.current = null;
    if (inspectorDirty && !approved) return;
    const currentVisible = leafFiles.find((file) => file.path === inspectorPath && file.editable);
    const nextFile = currentVisible ?? leafFiles.find((file) => file.editable);
    if (nextFile?.path === inspectorPath) {
      // An external selection has already loaded this file into the shared store.
      // Sync the local view without requesting it a second time.
      setViewedPath(inspectorPath);
    } else if (nextFile) {
      void selectFile(nextFile, approved);
    }
    // Selection runs once per tab; store changes while typing must not reload a file.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeTab, listLoaded, busy, isSessionBusy]);

  const iconButton = "inline-flex h-7 w-7 shrink-0 items-center justify-center rounded-md border border-[#e4edf7] bg-white text-[#6480a1] transition hover:border-[#b9dce4] hover:bg-[#f2f9fc] hover:text-[#006c80] disabled:cursor-not-allowed disabled:opacity-40";

  return (
    <section aria-label="文件工作台" className="flex h-full min-h-0 min-w-0 flex-col gap-3 overflow-hidden bg-[#fbfdff] p-3 text-[#122b49]">
      <header className="flex shrink-0 items-center gap-2">
        <div className="grid min-w-0 flex-1 grid-cols-4 gap-1" aria-label="工作台分类">
          {TABS.map((tab) => <button aria-pressed={activeTab === tab.id} className={`relative flex min-h-10 items-center justify-center gap-1 whitespace-nowrap rounded-lg px-1 text-[11px] transition disabled:cursor-wait ${activeTab === tab.id ? "border border-[#e1edf4] bg-white font-semibold text-[#006c80] shadow-[0_2px_8px_rgba(20,57,93,0.03)] after:absolute after:inset-x-2 after:bottom-0 after:h-[2px] after:rounded-full after:bg-[#006c80]" : "border border-transparent bg-[#f0f5fd] text-[#6681a3] hover:bg-[#e9f1fb]"}`} disabled={busy} key={tab.id} onClick={() => changeTab(tab.id)} type="button">{tab.label}{tab.english && <span className="hidden text-[10px] font-normal min-[390px]:inline">{tab.english}</span>}</button>)}
        </div>
        {onClose && <button aria-label="关闭文件工作台" className={iconButton} disabled={busy} onClick={onClose} type="button"><X size={14} /></button>}
      </header>

      <div className="flex h-[215px] min-h-[130px] shrink-0 flex-col overflow-hidden rounded-xl border border-[#e4edf7] bg-white">
        <div className="flex h-11 shrink-0 items-center justify-between gap-2 px-3">
          <h2 title="最多展示 500 个文件和目录" className="flex min-w-0 items-center gap-2 text-xs font-semibold"><Files size={14} className="text-[#4e7097]" />文件<span className="text-[10px] font-normal text-[#8b9fb6]">{listLoaded ? leafFiles.length : ""}</span></h2>
          <div className="flex items-center gap-1.5">
            <button aria-label="刷新文件列表与运行信息" className={iconButton} disabled={busy} onClick={() => void refreshFiles()} title="刷新文件列表与运行信息" type="button"><RefreshCw size={13} className={requestKind === "refresh" ? "animate-spin" : ""} /></button>
            <button aria-label="新建文本文件" className={iconButton} disabled={busy} onClick={openNewFile} title="新建文本文件" type="button"><FilePlus2 size={13} /></button>
          </div>
        </div>
        <nav aria-label={`${TABS.find((tab) => tab.id === activeTab)?.label}文件`} className="min-h-0 flex-1 overflow-auto px-2 pb-2">
          {listError && <p className="mb-2 rounded-lg bg-red-50 p-2 text-[11px] leading-5 text-red-700" role="alert">{listError}</p>}
          {!listLoaded && requestKind === "refresh" ? <div className="flex h-full items-center justify-center gap-2 text-xs text-[#7b91ab]" role="status"><Loader2 size={14} className="animate-spin" />正在读取文件…</div> : visibleFiles.length ? <FileTree nodes={visibleFiles} expanded={expanded} selectedPath={viewedPath} disabled={busy} onToggle={(path) => setExpanded((previous) => { const next = new Set(previous); if (next.has(path)) next.delete(path); else next.add(path); return next; })} onSelect={(file) => void selectFile(file)} /> : listLoaded && <div className="flex h-full items-center justify-center px-3"><p className="text-xs leading-6 text-[#7b91ab]">{EMPTY_HINTS[activeTab]}</p></div>}
        </nav>
      </div>

      <div className="flex min-h-[180px] min-w-0 flex-1 flex-col overflow-hidden rounded-xl border border-[#e4edf7] bg-white">
        <div className="flex shrink-0 items-center justify-between gap-2 px-3 pb-1 pt-3">
          <h2 className="flex items-center gap-2 text-xs font-semibold"><FileCode2 size={14} className="text-[#4e7097]" />当前文件</h2>
          <button aria-label="保存当前文件" className="inline-flex h-7 shrink-0 items-center gap-1.5 rounded-md bg-[#006c80] px-2.5 text-[11px] font-medium text-white transition hover:bg-[#005c6e] disabled:cursor-not-allowed disabled:opacity-40" disabled={!currentEditable || !inspectorDirty || busy} onClick={() => void save()} type="button">{requestKind === "save" ? <Loader2 size={12} className="animate-spin" /> : <Save size={12} />}{requestKind === "save" ? "保存中" : "保存"}</button>
        </div>
        <div className="flex shrink-0 items-center justify-between gap-2 border-b border-[#edf2f8] px-3 py-2">
          <div className="min-w-0 flex-1"><p className="truncate text-[12px] font-medium" title={viewedPath}>{currentName}</p><p className="mono mt-0.5 truncate text-[9px] text-[#7b91ab]" title={viewedPath}>{viewedPath}</p></div>
          {currentEditable && <button aria-pressed={plainText} className="shrink-0 rounded-md bg-[#f3f7fc] px-2 py-1 text-[10px] text-[#6480a1] transition hover:bg-[#e8f2ff] disabled:opacity-40" disabled={busy} onClick={() => setPlainText((current) => !current)} type="button">{plainText ? "代码编辑器" : "纯文本"}</button>}
        </div>
        <div className="relative min-h-0 flex-1 overflow-hidden">
          {requestKind === "load" || externalBusy || awaitingStartup ? <div className="flex h-full items-center justify-center gap-2 text-xs text-[#7b91ab]" role="status"><Loader2 size={15} className="animate-spin" />正在读取文件…</div> : currentEditable ? plainText ? <textarea aria-label={`编辑 ${currentName}`} className="mono h-full w-full resize-none border-0 bg-[#fcfdff] p-3 text-[12px] leading-[22px] text-[#243e60] outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-[#006c80]" onChange={(event) => edit(event.target.value)} readOnly={busy} spellCheck={false} value={inspectorContent} /> : <Editor height="100%" language={editorLanguage(viewedPath)} loading={<div className="flex flex-col items-center gap-2 px-4 text-center text-xs text-[#7b91ab]" role="status"><Loader2 size={17} className="animate-spin" /><span>编辑器加载中</span><span className="text-[10px] leading-5">可切换「纯文本」继续编辑。</span></div>} onChange={(value) => edit(value ?? "")} options={{ automaticLayout: true, fontFamily: "IBM Plex Mono, Menlo, monospace", fontSize: 12, lineHeight: 22, lineNumbersMinChars: 2, minimap: { enabled: false }, padding: { top: 10, bottom: 10 }, readOnly: busy, renderLineHighlight: "none", scrollBeyondLastLine: false, wordWrap: "on" }} path={viewedPath} theme="vs-light" value={inspectorContent} /> : <div className="flex h-full flex-col items-center justify-center gap-2 px-5 text-center text-xs leading-6 text-[#7b91ab]"><FileText size={23} className="text-[#a0b3c9]" /><p>{currentFile && !currentFile.editable ? "此文件为二进制文件或不支持的文本格式，无法直接编辑。" : "从上方文件列表选择一个文本文件。"}</p>{currentFile && <span className="text-[10px]">文件大小：{sizeLabel(currentFile.size)}</span>}</div>}
        </div>
        <footer className="flex min-h-8 shrink-0 items-center gap-1.5 border-t border-[#edf2f8] px-3 py-1.5 text-[10px] text-[#7b91ab]" role={feedback?.error ? "alert" : "status"}>
          {feedback ? <span className={`flex items-center gap-1.5 ${feedback.error ? "text-red-600" : "text-[#007786]"}`}>{!feedback.error && <Check size={12} />}{feedback.text}</span> : <><span className={`h-1.5 w-1.5 shrink-0 rounded-full ${inspectorDirty ? "bg-amber-500" : "bg-[#12a591]"}`} /><span>{requestKind === "save" ? "正在保存…" : inspectorDirty ? "当前编辑内容尚未保存" : currentEditable ? "文件已同步 · 修改后点击保存" : "支持本地文本文件编辑"}</span></>}
        </footer>
      </div>

      <div className="shrink-0 rounded-xl border border-[#e4edf7] bg-white px-3 py-3">
        <h2 className="mb-2 flex items-center gap-2 text-xs font-semibold"><Cpu size={14} className="text-[#4e7097]" />运行配置</h2>
        {runtime ? <><div className="flex items-center gap-2 rounded-lg border border-[#e4edf7] bg-[#fcfdff] px-2.5 py-2"><Code2 size={14} className="shrink-0 text-[#006c80]" /><span className="min-w-0 flex-1 truncate text-[11px]" title={runtime.python_environment}>Python {runtime.python_version} · {runtime.python_environment}</span><span className="h-1.5 w-1.5 shrink-0 rounded-full bg-[#12a591]" /></div><div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-[10px] text-[#708aab]"><span title={`${runtime.provider} / ${runtime.model}`}>{runtime.model}</span><span>记忆：{runtime.memory_backend === "off" ? "关闭" : runtime.memory_backend}</span><span>RAG：{ragMode ? "开启" : "关闭"}</span></div></> : <p className="text-[11px] leading-5 text-[#7b91ab]">{runtimeError ? "运行信息读取失败，请点击上方刷新按钮重试。" : "正在读取运行信息…"}</p>}
        {runtime && runtimeError && <p className="mt-1 text-[10px] text-amber-700" role="status">运行信息刷新失败，当前显示上次读取的配置。</p>}
        <p className="mt-2 text-[10px] leading-5 text-[#8096b0]">代码与工具由本地后端执行，模型回答通过 API 生成。</p>
      </div>

      {newFileOpen && <Modal labelledBy="new-workspace-file-title" onClose={() => { if (!busyRef.current && !externalBusyRef.current) setNewFileOpen(false); }} className="w-full max-w-md border border-[#e4edf7] bg-white p-5"><form onSubmit={(event) => void createFile(event)}><div className="flex items-center justify-between gap-3"><h2 className="text-base font-semibold text-[#122b49]" id="new-workspace-file-title">新建文本文件</h2><button aria-label="关闭新建文件窗口" className={iconButton} disabled={busy} onClick={() => setNewFileOpen(false)} type="button"><X size={14} /></button></div><label className="mb-2 mt-5 block text-xs font-medium text-[#526f91]" htmlFor="new-workspace-file-path">文件路径</label><input autoComplete="off" className="mono h-10 w-full rounded-lg border border-[#dce7f3] bg-[#fcfdff] px-3 text-xs text-[#122b49] outline-none focus:border-[#006c80] focus:ring-2 focus:ring-[#006c80]/10" data-autofocus disabled={busy} id="new-workspace-file-path" onChange={(event) => setNewFilePath(event.target.value)} placeholder="workspace/notes.md" required value={newFilePath} /><p className="mt-2 text-[11px] leading-5 text-[#7b91ab]">以 workspace/、knowledge/、memory/ 或 skills/ 开头。可以包含子目录，已有文件不会被覆盖。</p>{newFileError && <p className="mt-3 rounded-lg bg-red-50 px-3 py-2 text-xs leading-5 text-red-700" role="alert">{newFileError}</p>}<div className="mt-5 flex justify-end gap-2"><button className="rounded-lg border border-[#e4edf7] px-4 py-2 text-xs text-[#5c7798] hover:bg-[#f3f7fc] disabled:opacity-40" disabled={busy} onClick={() => setNewFileOpen(false)} type="button">取消</button><button className="inline-flex items-center gap-2 rounded-lg bg-[#006c80] px-4 py-2 text-xs font-medium text-white hover:bg-[#005c6e] disabled:opacity-40" disabled={busy} type="submit">{requestKind === "create" && <Loader2 size={13} className="animate-spin" />}{requestKind === "create" ? "创建中" : "创建并打开"}</button></div></form></Modal>}
    </section>
  );
}
