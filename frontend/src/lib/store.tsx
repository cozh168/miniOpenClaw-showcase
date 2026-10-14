"use client";

import { createContext, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";

import {
  compressSession,
  createSession,
  deleteSession,
  getRagMode,
  getSessionHistory,
  getSessionTokens,
  listSessions,
  listSkills,
  loadFile,
  renameSession,
  restoreSession,
  saveFile,
  setRagMode,
  streamChat,
  type RetrievalResult,
  type SessionSummary,
  type ToolCall
} from "@/lib/api";

type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  toolCalls: ToolCall[];
  retrievals: RetrievalResult[];
};

type TokenStats = {
  system_tokens: number;
  message_tokens: number;
  total_tokens: number;
};

type AppStore = {
  sessions: SessionSummary[];
  currentSessionId: string | null;
  messages: Message[];
  isStreaming: boolean;
  isSessionBusy: boolean;
  sessionError: string | null;
  ragMode: boolean;
  skills: Array<{ name: string; description: string; path: string }>;
  editableFiles: string[];
  inspectorPath: string;
  inspectorContent: string;
  inspectorDirty: boolean;
  sidebarWidth: number;
  inspectorWidth: number;
  tokenStats: TokenStats | null;
  createNewSession: () => Promise<void>;
  selectSession: (sessionId: string) => Promise<void>;
  sendMessage: (value: string) => Promise<void>;
  toggleRagMode: () => Promise<void>;
  renameCurrentSession: (title: string) => Promise<void>;
  removeSession: (sessionId: string) => Promise<void>;
  refreshSessions: () => Promise<void>;
  restoreDeletedSession: (sessionId: string) => Promise<void>;
  loadInspectorFile: (path: string) => Promise<void>;
  updateInspectorContent: (value: string) => void;
  saveInspector: () => Promise<void>;
  compressCurrentSession: () => Promise<void>;
  setSidebarWidth: (width: number) => void;
  setInspectorWidth: (width: number) => void;
};

const FIXED_FILES = [
  "workspace/SOUL.md",
  "workspace/IDENTITY.md",
  "workspace/USER.md",
  "workspace/AGENTS.md",
  "memory_module_v1/long_term_memory/MEMORY.md",
  "SKILLS_SNAPSHOT.md"
];

const StoreContext = createContext<AppStore | null>(null);

function makeId() {
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function mergeRetrievals(
  current: RetrievalResult[],
  incoming: RetrievalResult[]
) {
  const merged = [...current];
  const seen = new Set(
    current.map((item) => `${item.kind ?? "memory"}::${item.source}::${item.text}`)
  );

  for (const item of incoming) {
    const key = `${item.kind ?? "memory"}::${item.source}::${item.text}`;
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    merged.push(item);
  }

  return merged;
}

function toUiMessages(history: Awaited<ReturnType<typeof getSessionHistory>>["messages"]) {
  return history.map((message) => ({
    id: makeId(),
    role: message.role,
    content: message.content ?? "",
    toolCalls: (message.tool_calls ?? []).map((call) => ({ ...call, completed: true })),
    retrievals: []
  }));
}

export function AppProvider({ children }: { children: ReactNode }) {
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const sendLockRef = useRef(false);
  const sessionLockRef = useRef(true);
  const sessionListRequestRef = useRef(0);
  const ragLockRef = useRef(false);
  const initializedRef = useRef(false);
  const [isSessionBusy, setIsSessionBusy] = useState(true);
  const [sessionError, setSessionError] = useState<string | null>(null);
  const [ragMode, setRagModeState] = useState(false);
  const [skills, setSkills] = useState<Array<{ name: string; description: string; path: string }>>([]);
  const [inspectorPath, setInspectorPath] = useState(
    "memory_module_v1/long_term_memory/MEMORY.md"
  );
  const [inspectorContent, setInspectorContent] = useState("");
  const [inspectorDirty, setInspectorDirty] = useState(false);
  const inspectorLockRef = useRef(false);
  const inspectorStateRef = useRef({ path: inspectorPath, content: inspectorContent });
  inspectorStateRef.current = { path: inspectorPath, content: inspectorContent };
  const [sidebarWidth, setSidebarWidth] = useState(264);
  const [inspectorWidth, setInspectorWidth] = useState(390);
  const [tokenStats, setTokenStats] = useState<TokenStats | null>(null);

  const editableFiles = useMemo(
    () => [...FIXED_FILES, ...skills.map((skill) => skill.path)],
    [skills]
  );

  async function refreshSessions() {
    const sequence = ++sessionListRequestRef.current;
    const result = await listSessions();
    if (sequence === sessionListRequestRef.current) setSessions(result);
  }

  async function refreshSkills() {
    setSkills(await listSkills());
  }

  async function refreshSessionDetails(sessionId: string) {
    const [history, tokens] = await Promise.allSettled([
      getSessionHistory(sessionId),
      getSessionTokens(sessionId)
    ]);
    if (history.status === "rejected") throw history.reason;
    setMessages(toUiMessages(history.value.messages));
    setTokenStats(tokens.status === "fulfilled" ? tokens.value : null);
  }

  async function createNewSession() {
    if (sessionLockRef.current || sendLockRef.current) throw new Error("会话操作尚未完成，请稍后重试。");
    sessionLockRef.current = true;
    setIsSessionBusy(true);
    setSessionError(null);
    try {
      const created = await createSession();
      setCurrentSessionId(created.id);
      setMessages([]);
      setTokenStats(null);
      // The session is already created even if a later list refresh fails.
      setSessions((previous) => [created, ...previous.filter((item) => item.id !== created.id)]);
      await refreshSessions().catch(() => setSessionError("会话已创建，但列表刷新失败。请稍后刷新页面。"));
    } finally {
      sessionLockRef.current = false;
      setIsSessionBusy(false);
    }
  }

  async function selectSession(sessionId: string) {
    if (sessionLockRef.current || sendLockRef.current) throw new Error("会话操作尚未完成，请稍后重试。");
    sessionLockRef.current = true;
    setIsSessionBusy(true);
    setSessionError(null);
    try {
      await refreshSessionDetails(sessionId);
      setCurrentSessionId(sessionId);
    } finally {
      sessionLockRef.current = false;
      setIsSessionBusy(false);
    }
  }

  async function ensureSession() {
    if (currentSessionId) {
      return currentSessionId;
    }

    const created = await createSession();
    setCurrentSessionId(created.id);
    setSessions((previous) => [created, ...previous.filter((item) => item.id !== created.id)]);
    await refreshSessions().catch(() => undefined);
    return created.id;
  }

  async function sendMessage(value: string) {
    if (!value.trim()) return;
    if (sendLockRef.current || isStreaming) throw new Error("正在回复，请稍后发送。");
    if (sessionLockRef.current || ragLockRef.current) throw new Error("正在更新会话或配置，请稍后发送。");
    sendLockRef.current = true;
    setIsStreaming(true);
    try { await sendMessageUnlocked(value); }
    finally { sendLockRef.current = false; setIsStreaming(false); }
  }

  async function sendMessageUnlocked(value: string) {
    const sessionId = await ensureSession();
    setSessionError(null);
    const userMessage: Message = {
      id: makeId(),
      role: "user",
      content: value.trim(),
      toolCalls: [],
      retrievals: []
    };
    const assistantMessage: Message = {
      id: makeId(),
      role: "assistant",
      content: "",
      toolCalls: [],
      retrievals: []
    };

    setMessages((prev) => [...prev, userMessage, assistantMessage]);
    setIsStreaming(true);

    let activeAssistantId = assistantMessage.id;
    let streamError: Error | null = null;
    let streamFailed = false;
    let receivedDone = false;

    const patchAssistant = (updater: (message: Message) => Message) => {
      const assistantId = activeAssistantId;
      setMessages((prev) =>
        prev.map((message) => (message.id === assistantId ? updater(message) : message))
      );
    };

    try {
      await streamChat(
        { message: value.trim(), session_id: sessionId },
        {
          onEvent(event, data) {
            if (event === "retrieval") {
              patchAssistant((message) => ({
                ...message,
                retrievals: mergeRetrievals(
                  message.retrievals,
                  (data.results as RetrievalResult[]) ?? []
                )
              }));
              return;
            }

            if (event === "token") {
              patchAssistant((message) => ({
                ...message,
                content: `${message.content}${String(data.content ?? "")}`
              }));
              return;
            }

            if (event === "tool_start") {
              patchAssistant((message) => ({
                ...message,
                toolCalls: [
                  ...message.toolCalls,
                  {
                    tool: String(data.tool ?? "tool"),
                    input: String(data.input ?? ""),
                    output: "",
                    completed: false,
                    ...(typeof data.call_id === "string" ? { call_id: data.call_id } : {})
                  }
                ]
              }));
              return;
            }

            if (event === "tool_end") {
              patchAssistant((message) => {
                const callId = typeof data.call_id === "string" ? data.call_id : undefined;
                let targetIndex = callId ? message.toolCalls.findIndex((call) => call.call_id === callId) : -1;
                if (targetIndex < 0) targetIndex = message.toolCalls.findIndex((call) => call.tool === data.tool && call.completed !== true);
                if (targetIndex < 0) targetIndex = message.toolCalls.length - 1;
                const updatedToolCalls = message.toolCalls.map((toolCall, index) =>
                  index === targetIndex
                    ? { ...toolCall, output: String(data.output ?? ""), completed: true }
                    : toolCall
                );
                return {
                  ...message,
                  toolCalls: updatedToolCalls
                };
              });
              return;
            }

            if (event === "new_response") {
              const nextAssistant: Message = {
                id: makeId(),
                role: "assistant",
                content: "",
                toolCalls: [],
                retrievals: []
              };
              activeAssistantId = nextAssistant.id;
              setMessages((prev) => [...prev, nextAssistant]);
              return;
            }

            if (event === "done") {
              receivedDone = true;
              const finalContent = String(data.content ?? "");
              patchAssistant((message) =>
                message.content
                  ? message
                  : {
                      ...message,
                      content: finalContent
                    }
              );
              return;
            }

            if (event === "title") {
              void refreshSessions().catch(() => undefined);
              return;
            }

            if (event === "error" && !receivedDone) {
              streamError = new Error(String(data.error ?? "模型请求失败"));
              patchAssistant((message) => ({
                ...message,
                content: `${message.content ? `${message.content}\n\n` : ""}请求失败：${streamError?.message}`
              }));
            }
          }
        }
      );
      if (!receivedDone) throw streamError ?? new Error("回复连接中断，请重试。");
    } catch (error) {
      if (!receivedDone) {
        streamFailed = true;
        if (!streamError) {
          patchAssistant((message) => ({ ...message, content: "发送失败，请检查连接后重试。" }));
        }
        throw error;
      }
    } finally {
      try {
        await refreshSessions();
        if (!streamFailed) await refreshSessionDetails(sessionId);
      } catch {
        // Keep the live reply if refreshing the persisted history fails.
      }
    }
  }

  async function toggleRagMode() {
    if (ragLockRef.current || sessionLockRef.current || sendLockRef.current) throw new Error("当前操作尚未完成，请稍后重试。");
    ragLockRef.current = true;
    const next = !ragMode;
    setRagModeState(next);
    try {
      const result = await setRagMode(next);
      setRagModeState(result.enabled);
    } catch (error) {
      setRagModeState(!next);
      throw error;
    } finally {
      ragLockRef.current = false;
    }
  }

  async function renameCurrentSession(title: string) {
    if (!currentSessionId || !title.trim()) {
      return;
    }
    if (sessionLockRef.current || sendLockRef.current) throw new Error("会话操作尚未完成，请稍后重试。");
    sessionLockRef.current = true;
    setIsSessionBusy(true);
    try {
      const renamed = await renameSession(currentSessionId, title.trim());
      setSessions((previous) => previous.map((item) => item.id === renamed.id ? renamed : item));
      await refreshSessions().catch(() => setSessionError("名称已保存，但列表刷新失败。请稍后刷新页面。"));
    } finally {
      sessionLockRef.current = false;
      setIsSessionBusy(false);
    }
  }

  async function removeSession(sessionId: string) {
    if (sessionLockRef.current || sendLockRef.current) throw new Error("会话操作尚未完成，请稍后重试。");
    sessionLockRef.current = true;
    setIsSessionBusy(true);
    try {
      await deleteSession(sessionId);
      setSessions((previous) => previous.filter((item) => item.id !== sessionId));
      if (currentSessionId === sessionId) {
        // Never keep a deleted ID active if refreshing the replacement fails.
        setCurrentSessionId(null);
        setMessages([]);
        setTokenStats(null);
      }
      const sequence = ++sessionListRequestRef.current;
      let nextSessions: SessionSummary[];
      try { nextSessions = await listSessions(); }
      catch { setSessionError("会话已移入回收站，但列表刷新失败。请稍后刷新页面。"); return; }
      if (sequence === sessionListRequestRef.current) setSessions(nextSessions);
      if (currentSessionId === sessionId) {
        if (nextSessions.length) {
          try {
            await refreshSessionDetails(nextSessions[0].id);
            setCurrentSessionId(nextSessions[0].id);
          } catch { setSessionError("会话已移入回收站，请从侧栏重新选择一个会话。"); }
        } else {
          setCurrentSessionId(null);
          setMessages([]);
          setTokenStats(null);
        }
      }
    } finally {
      sessionLockRef.current = false;
      setIsSessionBusy(false);
    }
  }

  async function loadInspectorFile(path: string) {
    if (inspectorLockRef.current) throw new Error("文件操作尚未完成，请稍后重试。");
    inspectorLockRef.current = true;
    try {
      const file = await loadFile(path);
      inspectorStateRef.current = { path, content: file.content };
      setInspectorPath(path);
      setInspectorContent(file.content);
      setInspectorDirty(false);
    } finally { inspectorLockRef.current = false; }
  }

  async function restoreDeletedSession(sessionId: string) {
    if (sessionLockRef.current || sendLockRef.current) throw new Error("会话操作尚未完成");
    sessionLockRef.current = true;
    setIsSessionBusy(true);
    try {
      const restored = await restoreSession(sessionId);
      setSessions((previous) => [restored, ...previous.filter((item) => item.id !== restored.id)]);
      await refreshSessions().catch(() => setSessionError("会话已恢复，但列表刷新失败。请稍后刷新页面。"));
    } finally { sessionLockRef.current = false; setIsSessionBusy(false); }
  }

  function updateInspectorContent(value: string) {
    inspectorStateRef.current = { ...inspectorStateRef.current, content: value };
    setInspectorContent(value);
    setInspectorDirty(true);
  }

  async function saveInspector() {
    if (inspectorLockRef.current) throw new Error("文件操作尚未完成，请稍后重试。");
    inspectorLockRef.current = true;
    const snapshot = { ...inspectorStateRef.current };
    try {
      await saveFile(snapshot.path, snapshot.content);
      if (inspectorStateRef.current.path === snapshot.path && inspectorStateRef.current.content === snapshot.content) setInspectorDirty(false);
      try { await refreshSkills(); } catch { /* The file has already been saved. */ }
    } finally { inspectorLockRef.current = false; }
  }

  async function compressCurrentSession() {
    if (!currentSessionId) {
      return;
    }
    if (sessionLockRef.current || sendLockRef.current) throw new Error("会话操作尚未完成，请稍后重试。");
    sessionLockRef.current = true;
    setIsSessionBusy(true);
    try {
      await compressSession(currentSessionId);
      await refreshSessionDetails(currentSessionId);
      await refreshSessions();
    } finally { sessionLockRef.current = false; setIsSessionBusy(false); }
  }

  useEffect(() => {
    // React Strict Mode reruns mount effects; initialization must remain single.
    if (initializedRef.current) return;
    initializedRef.current = true;
    void (async () => {
      try {
      const [sessionResult, rag, initialSkills] = await Promise.allSettled([
        listSessions(),
        getRagMode(),
        listSkills()
      ]);

      if (sessionResult.status === "rejected") throw sessionResult.reason;
      const initialSessions = sessionResult.value;
      setSessions(initialSessions);
      if (rag.status === "fulfilled") setRagModeState(rag.value.enabled);
      if (initialSkills.status === "fulfilled") setSkills(initialSkills.value);
      if (rag.status === "rejected" || initialSkills.status === "rejected") setSessionError("部分配置加载失败；会话可以使用，请稍后刷新页面重试。");

      if (initialSessions.length) {
        setCurrentSessionId(initialSessions[0].id);
        await refreshSessionDetails(initialSessions[0].id);
      } else {
        const created = await createSession();
        setCurrentSessionId(created.id);
        setSessions([created]);
      }

      try {
        const file = await loadFile("memory_module_v1/long_term_memory/MEMORY.md");
        setInspectorPath(file.path);
        setInspectorContent(file.content);
      } catch { /* A missing optional memory file must not prevent chatting. */ }
      } catch {
        setSessionError("工作台加载失败，请确认后端服务已启动后刷新页面。");
      } finally {
        sessionLockRef.current = false;
        setIsSessionBusy(false);
      }
    })();
  }, []);

  const value: AppStore = {
    sessions,
    currentSessionId,
    messages,
    isStreaming,
    isSessionBusy,
    sessionError,
    ragMode,
    skills,
    editableFiles,
    inspectorPath,
    inspectorContent,
    inspectorDirty,
    sidebarWidth,
    inspectorWidth,
    tokenStats,
    createNewSession,
    selectSession,
    sendMessage,
    toggleRagMode,
    renameCurrentSession,
    removeSession,
    refreshSessions,
    restoreDeletedSession,
    loadInspectorFile,
    updateInspectorContent,
    saveInspector,
    compressCurrentSession,
    setSidebarWidth,
    setInspectorWidth
  };

  return <StoreContext.Provider value={value}>{children}</StoreContext.Provider>;
}

export function useAppStore() {
  const value = useContext(StoreContext);
  if (!value) {
    throw new Error("useAppStore must be used inside AppProvider");
  }
  return value;
}
