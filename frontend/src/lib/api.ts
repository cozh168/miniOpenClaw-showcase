export type ToolCall = {
  tool: string;
  input: string;
  output: string;
  call_id?: string;
  /** UI completion marker; historical tool records are already complete. */
  completed?: boolean;
};

export type RetrievalResult = {
  text: string;
  score: number;
  source: string;
  kind?: string;
};

export type SessionSummary = {
  id: string;
  title: string;
  created_at: number;
  updated_at: number;
  message_count: number;
};

export type SessionHistory = {
  id: string;
  title: string;
  created_at: number;
  updated_at: number;
  compressed_context?: string;
  messages: Array<{
    role: "user" | "assistant";
    content: string;
    tool_calls?: ToolCall[];
  }>;
};

export type StreamHandlers = {
  onEvent: (event: string, data: Record<string, unknown>) => void;
};

export type WorkspaceFileNode = {
  path: string;
  name: string;
  type: "file" | "directory";
  size: number;
  editable: boolean;
  children?: WorkspaceFileNode[];
};

export type RuntimeInfo = {
  python_version: string;
  python_environment: string;
  provider: string;
  model: string;
  memory_backend: string;
  rag_enabled: boolean;
  tools: Array<{ name: string; enabled: boolean }>;
};

export type ToolAuditEntry = {
  timestamp: string;
  tool: string;
  status: string;
  input_preview: string;
  output_preview: string;
  reason: string | null;
  duration_ms: number | null;
  metadata: Record<string, unknown>;
};

export function getWorkspaceFiles() {
  return request<{ files: WorkspaceFileNode[] }>("/workbench/files");
}

export function createWorkspaceFile(path: string, content: string) {
  return request<{ ok: boolean; path: string }>("/workbench/files", {
    method: "POST", body: JSON.stringify({ path, content })
  });
}

export function getRuntimeInfo() {
  return request<RuntimeInfo>("/workbench/runtime");
}

export function getToolAudit(limit = 100) {
  return request<{ items: ToolAuditEntry[] }>(`/tool-audit?limit=${limit}`);
}

export function listTrashedSessions() {
  return request<SessionSummary[]>("/sessions/trash");
}

export function restoreSession(sessionId: string) {
  return request<SessionSummary>(`/sessions/${encodeURIComponent(sessionId)}/restore`, { method: "POST" });
}

function getApiBase() {
  if (typeof window === "undefined") {
    return "http://127.0.0.1:8002/api";
  }

  return `http://${window.location.hostname}:8002/api`;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${getApiBase()}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {})
    }
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Request failed: ${response.status}`);
  }

  return (await response.json()) as T;
}

export async function listSessions() {
  return request<SessionSummary[]>("/sessions");
}

export async function createSession(title = "新会话") {
  return request<SessionSummary>("/sessions", {
    method: "POST",
    body: JSON.stringify({ title })
  });
}

export async function renameSession(sessionId: string, title: string) {
  return request<SessionSummary>(`/sessions/${sessionId}`, {
    method: "PUT",
    body: JSON.stringify({ title })
  });
}

export async function deleteSession(sessionId: string) {
  return request<{ ok: boolean }>(`/sessions/${sessionId}`, {
    method: "DELETE"
  });
}

export async function getSessionHistory(sessionId: string) {
  return request<SessionHistory>(`/sessions/${sessionId}/history`);
}

export async function getSessionTokens(sessionId: string) {
  return request<{
    system_tokens: number;
    message_tokens: number;
    total_tokens: number;
  }>(`/tokens/session/${sessionId}`);
}

export async function listSkills() {
  return request<Array<{ name: string; description: string; path: string }>>("/skills");
}

export async function loadFile(path: string) {
  return request<{ path: string; content: string }>(
    `/files?path=${encodeURIComponent(path)}`
  );
}

export async function saveFile(path: string, content: string) {
  return request<{ ok: boolean; path: string }>("/files", {
    method: "POST",
    body: JSON.stringify({ path, content })
  });
}

export async function getRagMode() {
  return request<{ enabled: boolean }>("/config/rag-mode");
}

export async function setRagMode(enabled: boolean) {
  return request<{ enabled: boolean }>("/config/rag-mode", {
    method: "PUT",
    body: JSON.stringify({ enabled })
  });
}

export async function compressSession(sessionId: string) {
  return request<{ archived_count: number; remaining_count: number }>(
    `/sessions/${sessionId}/compress`,
    { method: "POST" }
  );
}

export async function streamChat(
  payload: {
    message: string;
    session_id: string;
  },
  handlers: StreamHandlers
) {
  const response = await fetch(`${getApiBase()}/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      ...payload,
      stream: true
    })
  });

  if (!response.ok || !response.body) {
    throw new Error(`Chat request failed: ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let blockLines: string[] = [];

  const flushBlock = (block: string) => {
    const lines = block.split(/\r\n|\r|\n/);
    let event = "message";
    const dataLines: string[] = [];

    for (const line of lines) {
      if (line.startsWith("event:")) {
        event = line.slice(6).trim();
      }
      if (line.startsWith("data:")) {
        dataLines.push(line.slice(5).trim());
      }
    }

    if (!dataLines.length) {
      return;
    }

    const data = JSON.parse(dataLines.join("\n")) as Record<string, unknown>;
    handlers.onEvent(event, data);
  };

  let completed = false;
  try {
    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value ?? new Uint8Array(), { stream: !done });
      let newline = /\r\n|\r|\n/.exec(buffer);
      while (newline) {
        // A CR at a chunk boundary may be the first half of CRLF.
        if (!done && newline[0] === "\r" && newline.index === buffer.length - 1) break;
        const line = buffer.slice(0, newline.index);
        buffer = buffer.slice(newline.index + newline[0].length);
        if (line === "") {
          flushBlock(blockLines.join("\n"));
          blockLines = [];
        } else {
          blockLines.push(line);
        }
        newline = /\r\n|\r|\n/.exec(buffer);
      }

      if (done) {
        if (buffer) blockLines.push(buffer);
        if (blockLines.length) flushBlock(blockLines.join("\n"));
        completed = true;
        break;
      }
    }
  } finally {
    if (!completed) await reader.cancel().catch(() => undefined);
    reader.releaseLock();
  }
}
