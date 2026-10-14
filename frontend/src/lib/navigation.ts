import type { WorkspaceFileNode } from "@/lib/api";

export type WorkbenchView = "chat" | "agent" | "knowledge" | "tools" | "workspace" | "observability" | "logs" | "settings" | "trash";
export const VIEW_LABELS: Record<WorkbenchView, string> = {
  chat: "工作空间", agent: "智能体", knowledge: "知识库", tools: "工具集", workspace: "工作空间", observability: "观测分析", logs: "运行日志", settings: "设置", trash: "回收站"
};
export function flattenFiles(nodes: WorkspaceFileNode[]): WorkspaceFileNode[] {
  return nodes.flatMap((node) => node.type === "directory" ? flattenFiles(node.children ?? []) : [node]);
}
