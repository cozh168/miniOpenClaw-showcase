"use client";

import { ChatPanel } from "@/components/chat/ChatPanel";
import { Navbar } from "@/components/layout/Navbar";
import { ResizeHandle } from "@/components/layout/ResizeHandle";
import { Sidebar } from "@/components/layout/Sidebar";
import { AppProvider, useAppStore } from "@/lib/store";

function Workspace() {
  const { sidebarWidth, setSidebarWidth } = useAppStore();

  return (
    <main className="h-screen p-4 md:p-6 flex flex-col">
      <div className="mx-auto flex w-full max-w-[1800px] flex-1 flex-col gap-4 min-h-0">
        <Navbar />
        <div className="flex flex-1 gap-0 min-h-0">
          <div style={{ width: sidebarWidth, flexShrink: 0 }} className="h-full">
            <Sidebar />
          </div>
          <ResizeHandle onResize={(delta) => setSidebarWidth(Math.max(260, sidebarWidth + delta))} />
          <div className="flex-1 min-w-0 h-full">
            <ChatPanel />
          </div>
        </div>
      </div>
    </main>
  );
}

export default function Page() {
  return (
    <AppProvider>
      <Workspace />
    </AppProvider>
  );
}
