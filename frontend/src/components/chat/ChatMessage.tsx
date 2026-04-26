"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { RetrievalCard } from "@/components/chat/RetrievalCard";
import { ThoughtChain } from "@/components/chat/ThoughtChain";
import type { RetrievalResult, ToolCall } from "@/lib/api";

export function ChatMessage({
  role,
  content,
  toolCalls,
  retrievals
}: {
  role: "user" | "assistant";
  content: string;
  toolCalls: ToolCall[];
  retrievals: RetrievalResult[];
}) {
  const isUser = role === "user";

  return (
    <article
      className={`max-w-[90%] rounded-[28px] px-5 py-4 ${
        isUser
          ? "ml-auto bg-[rgba(13,37,48,0.92)] text-white"
          : "panel mr-auto text-[var(--color-ink)]"
      }`}
    >
      {!isUser && <RetrievalCard results={retrievals} />}
      {!isUser && <ThoughtChain toolCalls={toolCalls} />}
      {(!isUser && toolCalls.length > 0 && (!content || content.trim() === "") && !toolCalls.some(tc => tc.output === content)) ? null : (
        (content && content.trim() !== "" && (!toolCalls.length || !toolCalls.some(tc => tc.output.trim() === content.trim()))) && (
          <div className={isUser ? "whitespace-pre-wrap leading-7" : "markdown"}>
            {isUser ? (
              content
            ) : (
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {content}
              </ReactMarkdown>
            )}
          </div>
        )
      )}
      {!isUser && (!content || content.trim() === "") && !toolCalls.length && (
        <div className="text-[var(--color-ink-soft)]">正在思考...</div>
      )}
    </article>
  );
}
