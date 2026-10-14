"use client";

import { useEffect, useLayoutEffect, useRef, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";

let activeModalCount = 0;
export function isModalOpen() { return activeModalCount > 0; }
const useBrowserLayoutEffect = typeof window === "undefined" ? useEffect : useLayoutEffect;

export function Modal({ children, labelledBy, onClose, className = "" }: {
  children: ReactNode;
  labelledBy: string;
  onClose: () => void;
  className?: string;
}) {
  const [mounted, setMounted] = useState(false);
  const panelRef = useRef<HTMLDivElement>(null);
  const onCloseRef = useRef(onClose);
  onCloseRef.current = onClose;

  useBrowserLayoutEffect(() => {
    activeModalCount++;
    return () => { activeModalCount--; };
  }, []);

  useEffect(() => setMounted(true), []);
  useEffect(() => {
    if (!mounted) return;
    const previous = document.activeElement as HTMLElement | null;
    const panel = panelRef.current;
    const focusable = () => Array.from(panel?.querySelectorAll<HTMLElement>(
      'button:not([disabled]), input:not([disabled]), textarea:not([disabled]), select:not([disabled]), a[href], [tabindex="0"]'
    ) ?? []).filter((element) => element.getClientRects().length > 0);
    const first = panel?.querySelector<HTMLElement>("[data-autofocus]") ?? focusable()[0] ?? panel;
    first?.focus();
    const handleKey = (event: KeyboardEvent) => {
      if (event.key === "Escape" && !event.isComposing) {
        event.preventDefault(); onCloseRef.current();
      }
      if (event.key !== "Tab") return;
      const items = focusable();
      const firstItem = items[0];
      const lastItem = items[items.length - 1];
      if (!firstItem) {
        event.preventDefault(); panel?.focus();
      } else if (!panel?.contains(document.activeElement)) {
        event.preventDefault(); (event.shiftKey ? lastItem : firstItem).focus();
      } else if (event.shiftKey && (document.activeElement === firstItem || document.activeElement === panel)) {
        event.preventDefault(); lastItem.focus();
      } else if (!event.shiftKey && document.activeElement === lastItem) {
        event.preventDefault(); firstItem.focus();
      }
    };
    document.addEventListener("keydown", handleKey);
    return () => {
      document.removeEventListener("keydown", handleKey);
      const active = document.activeElement;
      // A confirm action may deliberately focus the composer or file editor.
      if (previous?.isConnected && (active === document.body || panel?.contains(active))) previous.focus();
    };
  }, [mounted]);

  if (!mounted) return null;
  return createPortal(
    <div className="modal-overlay">
      <div className="modal-backdrop" onClick={() => onCloseRef.current()} aria-hidden="true" />
      <div ref={panelRef} className={`modal-panel ${className}`} role="dialog" aria-modal="true" aria-labelledby={labelledBy} tabIndex={-1}>
        {children}
      </div>
    </div>, document.body
  );
}
