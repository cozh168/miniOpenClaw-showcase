"use client";

import { useEffect, useState } from "react";

export function ResizeHandle({
  onResize, width
}: {
  onResize: (delta: number) => void;
  width: number;
}) {
  const [dragging, setDragging] = useState(false);

  useEffect(() => {
    if (!dragging) {
      return;
    }

    const onMouseMove = (event: MouseEvent) => {
      onResize(event.movementX);
    };
    const onMouseUp = () => {
      setDragging(false);
    };

    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mouseup", onMouseUp);

    return () => {
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("mouseup", onMouseUp);
    };
  }, [dragging, onResize]);

  return (
    <div
      className={`resize-handle ${dragging ? "is-dragging" : ""}`}
      role="separator" aria-label="调整会话侧栏宽度" aria-orientation="vertical"
      aria-valuemin={224} aria-valuemax={380} aria-valuenow={width} tabIndex={0}
      onMouseDown={(event) => { event.preventDefault(); setDragging(true); }}
      onKeyDown={(event) => {
        if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
          event.preventDefault(); onResize(event.key === "ArrowLeft" ? -16 : 16);
        }
      }}
    />
  );
}
