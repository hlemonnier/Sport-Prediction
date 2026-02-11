"use client";

import { useEffect, useState } from "react";
import { APP_TOAST_EVENT, type ToastKind, type ToastPayload } from "@/lib/toast";

type ToastItem = {
  id: string;
  message: string;
  kind: ToastKind;
};

function nextId(): string {
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}

export default function ToastHost() {
  const [toasts, setToasts] = useState<ToastItem[]>([]);

  useEffect(() => {
    const handleToast = (event: Event) => {
      const customEvent = event as CustomEvent<ToastPayload>;
      const detail = customEvent.detail;
      if (!detail || typeof detail.message !== "string" || !detail.message.trim()) {
        return;
      }

      const id = nextId();
      const toast: ToastItem = {
        id,
        message: detail.message.trim(),
        kind: detail.kind ?? "info",
      };
      setToasts((prev) => [...prev, toast]);

      const timeout = typeof detail.durationMs === "number" ? detail.durationMs : 3000;
      window.setTimeout(() => {
        setToasts((prev) => prev.filter((item) => item.id !== id));
      }, timeout);
    };

    window.addEventListener(APP_TOAST_EVENT, handleToast as EventListener);
    return () => {
      window.removeEventListener(APP_TOAST_EVENT, handleToast as EventListener);
    };
  }, []);

  if (toasts.length === 0) {
    return null;
  }

  return (
    <div className="toast-host" role="status" aria-live="polite">
      {toasts.map((toast) => (
        <div key={toast.id} className={`toast-item ${toast.kind}`}>
          <span className={`chip-led ${toast.kind === "success" ? "green" : toast.kind === "error" ? "red" : "amber"}`} />
          <span>{toast.message}</span>
        </div>
      ))}
    </div>
  );
}
