export const APP_TOAST_EVENT = "app:toast";

export type ToastKind = "info" | "success" | "error";

export type ToastPayload = {
  message: string;
  kind?: ToastKind;
  durationMs?: number;
};

export function showToast(payload: ToastPayload): void {
  if (typeof window === "undefined") {
    return;
  }
  window.dispatchEvent(new CustomEvent(APP_TOAST_EVENT, { detail: payload }));
}
