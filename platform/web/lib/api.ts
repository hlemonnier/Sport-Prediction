import type { UiPreferences } from "@/lib/uiPreferences";

export const API_BASE =
  process.env.NEXT_PUBLIC_API_URL || process.env.API_URL || "http://localhost:4000";

export async function apiGet<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, { cache: "no-store" });
  if (!res.ok) {
    throw new Error(`Request failed: ${res.status}`);
  }
  return (await res.json()) as T;
}

export async function apiPost<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    throw new Error(`Request failed: ${res.status}`);
  }
  return (await res.json()) as T;
}

export function fileUrl(path: string): string {
  const encoded = encodeURI(path);
  return `${API_BASE}/api/files/${encoded}`;
}

export type UserPreferencesResponse = {
  preferences: unknown | null;
  savings: Record<string, unknown>;
  updatedAt: string | null;
};

export type SaveUserPreferencesRequest = {
  preferences: UiPreferences;
  savings?: Record<string, unknown>;
};

export type SaveUserPreferencesResponse = {
  status: string;
  updatedAt: string;
};

export function getUserPreferences(): Promise<UserPreferencesResponse> {
  return apiGet<UserPreferencesResponse>("/api/user-preferences");
}

export function saveUserPreferences(
  payload: SaveUserPreferencesRequest
): Promise<SaveUserPreferencesResponse> {
  return apiPost<SaveUserPreferencesResponse>("/api/user-preferences", payload);
}
