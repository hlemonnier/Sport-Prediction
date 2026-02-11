import type { UiPreferences, UserSavings } from "@/lib/uiPreferences";
import type {
  CatalogResponse,
  DataStatus,
  FixtureResponse,
  NotebookEntry,
  PaperEntry,
  RunDetail,
  RunSummary,
  SweepDetail,
  SweepRow,
} from "@/lib/types";

export const API_BASE =
  process.env.NEXT_PUBLIC_API_URL || process.env.API_URL || "http://localhost:4000";

function isObject(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

async function parseApiError(response: Response): Promise<string> {
  try {
    const payload = (await response.json()) as unknown;
    if (isObject(payload) && typeof payload.error === "string" && payload.error.trim()) {
      return payload.error;
    }
  } catch {
    // Ignore JSON parse errors and use fallback message.
  }
  return `Request failed: ${response.status}`;
}

async function apiRequest<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    cache: "no-store",
    ...init,
  });

  if (!response.ok) {
    throw new Error(await parseApiError(response));
  }

  return (await response.json()) as T;
}

export async function apiGet<T>(path: string): Promise<T> {
  return apiRequest<T>(path);
}

export async function apiPost<T>(path: string, body: unknown): Promise<T> {
  return apiRequest<T>(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export function fileUrl(path: string): string {
  const encoded = encodeURI(path);
  return `${API_BASE}/api/files/${encoded}`;
}

export type UserPreferencesResponse = {
  preferences: unknown | null;
  savings: UserSavings;
  updatedAt: string | null;
};

export type SaveUserPreferencesRequest = {
  preferences: UiPreferences;
  savings?: UserSavings;
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

export function getCatalog(): Promise<CatalogResponse> {
  return apiGet<CatalogResponse>("/api/catalog");
}

export function getDataStatus(): Promise<DataStatus> {
  return apiGet<DataStatus>("/api/data-status");
}

export function getFootballFixtures(limit = 5): Promise<FixtureResponse> {
  return apiGet<FixtureResponse>(`/api/football/fixtures?limit=${limit}`);
}

export function getPapers(): Promise<PaperEntry[]> {
  return apiGet<PaperEntry[]>("/api/papers");
}

export function getNotebooks(): Promise<NotebookEntry[]> {
  return apiGet<NotebookEntry[]>("/api/notebooks");
}

export function listRuns(): Promise<RunSummary[]> {
  return apiGet<RunSummary[]>("/api/runs");
}

export function getRun(runId: string): Promise<RunDetail> {
  return apiGet<RunDetail>(`/api/runs/${encodeURIComponent(runId)}`);
}

export type CreateRunRequest = {
  sport: string;
  project: string;
  params: Record<string, unknown>;
  cacheDir?: string;
  cache_dir?: string;
  tags?: unknown[];
};

export type CreateRunResponse = {
  runId: string;
  status: "queued" | "running" | "done" | "error";
};

export function createRun(payload: CreateRunRequest): Promise<CreateRunResponse> {
  return apiPost<CreateRunResponse>("/api/runs", payload);
}

export function listSweeps(): Promise<SweepRow[]> {
  return apiGet<SweepRow[]>("/api/sweeps");
}

export function getSweep(sweepId: string): Promise<SweepDetail> {
  return apiGet<SweepDetail>(`/api/sweeps/${encodeURIComponent(sweepId)}`);
}

export type CreateSweepRequest = {
  sport: string;
  project: string;
  baseParams: Record<string, unknown>;
  sweep: {
    param: string;
    values: unknown[];
  };
};

export type CreateSweepResponse = {
  sweepId: string;
  runIds: string[];
  status: "queued" | "running" | "done" | "partial" | "error";
};

export function createSweep(payload: CreateSweepRequest): Promise<CreateSweepResponse> {
  return apiPost<CreateSweepResponse>("/api/sweeps", payload);
}

export function openSystemPath(path: string): Promise<{ status: string }> {
  return apiPost<{ status: string }>("/api/open", { path });
}

type WaitForRunOptions = {
  pollMs?: number;
  timeoutMs?: number;
  onTick?: (run: RunDetail) => void;
};

type WaitForSweepOptions = {
  pollMs?: number;
  timeoutMs?: number;
  onTick?: (sweep: SweepDetail) => void;
};

const RUN_TERMINAL_STATUSES = new Set(["done", "error"]);
const SWEEP_TERMINAL_STATUSES = new Set(["done", "partial", "error"]);

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

export async function waitForRunCompletion(
  runId: string,
  options: WaitForRunOptions = {}
): Promise<RunDetail> {
  const pollMs = options.pollMs ?? 1500;
  const timeoutMs = options.timeoutMs ?? 15 * 60 * 1000;
  const startedAt = Date.now();

  while (true) {
    const run = await getRun(runId);
    options.onTick?.(run);
    if (RUN_TERMINAL_STATUSES.has(run.status)) {
      return run;
    }
    if (Date.now() - startedAt > timeoutMs) {
      throw new Error("Timed out while waiting for run completion");
    }
    await sleep(pollMs);
  }
}

export async function waitForSweepCompletion(
  sweepId: string,
  options: WaitForSweepOptions = {}
): Promise<SweepDetail> {
  const pollMs = options.pollMs ?? 2000;
  const timeoutMs = options.timeoutMs ?? 30 * 60 * 1000;
  const startedAt = Date.now();

  while (true) {
    const sweep = await getSweep(sweepId);
    options.onTick?.(sweep);
    if (SWEEP_TERMINAL_STATUSES.has(sweep.status)) {
      return sweep;
    }
    if (Date.now() - startedAt > timeoutMs) {
      throw new Error("Timed out while waiting for sweep completion");
    }
    await sleep(pollMs);
  }
}
