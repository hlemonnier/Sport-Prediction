export type ParamDef = {
  name: string;
  label: string;
  kind: "string" | "int" | "bool" | "select";
  required: boolean;
  default?: string | number | boolean | null;
  options?: string[] | null;
};

export type CatalogProject = {
  sport: string;
  name: string;
  pythonDir: string;
  notebook?: string | null;
  params: ParamDef[];
};

export type CatalogResponse = {
  projects: CatalogProject[];
};

export type PaperEntry = {
  sport: string;
  title: string;
  file: string;
  source?: string | null;
};

export type NotebookEntry = {
  sport: string;
  project: string;
  file: string;
};

export type RunSummary = {
  id: string;
  createdAt: string;
  sport: string;
  project: string;
  status: "queued" | "running" | "done" | "error";
  durationMs?: number | null;
  sweepId?: string | null;
};

export type RunDetail = {
  id: string;
  createdAt: string;
  sport: string;
  project: string;
  status: "queued" | "running" | "done" | "error";
  durationMs?: number | null;
  sweepId?: string | null;
  config: Record<string, unknown>;
  result?: {
    rows?: Array<Record<string, unknown>>;
    notes?: string[];
    version?: string;
  } | null;
  stdout?: string | null;
  stderr?: string | null;
  resultPath?: string | null;
};

export type SweepRow = {
  id: string;
  createdAt: string;
  sport: string;
  project: string;
  param: string;
  valuesJson: string;
  baseConfigJson: string;
  status: "queued" | "running" | "done" | "partial" | "error";
};

export type SweepDetail = {
  id: string;
  createdAt: string;
  sport: string;
  project: string;
  param: string;
  values: unknown;
  status: "queued" | "running" | "done" | "partial" | "error";
  runs: Array<{
    id: string;
    createdAt: string;
    status: "queued" | "running" | "done" | "error";
    paramValue: string;
    result?: Record<string, unknown> | null;
  }>;
  summary: Array<{
    paramValue: string;
    score?: number | null;
  }>;
};

export type DataStatus = {
  football: {
    teams: { path: string; format: string; exists: boolean };
    matches: { path: string; format: string; exists: boolean };
    fixtures: { path: string; format: string; exists: boolean };
  };
};

export type Fixture = {
  matchId: string;
  date: string;
  season: string;
  league: string;
  homeTeamId: string;
  awayTeamId: string;
};

export type FixtureResponse = {
  fixtures: Fixture[];
  warning?: string | null;
};
