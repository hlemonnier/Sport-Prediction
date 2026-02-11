import fs from "node:fs";
import fsp from "node:fs/promises";
import path from "node:path";
import { randomUUID } from "node:crypto";
import { spawnSync } from "node:child_process";
import postgres from "postgres";

function parseEnvValue(raw) {
  const trimmed = raw.trim();
  if (
    (trimmed.startsWith("\"") && trimmed.endsWith("\"")) ||
    (trimmed.startsWith("'") && trimmed.endsWith("'"))
  ) {
    return trimmed.slice(1, -1);
  }
  return trimmed;
}

function loadEnvFile(filePath) {
  if (!fs.existsSync(filePath)) {
    return;
  }

  let content;
  try {
    content = fs.readFileSync(filePath, "utf8");
  } catch {
    return;
  }

  for (const rawLine of content.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#")) {
      continue;
    }

    const equalIndex = line.indexOf("=");
    if (equalIndex <= 0) {
      continue;
    }

    const key = line.slice(0, equalIndex).trim();
    if (!key) {
      continue;
    }

    const existingValue = process.env[key];
    if (existingValue !== undefined && existingValue !== "") {
      continue;
    }

    const rawValue = line.slice(equalIndex + 1);
    process.env[key] = parseEnvValue(rawValue);
  }
}

class AppError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }

  static badRequest(message) {
    return new AppError(400, message);
  }

  static notFound(message) {
    return new AppError(404, message);
  }

  static forbidden(message) {
    return new AppError(403, message);
  }

  static internal(message) {
    return new AppError(500, message);
  }
}

const CORS_HEADERS = {
  "access-control-allow-origin": "*",
  "access-control-allow-methods": "GET,POST,OPTIONS",
  "access-control-allow-headers": "*",
};

function responseHeaders(extra = {}) {
  return {
    ...CORS_HEADERS,
    ...extra,
  };
}

function jsonResponse(payload, status = 200) {
  return new Response(JSON.stringify(payload), {
    status,
    headers: responseHeaders({ "content-type": "application/json" }),
  });
}

function errorResponse(error) {
  if (error instanceof AppError) {
    return jsonResponse({ error: error.message }, error.status);
  }

  const message = error instanceof Error ? error.message : "Internal server error";
  console.error(error);
  return jsonResponse({ error: message }, 500);
}

async function parseJsonBody(request) {
  let raw;
  try {
    raw = await request.text();
  } catch {
    throw AppError.badRequest("Invalid request body");
  }

  if (!raw) {
    return {};
  }

  try {
    return JSON.parse(raw);
  } catch {
    throw AppError.badRequest("Invalid JSON body");
  }
}

function isObject(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

const DEFAULT_USER_SAVINGS = {
  bankroll: 0,
  monthlySavingsTarget: 0,
  reserveBalance: 0,
  defaultStake: 10,
  maxStakePercent: 2,
  autoSaveProfit: true,
};

function coerceNumber(value, fallback, min = 0, max = Number.POSITIVE_INFINITY) {
  const asNumber =
    typeof value === "number" && Number.isFinite(value)
      ? value
      : typeof value === "string"
        ? Number.parseFloat(value)
        : Number.NaN;

  if (!Number.isFinite(asNumber)) {
    return fallback;
  }

  return Math.min(max, Math.max(min, asNumber));
}

function normalizeRunStatus(status) {
  const normalized = String(status ?? "").toLowerCase();
  if (normalized === "success") return "done";
  if (normalized === "failed") return "error";
  if (normalized === "done" || normalized === "error" || normalized === "running" || normalized === "queued") {
    return normalized;
  }
  return normalized || "queued";
}

function normalizeSweepStatus(status) {
  const normalized = String(status ?? "").toLowerCase();
  if (normalized === "success") return "done";
  if (normalized === "failed") return "error";
  if (
    normalized === "done" ||
    normalized === "error" ||
    normalized === "partial" ||
    normalized === "running" ||
    normalized === "queued"
  ) {
    return normalized;
  }
  return normalized || "queued";
}

function sanitizeUserSavings(raw) {
  const source = isObject(raw) ? raw : {};
  return {
    bankroll: coerceNumber(source.bankroll, DEFAULT_USER_SAVINGS.bankroll, 0),
    monthlySavingsTarget: coerceNumber(
      source.monthlySavingsTarget,
      DEFAULT_USER_SAVINGS.monthlySavingsTarget,
      0
    ),
    reserveBalance: coerceNumber(source.reserveBalance, DEFAULT_USER_SAVINGS.reserveBalance, 0),
    defaultStake: coerceNumber(source.defaultStake, DEFAULT_USER_SAVINGS.defaultStake, 0),
    maxStakePercent: coerceNumber(source.maxStakePercent, DEFAULT_USER_SAVINGS.maxStakePercent, 0, 100),
    autoSaveProfit:
      typeof source.autoSaveProfit === "boolean"
        ? source.autoSaveProfit
        : DEFAULT_USER_SAVINGS.autoSaveProfit,
  };
}

function enqueueBackgroundJob(state, name, job) {
  state.jobQueue.push({ name, job });
  drainBackgroundJobs(state);
}

function drainBackgroundJobs(state) {
  if (state.jobQueueRunning) {
    return;
  }

  const next = state.jobQueue.shift();
  if (!next) {
    return;
  }

  state.jobQueueRunning = true;

  void (async () => {
    try {
      await next.job();
    } catch (error) {
      console.error(`Background job failed: ${next.name}`);
      console.error(error);
    } finally {
      state.jobQueueRunning = false;
      drainBackgroundJobs(state);
    }
  })();
}

async function initDb(db) {
  await db.unsafe(`
    CREATE TABLE IF NOT EXISTS runs (
      id TEXT PRIMARY KEY,
      created_at TEXT NOT NULL,
      sport TEXT NOT NULL,
      project TEXT NOT NULL,
      status TEXT NOT NULL,
      config_json TEXT NOT NULL,
      result_path TEXT,
      stdout_path TEXT,
      stderr_path TEXT,
      duration_ms INTEGER,
      sweep_id TEXT
    );
  `);

  await db.unsafe(`
    CREATE TABLE IF NOT EXISTS sweeps (
      id TEXT PRIMARY KEY,
      created_at TEXT NOT NULL,
      sport TEXT NOT NULL,
      project TEXT NOT NULL,
      param TEXT NOT NULL,
      values_json TEXT NOT NULL,
      base_config_json TEXT NOT NULL,
      status TEXT NOT NULL
    );
  `);

  await db.unsafe(`
    CREATE TABLE IF NOT EXISTS sweep_runs (
      sweep_id TEXT NOT NULL,
      run_id TEXT NOT NULL,
      param_value TEXT NOT NULL,
      PRIMARY KEY (sweep_id, run_id)
    );
  `);

  await db.unsafe(`
    CREATE TABLE IF NOT EXISTS user_preferences (
      id INTEGER PRIMARY KEY CHECK (id = 1),
      updated_at TEXT NOT NULL,
      preferences_json TEXT NOT NULL,
      savings_json TEXT NOT NULL DEFAULT '{}'
    );
  `);

  await hardenDbSecurity(db);
}

async function hardenDbSecurity(db) {
  await db.unsafe("ALTER TABLE public.runs ENABLE ROW LEVEL SECURITY");
  await db.unsafe("ALTER TABLE public.sweeps ENABLE ROW LEVEL SECURITY");
  await db.unsafe("ALTER TABLE public.sweep_runs ENABLE ROW LEVEL SECURITY");
  await db.unsafe("ALTER TABLE public.user_preferences ENABLE ROW LEVEL SECURITY");

  await db.unsafe(
    "REVOKE ALL PRIVILEGES ON TABLE public.runs, public.sweeps, public.sweep_runs, public.user_preferences FROM PUBLIC"
  );

  await db.unsafe(`
    DO $$
    BEGIN
      IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'anon') THEN
        REVOKE ALL PRIVILEGES ON TABLE public.runs, public.sweeps, public.sweep_runs, public.user_preferences FROM anon;
      END IF;

      IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'authenticated') THEN
        REVOKE ALL PRIVILEGES ON TABLE public.runs, public.sweeps, public.sweep_runs, public.user_preferences FROM authenticated;
      END IF;

      IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'service_role') THEN
        GRANT ALL PRIVILEGES ON TABLE public.runs, public.sweeps, public.sweep_runs, public.user_preferences TO service_role;
      END IF;
    END
    $$;
  `);
}

function resolveProjectsRoot(repoRoot) {
  const candidate = path.join(repoRoot, "research", "projects");
  if (fs.existsSync(candidate)) {
    return candidate;
  }

  const legacy = path.join(repoRoot, "projects");
  if (fs.existsSync(legacy)) {
    return legacy;
  }

  return repoRoot;
}

function resolvePapersRoot(repoRoot) {
  const candidate = path.join(repoRoot, "research", "papers");
  if (fs.existsSync(candidate)) {
    return candidate;
  }

  const legacy = path.join(repoRoot, "Research");
  if (fs.existsSync(legacy)) {
    return legacy;
  }

  return path.join(repoRoot, "papers");
}

function normalize(value) {
  return String(value ?? "")
    .toLowerCase()
    .replaceAll("_", " ")
    .trim();
}

function isSportDir(dirPath) {
  if (!fs.existsSync(dirPath) || !fs.statSync(dirPath).isDirectory()) {
    return false;
  }

  const name = path.basename(dirPath);
  if (name.startsWith(".")) {
    return false;
  }

  if (
    ["Research", "research", "platform", "papers", "projects", ".git"].includes(name)
  ) {
    return false;
  }

  let children;
  try {
    children = fs.readdirSync(dirPath, { withFileTypes: true });
  } catch {
    return false;
  }

  for (const child of children) {
    if (!child.isDirectory() || child.name.startsWith(".")) {
      continue;
    }

    const childPath = path.join(dirPath, child.name);
    if (
      fs.existsSync(path.join(childPath, "Python")) ||
      fs.existsSync(path.join(childPath, "Jupyter"))
    ) {
      return true;
    }
  }

  return false;
}

function paramsForProject(kind) {
  if (kind === "F1") {
    return [
      {
        name: "mode",
        label: "Mode",
        kind: "select",
        required: true,
        default: "qualifying",
        options: ["qualifying", "race"],
      },
      {
        name: "source",
        label: "Source",
        kind: "select",
        required: true,
        default: "fastf1",
        options: ["fastf1", "openf1"],
      },
      {
        name: "year",
        label: "Year",
        kind: "int",
        required: true,
        default: 2026,
      },
      {
        name: "round_number",
        label: "Round",
        kind: "int",
        required: true,
        default: 1,
      },
      {
        name: "train_seasons",
        label: "Train Seasons",
        kind: "string",
        required: false,
        default: "auto",
      },
      {
        name: "include_standings",
        label: "Include Standings",
        kind: "bool",
        required: false,
        default: false,
      },
      {
        name: "cache_dir",
        label: "Cache Dir",
        kind: "string",
        required: false,
        default: ".cache/fastf1",
      },
      {
        name: "meeting_name",
        label: "Meeting Name",
        kind: "string",
        required: false,
        default: null,
      },
      {
        name: "country_name",
        label: "Country Name",
        kind: "string",
        required: false,
        default: null,
      },
    ];
  }

  if (kind === "Football") {
    return [
      {
        name: "mode",
        label: "Mode",
        kind: "select",
        required: true,
        default: "match_result",
        options: ["match_result", "scoreline"],
      },
      {
        name: "league",
        label: "League",
        kind: "string",
        required: true,
        default: "epl",
      },
      {
        name: "season",
        label: "Season",
        kind: "int",
        required: true,
        default: 2026,
      },
      {
        name: "round_number",
        label: "Round",
        kind: "int",
        required: true,
        default: 1,
      },
      {
        name: "data_source",
        label: "Data Source",
        kind: "string",
        required: false,
        default: "placeholder",
      },
      {
        name: "train_seasons",
        label: "Train Seasons",
        kind: "string",
        required: false,
        default: "auto",
      },
      {
        name: "cache_dir",
        label: "Cache Dir",
        kind: "string",
        required: false,
        default: null,
      },
    ];
  }

  return [];
}

function buildCatalog(repoRoot) {
  const projectsRoot = resolveProjectsRoot(repoRoot);
  const projects = [];

  let sportEntries;
  try {
    sportEntries = fs.readdirSync(projectsRoot, { withFileTypes: true });
  } catch {
    throw AppError.internal("Failed to scan repo");
  }

  for (const sportEntry of sportEntries) {
    if (!sportEntry.isDirectory()) {
      continue;
    }

    const sportPath = path.join(projectsRoot, sportEntry.name);
    if (!isSportDir(sportPath)) {
      continue;
    }

    let projectEntries;
    try {
      projectEntries = fs.readdirSync(sportPath, { withFileTypes: true });
    } catch {
      throw AppError.internal("Failed to scan sport");
    }

    for (const projectEntry of projectEntries) {
      if (!projectEntry.isDirectory()) {
        continue;
      }

      const projectName = projectEntry.name;
      const projectPath = path.join(sportPath, projectName);
      const pythonDir = path.join(projectPath, "Python");
      if (!fs.existsSync(pythonDir)) {
        continue;
      }

      const notebook = path.join(projectPath, "Jupyter", "model-research.ipynb");
      let kind = "Unknown";
      if (sportEntry.name === "F1" && projectName === "Rising Qualification Prediction") {
        kind = "F1";
      }
      if (sportEntry.name === "Football" && projectName === "Match Result Prediction") {
        kind = "Football";
      }

      projects.push({
        sport: sportEntry.name,
        name: projectName,
        pythonDir,
        notebook: fs.existsSync(notebook) ? notebook : null,
        kind,
      });
    }
  }

  return projects;
}

function resolveProject(repoRoot, sport, project) {
  const projects = buildCatalog(repoRoot);
  const match = projects.find(
    (item) => normalize(item.sport) === normalize(sport) && normalize(item.name) === normalize(project)
  );

  if (!match) {
    throw AppError.notFound("Project not found");
  }

  return match;
}

function toRelative(repoRoot, filePath) {
  const absolute = path.isAbsolute(filePath) ? filePath : path.resolve(repoRoot, filePath);
  const relative = path.relative(repoRoot, absolute);

  if (relative.startsWith("..") || path.isAbsolute(relative)) {
    return null;
  }

  return relative.split(path.sep).join("/");
}

function sanitizePath(repoRoot, input) {
  const relative = String(input ?? "").replace(/^\/+/, "");
  const candidate = path.resolve(repoRoot, relative);

  let resolvedCandidate;
  try {
    resolvedCandidate = fs.realpathSync(candidate);
  } catch {
    throw AppError.notFound("File not found");
  }

  let resolvedRoot;
  try {
    resolvedRoot = fs.realpathSync(repoRoot);
  } catch {
    throw AppError.internal("Failed to resolve repo");
  }

  const allowedPrefix = `${resolvedRoot}${path.sep}`;
  if (resolvedCandidate !== resolvedRoot && !resolvedCandidate.startsWith(allowedPrefix)) {
    throw AppError.forbidden("Path not allowed");
  }

  return resolvedCandidate;
}

function parseResearchIndex(repoRoot) {
  const index = new Map();
  const readmePath = path.join(resolvePapersRoot(repoRoot), "README.md");
  if (!fs.existsSync(readmePath)) {
    return index;
  }

  const content = fs.readFileSync(readmePath, "utf8");
  const lines = content.split(/\r?\n/);

  let currentSport = null;
  let currentTitle = null;
  let currentFile = null;

  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (!line) {
      continue;
    }

    if (line.startsWith("**") && line.endsWith("**") && (line.match(/\*\*/g) ?? []).length >= 2) {
      const label = line.replaceAll("*", "").trim();
      if (label.toLowerCase() !== "research papers") {
        currentSport = label;
      }
      continue;
    }

    if (line.startsWith("- ") && !line.startsWith("- File:") && !line.startsWith("- Source:")) {
      currentTitle = line.slice(2).trim();
      currentFile = null;
      continue;
    }

    if (line.startsWith("- File:")) {
      const file = line.split("`")[1]?.trim() ?? "";
      const fileName = path.basename(file);
      if (!fileName) {
        continue;
      }

      currentFile = fileName;
      if (!index.has(fileName)) {
        index.set(fileName, new Map());
      }

      const meta = index.get(fileName);
      if (currentTitle) {
        meta.set("title", currentTitle);
      }
      if (currentSport) {
        meta.set("sport", currentSport);
      }

      continue;
    }

    if (line.startsWith("- Source:") && currentFile) {
      if (!index.has(currentFile)) {
        index.set(currentFile, new Map());
      }
      const meta = index.get(currentFile);
      const source = line.split(":").slice(1).join(":").trim();
      meta.set("source", source);
    }
  }

  return index;
}

function loadPapers(repoRoot) {
  const papersRoot = resolvePapersRoot(repoRoot);
  const index = parseResearchIndex(repoRoot);
  const papers = [];

  if (!fs.existsSync(papersRoot)) {
    return papers;
  }

  let sportEntries;
  try {
    sportEntries = fs.readdirSync(papersRoot, { withFileTypes: true });
  } catch {
    throw AppError.internal("Failed to read research");
  }

  for (const sportEntry of sportEntries) {
    if (!sportEntry.isDirectory()) {
      continue;
    }

    const sport = sportEntry.name;
    const sportPath = path.join(papersRoot, sport);

    let paperEntries;
    try {
      paperEntries = fs.readdirSync(sportPath, { withFileTypes: true });
    } catch {
      throw AppError.internal("Failed to read papers");
    }

    for (const paperEntry of paperEntries) {
      if (!paperEntry.isFile() || path.extname(paperEntry.name) !== ".pdf") {
        continue;
      }

      const paperPath = path.join(sportPath, paperEntry.name);
      const meta = index.get(paperEntry.name) ?? new Map();
      const title =
        meta.get("title") ??
        path
          .basename(paperEntry.name, ".pdf")
          .replaceAll("-", " ");
      const source = meta.get("source") ?? null;
      const file = toRelative(repoRoot, paperPath) ?? paperPath;

      papers.push({
        sport: meta.get("sport") ?? sport,
        title,
        file,
        source,
      });
    }
  }

  return papers;
}

function shouldSkipDir(name) {
  return [
    ".git",
    "node_modules",
    ".next",
    "target",
    "data",
    ".turbo",
    "platform",
    "papers",
  ].includes(name);
}

function walkDirectory(rootDir, callback) {
  const entries = fs.readdirSync(rootDir, { withFileTypes: true });
  for (const entry of entries) {
    const entryPath = path.join(rootDir, entry.name);

    if (entry.isDirectory()) {
      if (shouldSkipDir(entry.name)) {
        continue;
      }
      walkDirectory(entryPath, callback);
      continue;
    }

    callback(entryPath);
  }
}

function loadNotebooks(repoRoot) {
  const projectsRoot = resolveProjectsRoot(repoRoot);
  const notebooks = [];

  if (!fs.existsSync(projectsRoot)) {
    return notebooks;
  }

  walkDirectory(projectsRoot, (filePath) => {
    if (!filePath.endsWith(".ipynb")) {
      return;
    }

    const file = toRelative(repoRoot, filePath) ?? filePath;
    const relToProjects = path.relative(projectsRoot, filePath).split(path.sep);
    const sport = relToProjects[0] ?? "Unknown";
    const project = relToProjects[1] ?? "Unknown";

    notebooks.push({
      sport,
      project,
      file,
    });
  });

  return notebooks;
}

function dataFileStatusRaw(root, name, formats) {
  for (const format of formats) {
    const candidate = path.join(root, `${name}.${format}`);
    if (fs.existsSync(candidate)) {
      return [candidate, format, true];
    }
  }

  const defaultFormat = formats[0] ?? "csv";
  return [path.join(root, `${name}.${defaultFormat}`), defaultFormat, false];
}

function dataFileStatus(root, name, formats) {
  const [filePath, format, exists] = dataFileStatusRaw(root, name, formats);
  return {
    path: filePath,
    format,
    exists,
  };
}

function parseCsvLine(line) {
  const values = [];
  let current = "";
  let inQuotes = false;

  for (let i = 0; i < line.length; i += 1) {
    const char = line[i];

    if (char === '"') {
      if (inQuotes && line[i + 1] === '"') {
        current += '"';
        i += 1;
      } else {
        inQuotes = !inQuotes;
      }
      continue;
    }

    if (char === "," && !inQuotes) {
      values.push(current);
      current = "";
      continue;
    }

    current += char;
  }

  values.push(current);
  return values;
}

function readFixturesCsv(filePath, limit) {
  let content;
  try {
    content = fs.readFileSync(filePath, "utf8");
  } catch {
    throw AppError.internal("Failed to read fixtures CSV");
  }

  const lines = content
    .split(/\r?\n/)
    .map((line) => line.trimEnd())
    .filter(Boolean);

  if (lines.length === 0) {
    return [];
  }

  const headers = parseCsvLine(lines[0]);
  const fixtures = [];

  for (const line of lines.slice(1)) {
    const values = parseCsvLine(line);
    const record = {};

    for (let i = 0; i < headers.length; i += 1) {
      record[headers[i]] = values[i] ?? "";
    }

    fixtures.push({
      matchId: record.match_id ?? "",
      date: record.date ?? "",
      season: record.season ?? "",
      league: record.league ?? "",
      homeTeamId: record.home_team_id ?? "",
      awayTeamId: record.away_team_id ?? "",
    });

    if (fixtures.length >= limit) {
      break;
    }
  }

  return fixtures;
}

function parseJsonSafely(text, fallback) {
  try {
    return JSON.parse(text);
  } catch {
    return fallback;
  }
}

async function readJsonPath(repoRoot, filePath) {
  const absolute = path.isAbsolute(filePath) ? filePath : path.join(repoRoot, filePath);
  if (!fs.existsSync(absolute)) {
    return null;
  }

  let content;
  try {
    content = await fsp.readFile(absolute, "utf8");
  } catch {
    throw AppError.internal("Failed to read result");
  }

  try {
    return JSON.parse(content);
  } catch {
    throw AppError.internal("Invalid JSON");
  }
}

async function readTextPath(repoRoot, filePath) {
  const absolute = path.isAbsolute(filePath) ? filePath : path.join(repoRoot, filePath);
  if (!fs.existsSync(absolute)) {
    return null;
  }

  try {
    return await fsp.readFile(absolute, "utf8");
  } catch {
    throw AppError.internal("Failed to read log");
  }
}

async function writeJson(filePath, payload) {
  const content = JSON.stringify(payload, null, 2);
  try {
    await fsp.writeFile(filePath, content, "utf8");
  } catch {
    throw AppError.internal("Failed to write JSON");
  }
}

function getString(params, key, defaultValue, required) {
  if (Object.hasOwn(params, key) && params[key] !== undefined && params[key] !== null) {
    if (typeof params[key] === "string") {
      return params[key];
    }
    return String(params[key]);
  }

  if (defaultValue !== undefined && defaultValue !== null) {
    return defaultValue;
  }

  if (required) {
    throw AppError.badRequest(`Missing param: ${key}`);
  }

  return "";
}

function getInt(params, key, fallback, required) {
  if (Object.hasOwn(params, key) && params[key] !== undefined && params[key] !== null) {
    const value = params[key];
    if (typeof value === "number" && Number.isFinite(value)) {
      return Math.trunc(value);
    }

    const parsed = Number.parseInt(String(value), 10);
    if (Number.isFinite(parsed)) {
      return parsed;
    }

    throw AppError.badRequest(`Invalid param: ${key}`);
  }

  if (fallback !== undefined && fallback !== null) {
    return fallback;
  }

  if (required) {
    throw AppError.badRequest(`Missing param: ${key}`);
  }

  return 0;
}

function getBool(params, key, defaultValue) {
  if (Object.hasOwn(params, key) && typeof params[key] === "boolean") {
    return params[key];
  }

  return defaultValue;
}

function optionalInt(value) {
  if (typeof value === "number" && Number.isFinite(value)) {
    return Math.trunc(value);
  }

  if (typeof value === "string" && value.trim()) {
    const parsed = Number.parseInt(value, 10);
    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }

  return null;
}

function buildCommand(project, params, outputPath) {
  const script = path.join(project.pythonDir, "run_prediction.py");
  if (!fs.existsSync(script)) {
    throw AppError.internal("run_prediction.py not found");
  }

  const args = [script];

  if (project.kind === "F1") {
    const mode = getString(params, "mode", undefined, true);
    const source = getString(params, "source", undefined, true);
    const year = getInt(params, "year", 2026, true);
    const round = getInt(params, "round_number", optionalInt(params.round), true);
    const trainSeasons = getString(params, "train_seasons", "auto", false);
    const includeStandings = getBool(params, "include_standings", false);

    args.push("--mode", mode);
    args.push("--source", source);
    args.push("--year", String(year));
    args.push("--round", String(round));
    args.push("--train-seasons", trainSeasons);

    if (includeStandings) {
      args.push("--include-standings");
    }

    const cacheDir = typeof params.cache_dir === "string" ? params.cache_dir.trim() : "";
    if (cacheDir) {
      args.push("--cache-dir", cacheDir);
    }

    const meetingName = typeof params.meeting_name === "string" ? params.meeting_name.trim() : "";
    if (meetingName) {
      args.push("--meeting-name", meetingName);
    }

    const countryName = typeof params.country_name === "string" ? params.country_name.trim() : "";
    if (countryName) {
      args.push("--country-name", countryName);
    }
  } else if (project.kind === "Football") {
    const mode = getString(params, "mode", undefined, true);
    const league = getString(params, "league", undefined, true);
    const season = getInt(params, "season", 2026, true);
    const round = getInt(params, "round_number", optionalInt(params.round), true);
    const dataSource = getString(params, "data_source", "placeholder", false);
    const trainSeasons = getString(params, "train_seasons", "auto", false);

    args.push("--mode", mode);
    args.push("--league", league);
    args.push("--season", String(season));
    args.push("--round", String(round));
    args.push("--data-source", dataSource);
    args.push("--train-seasons", trainSeasons);

    const cacheDir = typeof params.cache_dir === "string" ? params.cache_dir.trim() : "";
    if (cacheDir) {
      args.push("--cache-dir", cacheDir);
    }
  } else {
    throw AppError.badRequest("Unknown project kind");
  }

  args.push("--output-format", "json");
  args.push("--output-path", outputPath);
  args.push("--quiet");

  return ["python", args];
}

async function executePythonRun(project, config, outputPath, stdoutPath, stderrPath) {
  if (!isObject(config.params)) {
    throw AppError.badRequest("Params missing");
  }

  const [command, args] = buildCommand(project, config.params, outputPath);

  try {
    await Bun.write(stdoutPath, "");
    await Bun.write(stderrPath, "");

    const process = Bun.spawn([command, ...args], {
      cwd: project.pythonDir,
      stdout: Bun.file(stdoutPath),
      stderr: Bun.file(stderrPath),
    });

    const exitCode = await process.exited;
    return exitCode === 0 && fs.existsSync(outputPath);
  } catch {
    throw AppError.internal("Failed to run python");
  }
}

function scoreFromResult(result) {
  if (!isObject(result) || !Array.isArray(result.rows)) {
    return null;
  }

  const values = [];
  for (const row of result.rows) {
    if (isObject(row) && typeof row.pred === "number") {
      values.push(row.pred);
    }
  }

  if (values.length === 0) {
    return null;
  }

  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

async function handleCatalog(state) {
  const projects = buildCatalog(state.repoRoot);
  const responseProjects = projects.map((project) => ({
    sport: project.sport,
    name: project.name,
    pythonDir: toRelative(state.repoRoot, project.pythonDir) ?? project.pythonDir,
    notebook: project.notebook ? toRelative(state.repoRoot, project.notebook) : null,
    params: paramsForProject(project.kind),
  }));

  return jsonResponse({ projects: responseProjects });
}

async function handleDataStatus(state) {
  const dataRoot = path.join(state.repoRoot, "data", "football");
  const teams = dataFileStatus(dataRoot, "teams", ["parquet", "csv"]);
  const matches = dataFileStatus(dataRoot, "matches", ["parquet", "csv"]);
  const fixtures = dataFileStatus(dataRoot, "fixtures", ["parquet", "csv"]);

  return jsonResponse({
    football: {
      teams,
      matches,
      fixtures,
    },
  });
}

async function handleFootballFixtures(state, url) {
  const dataRoot = path.join(state.repoRoot, "data", "football");
  const [fixturesPath, format, exists] = dataFileStatusRaw(dataRoot, "fixtures", ["parquet", "csv"]);

  if (!exists) {
    return jsonResponse({
      fixtures: [],
      warning: "Fixtures file not found.",
    });
  }

  if (format !== "csv") {
    return jsonResponse({
      fixtures: [],
      warning: "Parquet fixtures detected. CSV parsing only in MVP.",
    });
  }

  const rawLimit = Number.parseInt(url.searchParams.get("limit") ?? "5", 10);
  const limit = Number.isFinite(rawLimit) && rawLimit > 0 ? rawLimit : 5;
  const fixtures = readFixturesCsv(fixturesPath, limit);

  return jsonResponse({ fixtures, warning: null });
}

async function handleServeFile(state, pathname) {
  const rawPath = pathname.slice("/api/files/".length);
  const decodedPath = decodeURIComponent(rawPath);
  const safePath = sanitizePath(state.repoRoot, decodedPath);

  if (!fs.existsSync(safePath)) {
    throw AppError.notFound("File not found");
  }

  const file = Bun.file(safePath);
  const headers = responseHeaders({
    "content-type": file.type || "application/octet-stream",
  });

  return new Response(file, { status: 200, headers });
}

async function handleOpenPath(state, request) {
  const payload = await parseJsonBody(request);
  const requestedPath = typeof payload.path === "string" ? payload.path : "";
  if (!requestedPath) {
    throw AppError.badRequest("Path is required");
  }

  const safePath = sanitizePath(state.repoRoot, requestedPath);
  if (!fs.existsSync(safePath)) {
    throw AppError.notFound("File not found");
  }

  const result = spawnSync("open", [safePath]);
  if (result.error || result.status !== 0) {
    throw AppError.internal("Open command failed");
  }

  return jsonResponse({ status: "ok" });
}

async function handleListRuns(state) {
  const rows = await state.db`
    SELECT id, created_at, sport, project, status, duration_ms, sweep_id
    FROM runs
    ORDER BY created_at DESC
  `;

  const runs = rows.map((row) => ({
    id: row.id ?? "",
    createdAt: row.created_at ?? "",
    sport: row.sport ?? "",
    project: row.project ?? "",
    status: normalizeRunStatus(row.status),
    durationMs: row.duration_ms ?? null,
    sweepId: row.sweep_id ?? null,
  }));

  return jsonResponse(runs);
}

async function handleGetRun(state, runId) {
  const rows = await state.db`
    SELECT id, created_at, sport, project, status, config_json, result_path, stdout_path, stderr_path, duration_ms, sweep_id
    FROM runs
    WHERE id = ${runId}
  `;

  const row = rows[0];
  if (!row) {
    throw AppError.notFound("Run not found");
  }

  const config = parseJsonSafely(row.config_json ?? "{}", {});
  const result = row.result_path ? await readJsonPath(state.repoRoot, row.result_path) : null;
  const stdout = row.stdout_path ? await readTextPath(state.repoRoot, row.stdout_path) : null;
  const stderr = row.stderr_path ? await readTextPath(state.repoRoot, row.stderr_path) : null;

  return jsonResponse({
    id: row.id ?? "",
    createdAt: row.created_at ?? "",
    sport: row.sport ?? "",
    project: row.project ?? "",
    status: normalizeRunStatus(row.status),
    durationMs: row.duration_ms ?? null,
    sweepId: row.sweep_id ?? null,
    config,
    result,
    stdout,
    stderr,
    resultPath: row.result_path ? toRelative(state.repoRoot, row.result_path) : null,
  });
}

async function handleCreateRun(state, request) {
  const payload = await parseJsonBody(request);
  if (!isObject(payload)) {
    throw AppError.badRequest("Invalid request");
  }

  const sport = typeof payload.sport === "string" ? payload.sport : "";
  const projectName = typeof payload.project === "string" ? payload.project : "";
  if (!sport || !projectName) {
    throw AppError.badRequest("sport and project are required");
  }

  if (!isObject(payload.params)) {
    throw AppError.badRequest("params must be an object");
  }

  const project = resolveProject(state.repoRoot, sport, projectName);
  const runId = randomUUID();
  const runDir = path.join(state.dataDir, "runs", runId);
  await fsp.mkdir(runDir, { recursive: true });

  const params = { ...payload.params };
  const cacheDir =
    typeof payload.cacheDir === "string"
      ? payload.cacheDir
      : typeof payload.cache_dir === "string"
        ? payload.cache_dir
        : null;

  if (cacheDir && !Object.hasOwn(params, "cache_dir")) {
    params.cache_dir = cacheDir;
  }

  const tags = Array.isArray(payload.tags) ? payload.tags : null;
  const config = {
    sport,
    project: projectName,
    params,
    tags,
  };

  const configPath = path.join(runDir, "config.json");
  const outputPath = path.join(runDir, "result.json");
  const stdoutPath = path.join(runDir, "stdout.log");
  const stderrPath = path.join(runDir, "stderr.log");

  await writeJson(configPath, config);

  const createdAt = new Date().toISOString();
  await state.db`
    INSERT INTO runs (id, created_at, sport, project, status, config_json, result_path, stdout_path, stderr_path, duration_ms, sweep_id)
    VALUES (${runId}, ${createdAt}, ${project.sport}, ${project.name}, ${"queued"}, ${JSON.stringify(
      config
    )}, ${outputPath}, ${stdoutPath}, ${stderrPath}, ${null}, ${null})
  `;

  enqueueBackgroundJob(state, `run:${runId}`, async () => {
    await state.db`
      UPDATE runs
      SET status = ${"running"}
      WHERE id = ${runId}
    `;

    const startedAt = Date.now();
    let ok = false;

    try {
      ok = await executePythonRun(project, config, outputPath, stdoutPath, stderrPath);
    } catch (error) {
      console.error(`Run job failed (${runId})`);
      console.error(error);
      try {
        await fsp.appendFile(stderrPath, `\n${String(error)}\n`, "utf8");
      } catch {
        // Ignore best-effort stderr append failures.
      }
    }

    const durationMs = Date.now() - startedAt;
    const status = ok ? "done" : "error";

    await state.db`
      UPDATE runs
      SET status = ${status},
          duration_ms = ${durationMs},
          result_path = ${outputPath},
          stdout_path = ${stdoutPath},
          stderr_path = ${stderrPath}
      WHERE id = ${runId}
    `;
  });

  return jsonResponse({ runId, status: "queued" }, 202);
}

async function handleListSweeps(state) {
  const rows = await state.db`
    SELECT id, created_at, sport, project, param, values_json, base_config_json, status
    FROM sweeps
    ORDER BY created_at DESC
  `;

  const sweeps = rows.map((row) => ({
    id: row.id ?? "",
    createdAt: row.created_at ?? "",
    sport: row.sport ?? "",
    project: row.project ?? "",
    param: row.param ?? "",
    valuesJson: row.values_json ?? "[]",
    baseConfigJson: row.base_config_json ?? "{}",
    status: normalizeSweepStatus(row.status),
  }));

  return jsonResponse(sweeps);
}

async function handleCreateSweep(state, request) {
  const payload = await parseJsonBody(request);
  if (!isObject(payload)) {
    throw AppError.badRequest("Invalid request");
  }

  const sport = typeof payload.sport === "string" ? payload.sport : "";
  const projectName = typeof payload.project === "string" ? payload.project : "";
  if (!sport || !projectName) {
    throw AppError.badRequest("sport and project are required");
  }

  if (!isObject(payload.baseParams)) {
    throw AppError.badRequest("baseParams must be an object");
  }

  if (!isObject(payload.sweep)) {
    throw AppError.badRequest("sweep is required");
  }

  const sweepParam = typeof payload.sweep.param === "string" ? payload.sweep.param : "";
  const sweepValues = Array.isArray(payload.sweep.values) ? payload.sweep.values : [];
  if (!sweepParam || sweepValues.length === 0) {
    throw AppError.badRequest("sweep.param and sweep.values are required");
  }

  const project = resolveProject(state.repoRoot, sport, projectName);
  const sweepId = randomUUID();

  const baseConfig = {
    sport,
    project: projectName,
    params: payload.baseParams,
  };

  const createdAt = new Date().toISOString();
  await state.db`
    INSERT INTO sweeps (id, created_at, sport, project, param, values_json, base_config_json, status)
    VALUES (
      ${sweepId},
      ${createdAt},
      ${project.sport},
      ${project.name},
      ${sweepParam},
      ${JSON.stringify(sweepValues)},
      ${JSON.stringify(baseConfig)},
      ${"queued"}
    )
  `;

  const runIds = [];
  const runPlans = [];

  for (const value of sweepValues) {
    const runId = randomUUID();
    runIds.push(runId);

    const runDir = path.join(state.dataDir, "runs", runId);
    await fsp.mkdir(runDir, { recursive: true });

    const params = {
      ...payload.baseParams,
      [sweepParam]: value,
    };

    const config = {
      sport,
      project: projectName,
      params,
      sweep_id: sweepId,
    };

    const configPath = path.join(runDir, "config.json");
    const outputPath = path.join(runDir, "result.json");
    const stdoutPath = path.join(runDir, "stdout.log");
    const stderrPath = path.join(runDir, "stderr.log");

    await writeJson(configPath, config);

    await state.db`
      INSERT INTO runs (id, created_at, sport, project, status, config_json, result_path, stdout_path, stderr_path, duration_ms, sweep_id)
      VALUES (
        ${runId},
        ${createdAt},
        ${project.sport},
        ${project.name},
        ${"queued"},
        ${JSON.stringify(config)},
        ${outputPath},
        ${stdoutPath},
        ${stderrPath},
        ${null},
        ${sweepId}
      )
    `;

    const paramValue =
      typeof value === "string" || typeof value === "number" || typeof value === "boolean"
        ? String(value)
        : JSON.stringify(value);

    await state.db`
      INSERT INTO sweep_runs (sweep_id, run_id, param_value)
      VALUES (${sweepId}, ${runId}, ${paramValue})
    `;

    runPlans.push({
      runId,
      config,
      outputPath,
      stdoutPath,
      stderrPath,
    });
  }

  enqueueBackgroundJob(state, `sweep:${sweepId}`, async () => {
    try {
      await state.db`
        UPDATE sweeps
        SET status = ${"running"}
        WHERE id = ${sweepId}
      `;

      let successfulRuns = 0;
      let failedRuns = 0;

      for (const plan of runPlans) {
        await state.db`
          UPDATE runs
          SET status = ${"running"}
          WHERE id = ${plan.runId}
        `;

        const startedAt = Date.now();
        let ok = false;
        try {
          ok = await executePythonRun(
            project,
            plan.config,
            plan.outputPath,
            plan.stdoutPath,
            plan.stderrPath
          );
        } catch (error) {
          console.error(`Sweep run failed (${plan.runId})`);
          console.error(error);
          try {
            await fsp.appendFile(plan.stderrPath, `\n${String(error)}\n`, "utf8");
          } catch {
            // Ignore best-effort stderr append failures.
          }
        }

        const durationMs = Date.now() - startedAt;
        const runStatus = ok ? "done" : "error";
        if (ok) {
          successfulRuns += 1;
        } else {
          failedRuns += 1;
        }

        await state.db`
          UPDATE runs
          SET status = ${runStatus},
              duration_ms = ${durationMs},
              result_path = ${plan.outputPath},
              stdout_path = ${plan.stdoutPath},
              stderr_path = ${plan.stderrPath}
          WHERE id = ${plan.runId}
        `;
      }

      const sweepStatus =
        failedRuns === 0 ? "done" : successfulRuns === 0 ? "error" : "partial";
      await state.db`
        UPDATE sweeps
        SET status = ${sweepStatus}
        WHERE id = ${sweepId}
      `;
    } catch (error) {
      console.error(`Sweep job failed (${sweepId})`);
      console.error(error);
      await state.db`
        UPDATE runs
        SET status = ${"error"}
        WHERE sweep_id = ${sweepId} AND status IN (${"queued"}, ${"running"})
      `;
      await state.db`
        UPDATE sweeps
        SET status = ${"error"}
        WHERE id = ${sweepId}
      `;
    }
  });

  return jsonResponse({
    sweepId,
    runIds,
    status: "queued",
  }, 202);
}

async function handleGetSweep(state, sweepId) {
  const sweepRows = await state.db`
    SELECT id, created_at, sport, project, param, values_json, status
    FROM sweeps
    WHERE id = ${sweepId}
  `;

  const sweep = sweepRows[0];
  if (!sweep) {
    throw AppError.notFound("Sweep not found");
  }

  const runRows = await state.db`
    SELECT r.id, r.created_at, r.status, r.result_path, sr.param_value
    FROM runs r
    JOIN sweep_runs sr ON r.id = sr.run_id
    WHERE sr.sweep_id = ${sweepId}
    ORDER BY r.created_at ASC
  `;

  const runs = [];
  const summary = [];

  for (const row of runRows) {
    const result = row.result_path ? await readJsonPath(state.repoRoot, row.result_path) : null;
    const score = result ? scoreFromResult(result) : null;

    runs.push({
      id: row.id ?? "",
      createdAt: row.created_at ?? "",
      status: normalizeRunStatus(row.status),
      paramValue: row.param_value ?? "",
      result,
    });

    summary.push({
      paramValue: row.param_value ?? "",
      score,
    });
  }

  const values = parseJsonSafely(sweep.values_json ?? "[]", []);

  return jsonResponse({
    id: sweep.id ?? "",
    createdAt: sweep.created_at ?? "",
    sport: sweep.sport ?? "",
    project: sweep.project ?? "",
    param: sweep.param ?? "",
    values,
    status: normalizeSweepStatus(sweep.status),
    runs,
    summary,
  });
}

async function handleGetUserPreferences(state) {
  const rows = await state.db`
    SELECT preferences_json, savings_json, updated_at
    FROM user_preferences
    WHERE id = 1
  `;

  const row = rows[0];
  if (!row) {
    return jsonResponse({
      preferences: null,
      savings: DEFAULT_USER_SAVINGS,
      updatedAt: null,
    });
  }

  return jsonResponse({
    preferences: parseJsonSafely(row.preferences_json ?? "{}", {}),
    savings: sanitizeUserSavings(parseJsonSafely(row.savings_json ?? "{}", {})),
    updatedAt: row.updated_at ?? null,
  });
}

async function handleSaveUserPreferences(state, request) {
  const payload = await parseJsonBody(request);
  if (!isObject(payload)) {
    throw AppError.badRequest("Invalid request");
  }

  if (!isObject(payload.preferences)) {
    throw AppError.badRequest("preferences must be an object");
  }

  if (payload.savings !== undefined && !isObject(payload.savings)) {
    throw AppError.badRequest("savings must be an object when provided");
  }

  const updatedAt = new Date().toISOString();
  const preferencesJson = JSON.stringify(payload.preferences);
  const savingsJson = JSON.stringify(sanitizeUserSavings(payload.savings));

  await state.db`
    INSERT INTO user_preferences (id, updated_at, preferences_json, savings_json)
    VALUES (1, ${updatedAt}, ${preferencesJson}, ${savingsJson})
    ON CONFLICT (id) DO UPDATE
    SET updated_at = EXCLUDED.updated_at,
        preferences_json = EXCLUDED.preferences_json,
        savings_json = EXCLUDED.savings_json
  `;

  return jsonResponse({
    status: "ok",
    updatedAt,
  });
}

async function handleRequest(request, state) {
  if (request.method === "OPTIONS") {
    return new Response(null, {
      status: 204,
      headers: responseHeaders(),
    });
  }

  const url = new URL(request.url);
  const { pathname } = url;

  if (request.method === "GET" && pathname === "/api/health") {
    return jsonResponse({ status: "ok" });
  }

  if (request.method === "GET" && pathname === "/api/catalog") {
    return handleCatalog(state);
  }

  if (request.method === "GET" && pathname === "/api/data-status") {
    return handleDataStatus(state);
  }

  if (request.method === "GET" && pathname === "/api/papers") {
    return jsonResponse(loadPapers(state.repoRoot));
  }

  if (request.method === "GET" && pathname === "/api/notebooks") {
    return jsonResponse(loadNotebooks(state.repoRoot));
  }

  if (request.method === "GET" && pathname === "/api/football/fixtures") {
    return handleFootballFixtures(state, url);
  }

  if (request.method === "GET" && pathname.startsWith("/api/files/")) {
    return handleServeFile(state, pathname);
  }

  if (request.method === "POST" && pathname === "/api/open") {
    return handleOpenPath(state, request);
  }

  if (request.method === "GET" && pathname === "/api/user-preferences") {
    return handleGetUserPreferences(state);
  }

  if (request.method === "POST" && pathname === "/api/user-preferences") {
    return handleSaveUserPreferences(state, request);
  }

  if (request.method === "GET" && pathname === "/api/runs") {
    return handleListRuns(state);
  }

  if (request.method === "POST" && pathname === "/api/runs") {
    return handleCreateRun(state, request);
  }

  if (request.method === "GET" && pathname.startsWith("/api/runs/")) {
    const runId = decodeURIComponent(pathname.slice("/api/runs/".length));
    if (!runId) {
      throw AppError.notFound("Run not found");
    }
    return handleGetRun(state, runId);
  }

  if (request.method === "GET" && pathname === "/api/sweeps") {
    return handleListSweeps(state);
  }

  if (request.method === "POST" && pathname === "/api/sweeps") {
    return handleCreateSweep(state, request);
  }

  if (request.method === "GET" && pathname.startsWith("/api/sweeps/")) {
    const sweepId = decodeURIComponent(pathname.slice("/api/sweeps/".length));
    if (!sweepId) {
      throw AppError.notFound("Sweep not found");
    }
    return handleGetSweep(state, sweepId);
  }

  throw AppError.notFound("Not found");
}

function findRepoRoot(startPath) {
  let current = path.resolve(startPath);

  while (true) {
    if (
      fs.existsSync(path.join(current, "research")) ||
      fs.existsSync(path.join(current, "Research")) ||
      fs.existsSync(path.join(current, ".git"))
    ) {
      return current;
    }

    const parent = path.dirname(current);
    if (parent === current) {
      break;
    }

    current = parent;
  }

  return path.resolve(startPath);
}

async function main() {
  const currentDir = process.cwd();
  const repoRoot = process.env.REPO_ROOT
    ? path.resolve(process.env.REPO_ROOT)
    : findRepoRoot(currentDir);

  loadEnvFile(path.join(repoRoot, ".env"));
  loadEnvFile(path.join(repoRoot, ".env.local"));
  loadEnvFile(path.join(currentDir, ".env"));
  loadEnvFile(path.join(currentDir, ".env.local"));

  const dataDir = path.join(repoRoot, "platform", "backend", "data");
  await fsp.mkdir(dataDir, { recursive: true });

  const databaseUrl =
    process.env.DATABASE_URL ||
    process.env.SUPABASE_DIRECT_CONNECTION_STRING ||
    process.env.SUPABASE_DB_URL ||
    process.env.SUPABASE_SESSION_POOLER_CONNECTION_STRING ||
    process.env.SUPABASE_TRANSACTION_POOLER_CONNECTION_STRING;
  if (!databaseUrl) {
    throw new Error("DATABASE_URL is required (use your Supabase Postgres URI)");
  }

  const db = postgres(databaseUrl, {
    max: 5,
  });

  await initDb(db);

  const state = {
    repoRoot,
    dataDir,
    db,
    jobQueue: [],
    jobQueueRunning: false,
  };

  const port = Number.parseInt(process.env.PORT ?? "4000", 10) || 4000;

  Bun.serve({
    hostname: "0.0.0.0",
    port,
    async fetch(request) {
      try {
        return await handleRequest(request, state);
      } catch (error) {
        return errorResponse(error);
      }
    },
  });

  console.log(`Backend running on http://0.0.0.0:${port}`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
