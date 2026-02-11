"use client";

import { useEffect, useMemo, useState } from "react";
import { createRun, getRun, waitForRunCompletion } from "@/lib/api";
import { showToast } from "@/lib/toast";
import type { CatalogProject, ParamDef, RunDetail } from "@/lib/types";
import RunResult from "./RunResult";

function initValues(params: ParamDef[]): Record<string, string | number | boolean> {
  const values: Record<string, string | number | boolean> = {};
  for (const param of params) {
    if (param.default !== undefined && param.default !== null) {
      values[param.name] = param.default as string | number | boolean;
    } else if (param.kind === "bool") {
      values[param.name] = false;
    } else {
      values[param.name] = "";
    }
  }
  return values;
}

function parseValue(param: ParamDef, raw: string | number | boolean) {
  if (param.kind === "int") {
    if (typeof raw === "number") {
      return raw;
    }
    const parsed = parseInt(String(raw), 10);
    return Number.isFinite(parsed) ? parsed : 0;
  }
  if (param.kind === "bool") {
    if (typeof raw === "boolean") {
      return raw;
    }
    return String(raw).toLowerCase() === "true";
  }
  return String(raw);
}

export default function RunForm({
  project,
  title,
  description,
  defaults,
  locked,
  hidden,
}: {
  project: CatalogProject;
  title?: string;
  description?: string;
  defaults?: Record<string, string | number | boolean>;
  locked?: string[];
  hidden?: string[];
}) {
  const [values, setValues] = useState(() => initValues(project.params));
  const [loading, setLoading] = useState(false);
  const [run, setRun] = useState<RunDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [runProgress, setRunProgress] = useState<string | null>(null);

  const formFields = useMemo(() => project.params, [project.params]);

  useEffect(() => {
    const initial = initValues(project.params);
    if (defaults) {
      for (const [key, value] of Object.entries(defaults)) {
        initial[key] = value;
      }
    }
    setValues(initial);
  }, [project, defaults]);

  const handleChange = (name: string, value: string | number | boolean) => {
    setValues((prev) => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const params: Record<string, unknown> = {};
      for (const param of formFields) {
        const raw = values[param.name];
        if (param.required && (raw === "" || raw === undefined)) {
          throw new Error(`Missing parameter: ${param.label}`);
        }
        params[param.name] = parseValue(param, raw ?? "");
      }

      const runData = await createRun({
        sport: project.sport,
        project: project.name,
        params,
      });

      setRunProgress(`Queued run ${runData.runId.slice(0, 8)}...`);
      showToast({ message: "Run queued. Processing started.", kind: "info" });

      const firstDetail = await getRun(runData.runId);
      setRun(firstDetail);

      const finalDetail = await waitForRunCompletion(runData.runId, {
        pollMs: 1500,
        timeoutMs: 15 * 60 * 1000,
        onTick: (detail) => {
          setRun(detail);
          setRunProgress(
            detail.status === "queued" || detail.status === "running"
              ? `${detail.status.toUpperCase()} — ${detail.id.slice(0, 8)}`
              : null
          );
        },
      });
      setRun(finalDetail);
      setRunProgress(null);

      if (finalDetail.status === "done") {
        showToast({ message: "Run completed", kind: "success" });
      } else {
        showToast({ message: "Run finished with errors", kind: "error" });
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unexpected error";
      setRunProgress(null);
      setError(message);
      showToast({ message, kind: "error" });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="run-layout">
      {/* Left: Run Console */}
      <form className="panel" onSubmit={handleSubmit} style={{ alignSelf: "start" }}>
        <div className="panel-header">
          <div className="panel-header-left">
            <h2 className="module-title">{title ?? project.name}</h2>
            <span className="module-subtitle">
              {description ?? "Configure, run, and inspect results"}
            </span>
          </div>
        </div>
        <div className="panel-body">
          <div className="stack">
            <div className="form-grid">
              {formFields
                .filter((param) => !(hidden ?? []).includes(param.name))
                .map((param) => (
                <div className="field" key={param.name}>
                  <label htmlFor={param.name}>{param.label}</label>
                  {param.kind === "select" ? (
                    <select
                      id={param.name}
                      value={String(values[param.name] ?? "")}
                      onChange={(event) => handleChange(param.name, event.target.value)}
                      disabled={(locked ?? []).includes(param.name)}
                    >
                      {param.options?.map((option) => (
                        <option key={option} value={option}>
                          {option}
                        </option>
                      ))}
                    </select>
                  ) : param.kind === "bool" ? (
                    <select
                      id={param.name}
                      value={String(values[param.name] ?? false)}
                      onChange={(event) => handleChange(param.name, event.target.value === "true")}
                      disabled={(locked ?? []).includes(param.name)}
                    >
                      <option value="false">false</option>
                      <option value="true">true</option>
                    </select>
                  ) : (
                    <input
                      id={param.name}
                      type={param.kind === "int" ? "number" : "text"}
                      value={String(values[param.name] ?? "")}
                      onChange={(event) =>
                        handleChange(
                          param.name,
                          param.kind === "int" ? Number(event.target.value) : event.target.value
                        )
                      }
                      disabled={(locked ?? []).includes(param.name)}
                    />
                  )}
                </div>
              ))}
            </div>
            <div className="row" style={{ gap: 12 }}>
              <button className="button" type="submit" disabled={loading}>
                {loading ? "Running..." : "Launch Run"}
              </button>
              {runProgress && (
                <span className="chip">
                  <span className="chip-led amber" />
                  {runProgress}
                </span>
              )}
              {error && (
                <span className="chip">
                  <span className="chip-led red" />
                  {error}
                </span>
              )}
            </div>
          </div>
        </div>
      </form>

      {/* Right: Results Panel */}
      <RunResult run={run} />
    </div>
  );
}
