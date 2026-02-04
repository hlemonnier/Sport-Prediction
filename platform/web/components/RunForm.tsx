"use client";

import { useEffect, useMemo, useState } from "react";
import { API_BASE } from "@/lib/api";
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
          throw new Error(`Missing param: ${param.label}`);
        }
        params[param.name] = parseValue(param, raw ?? "");
      }

      const runRes = await fetch(`${API_BASE}/api/runs`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sport: project.sport,
          project: project.name,
          params,
        }),
      });

      if (!runRes.ok) {
        throw new Error(`Run failed: ${runRes.status}`);
      }
      const runData = (await runRes.json()) as { runId: string };
      const detailRes = await fetch(`${API_BASE}/api/runs/${runData.runId}`, {
        cache: "no-store",
      });
      if (!detailRes.ok) {
        throw new Error(`Detail failed: ${detailRes.status}`);
      }
      const detail = (await detailRes.json()) as RunDetail;
      setRun(detail);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unexpected error");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="stack">
      <form className="card stack" onSubmit={handleSubmit}>
        <div>
          <h2 className="section-title">{title ?? project.name}</h2>
          <p className="section-subtitle">
            {description ?? "Configure the run, launch the pipeline, and inspect results instantly."}
          </p>
        </div>
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
        <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
          <button className="button" type="submit" disabled={loading}>
            {loading ? "Running..." : "Launch run"}
          </button>
          {error ? <span className="pill">{error}</span> : null}
        </div>
      </form>
      <RunResult run={run} />
    </div>
  );
}
