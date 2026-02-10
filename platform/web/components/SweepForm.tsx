"use client";

import { useEffect, useMemo, useState } from "react";
import { API_BASE } from "@/lib/api";
import type { CatalogProject, ParamDef } from "@/lib/types";

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

export default function SweepForm({ projects }: { projects: CatalogProject[] }) {
  const [projectKey, setProjectKey] = useState(
    projects[0] ? `${projects[0].sport}::${projects[0].name}` : ""
  );
  const [baseValues, setBaseValues] = useState<Record<string, string | number | boolean>>(
    projects[0] ? initValues(projects[0].params) : {}
  );
  const [paramName, setParamName] = useState(projects[0]?.params[0]?.name ?? "");
  const [valuesInput, setValuesInput] = useState("1,2,3");
  const [status, setStatus] = useState<string | null>(null);

  const selectedProject = useMemo(
    () => projects.find((p) => `${p.sport}::${p.name}` === projectKey),
    [projects, projectKey]
  );

  useEffect(() => {
    if (selectedProject) {
      setBaseValues(initValues(selectedProject.params));
      setParamName(selectedProject.params[0]?.name ?? "");
    }
  }, [selectedProject]);

  const handleChange = (name: string, value: string | number | boolean) => {
    setBaseValues((prev) => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!selectedProject) {
      return;
    }
    setStatus("running");
    const params: Record<string, unknown> = {};
    for (const param of selectedProject.params) {
      params[param.name] = parseValue(param, baseValues[param.name] ?? "");
    }
    const sweepParam = selectedProject.params.find((p) => p.name === paramName);
    const values = valuesInput
      .split(",")
      .map((value) => value.trim())
      .filter(Boolean)
      .map((value) => (sweepParam ? parseValue(sweepParam, value) : value));

    const res = await fetch(`${API_BASE}/api/sweeps`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        sport: selectedProject.sport,
        project: selectedProject.name,
        baseParams: params,
        sweep: { param: paramName, values },
      }),
    });

    if (!res.ok) {
      setStatus("failed");
      return;
    }

    const data = (await res.json()) as { sweepId: string };
    setStatus(`created: ${data.sweepId}`);
  };

  return (
    <form className="panel" onSubmit={handleSubmit}>
      <div className="panel-header">
        <div className="panel-header-left">
          <h2 className="module-title">New Sweep</h2>
          <span className="module-subtitle">Launch a parameter sweep and compare runs</span>
        </div>
      </div>
      <div className="panel-body">
        <div className="stack">
          <div className="form-grid">
            <div className="field">
              <label>Project</label>
              <select value={projectKey} onChange={(event) => setProjectKey(event.target.value)}>
                {projects.map((project) => (
                  <option key={`${project.sport}::${project.name}`} value={`${project.sport}::${project.name}`}>
                    {project.sport} — {project.name}
                  </option>
                ))}
              </select>
            </div>
            <div className="field">
              <label>Sweep Param</label>
              <select value={paramName} onChange={(event) => setParamName(event.target.value)}>
                {selectedProject?.params.map((param) => (
                  <option key={param.name} value={param.name}>
                    {param.label}
                  </option>
                ))}
              </select>
            </div>
            <div className="field">
              <label>Values (comma-separated)</label>
              <input
                value={valuesInput}
                onChange={(event) => setValuesInput(event.target.value)}
                placeholder="1,2,3"
              />
            </div>
          </div>

          <div className="form-grid">
            {selectedProject?.params.map((param) => (
              <div className="field" key={param.name}>
                <label>{param.label}</label>
                {param.kind === "select" ? (
                  <select
                    value={String(baseValues[param.name] ?? "")}
                    onChange={(event) => handleChange(param.name, event.target.value)}
                  >
                    {param.options?.map((option) => (
                      <option key={option} value={option}>
                        {option}
                      </option>
                    ))}
                  </select>
                ) : param.kind === "bool" ? (
                  <select
                    value={String(baseValues[param.name] ?? false)}
                    onChange={(event) => handleChange(param.name, event.target.value === "true")}
                  >
                    <option value="false">false</option>
                    <option value="true">true</option>
                  </select>
                ) : (
                  <input
                    type={param.kind === "int" ? "number" : "text"}
                    value={String(baseValues[param.name] ?? "")}
                    onChange={(event) =>
                      handleChange(
                        param.name,
                        param.kind === "int" ? Number(event.target.value) : event.target.value
                      )
                    }
                  />
                )}
              </div>
            ))}
          </div>

          <div className="row" style={{ gap: 12 }}>
            <button className="button" type="submit">Launch Sweep</button>
            {status && (
              <span className="chip">
                <span className={`chip-led ${status === "failed" ? "red" : status === "running" ? "amber" : "green"}`} />
                {status}
              </span>
            )}
          </div>
        </div>
      </div>
    </form>
  );
}
