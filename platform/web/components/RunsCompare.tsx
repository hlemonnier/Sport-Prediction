"use client";

import { useEffect, useState } from "react";
import { API_BASE } from "@/lib/api";
import type { RunDetail, RunSummary } from "@/lib/types";
import RunResult from "./RunResult";

export default function RunsCompare({ runs }: { runs: RunSummary[] }) {
  const [selected, setSelected] = useState<string[]>([]);
  const [details, setDetails] = useState<RunDetail[]>([]);

  useEffect(() => {
    if (selected.length === 0) {
      setDetails([]);
      return;
    }
    let active = true;
    const fetchDetails = async () => {
      const results = await Promise.all(
        selected.map(async (id) => {
          const res = await fetch(`${API_BASE}/api/runs/${id}`, { cache: "no-store" });
          if (!res.ok) {
            return null;
          }
          return (await res.json()) as RunDetail;
        })
      );
      if (active) {
        setDetails(results.filter(Boolean) as RunDetail[]);
      }
    };
    fetchDetails();
    return () => {
      active = false;
    };
  }, [selected]);

  const toggle = (id: string) => {
    setSelected((prev) =>
      prev.includes(id) ? prev.filter((item) => item !== id) : [...prev, id]
    );
  };

  return (
    <div className="stack">
      <div className="panel">
        <div className="panel-header">
          <div className="panel-header-left">
            <h2 className="module-title">Runs</h2>
            <span className="module-subtitle">
              Select multiple runs to compare outputs
            </span>
          </div>
          {selected.length > 0 && (
            <span className="chip">
              <span className="chip-led accent" />
              {selected.length} selected
            </span>
          )}
        </div>
        <div className="panel-body" style={{ padding: 0 }}>
          {runs.length === 0 ? (
            <div className="panel-body">
              <div className="empty-state">
                <span className="empty-state-text">No runs available</span>
                <span className="empty-state-hint">Launch runs from the F1 or Football modules</span>
              </div>
            </div>
          ) : (
            <table className="table">
              <thead>
                <tr>
                  <th style={{ width: 40 }}>Sel</th>
                  <th>ID</th>
                  <th>Sport</th>
                  <th>Project</th>
                  <th>Status</th>
                  <th>Date</th>
                </tr>
              </thead>
              <tbody>
                {runs.map((run) => (
                  <tr key={run.id}>
                    <td>
                      <input
                        type="checkbox"
                        checked={selected.includes(run.id)}
                        onChange={() => toggle(run.id)}
                        style={{ accentColor: "var(--accent)" }}
                      />
                    </td>
                    <td className="mono" style={{ fontSize: 12 }}>{run.id.slice(0, 8)}</td>
                    <td>{run.sport}</td>
                    <td>{run.project}</td>
                    <td>
                      <span className="chip">
                        <span className={`chip-led ${run.status === "done" ? "green" : run.status === "error" ? "red" : "amber"}`} />
                        {run.status}
                      </span>
                    </td>
                    <td className="mono" style={{ fontSize: 12, color: "var(--muted)" }}>
                      {new Date(run.createdAt).toLocaleString()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>
      {details.length > 0 && (
        <div className="grid-two">
          {details.map((detail) => (
            <RunResult key={detail.id} run={detail} />
          ))}
        </div>
      )}
    </div>
  );
}
