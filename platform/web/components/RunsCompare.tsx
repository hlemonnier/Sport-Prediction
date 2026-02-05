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
      <div className="card">
        <h2 className="section-title">Runs</h2>
        <p className="section-subtitle">
          Selectionne plusieurs runs pour comparer les sorties.
        </p>
        {runs.length === 0 ? (
          <p className="section-subtitle">Aucune run disponible.</p>
        ) : (
          <table className="table">
            <thead>
              <tr>
                <th>Sel</th>
                <th>ID</th>
                <th>Sport</th>
                <th>Projet</th>
                <th>Statut</th>
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
                    />
                  </td>
                  <td>{run.id.slice(0, 8)}</td>
                  <td>{run.sport}</td>
                  <td>{run.project}</td>
                  <td>{run.status}</td>
                  <td>{new Date(run.createdAt).toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
      {details.length > 0 ? (
        <div className="grid-two">
          {details.map((detail) => (
            <RunResult key={detail.id} run={detail} />
          ))}
        </div>
      ) : null}
    </div>
  );
}
