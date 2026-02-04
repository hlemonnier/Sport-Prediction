"use client";

import ReactECharts from "echarts-for-react";
import type { RunDetail } from "@/lib/types";

function toNumber(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

export default function RunResult({ run }: { run: RunDetail | null }) {
  if (!run) {
    return (
      <div className="card">
        <h3 className="section-title">Results</h3>
        <p className="section-subtitle">No run yet. Launch a run to see results here.</p>
      </div>
    );
  }

  const rows = run.result?.rows ?? [];
  const columns = rows.length > 0 ? Object.keys(rows[0] ?? {}) : [];
  const chartData = rows
    .map((row) => ({
      label: String(
        row.driver_name ?? row.team ?? row.player ?? row.rank ?? row.id ?? "Item"
      ),
      value: toNumber(row.pred ?? row.score ?? row.value),
    }))
    .filter((item) => item.value !== null);

  const chartOption = {
    grid: { left: 40, right: 20, top: 30, bottom: 50 },
    textStyle: { color: "#f2f4f8" },
    xAxis: {
      type: "category",
      data: chartData.map((item) => item.label),
      axisLabel: { rotate: 35, color: "#9aa3b2" },
      axisLine: { lineStyle: { color: "#2a3242" } },
    },
    yAxis: {
      type: "value",
      axisLabel: { color: "#9aa3b2" },
      splitLine: { lineStyle: { color: "rgba(125, 249, 255, 0.08)" } },
    },
    series: [
      {
        type: "bar",
        data: chartData.map((item) => item.value),
        itemStyle: { color: "#7df9ff" },
      },
    ],
    tooltip: {
      trigger: "axis",
      backgroundColor: "#0b0d14",
      borderColor: "rgba(125, 249, 255, 0.3)",
      textStyle: { color: "#f2f4f8" },
    },
  };

  return (
    <div className="stack">
      <div className="card">
        <h3 className="section-title">Run {run.id}</h3>
        <p className="section-subtitle">
          Status: <span className="pill">{run.status}</span> | Project: {run.project}
        </p>
        {run.result?.notes && run.result.notes.length > 0 ? (
          <div className="stack">
            {run.result.notes.map((note, index) => (
              <div className="pill" key={`${note}-${index}`}>
                {note}
              </div>
            ))}
          </div>
        ) : null}
      </div>

      <div className="grid-two">
        <div className="card">
          <h4 className="section-title">Table</h4>
          {rows.length === 0 ? (
            <p className="section-subtitle">No rows available.</p>
          ) : (
            <table className="table">
              <thead>
                <tr>
                  {columns.map((col) => (
                    <th key={col}>{col}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.map((row, idx) => (
                  <tr key={`row-${idx}`}>
                    {columns.map((col) => (
                      <td key={`${idx}-${col}`}>{String(row[col] ?? "")}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
        <div className="card">
          <h4 className="section-title">Signal</h4>
          {chartData.length === 0 ? (
            <p className="section-subtitle">No numeric signal to plot yet.</p>
          ) : (
            <ReactECharts option={chartOption} style={{ height: 320 }} />
          )}
        </div>
      </div>
    </div>
  );
}
