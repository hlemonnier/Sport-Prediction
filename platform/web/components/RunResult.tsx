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
      <div className="panel">
        <div className="panel-header">
          <h3 className="module-title">Results</h3>
        </div>
        <div className="panel-body">
          <div className="empty-state">
            <span className="empty-state-text">No run executed</span>
            <span className="empty-state-hint">Launch a run to see results here</span>
          </div>
        </div>
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
    grid: { left: 40, right: 16, top: 24, bottom: 44 },
    textStyle: { color: "#1a1a19", fontFamily: "JetBrains Mono, monospace" },
    xAxis: {
      type: "category",
      data: chartData.map((item) => item.label),
      axisLabel: { rotate: 35, color: "#66655f", fontSize: 10 },
      axisLine: { lineStyle: { color: "#e5e4e2" } },
    },
    yAxis: {
      type: "value",
      axisLabel: { color: "#66655f", fontSize: 10 },
      splitLine: { lineStyle: { color: "rgba(220, 38, 38, 0.08)" } },
    },
    series: [
      {
        type: "bar",
        data: chartData.map((item) => item.value),
        itemStyle: { color: "#dc2626", borderRadius: [2, 2, 0, 0] },
        barMaxWidth: 32,
      },
    ],
    tooltip: {
      trigger: "axis",
      backgroundColor: "#ffffff",
      borderColor: "#e5e4e2",
      textStyle: { color: "#1a1a19", fontSize: 11 },
    },
  };

  return (
    <div className="stack">
      {/* Run info */}
      <div className="panel">
        <div className="panel-header">
          <div className="panel-header-left">
            <h3 className="module-title">Run {run.id.slice(0, 8)}</h3>
            <span className="module-subtitle">{run.project}</span>
          </div>
          <span className="chip">
            <span className={`chip-led ${run.status === "done" ? "green" : run.status === "error" ? "red" : "amber"}`} />
            {run.status}
          </span>
        </div>
        {run.result?.notes && run.result.notes.length > 0 && (
          <div className="panel-body">
            <div className="row" style={{ flexWrap: "wrap", gap: 8 }}>
              {run.result.notes.map((note, index) => (
                <span className="chip" key={`${note}-${index}`}>{note}</span>
              ))}
            </div>
          </div>
        )}
      </div>

      <div className="grid-two">
        {/* Table */}
        <div className="panel">
          <div className="panel-header">
            <h4 className="module-title">Table</h4>
          </div>
          <div className="panel-body" style={{ padding: 0 }}>
            {rows.length === 0 ? (
              <div className="panel-body">
                <div className="empty-state">
                  <span className="empty-state-text">No rows available</span>
                </div>
              </div>
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
        </div>

        {/* Chart */}
        <div className="panel">
          <div className="panel-header">
            <h4 className="module-title">Signal</h4>
          </div>
          <div className="panel-body">
            {chartData.length === 0 ? (
              <div className="empty-state">
                <span className="empty-state-text">No numeric signal available</span>
              </div>
            ) : (
              <ReactECharts option={chartOption} style={{ height: 300 }} />
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
