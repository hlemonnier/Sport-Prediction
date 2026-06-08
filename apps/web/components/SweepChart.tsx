"use client";

import ReactECharts from "echarts-for-react";

export default function SweepChart({
  summary,
}: {
  summary: Array<{ paramValue: string; score?: number | null }>;
}) {
  const data = summary.map((item) => ({
    label: item.paramValue,
    value: item.score ?? null,
  }));

  const option = {
    grid: { left: 40, right: 16, top: 24, bottom: 44 },
    textStyle: { color: "#1a1a19", fontFamily: "JetBrains Mono, monospace" },
    xAxis: {
      type: "category",
      data: data.map((item) => item.label),
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
        type: "line",
        smooth: true,
        data: data.map((item) => item.value),
        itemStyle: { color: "#dc2626" },
        lineStyle: { color: "#dc2626", width: 2 },
        areaStyle: {
          color: {
            type: "linear",
            x: 0, y: 0, x2: 0, y2: 1,
            colorStops: [
              { offset: 0, color: "rgba(220, 38, 38, 0.15)" },
              { offset: 1, color: "rgba(220, 38, 38, 0)" },
            ],
          },
        },
        symbolSize: 6,
      },
    ],
    tooltip: {
      trigger: "axis",
      backgroundColor: "#ffffff",
      borderColor: "#e5e4e2",
      textStyle: { color: "#1a1a19", fontSize: 11 },
    },
  };

  return <ReactECharts option={option} style={{ height: 300 }} />;
}
