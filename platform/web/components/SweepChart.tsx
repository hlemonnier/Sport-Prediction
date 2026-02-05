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
    grid: { left: 40, right: 20, top: 30, bottom: 50 },
    textStyle: { color: "#f2f4f8" },
    xAxis: {
      type: "category",
      data: data.map((item) => item.label),
      axisLabel: { rotate: 35, color: "#9aa3b2" },
      axisLine: { lineStyle: { color: "#2a3242" } },
    },
    yAxis: {
      type: "value",
      axisLabel: { color: "#9aa3b2" },
      splitLine: { lineStyle: { color: "rgba(110, 231, 255, 0.12)" } },
    },
    series: [
      {
        type: "line",
        smooth: true,
        data: data.map((item) => item.value),
        itemStyle: { color: "#6ee7ff" },
        lineStyle: { color: "#6ee7ff" },
      },
    ],
    tooltip: {
      trigger: "axis",
      backgroundColor: "#0b0d14",
      borderColor: "rgba(110, 231, 255, 0.3)",
      textStyle: { color: "#f2f4f8" },
    },
  };

  return <ReactECharts option={option} style={{ height: 320 }} />;
}
