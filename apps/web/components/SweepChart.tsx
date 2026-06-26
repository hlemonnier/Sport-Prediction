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
    textStyle: { color: "#fefefe", fontFamily: "JetBrains Mono, monospace" },
    xAxis: {
      type: "category",
      data: data.map((item) => item.label),
      axisLabel: { rotate: 35, color: "#a7a7a7", fontSize: 10 },
      axisLine: { lineStyle: { color: "#292929" } },
    },
    yAxis: {
      type: "value",
      axisLabel: { color: "#a7a7a7", fontSize: 10 },
      splitLine: { lineStyle: { color: "rgba(255, 99, 99, 0.1)" } },
    },
    series: [
      {
        type: "line",
        smooth: true,
        data: data.map((item) => item.value),
        itemStyle: { color: "#ff6363" },
        lineStyle: { color: "#ff6363", width: 2 },
        areaStyle: {
          color: {
            type: "linear",
            x: 0, y: 0, x2: 0, y2: 1,
            colorStops: [
              { offset: 0, color: "rgba(255, 99, 99, 0.18)" },
              { offset: 1, color: "rgba(255, 99, 99, 0)" },
            ],
          },
        },
        symbolSize: 6,
      },
    ],
    tooltip: {
      trigger: "axis",
      backgroundColor: "#151515",
      borderColor: "#292929",
      textStyle: { color: "#fefefe", fontSize: 11 },
    },
  };

  return <ReactECharts option={option} style={{ height: 300 }} />;
}
