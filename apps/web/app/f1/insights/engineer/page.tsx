import F1LivePlatformDashboard from "@/components/F1LivePlatformDashboard";

export default function F1InsightsEngineerPage() {
  return (
    <F1LivePlatformDashboard
      initialTab="engineer"
      surfaceTitle="F1 Engineer Dashboard"
      surfaceStatus="Telemetry, FastF1 track geometry, and engineering artifacts"
      showOperationsControls={false}
    />
  );
}
