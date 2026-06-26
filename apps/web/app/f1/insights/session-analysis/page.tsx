import F1LivePlatformDashboard from "@/components/F1LivePlatformDashboard";

export default function F1InsightsSessionAnalysisPage() {
  return (
    <F1LivePlatformDashboard
      initialTab="lapChart"
      surfaceTitle="F1 Session Analysis"
      surfaceStatus="Select an imported or replay session for lap chart, stint timeline, pace analysis, and micro-sector review"
      showOperationsControls={false}
      sessionMode="selected"
    />
  );
}
