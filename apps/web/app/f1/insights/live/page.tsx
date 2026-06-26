import F1LivePlatformDashboard from "@/components/F1LivePlatformDashboard";

export default function F1InsightsLivePage() {
  return (
    <F1LivePlatformDashboard
      initialTab="standings"
      surfaceTitle="F1 Live Dashboard"
      surfaceStatus="Near-live timing, standings, sectors, and race operations"
      showOperationsControls={false}
    />
  );
}
