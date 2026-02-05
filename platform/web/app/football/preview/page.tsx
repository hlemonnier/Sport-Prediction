import FootballPreview from "@/components/FootballPreview";

export default function FootballPreviewPage() {
  return (
    <div className="stack">
      <div>
        <h1 className="section-title">Football Apercu</h1>
        <p className="section-subtitle">
          Apercu rapide des matchs a venir et facteurs principaux.
        </p>
      </div>
      <FootballPreview />
    </div>
  );
}
