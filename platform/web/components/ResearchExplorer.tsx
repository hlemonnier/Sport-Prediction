"use client";

import { useEffect, useState } from "react";
import { API_BASE, fileUrl } from "@/lib/api";
import type { NotebookEntry, PaperEntry } from "@/lib/types";
import NotebookViewer from "./NotebookViewer";

export default function ResearchExplorer({
  papers,
  notebooks,
}: {
  papers: PaperEntry[];
  notebooks: NotebookEntry[];
}) {
  const [activePaper, setActivePaper] = useState<PaperEntry | null>(papers[0] ?? null);
  const [activeNotebook, setActiveNotebook] = useState<NotebookEntry | null>(notebooks[0] ?? null);
  const [notebookData, setNotebookData] = useState<Record<string, unknown> | null>(null);

  useEffect(() => {
    if (!activeNotebook) {
      setNotebookData(null);
      return;
    }
    let active = true;
    const load = async () => {
      const res = await fetch(fileUrl(activeNotebook.file), { cache: "no-store" });
      if (!res.ok) {
        return;
      }
      const json = (await res.json()) as Record<string, unknown>;
      if (active) {
        setNotebookData(json);
      }
    };
    load();
    return () => {
      active = false;
    };
  }, [activeNotebook]);

  const openSystem = async (path: string) => {
    await fetch(`${API_BASE}/api/open`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path }),
    });
  };

  return (
    <div className="stack">
      <div className="grid-two">
        <div className="card stack">
          <h2 className="section-title">Papers</h2>
          <div className="stack">
            {papers.map((paper) => (
              <button
                key={paper.file}
                className="button secondary"
                onClick={() => setActivePaper(paper)}
              >
                {paper.title}
              </button>
            ))}
          </div>
        </div>
        <div className="card stack">
          <h2 className="section-title">Paper Preview</h2>
          {activePaper ? (
            <>
              <p className="section-subtitle">{activePaper.title}</p>
              <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
                {activePaper.source ? (
                  <a
                    className="button secondary"
                    href={activePaper.source}
                    target="_blank"
                    rel="noreferrer"
                  >
                    Source
                  </a>
                ) : null}
                <button className="button" onClick={() => openSystem(activePaper.file)}>
                  Open system
                </button>
              </div>
              <iframe className="viewer" src={fileUrl(activePaper.file)} />
            </>
          ) : (
            <p className="section-subtitle">No paper selected.</p>
          )}
        </div>
      </div>

      <div className="grid-two">
        <div className="card stack">
          <h2 className="section-title">Notebooks</h2>
          <div className="stack">
            {notebooks.map((notebook) => (
              <button
                key={notebook.file}
                className="button secondary"
                onClick={() => setActiveNotebook(notebook)}
              >
                {notebook.sport} — {notebook.project}
              </button>
            ))}
          </div>
        </div>
        <div className="card stack">
          <h2 className="section-title">Notebook Viewer</h2>
          {activeNotebook ? (
            <>
              <p className="section-subtitle">
                {activeNotebook.sport} — {activeNotebook.project}
              </p>
              <button className="button" onClick={() => openSystem(activeNotebook.file)}>
                Open system
              </button>
              <NotebookViewer notebook={notebookData} />
            </>
          ) : (
            <p className="section-subtitle">No notebook selected.</p>
          )}
        </div>
      </div>
    </div>
  );
}
