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
    <div className="stack-lg">
      <div>
        <h1 className="page-title">Research</h1>
        <p className="page-status">Papers and notebooks explorer</p>
      </div>

      <div className="grid-two">
        <div className="panel">
          <div className="panel-header">
            <h2 className="module-title">Papers</h2>
          </div>
          <div className="panel-body">
            {papers.length === 0 ? (
              <div className="empty-state">
                <span className="empty-state-text">No papers found</span>
                <span className="empty-state-hint">Add PDF files to the research directory</span>
              </div>
            ) : (
              <div className="stack-sm">
                {papers.map((paper) => (
                  <button
                    key={paper.file}
                    className={`button secondary${activePaper?.file === paper.file ? "" : " ghost"}`}
                    onClick={() => setActivePaper(paper)}
                    style={{ justifyContent: "flex-start", textAlign: "left" }}
                  >
                    {paper.title}
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>
        <div className="panel">
          <div className="panel-header">
            <div className="panel-header-left">
              <h2 className="module-title">Paper Preview</h2>
              {activePaper && <span className="module-subtitle">{activePaper.title}</span>}
            </div>
            {activePaper && (
              <div className="panel-header-actions">
                {activePaper.source && (
                  <a
                    className="button secondary button-sm"
                    href={activePaper.source}
                    target="_blank"
                    rel="noreferrer"
                  >
                    Source
                  </a>
                )}
                <button className="button button-sm" onClick={() => openSystem(activePaper.file)}>
                  Open in System
                </button>
              </div>
            )}
          </div>
          <div className="panel-body" style={{ padding: 0 }}>
            {activePaper ? (
              <iframe className="viewer" src={fileUrl(activePaper.file)} style={{ border: "none" }} />
            ) : (
              <div className="panel-body">
                <div className="empty-state">
                  <span className="empty-state-text">No paper selected</span>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      <div className="grid-two">
        <div className="panel">
          <div className="panel-header">
            <h2 className="module-title">Notebooks</h2>
          </div>
          <div className="panel-body">
            {notebooks.length === 0 ? (
              <div className="empty-state">
                <span className="empty-state-text">No notebooks found</span>
                <span className="empty-state-hint">Add .ipynb files to the research directory</span>
              </div>
            ) : (
              <div className="stack-sm">
                {notebooks.map((notebook) => (
                  <button
                    key={notebook.file}
                    className={`button secondary${activeNotebook?.file === notebook.file ? "" : " ghost"}`}
                    onClick={() => setActiveNotebook(notebook)}
                    style={{ justifyContent: "flex-start", textAlign: "left" }}
                  >
                    {notebook.sport} — {notebook.project}
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>
        <div className="panel">
          <div className="panel-header">
            <div className="panel-header-left">
              <h2 className="module-title">Notebook Viewer</h2>
              {activeNotebook && (
                <span className="module-subtitle">
                  {activeNotebook.sport} — {activeNotebook.project}
                </span>
              )}
            </div>
            {activeNotebook && (
              <button className="button button-sm" onClick={() => openSystem(activeNotebook.file)}>
                Open in System
              </button>
            )}
          </div>
          <div className="panel-body">
            {activeNotebook ? (
              <NotebookViewer notebook={notebookData} />
            ) : (
              <div className="empty-state">
                <span className="empty-state-text">No notebook selected</span>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
