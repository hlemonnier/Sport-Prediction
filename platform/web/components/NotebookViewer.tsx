"use client";

import { marked } from "marked";

type NotebookCell = {
  cell_type: "markdown" | "code" | string;
  source: string[] | string;
};

type Notebook = {
  cells?: NotebookCell[];
};

function normalizeSource(source: string[] | string | undefined): string {
  if (!source) {
    return "";
  }
  if (Array.isArray(source)) {
    return source.join("");
  }
  return source;
}

export default function NotebookViewer({ notebook }: { notebook: Notebook | null }) {
  if (!notebook || !notebook.cells) {
    return (
      <div className="empty-state">
        <span className="empty-state-text">Loading notebook...</span>
      </div>
    );
  }

  return (
    <div className="stack-sm">
      {notebook.cells.map((cell, index) => {
        const content = normalizeSource(cell.source);
        if (cell.cell_type === "markdown") {
          return (
            <div
              key={`cell-${index}`}
              style={{ fontSize: 13, lineHeight: 1.6, color: "var(--ink-2)" }}
              dangerouslySetInnerHTML={{ __html: marked.parse(content) }}
            />
          );
        }
        if (cell.cell_type === "code") {
          return (
            <pre
              key={`cell-${index}`}
              style={{
                background: "var(--input-bg)",
                color: "var(--ink)",
                padding: 12,
                borderRadius: 4,
                border: "1px solid var(--border)",
                overflowX: "auto",
                fontSize: 12,
                fontFamily: "var(--font-mono), JetBrains Mono, monospace",
                lineHeight: 1.5,
              }}
            >
              <code>{content}</code>
            </pre>
          );
        }
        return (
          <pre
            key={`cell-${index}`}
            style={{
              fontSize: 12,
              fontFamily: "var(--font-mono), JetBrains Mono, monospace",
              color: "var(--muted)",
            }}
          >
            <code>{content}</code>
          </pre>
        );
      })}
    </div>
  );
}
