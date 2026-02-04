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
    return <div className="viewer" />;
  }

  return (
    <div className="viewer" style={{ padding: 20, overflow: "auto" }}>
      <div className="stack">
        {notebook.cells.map((cell, index) => {
          const content = normalizeSource(cell.source);
          if (cell.cell_type === "markdown") {
            return (
              <div
                key={`cell-${index}`}
                dangerouslySetInnerHTML={{ __html: marked.parse(content) }}
              />
            );
          }
          if (cell.cell_type === "code") {
            return (
              <pre
                key={`cell-${index}`}
                style={{
                  background: "#1f1b16",
                  color: "#f5f2ee",
                  padding: 16,
                  borderRadius: 12,
                  overflowX: "auto",
                }}
              >
                <code>{content}</code>
              </pre>
            );
          }
          return (
            <pre key={`cell-${index}`}>
              <code>{content}</code>
            </pre>
          );
        })}
      </div>
    </div>
  );
}
