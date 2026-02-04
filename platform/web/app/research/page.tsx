import { API_BASE } from "@/lib/api";
import type { NotebookEntry, PaperEntry } from "@/lib/types";
import ResearchExplorer from "@/components/ResearchExplorer";

async function fetchPapers(): Promise<PaperEntry[]> {
  try {
    const res = await fetch(`${API_BASE}/api/papers`, { cache: "no-store" });
    if (!res.ok) {
      return [];
    }
    return (await res.json()) as PaperEntry[];
  } catch {
    return [];
  }
}

async function fetchNotebooks(): Promise<NotebookEntry[]> {
  try {
    const res = await fetch(`${API_BASE}/api/notebooks`, { cache: "no-store" });
    if (!res.ok) {
      return [];
    }
    return (await res.json()) as NotebookEntry[];
  } catch {
    return [];
  }
}

export default async function ResearchPage() {
  const [papers, notebooks] = await Promise.all([fetchPapers(), fetchNotebooks()]);
  return <ResearchExplorer papers={papers} notebooks={notebooks} />;
}
