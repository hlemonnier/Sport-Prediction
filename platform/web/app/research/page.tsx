import { getNotebooks, getPapers } from "@/lib/api";
import type { NotebookEntry, PaperEntry } from "@/lib/types";
import ResearchExplorer from "@/components/ResearchExplorer";

async function fetchPapers(): Promise<PaperEntry[]> {
  try {
    return await getPapers();
  } catch {
    return [];
  }
}

async function fetchNotebooks(): Promise<NotebookEntry[]> {
  try {
    return await getNotebooks();
  } catch {
    return [];
  }
}

export default async function ResearchPage() {
  const [papers, notebooks] = await Promise.all([fetchPapers(), fetchNotebooks()]);
  return <ResearchExplorer papers={papers} notebooks={notebooks} />;
}
