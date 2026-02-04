#!/usr/bin/env python3
"""CLI pour naviguer dans le pipeline de recherche ML/RL par sport."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent
PROJECTS_DIR = ROOT / "projects"
PAPERS_DIR = ROOT / "papers"

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    RICH = True
    CONSOLE = Console()
except Exception:
    RICH = False
    CONSOLE = None


@dataclass(frozen=True)
class Project:
    sport: str
    name: str
    path: Path
    notebook: Path | None
    python_dir: Path | None
    python_readme: Path | None


@dataclass(frozen=True)
class Paper:
    sport: str
    path: Path
    title: str
    source: str | None


def is_sport_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if path.name.startswith("."):
        return False
    if path.name in {"Research", "research", "papers", "projects", ".git"}:
        return False
    for child in path.iterdir():
        if child.is_dir() and not child.name.startswith("."):
            if (child / "Jupyter").exists() or (child / "Python").exists():
                return True
    return False


def iter_sport_dirs() -> list[Path]:
    base_dir = PROJECTS_DIR if PROJECTS_DIR.exists() else ROOT
    return sorted([path for path in base_dir.iterdir() if is_sport_dir(path)], key=lambda p: p.name)


def load_projects() -> list[Project]:
    projects: list[Project] = []
    for sport_dir in iter_sport_dirs():
        for project_dir in sorted(
            [p for p in sport_dir.iterdir() if p.is_dir() and not p.name.startswith(".")],
            key=lambda p: p.name,
        ):
            notebook = project_dir / "Jupyter" / "model-research.ipynb"
            python_dir = project_dir / "Python"
            if not notebook.exists() and not python_dir.exists():
                continue
            python_readme = python_dir / "README.md" if python_dir.exists() else None
            projects.append(
                Project(
                    sport=sport_dir.name,
                    name=project_dir.name,
                    path=project_dir,
                    notebook=notebook if notebook.exists() else None,
                    python_dir=python_dir if python_dir.exists() else None,
                    python_readme=python_readme if python_readme and python_readme.exists() else None,
                )
            )
    return projects


def parse_research_index() -> dict[str, dict[str, str]]:
    index: dict[str, dict[str, str]] = {}
    readme = PAPERS_DIR / "README.md"
    if not readme.exists():
        return index
    current_sport: str | None = None
    current_title: str | None = None
    current_file: str | None = None

    def extract_backtick(text: str) -> str:
        if "`" in text:
            parts = text.split("`")
            if len(parts) >= 2:
                return parts[1].strip()
        return text.split(":", 1)[1].strip()

    for raw_line in readme.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("**") and line.endswith("**") and line.count("**") >= 2:
            label = line.strip("*").strip()
            if label.lower() != "research papers":
                current_sport = label
            continue
        if line.startswith("- ") and not line.startswith("- File:") and not line.startswith("- Source:"):
            current_title = line[2:].strip()
            current_file = None
            continue
        if line.startswith("- File:"):
            file_path = extract_backtick(line)
            current_file = Path(file_path).name
            index.setdefault(current_file, {})
            if current_title:
                index[current_file]["title"] = current_title
            if current_sport:
                index[current_file]["sport"] = current_sport
            continue
        if line.startswith("- Source:"):
            if current_file is None:
                continue
            source = line.split(":", 1)[1].strip()
            index.setdefault(current_file, {})
            index[current_file]["source"] = source
    return index


def load_papers(sport_filter: str | None = None) -> list[Paper]:
    index = parse_research_index()
    papers: list[Paper] = []
    if not PAPERS_DIR.exists():
        return papers

    sport_dirs: Iterable[Path]
    if sport_filter:
        sport_dirs = [PAPERS_DIR / sport_filter]
    else:
        sport_dirs = [p for p in PAPERS_DIR.iterdir() if p.is_dir() and not p.name.startswith(".")]

    for sport_dir in sorted(sport_dirs, key=lambda p: p.name):
        if not sport_dir.exists():
            continue
        for pdf in sorted(sport_dir.glob("*.pdf"), key=lambda p: p.name):
            meta = index.get(pdf.name, {})
            title = meta.get("title") or pdf.stem.replace("-", " ")
            source = meta.get("source")
            papers.append(Paper(sport=sport_dir.name, path=pdf, title=title, source=source))
    return papers


def normalize(text: str) -> str:
    return text.lower().replace("_", " ").strip()


def match_one(items: list[str], query: str) -> list[str]:
    q = normalize(query)
    return [item for item in items if q in normalize(item)]


def render_overview(projects: list[Project], papers: list[Paper]) -> None:
    counts: dict[str, dict[str, int]] = {}
    for project in projects:
        counts.setdefault(project.sport, {"projects": 0, "notebooks": 0, "python": 0})
        counts[project.sport]["projects"] += 1
        if project.notebook:
            counts[project.sport]["notebooks"] += 1
        if project.python_dir:
            counts[project.sport]["python"] += 1
    paper_counts: dict[str, int] = {}
    for paper in papers:
        paper_counts[paper.sport] = paper_counts.get(paper.sport, 0) + 1

    if RICH and CONSOLE:
        table = Table(title="Pipeline Recherche ML/RL", header_style="bold")
        table.add_column("Sport", style="bold")
        table.add_column("Projects")
        table.add_column("Notebooks")
        table.add_column("Code Python")
        table.add_column("Papers")
        for sport in sorted(counts.keys() | paper_counts.keys()):
            stats = counts.get(sport, {"projects": 0, "notebooks": 0, "python": 0})
            table.add_row(
                sport,
                str(stats["projects"]),
                str(stats["notebooks"]),
                str(stats["python"]),
                str(paper_counts.get(sport, 0)),
            )
        CONSOLE.print(table)
    else:
        print("Pipeline Recherche ML/RL")
        for sport in sorted(counts.keys() | paper_counts.keys()):
            stats = counts.get(sport, {"projects": 0, "notebooks": 0, "python": 0})
            print(
                f"- {sport}: projects={stats['projects']} notebooks={stats['notebooks']} "
                f"python={stats['python']} papers={paper_counts.get(sport, 0)}"
            )


def render_projects(projects: list[Project], sport_filter: str | None = None) -> None:
    items = [p for p in projects if not sport_filter or normalize(p.sport) == normalize(sport_filter)]
    if not items:
        print("Aucun projet trouve.")
        return

    if RICH and CONSOLE:
        table = Table(title="Projects", header_style="bold")
        table.add_column("Sport", style="bold")
        table.add_column("Project")
        table.add_column("Notebook")
        table.add_column("Python")
        for project in items:
            table.add_row(
                project.sport,
                project.name,
                "oui" if project.notebook else "non",
                "oui" if project.python_dir else "non",
            )
        CONSOLE.print(table)
    else:
        print("Projects")
        for project in items:
            print(
                f"- {project.sport} | {project.name} | notebook={bool(project.notebook)} | python={bool(project.python_dir)}"
            )


def render_project_detail(project: Project) -> None:
    quick_start = extract_quick_start(project.python_readme) if project.python_readme else None

    lines = [
        f"Sport: {project.sport}",
        f"Project: {project.name}",
        f"Notebook: {project.notebook if project.notebook else 'indisponible'}",
        f"Python: {project.python_dir if project.python_dir else 'indisponible'}",
        f"Python README: {project.python_readme if project.python_readme else 'indisponible'}",
    ]
    if quick_start:
        lines.append(f"Quick start: {quick_start}")

    if RICH and CONSOLE:
        CONSOLE.print(Panel("\n".join(lines), title="Project Detail", title_align="left"))
    else:
        print("\n".join(lines))


def extract_quick_start(readme_path: Path) -> str | None:
    try:
        text = readme_path.read_text(encoding="utf-8")
    except Exception:
        return None
    in_code = False
    commands: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("```"):
            in_code = not in_code
            continue
        if in_code and line:
            commands.append(line)
    if commands:
        return " ".join(commands)
    return None


def render_papers(papers: list[Paper], limit: int | None = None) -> None:
    if not papers:
        print("Aucun papier trouve.")
        return
    items = papers[:limit] if limit else papers

    if RICH and CONSOLE:
        table = Table(title="Research Papers", header_style="bold")
        table.add_column("#", style="bold")
        table.add_column("Sport", style="bold")
        table.add_column("Title")
        table.add_column("Source")
        for idx, paper in enumerate(items, start=1):
            table.add_row(str(idx), paper.sport, paper.title, paper.source or "-")
        CONSOLE.print(table)
    else:
        print("Research Papers")
        for idx, paper in enumerate(items, start=1):
            source = paper.source or "-"
            print(f"{idx}. {paper.sport} | {paper.title} | {source}")


def open_with_system(target: str) -> None:
    if sys.platform == "darwin":
        command = ["open", target]
    elif os.name == "nt":
        command = ["cmd", "/c", "start", "", target]
    else:
        command = ["xdg-open", target]
    subprocess.run(command, check=False)


def select_sport(sports: list[str]) -> str | None:
    if not sports:
        return None
    print("Choisissez un sport:")
    for idx, sport in enumerate(sports, start=1):
        print(f"{idx}. {sport}")
    choice = input("Numero (ou q): ").strip()
    if choice.lower() == "q":
        return None
    if not choice.isdigit():
        return None
    index = int(choice)
    if 1 <= index <= len(sports):
        return sports[index - 1]
    return None


def select_project(projects: list[Project]) -> Project | None:
    if not projects:
        return None
    print("Choisissez un project:")
    for idx, project in enumerate(projects, start=1):
        print(f"{idx}. {project.name}")
    choice = input("Numero (ou q): ").strip()
    if choice.lower() == "q":
        return None
    if not choice.isdigit():
        return None
    index = int(choice)
    if 1 <= index <= len(projects):
        return projects[index - 1]
    return None


def select_paper(papers: list[Paper]) -> Paper | None:
    if not papers:
        return None
    print("Choisissez un papier:")
    for idx, paper in enumerate(papers, start=1):
        print(f"{idx}. {paper.title}")
    choice = input("Numero (ou q): ").strip()
    if choice.lower() == "q":
        return None
    if not choice.isdigit():
        return None
    index = int(choice)
    if 1 <= index <= len(papers):
        return papers[index - 1]
    return None


def run_menu(projects: list[Project], papers: list[Paper]) -> None:
    while True:
        print("\nMenu pipeline research")
        print("1. Overview")
        print("2. Liste des projects")
        print("3. Liste des papers")
        print("4. Ouvrir un paper")
        print("5. Ouvrir un notebook")
        print("6. Ouvrir README Python")
        print("7. Search")
        print("8. Quit")

        choice = input("Choix: ").strip()
        if choice == "1":
            render_overview(projects, papers)
        elif choice == "2":
            render_projects(projects)
        elif choice == "3":
            render_papers(papers)
        elif choice == "4":
            sports = sorted({p.sport for p in papers})
            sport = select_sport(sports)
            if not sport:
                continue
            sport_papers = [p for p in papers if p.sport == sport]
            paper = select_paper(sport_papers)
            if paper:
                open_with_system(str(paper.path))
        elif choice == "5":
            sports = sorted({p.sport for p in projects})
            sport = select_sport(sports)
            if not sport:
                continue
            sport_projects = [p for p in projects if p.sport == sport and p.notebook]
            project = select_project(sport_projects)
            if project and project.notebook:
                open_with_system(str(project.notebook))
        elif choice == "6":
            sports = sorted({p.sport for p in projects})
            sport = select_sport(sports)
            if not sport:
                continue
            sport_projects = [p for p in projects if p.sport == sport and p.python_readme]
            project = select_project(sport_projects)
            if project and project.python_readme:
                open_with_system(str(project.python_readme))
        elif choice == "7":
            query = input("Mot-cle: ").strip()
            if not query:
                continue
            run_search(projects, papers, query, scope="all")
        elif choice == "8":
            break
        else:
            print("Choix invalide.")


def run_search(
    projects: list[Project],
    papers: list[Paper],
    query: str,
    scope: str = "all",
    sport_filter: str | None = None,
) -> None:
    q = normalize(query)
    if scope in {"projects", "all"}:
        matched_projects = [
            p
            for p in projects
            if (not sport_filter or normalize(p.sport) == normalize(sport_filter))
            and q in normalize(p.name)
        ]
        if matched_projects:
            render_projects(matched_projects)
    if scope in {"papers", "all"}:
        matched_papers = [
            p
            for p in papers
            if (not sport_filter or normalize(p.sport) == normalize(sport_filter))
            and (q in normalize(p.title) or q in normalize(p.path.name))
        ]
        if matched_papers:
            render_papers(matched_papers)


def resolve_project(projects: list[Project], sport: str | None, name: str) -> Project | None:
    if sport:
        filtered = [p for p in projects if normalize(p.sport) == normalize(sport)]
    else:
        filtered = projects
    matches = [p for p in filtered if normalize(name) in normalize(p.name)]
    if len(matches) == 1:
        return matches[0]
    return None


def resolve_paper(papers: list[Paper], sport: str | None, index: int | None, name: str | None) -> Paper | None:
    filtered = [p for p in papers if not sport or normalize(p.sport) == normalize(sport)]
    if index is not None:
        if 1 <= index <= len(filtered):
            return filtered[index - 1]
        return None
    if name:
        matches = [
            p
            for p in filtered
            if normalize(name) in normalize(p.title) or normalize(name) in normalize(p.path.name)
        ]
        if len(matches) == 1:
            return matches[0]
        return None
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sport-cli",
        description="Navigation du pipeline de recherche ML/RL (sport prediction)",
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("overview", help="Vue globale des sports/projects/papers")

    projects_parser = subparsers.add_parser("projects", help="Lister les projects")
    projects_parser.add_argument("--sport", default=None)

    project_parser = subparsers.add_parser("project", help="Details d'un project")
    project_parser.add_argument("--sport", default=None)
    project_parser.add_argument("--name", required=True)

    papers_parser = subparsers.add_parser("papers", help="Lister les papers")
    papers_parser.add_argument("--sport", default=None)
    papers_parser.add_argument("--limit", type=int, default=None)

    search_parser = subparsers.add_parser("search", help="Recherche dans papers/projects")
    search_parser.add_argument("query")
    search_parser.add_argument("--scope", choices=["papers", "projects", "all"], default="all")
    search_parser.add_argument("--sport", default=None)

    open_parser = subparsers.add_parser("open", help="Ouvrir un fichier (paper, notebook, readme)")
    open_parser.add_argument(
        "type",
        choices=[
            "paper",
            "notebook",
            "python",
            "project-readme",
            "research-readme",
            "root-readme",
            "source",
            "path",
        ],
    )
    open_parser.add_argument("--sport", default=None)
    open_parser.add_argument("--project", default=None)
    open_parser.add_argument("--index", type=int, default=None)
    open_parser.add_argument("--name", default=None)
    open_parser.add_argument("--path", default=None)
    open_parser.add_argument("--dry-run", action="store_true")

    subparsers.add_parser("menu", help="Mode interactif")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    projects = load_projects()
    papers = load_papers()

    if args.command is None:
        if sys.stdin.isatty() and sys.stdout.isatty():
            run_menu(projects, papers)
            return
        parser.print_help()
        return

    if args.command == "overview":
        render_overview(projects, papers)
        return

    if args.command == "projects":
        render_projects(projects, sport_filter=args.sport)
        return

    if args.command == "project":
        project = resolve_project(projects, args.sport, args.name)
        if not project:
            print("Project introuvable ou ambigu. Utilisez --sport et un nom plus precis.")
            return
        render_project_detail(project)
        return

    if args.command == "papers":
        sport = args.sport
        render_papers(load_papers(sport_filter=sport), limit=args.limit)
        return

    if args.command == "search":
        run_search(projects, papers, args.query, scope=args.scope, sport_filter=args.sport)
        return

    if args.command == "menu":
        run_menu(projects, papers)
        return

    if args.command == "open":
        if args.type == "research-readme":
            target = PAPERS_DIR / "README.md"
            if not target.exists():
                print("README de recherche introuvable.")
                return
            if args.dry_run:
                print(target)
                return
            open_with_system(str(target))
            return
        if args.type == "root-readme":
            target = ROOT / "README.md"
            if not target.exists():
                print("README racine introuvable.")
                return
            if args.dry_run:
                print(target)
                return
            open_with_system(str(target))
            return
        if args.type == "path":
            if not args.path:
                print("--path requis pour open path")
                return
            target = Path(args.path)
            if args.dry_run:
                print(target)
                return
            open_with_system(str(target))
            return
        if args.type == "paper":
            paper = resolve_paper(load_papers(args.sport), args.sport, args.index, args.name)
            if not paper:
                print("Paper introuvable ou ambigu. Utilisez --sport et --index/--name.")
                return
            if args.dry_run:
                print(paper.path)
                return
            open_with_system(str(paper.path))
            return
        if args.type == "source":
            paper = resolve_paper(load_papers(args.sport), args.sport, args.index, args.name)
            if not paper:
                print("Paper introuvable ou ambigu. Utilisez --sport et --index/--name.")
                return
            if not paper.source:
                print("Source indisponible pour ce paper.")
                return
            if args.dry_run:
                print(paper.source)
                return
            open_with_system(paper.source)
            return
        if args.type in {"notebook", "python", "project-readme"}:
            if not args.project:
                print("--project requis pour ce type")
                return
            project = resolve_project(projects, args.sport, args.project)
            if not project:
                print("Project introuvable ou ambigu. Utilisez --sport et un nom plus precis.")
                return
            if args.type == "notebook":
                target = project.notebook
            elif args.type == "python":
                target = project.python_dir
            else:
                target = project.python_readme
            if not target:
                print("Ressource indisponible pour ce project.")
                return
            if args.dry_run:
                print(target)
                return
            open_with_system(str(target))
            return


if __name__ == "__main__":
    main()
