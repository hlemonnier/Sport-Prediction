**Overview**
Sport Prediction regroupe des projets de prediction par sport. Chaque projet suit la meme structure et le meme flux de travail :
1. Recherche dans un notebook.
2. Stabilisation du modele.
3. Passage en code Python plus "prod".

**Structure commune**
- `Jupyter/` contient la recherche (exploration, features, tests d'hypotheses, evaluation).
- `Python/` contient le code "prod" (package, CLI, pipeline de prediction).
- Chaque projet a son propre `README.md` dans `Python/` avec les commandes de lancement.
- `research/papers/` centralise des papiers de recherche (PDF) par sport.

**Projects**
F1 — Rising Qualification Prediction. Objectif : predire les performances en qualification et en course.
- Recherche : `research/projects/F1/Rising Qualification Prediction/Jupyter/model-research.ipynb`
- Code prod : `research/projects/F1/Rising Qualification Prediction/Python/`
- Commandes : voir `research/projects/F1/Rising Qualification Prediction/Python/README.md`

Football — Match Result Prediction. Objectif : predire les resultats de match (1X2, scoreline).
- Recherche : `research/projects/Football/Match Result Prediction/Jupyter/model-research.ipynb`
- Code prod : `research/projects/Football/Match Result Prediction/Python/`
- Statut : scaffold (CLI + package + placeholders), a completer apres validation du notebook

**Plateforme locale (UI + API)**
- Backend Rust: `platform/backend` (Axum + SQLite)
- Frontend Next.js: `platform/web`
- Lancer le backend: `cargo run` dans `platform/backend`
- Lancer le frontend: `pnpm dev` dans `platform/web`

**CLI**
- `research/sport_cli.py`

**Data**
- `data/football/` contient `teams`, `matches`, `fixtures` (CSV/Parquet)

**Ajouter un nouveau sport**
- Creer un dossier racine par sport.
- Ajouter `Jupyter/model-research.ipynb` pour la recherche.
- Ajouter `Python/` avec un `run_prediction.py` et un package minimal.
- Ajouter des papiers dans `research/papers/<Sport>/`.
