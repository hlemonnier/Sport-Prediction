**Overview**
Sport Prediction regroupe des projets de prediction par sport. Chaque projet suit la meme structure et le meme flux de travail :
1. Recherche dans un notebook.
2. Stabilisation du modele.
3. Passage en code Python plus "prod".

**Structure commune**
- `Jupyter/` contient la recherche (exploration, features, tests d'hypotheses, evaluation).
- `Python/` contient le code "prod" (package, CLI, pipeline de prediction).
- Chaque projet a son propre `README.md` dans `Python/` avec les commandes de lancement.

**Projects**
F1 — Rising Qualification Prediction. Objectif : predire les performances en qualification et en course.
- Recherche : `F1/Rising Qualification Prediction/Jupyter/model-research.ipynb`
- Code prod : `F1/Rising Qualification Prediction/Python/`
- Commandes : voir `F1/Rising Qualification Prediction/Python/README.md`

Football — Match Result Prediction. Objectif : predire les resultats de match (1X2, scoreline).
- Recherche : `Football/Match Result Prediction/Jupyter/model-research.ipynb`
- Code prod : `Football/Match Result Prediction/Python/`
- Statut : scaffold (CLI + package + placeholders), a completer apres validation du notebook

**Ajouter un nouveau sport**
- Creer un dossier racine par sport.
- Ajouter `Jupyter/model-research.ipynb` pour la recherche.
- Ajouter `Python/` avec un `run_prediction.py` et un package minimal.
