# Ce qui a été ajouté ou amélioré dans ton projet

Ton projet possédait déjà **Streamlit, XGBoost, SHAP et un RAG hybride avec
reranking**. Le travail réalisé a surtout rendu ce prototype plus fiable,
mesurable et facile à vérifier sur GitHub.

## 1. Contrôle des données avant le scoring

- Validation commune des **15 indicateurs** extraits du texte ou lus dans un CSV.
- Refus des valeurs manquantes, doublons, textes à la place de nombres, NaN et infinis.
- Une donnée absente ne devient plus automatiquement zéro.
- Contrat d'entrée documenté à partir de ton notebook : pas de scaler identifié,
  et conversion des unités comptables brutes encore à préciser.

**Utilité :** éviter de produire un score à partir de données incomplètes ou mal interprétées.

## 2. Explications SHAP plus claires

Les facteurs qui **augmentent** le risque et ceux qui le **diminuent** sont
maintenant distingués dans l'interface et transmis avec leur sens au RAG.

**Utilité :** expliquer pourquoi le modèle donne son score, sans présenter tous
les indicateurs importants comme des facteurs aggravants.

## 3. Sources et recherche documentaire harmonisées

- Index reconstruit avec provenance : **102 pages, 557 passages**.
- BM25 et recherche vectorielle utilisent les mêmes passages et métadonnées.
- Réponses avec identifiants de citation, pages et extraits consultables.
- Consigne explicite d'abstention lorsque les passages ne permettent pas de répondre.
- Même code de recherche utilisé par l'application et le benchmark.

**Utilité :** pouvoir retrouver et vérifier les éléments utilisés dans la réponse.

## 4. Une vraie première évaluation du RAG

Création d'un jeu de **40 questions** : 32 répondables et 8 sans réponse.
Comparaison de trois configurations déjà disponibles dans la chaîne de recherche :

| Configuration | Page Hit@2 | Temps moyen local |
|---|---:|---:|
| Vectorielle seule | 62,50 % (20/32) | 0,247 s |
| Hybride | 68,75 % (22/32) | 0,230 s |
| Hybride + reranking | 96,875 % (31/32) | 2,585 s |

**Lecture :** le reranking retrouve davantage les pages attendues, avec une
latence supérieure. Ce score mesure la recherche de pages sur un jeu ciblant
quatre pages du corpus ; il ne représente pas la précision globale des réponses.

Intégration de **Ragas** et exécution d'un pilote de 2 questions × 3 configurations.
Les 6 réponses ont été générées, mais seulement 5 mesures du juge sur 9 ont abouti.
Faithfulness reste indisponible à cause de dépassements de délai.
Résultats bruts, erreurs et paramètres sont conservés pour vérifier les chiffres.

**Utilité :** comparer les choix techniques avec des mesures, en gardant visibles
les limites. Voir le [rapport détaillé](evaluation/RESULTS.md).

## 5. Tests, dépendances et pipelines CI sur GitHub

- **23 tests automatisés** couvrent notamment les entrées, le sens SHAP,
  les sources, la recherche simulée et le formulaire Streamlit.
- Versions des dépendances verrouillées avec **uv.lock**.
- Pipeline **GitHub Actions** à chaque push et pull request : installation,
  imports Ragas, lint Ruff, formatage et tests avec rapports de couverture.
- Première exécution sur GitHub **réussie** : [voir le run](https://github.com/zakarya-22/projet_analyse_pdf/actions/runs/40099893632).
- Pipeline d'évaluation RAG manuel publié : il nécessite une machine configurée
  comme runner avec Ollama et les modèles. Aucune CD ajoutée.

**Utilité :** détecter automatiquement les régressions et rendre l'installation reproductible.

## 6. Documentation et préparation à l'entretien

README principal mis à jour, résultats d'évaluation documentés et
[fiche d'entretien d'une page](output/pdf/fiche_entretien.pdf) créée.
Le dépôt était déjà connecté à GitHub ; les améliorations ont été publiées sur `main`.

## Ce qu'il reste à améliorer

Clarifier les unités des indicateurs, mieux extraire les tableaux, élargir le jeu
de test à tout le corpus avec validation humaine, stabiliser le juge Ragas et
améliorer le respect exact de l'abstention. Le modèle XGBoost n'a pas été réentraîné
ni réévalué pendant cette intervention.
