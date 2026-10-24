# Copilote de risque de crédit et lecture documentaire

[![CI](https://github.com/zakarya-22/projet_analyse_pdf/actions/workflows/ci.yml/badge.svg)](https://github.com/zakarya-22/projet_analyse_pdf/actions/workflows/ci.yml)

Prototype local : extraction de 15 indicateurs, score XGBoost, explications SHAP,
puis recherche documentaire dans un extrait du rapport BNP Paribas.
Le score ML n'est pas une décision réglementaire. Les réponses doivent être
vérifiées dans les passages cités ; aucune garantie de zéro hallucination.

Pour l'entretien : [fiche PDF d'une page](output/pdf/fiche_entretien.pdf)
et [version modifiable](FICHE_ENTRETIEN.md).
Pour les chiffres mesurés et leurs limites : [premier rapport](evaluation/RESULTS.md).
Pour comprendre les changements apportés : [résumé des nouveautés](README_NOUVEAUTES.md).

## Installation reproductible

Python 3.12, Windows ou Linux ; installer [uv](https://docs.astral.sh/uv/)
et [Ollama](https://ollama.com/), puis :

```bash
git clone https://github.com/zakarya-22/projet_analyse_pdf.git
cd projet_analyse_pdf
uv sync --locked --extra eval
ollama pull mistral
uv run --locked python -m scripts.build_vectordb
uv run --locked python -m streamlit run src/app.py
```

`pyproject.toml` décrit les dépendances ; `uv.lock` verrouille les versions et
les hashes. `requirements.txt` est un export généré pour les utilisateurs de pip :
`uv export --locked --no-dev --no-hashes --output-file requirements.txt`.
Les poids Hugging Face sont téléchargés au premier chargement. Ollama doit être
accessible sur localhost:11434. Le workflow ne télécharge pas de dossiers clients.

## Fonctionnement et limites

- Entrée client : texte collé, TXT ou Markdown. Le script `scripts/extract_doc.py`
  extrait séparément le PDF du corpus ; l'interface ne prend pas de PDF client.
- Validation commune au JSON et au CSV : noms normalisés, doublons refusés,
  15 valeurs numériques finies requises ; absence ou valeur invalide bloquante.
  Les zéros réellement fournis sont conservés. Le notebook fourni utilise les
  valeurs du CSV Kaggle directement, sans scaler. La conversion d'un bilan brut
  vers cette représentation reste inconnue : confirmation explicite dans l'interface,
  aucune plage ni conversion inventée. Voir `models/input_contract.json`.
- SHAP : facteurs aggravants et protecteurs distincts, transmis au RAG avec leur sens.
  Les contributions ne sont ni causales ni des points de probabilité.
- RAG : chunks de 1 000 caractères avec chevauchement de 200, CamemBERT + Chroma,
  BM25 (3 candidats) et vectoriel (4 candidats), fusion RRF pondérée 30/70,
  reranking MiniLM, puis 2 passages pour Mistral.
- Les deux recherches utilisent les mêmes chunks et métadonnées de Chroma.
  Les sources affichent page PDF, section si disponible et extrait exact.
  Les pages PDF diffèrent parfois de la pagination imprimée.
- L'abstention est explicite si le contexte ne permet pas de répondre. C'est une
  instruction au modèle, dont le respect est mesuré, pas une garantie.
- Le corpus extrait présente des coupures de mots et des tableaux dégradés.
  Le prototype n'est pas une certification de conformité bancaire ou RGPD.

### Environnement créé pendant cette intervention

L'environnement isolé `.venv-ci` contient les dépendances verrouillées, sans
remplacer l'environnement initial. Pour l'utiliser sous PowerShell :

```powershell
.\.venv-ci\Scripts\python.exe -m streamlit run src/app.py
```

## Tests et CI

```bash
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pytest --junitxml=reports/tests.xml --cov=src --cov-report=xml:reports/coverage.xml
```

Le workflow `.github/workflows/ci.yml` exécute ces étapes sur push et pull request,
avec installation verrouillée et publication des rapports XML. Les tests utilisent
des doublures pour les modèles RAG : aucun Ollama, téléchargement de poids ou
appel payant n'est nécessaire. Ils vérifient les contrats et le comportement du
code, pas la précision statistique du modèle XGBoost. Aucune CD.
Consulter les exécutions et leurs rapports dans [GitHub Actions](https://github.com/zakarya-22/projet_analyse_pdf/actions/workflows/ci.yml).
La CI standard s'exécute sur un runner Ubuntu hébergé par GitHub.

## Évaluation RAG

Voir `evaluation/README.md` pour le protocole, les limites et les commandes.
Le workflow manuel `.github/workflows/rag-evaluation.yml` nécessite un runner
**self-hosted** portant le label `rag-eval`, avec Bash, Ollama, Mistral et les poids
Hugging Face. Il n'est jamais déclenché par une PR et ne déploie rien.
Sans runner configuré, exécuter l'évaluation localement.

### Premiers résultats mesurés

40 questions de développement, dont 32 répondables ciblant les pages PDF 426–429
et 8 sans réponse ; corpus total de 102 pages et 557 passages.

| Recherche | Page Hit@2 (32 questions répondables) | Temps moyen local |
|---|---:|---:|
| Vectorielle | 62,50 % (20/32) | 0,247 s |
| Hybride BM25 + vectorielle | 68,75 % (22/32) | 0,230 s |
| Hybride + reranking | 96,875 % (31/32) | 2,585 s |

Page Hit@2 mesure la présence de la page attendue parmi les deux passages,
pas la précision des réponses. Jeu exploratoire ciblé, sans validation d'un expert
humain ; les temps proviennent d'un seul passage sur CPU.
Le pilote Ragas porte sur 2 questions × 3 variantes : 6 réponses générées,
5 mesures du juge réussies sur 9 tentées. Faithfulness reste indisponible à cause
de dépassements de délai. Voir le [rapport et les résultats bruts](evaluation/RESULTS.md).

L'index avec provenance est écrit dans `data/chroma_v2/` ; l'ancien
`data/chroma_db/` est conservé.

Le module historique `audit_reporter.py` n'est pas branché à l'interface.
