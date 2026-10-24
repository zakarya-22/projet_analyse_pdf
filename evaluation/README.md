# Protocole d'évaluation RAG

Le [premier rapport mesuré](RESULTS.md) distingue la comparaison de recherche
sur 40 questions du pilote Ragas sur 2 questions × 3 configurations.

## Jeu de développement

`questions.jsonl` : **40 questions**, dont **32 avec réponse** et **8 sans réponse**.
Les réponses de référence ont été rédigées à partir du corpus, indépendamment des
sorties du RAG. Chaque question répondable comporte une page PDF, une ancre exacte
et un extrait. Contrôle des ancres automatisé et lecture par l'assistant ; aucune
validation par un expert métier humain n'est revendiquée.

Ce premier jeu cible les procédures de crédit des pages PDF 426–429 : il ne
représente ni toutes les 102 pages du corpus, ni la performance en production.
Les questions sans réponse couvrent des seuils inventés, des profils fictifs et
des demandes hors corpus. Ne pas ajuster le système sur ce jeu puis le présenter
comme un test indépendant ; constituer ensuite un jeu de test tenu à l'écart.

## Comparaison contrôlée

Trois variantes, même corpus, mêmes questions, mêmes embeddings et même LLM :

1. Vectoriel seul : 4 candidats, 2 premiers passages.
2. Hybride : BM25 3 candidats + vectoriel 4, fusion RRF (k=60, poids 0,3/0,7), 2 passages.
3. Hybride + reranking : même ensemble de candidats, Cross-Encoder, 2 passages.

Paramètres de génération : Mistral, température 0, seed 42. Ce réglage réduit la
variabilité sans garantir le déterminisme bit à bit. Un échauffement de recherche
est exclu des temps ; le premier chargement Ollama peut encore affecter la génération.
La latence locale n'est pas un benchmark matériel universel.
Les recherches sont calculées avant la génération ; les modèles de recherche sont
ensuite libérés pour réduire la pression mémoire. Les passages sont sauvegardés
dans `retrieval.jsonl` même si le chargement du juge échoue.

## Commandes

Depuis la racine du projet, Ollama actif :

```bash
uv sync --locked --extra eval
uv run --locked python -m scripts.build_vectordb --db data/eval_chroma
uv run --locked --extra eval python -m scripts.evaluate_rag --db data/eval_chroma --output reports/rag-full
```

Contrôle rapide de l'intégration Ragas et de l'abstention :

```bash
uv run --locked --extra eval python -m scripts.evaluate_rag --db data/eval_chroma --ids q12 q33 --output reports/rag-smoke
```

Recherche seule (aucune génération ni jugement Ragas) :

```bash
uv run --locked --extra eval python -m scripts.evaluate_rag --db data/eval_chroma --retrieval-only --output reports/retrieval
```

Ajouter `--skip-ragas` pour mesurer génération/abstention sans le juge.
`--judge-model` permet de choisir un autre modèle **déjà installé dans Ollama**.
Le défaut utilise Mistral comme générateur et juge : risque de biais corrélé,
résultats exploratoires à contrôler humainement. Aucun envoi vers une API cloud.

## Mesures et lecture correcte

- **Context Precision / Context Recall (Ragas)** : pertinence des passages et
  couverture des affirmations de référence, sur les 32 questions répondables.
- **Faithfulness (Ragas)** : soutien des affirmations par les passages ; calculé
  sur les réponses non abstention aux questions répondables. Lire cette moyenne
  avec le taux de fausse abstention, sinon elle peut être trompeuse.
- **Page Hit@2** : au moins un des deux passages vient de la page annotée.
  Indicateur grossier de localisation, différent de Context Recall et ne prouvant
  pas que l'extrait contient la réponse.
- **Abstention correcte** : réponse égale à la phrase d'abstention attendue sur
  les 8 questions sans réponse. Mesure stricte, sensible aux reformulations.
- **Fausse abstention** : abstention sur une question répondable.
- **Citation ID validity** : présence et validité des identifiants [S1]/[S2] ;
  ne mesure PAS si chaque affirmation est réellement justifiée.
- Temps de recherche et génération en secondes.

Ragas 0.4.3 est verrouillé ; l'adaptateur LangChain historique est conservé pour
ChatOllama et reste déprécié. LangChain Community 0.3.31 est verrouillé pour
conserver les imports requis par cette version de Ragas. Les erreurs de parsing/timeout du juge sont explicites,
les scores manquants restent null, chaque moyenne indique `n_valid`.
Un run avec erreurs termine en échec, tout en conservant ses résultats partiels.

## Traçabilité et reprise

Chaque dossier contient : `manifest.json` (hashes du corpus, dataset, index,
code et lockfile, versions, révisions Hugging Face et digests Ollama), `results.jsonl` (réponses, passages,
mesures et erreurs) et `summary.json` (moyennes et effectifs valides).
Relancer la même commande reprend les paires question/variante non encore écrites.
Un échec du service de génération interrompt le run après sauvegarde, au lieu
de répéter la même erreur sur toutes les questions.
Une modification de configuration impose un nouveau dossier pour éviter le mélange
de résultats. Les lignes en échec restent enregistrées : relancer dans un nouveau
dossier pour les réévaluer. Les résultats de smoke test ne sont pas ceux du jeu complet.

## CV

Utiliser seulement les résultats réellement produits. Ne pas traduire Faithfulness
en « conformité » ni annoncer une réduction d'hallucinations à partir de Page Hit@2.
Sans benchmark complet, mentionner la réalisation :
« Conception d'un benchmark RAG de 40 questions (32 répondables, 8 hors corpus),
comparaison de 3 configurations et intégration de Ragas. »
Les gains chiffrés ne peuvent être ajoutés qu'après mesure et contrôle des erreurs.

## Option pour machine CPU

`--compact-judge` supprime les exemples few-shot des prompts Ragas et demande
des justifications courtes (sortie limitée à 768 tokens). Les schémas et formules
des métriques restent ceux de Ragas. Ce changement peut affecter la fiabilité du
juge ; il est enregistré dans le manifeste et doit être identique pour les trois
variantes. Ne pas comparer directement ces scores à ceux du juge par défaut.
