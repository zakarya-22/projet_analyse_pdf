# Copilote de risque de crédit — fiche entretien

**Pitch (30 secondes).** J’ai développé un prototype local qui extrait 15 indicateurs
financiers, calcule un score XGBoost, explique ce score avec SHAP et recherche des
passages du rapport BNP pour documenter l’analyse. L’interface Streamlit affiche
les facteurs aggravants/protecteurs et les sources. Le score reste une aide à
l’analyse, pas une décision bancaire ni une preuve de conformité.

**Chaîne technique.** Texte/TXT/Markdown → extraction JSON par Mistral/Ollama →
validation des 15 valeurs → XGBoost → SHAP signé → question documentaire →
recherche hybride → reranking → réponse Mistral avec citations et extraits.

**RAG.** 102 pages, 557 passages de 1 000 caractères, chevauchement de 200.
Embeddings CamemBERT et Chroma ; BM25 retrouve les termes précis, le vectoriel
les proximités sémantiques. Fusion RRF pondérée 30/70 de 3 candidats BM25 et
4 vectoriels ; Cross-Encoder MiniLM classe les candidats ; 2 passages alimentent
la réponse. Les deux index exploitent les mêmes passages et métadonnées.

**Fiabilité des entrées.** Valeur manquante ≠ zéro. Refus des doublons, champs
absents, booléens et nombres non finis. Le notebook utilise directement les
valeurs du CSV d’entraînement, sans scaler : la conversion des données comptables
brutes vers cette représentation reste à documenter. Une confirmation utilisateur
ne remplace pas cette validation métier.

**Explicabilité.** SHAP positif augmente la sortie du modèle pour la classe
faillite ; SHAP négatif la diminue. Ce n’est ni une causalité ni nécessairement
une variation en points de probabilité. Le seuil ML n’est pas un seuil réglementaire.

**Évaluation.** Jeu exploratoire de 40 questions : 32 répondables et 8 sans réponse.
Comparaison vectoriel / hybride / hybride + reranking. Page Hit@2 mesure la
présence de la page attendue, pas la qualité de la réponse.
**Résultat : 62,5 % → 96,9 % de Page Hit@2** (20/32 → 31/32), au prix d'une
recherche moyenne de 0,247 → 2,585 s. Jeu ciblé sur quatre pages, pas un test général.
Pilote Ragas sur
2 questions × 3 variantes : Context Precision, Context Recall et Faithfulness,
avec erreurs et effectifs valides conservés. Chiffres et limites :
[rapport](evaluation/RESULTS.md). Aucun score global de précision revendiqué.

**Industrialisation.** Dépendances verrouillées avec uv ; 23 tests locaux réussis ;
GitHub Actions préparé pour installation, lint, tests et rapports sur push/PR.
Évaluation RAG manuelle sur runner local équipé. Workflows non exécutés sur GitHub ;
aucune CD.

**Ce que j’améliorerais ensuite.** Validation des unités avec un expert ; meilleure
extraction des tableaux ; jeu de test indépendant couvrant tout le corpus ; juge
Ragas plus fiable ; comparaison d’un reranker multilingue ; contrôle humain des
citations et de l’abstention. Je mesure chaque gain avant de l’annoncer.
