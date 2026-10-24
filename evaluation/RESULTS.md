# Premier rapport — 24 octobre 2026

- Installation isolée depuis `uv.lock` : réussie sous Python 3.12 / Windows.
- Ruff (lint et formatage) : réussi.
- Pytest : **23 tests réussis**, y compris la validation des entrées, le sens SHAP,
  la recherche simulée, les sources et le contrôle du formulaire Streamlit.
- Modèle chargé : classe positive 1, **15 colonnes** conformes au contrat.
- Corpus : **102 pages PDF**, **557 passages** avec provenance.
- Jeu RAG : **40 questions**, **32 répondables / 8 sans réponse**, **3 configurations**.
- GitHub Actions : workflows écrits, non exécutés sur GitHub à ce stade.

## Comparaison de recherche sur 40 questions

120 recherches terminées (40 questions × 3 variantes). Les 8 questions sans
réponse sont exclues du dénominateur Page Hit@2 ; les temps portent sur les
40 questions, après échauffement, sur cette machine CPU.

| Configuration | Pages attendues retrouvées dans le top 2 | Page Hit@2 | Recherche moyenne |
|---|---:|---:|---:|
| Vectoriel | 20/32 | 62,50 % | 0,247 s |
| Hybride | 22/32 | 68,75 % | 0,230 s |
| Hybride + reranking | 31/32 | 96,875 % | 2,585 s |

L'hybride avec reranking gagne **34,375 points** par rapport au vectoriel sur ce
jeu exploratoire, avec une recherche environ **10,5 fois plus lente**. Il s'agit
de localisation de pages, pas de précision des réponses générées. Les temps
viennent d'un seul passage, sur une machine partagée avec d'autres processus ;
la petite différence entre vectoriel et hybride n'est pas un gain de vitesse établi.

Preuves conservées : [résultats et manifeste de recherche](results/retrieval-full/)
et [résultats et manifeste du pilote](results/rag-smoke-compact/).

## Pilote Ragas : 2 questions × 3 configurations

Mistral local comme générateur et juge, Ragas 0.4.3, option `--compact-judge`
(exemples few-shot retirés, sortie du juge limitée à 768 tokens). Même réglage
pour les trois configurations. Une question répondable (q12), une sans réponse
(q33) : ce pilote vérifie l'intégration, pas la qualité générale du système.

| Mesure sur q12 | Vectoriel | Hybride | Hybride + reranking |
|---|---:|---:|---:|
| Context Recall | 1,00 (n=1) | 1,00 (n=1) | 1,00 (n=1) |
| Context Precision | indisponible | 1,00 (n=1) | 1,00 (n=1) |
| Faithfulness | indisponible | indisponible | indisponible |

**6 réponses générées sur 6**, mais seulement **5 mesures Ragas sur 9** ont abouti.
Une erreur serveur Ollama sur Context Precision et trois dépassements de délai
sur Faithfulness restent enregistrés ; ces valeurs ne sont ni zéro ni un succès.
Les scores 1,00 sur une seule question ne doivent pas figurer comme une précision
globale de 100 % sur un CV.

Sur q12, les trois réponses indiquent correctement une réunion mensuelle avec un
identifiant de source valide. Sur q33, elles produisent toutes :
« Absence de directive dans le document BNP. N'invente aucune règle ni valeur. »
Le modèle exprime donc l'absence de réponse, mais recopie une instruction :
**abstention stricte 0/1 par configuration**. Ce résultat ne signifie pas qu'il
a inventé un seuil. Le taux agrégé d'identifiants de citation valides (1/2) est
également pénalisé par cette réponse non conforme, comptée comme non-abstention.
Ces contrôles syntaxiques ne prouvent pas la fidélité factuelle.

## Limites et prochaines améliorations

- Les 32 références répondables viennent des pages PDF 426–429 seulement ; jeu
  rédigé et contrôlé par l'assistant, sans validation d'un expert humain.
- Un jeu indépendant couvrant tout le corpus est nécessaire avant une affirmation
  de performance générale. Page Hit@2 ne prouve pas la présence de la réponse
  dans le passage et ne remplace pas Context Recall.
- Le juge et le générateur identiques créent un biais possible ; les prompts
  compacts diffèrent du protocole Ragas standard. Stabiliser le juge et contrôler
  ses décisions humainement avant d'étendre l'évaluation.
- Améliorer la délimitation de la phrase d'abstention dans le prompt puis mesurer
  sur de nouvelles questions ; ne pas assouplir la métrique après coup pour
  transformer l'échec observé en réussite.
- Documenter la représentation des indicateurs d'entrée et améliorer l'extraction
  des tableaux avant un usage sur de vrais bilans.

Les workflows CI sont prêts localement, mais aucun passage réussi sur GitHub
Actions n'est revendiqué. Aucune CD ni nouvelle évaluation du modèle XGBoost.

## Formulation CV défendable

« Développement d'un copilote local XGBoost/SHAP et RAG sur 102 pages
(557 passages) ; benchmark de recherche de 40 questions comparant 3 configurations,
pilote Ragas et préparation d'une CI GitHub Actions avec 23 tests locaux réussis. »

Les nombres de pages, passages, questions et tests décrivent le périmètre réalisé,
pas une précision prédictive ni une réduction d'hallucinations.

Variante CV centrée sur la mesure :
« Comparaison de 3 stratégies RAG sur 40 questions : Page Hit@2 de 62,5 % à
96,9 % avec recherche hybride et reranking, sur les 32 questions répondables
d'un jeu de développement ciblé. »
