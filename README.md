# 🏦 Copilote de Risque de Crédit & Conformité (Agentic Workflow)

## 📌 Présentation du Projet
Ce projet est une application d'intelligence artificielle complète combinant du **Machine Learning traditionnel** et de l'**IA Générative (LLM)**. Son objectif est d'assister les analystes financiers dans l'évaluation du risque de défaut d'une entreprise et de justifier cette évaluation en s'appuyant strictement sur des textes réglementaires (lois bancaires, accords de Bâle, etc.).

L'application est conçue pour fonctionner de manière sécurisée en local (on-premise) via Ollama, répondant ainsi aux exigences de confidentialité bancaire.

## 🏗️ Architecture du Système (Pipeline en 3 étapes)

Le système repose sur un **Agentic Workflow** de bout en bout :

1. **Extraction de Données (Agent LLM)** : 
   L'utilisateur fournit un document financier brut (texte ou PDF). Un Agent LLM lit le texte et extrait de manière déterministe (via *Structured Output JSON*) les 15 indicateurs financiers nécessaires au modèle ML.
2. **Scoring & Explicabilité (XGBoost + SHAP)** : 
   Les données sont injectées dans un modèle XGBoost préalablement entraîné avec *Forward Selection*. Le modèle évalue la probabilité de faillite. L'algorithme **SHAP (Explainable AI)** intervient ensuite pour isoler le « Top 3 » des variables qui tirent le client vers le défaut.
3. **Audit de Conformité (RAG Hybride)** : 
   Le diagnostic (Probabilité + Variables clés SHAP) est transformé en requête naturelle. Un système RAG couplant **ChromaDB** (recherche vectorielle) et **Okapi BM25** (recherche lexicale) interroge le corpus réglementaire de la banque. Un modèle *Cross-Encoder* (MiniLM) filtre les meilleurs résultats, avant qu'un LLM final rédige un avis de conformité sourcé, sans aucune hallucination.

## ⚙️ Technologies Utilisées
* **Interface** : Streamlit
* **Machine Learning** : XGBoost, SHAP (Explainable AI), Pandas, Scikit-Learn
* **Génération et Extraction (LLMs)** : LangChain, Ollama (Llama 3.2 / Mistral)
* **Vector Store & Hybrid Search** : ChromaDB, Rank_BM25, HuggingFace (sentence-camembert-base, ms-marco-MiniLM-L-6-v2)

## 🚀 Installation & Utilisation

\\\ash
# 1. Cloner le dépôt
git clone https://github.com/zakarya-22/projet_analyse_pdf.git
cd projet_analyse_pdf

# 2. Créer l'environnement virtuel et installer les dépendances
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
pip install rank_bm25

# 3. Construire la base vectorielle et l'index BM25 (RAG)
python scripts/build_vectordb.py

# 4. Lancer l'application web
streamlit run src/app.py
\\\

## 📊 Fonctionnalités de l'Interface
* **Onglet Nouveau Dossier** : Copiez-collez le texte d'un rapport financier. L'Agent extrait les chiffres, calcule le risque, génère le graphe SHAP, et rédige l'avis de conformité.
* **Onglet Portefeuille Existant** : Évaluation massive et instantanée d'une base de données CSV de clients. Détection automatique des clients à risque dépassant le seuil de tolérance de la banque, avec option d'audit profond pour chaque dossier.