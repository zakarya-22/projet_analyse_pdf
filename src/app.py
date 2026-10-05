import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from credit_scorer import CreditScorer
from regulatory_rag import RegulatoryRAG, build_rag_query
from audit_reporter import AuditReporter

ROOT         = Path(__file__).parent.parent
MODEL_PATH   = ROOT / "models" / "natixis_bankruptcy_xgb_model.pkl"
COLUMNS_PATH = ROOT / "models" / "colonnes_requises.pkl"
DB_PATH      = ROOT / "data"   / "chroma_db"
DATA_PATH    = ROOT / "data"   / "clients_test_db.csv"
LOG_PATH     = ROOT / "logs"   / "audit.log"

st.set_page_config(page_title="Copilote Risque Credit", page_icon="u\U0001F3E6", layout="wide")


@st.cache_resource(show_spinner="Chargement des modeles...")
def load_backend():
    scorer   = CreditScorer(MODEL_PATH, COLUMNS_PATH).load()
    rag      = RegulatoryRAG(DB_PATH).load()
    reporter = AuditReporter(LOG_PATH)
    return scorer, rag, reporter


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


try:
    scorer, rag, reporter = load_backend()
    df = load_data()
except Exception as e:
    st.error(f"Erreur de demarrage : {e}")
    st.stop()

for key, default in [("batch_done", False), ("batch_df", None),
                     ("preselected_client", None), ("audit_cache", {})]:
    if key not in st.session_state:
        st.session_state[key] = default

with st.sidebar:
    st.title("Copilote Risque Credit")
    st.caption("ML + RAG  |  100% Local  |  RGPD")
    st.divider()
    threshold = st.slider("Seuil d'alerte (%)", 30, 90, 50, 5,
                          help="Au-dessus de ce seuil, l'audit reglementaire est disponible.") / 100
    st.divider()
    st.caption("XGBoost + SHAP  |  ChromaDB  |  Mistral 7B")

st.header("Tableau de Bord - Analyse du Risque de Credit")

tab1, tab2 = st.tabs(["Portefeuille", "Audit Client"])

# ─── Onglet 1 : Analyse batch ─────────────────────────────────────────────────
with tab1:
    col_btn, col_stats = st.columns([1, 4], gap="large")

    with col_btn:
        run = st.button("Analyser le portefeuille", type="primary", use_container_width=True)

    if run:
        with st.spinner("Scoring en cours..."):
            features = df.drop(columns=["Client_ID"])
            scores   = scorer.predict_batch(features)
            batch = pd.DataFrame({
                "Client_ID":           df["Client_ID"].values,
                "Score de defaut (%)": (scores * 100).round(2),
                "Statut": [
                    "CRITIQUE" if s >= 0.70 else "ELEVE" if s >= threshold else "FAIBLE"
                    for s in scores
                ],
            }).sort_values("Score de defaut (%)", ascending=False)

        st.session_state.batch_df   = batch
        st.session_state.batch_done = True
        reporter.log_batch(len(batch), int((scores >= threshold).sum()), threshold)

    if st.session_state.batch_done and st.session_state.batch_df is not None:
        res    = st.session_state.batch_df
        n_risk = int((res["Score de defaut (%)"] >= threshold * 100).sum())

        with col_stats:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Clients analyses", len(res))
            c2.metric("A risque eleve",   n_risk)
            c3.metric("Taux de risque",   f"{n_risk / len(res):.1%}")
            c4.metric("Score median",     f"{res['Score de defaut (%)'].median():.1f}%")

        st.dataframe(res, use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("Lancer un audit individuel")
        sel_col, btn_col = st.columns([3, 1])
        with sel_col:
            client_pick = st.selectbox("Client ID :", options=res["Client_ID"].tolist(),
                                       key="batch_client_select")
        with btn_col:
            st.write(""); st.write("")
            if st.button("Aller a l'audit", use_container_width=True):
                st.session_state.preselected_client = client_pick
                st.info("Client pre-selectionne. Ouvrez l'onglet **Audit Client**.")

# ─── Onglet 2 : Audit individuel ──────────────────────────────────────────────
with tab2:
    client_ids  = df["Client_ID"].tolist()
    default_idx = (
        client_ids.index(st.session_state.preselected_client)
        if st.session_state.preselected_client in client_ids else 0
    )

    client_id  = st.selectbox("Client ID :", client_ids, index=default_idx, key="audit_sel")
    client_row = df[df["Client_ID"] == client_id].drop(columns=["Client_ID"]).copy()

    prob        = scorer.predict(client_row)
    explanation = scorer.explain(client_row)

    col_prob, col_status = st.columns(2)
    with col_prob:
        st.metric("Probabilite de defaut", f"{prob * 100:.2f}%")
    with col_status:
        if prob >= 0.70:
            st.error("RISQUE CRITIQUE - Audit obligatoire")
        elif prob >= threshold:
            st.warning(f"RISQUE ELEVE - Seuil depasse ({threshold:.0%})")
        else:
            st.success(f"RISQUE FAIBLE - En dessous du seuil ({threshold:.0%})")

    with st.expander("Donnees financieres du client"):
        st.dataframe(client_row, use_container_width=True, hide_index=True)

    st.subheader("Explication du score (SHAP)")

    sv      = explanation["shap_values"]
    f_names = explanation["feature_names"]
    n_show  = min(12, len(sv))
    top_idx = np.argsort(np.abs(sv))[-n_show:][::-1]

    fig, ax = plt.subplots(figsize=(9, max(3.5, n_show * 0.42)))
    colors  = ["#e74c3c" if sv[i] > 0 else "#3498db" for i in top_idx]
    ax.barh([f_names[i] for i in top_idx][::-1], [sv[i] for i in top_idx][::-1],
            color=colors[::-1], edgecolor="none")
    ax.axvline(0, color="#888888", linewidth=0.8, linestyle="--")
    ax.set_title("Variables les plus impactantes sur le score", pad=10)
    ax.set_xlabel("<-- reduit le risque    |    augmente le risque -->")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.divider()

    if prob < threshold:
        st.info(f"Score inferieur au seuil ({threshold:.0%}). Aucun audit reglementaire requis.")
    else:
        st.subheader("Audit Reglementaire")
        top_feat = explanation["top_feature"]
        top_val  = explanation["top_value"]
        top_shap = explanation["top_shap"]

        st.caption(
            f"Facteur principal : **{top_feat}** = {top_val:.4f}  "
            f"|  Contribution SHAP : {top_shap:+.4f}"
        )

        if st.button("Lancer l'analyse reglementaire", type="primary", key="rag_btn"):
            query = build_rag_query(prob, top_feat, top_val)
            with st.spinner("Analyse du document en cours..."):
                t0 = time.time()
                try:
                    response = rag.query(query)
                except Exception as e:
                    response = (
                        f"Erreur LLM - Verifiez qu'Ollama est actif (ollama serve).\n"
                        f"Detail : {e}"
                    )
                elapsed = time.time() - t0

            report = reporter.compile(client_id, prob, top_feat, top_val, response)
            st.session_state.audit_cache[client_id] = report
            st.caption(f"Genere en {elapsed:.1f}s")

        if client_id in st.session_state.audit_cache:
            report = st.session_state.audit_cache[client_id]
            st.info(report["regulatory_analysis"])
            st.download_button(
                label="Telecharger le rapport (.md)",
                data=reporter.to_markdown(report),
                file_name=f"audit_{client_id}_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                mime="text/markdown",
            )