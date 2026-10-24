import pickle

import numpy as np
import pandas as pd
import shap

from src.validation import validate_record


class CreditScorer:
    def __init__(self, model_path: str, columns_path: str):
        self.model_path = model_path
        self.columns_path = columns_path
        self.model = None
        self.columns = None
        self._explainer = None

    def load(self):
        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)
        with open(self.columns_path, "rb") as f:
            self.columns = pickle.load(f)

        self._explainer = shap.TreeExplainer(self.model)
        return self

    def _align(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            raise ValueError("Le dossier ne contient aucune ligne.")
        names = [str(c).strip().casefold() for c in df.columns]
        if len(names) != len(set(names)):
            raise ValueError("Colonnes dupliquées après normalisation.")
        rows = []
        for index, row in df.iterrows():
            try:
                rows.append(validate_record(row.to_dict(), self.columns, allow_extra=True))
            except ValueError as exc:
                raise ValueError(f"Client {index} : {exc}") from exc
        return pd.DataFrame(rows, index=df.index, columns=self.columns)

    def predict(self, df: pd.DataFrame) -> float:
        X = self._align(df)
        prob = self.model.predict_proba(X)[:, 1]
        return float(prob[0])

    def explain(self, client_row: pd.DataFrame) -> dict:
        X = self._align(client_row)
        raw = self._explainer.shap_values(X)
        sv = np.array(raw[1] if isinstance(raw, list) else raw).flatten()

        top_indices = np.argsort(np.abs(sv))[-3:][::-1]

        def factors(indices):
            return [
                {"feature": self.columns[i], "value": float(X.iloc[0, i]), "shap": float(sv[i])}
                for i in indices
            ]

        return {
            "risk_factors": factors(sorted(np.flatnonzero(sv > 0), key=lambda i: -sv[i])[:3]),
            "protective_factors": factors(sorted(np.flatnonzero(sv < 0), key=lambda i: sv[i])[:3]),
            "shap_values": sv,
            "feature_names": list(self.columns),
            "top_3_features": [self.columns[i] for i in top_indices],
            "top_3_values": [float(X.iloc[0, i]) for i in top_indices],
            "top_3_shaps": [float(sv[i]) for i in top_indices],
        }
