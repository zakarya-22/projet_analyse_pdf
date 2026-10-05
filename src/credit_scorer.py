import joblib
import numpy as np
import pandas as pd
import shap
from pathlib import Path


class CreditScorer:
    def __init__(self, model_path: str | Path, columns_path: str | Path):
        self.model_path = Path(model_path)
        self.columns_path = Path(columns_path)
        self.model = None
        self.columns: list[str] = []
        self._explainer = None

    def load(self) -> "CreditScorer":
        self.model = joblib.load(self.model_path)
        self.columns = list(joblib.load(self.columns_path))
        self._explainer = shap.TreeExplainer(self.model)
        return self

    def _align(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.reindex(columns=self.columns, fill_value=0)

    def predict(self, client_row: pd.DataFrame) -> float:
        return float(self.model.predict_proba(self._align(client_row))[0][1])

    def predict_batch(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self._align(df))[:, 1]

    def explain(self, client_row: pd.DataFrame) -> dict:
        X = self._align(client_row)
        raw = self._explainer.shap_values(X)
        sv = np.array(raw[1] if isinstance(raw, list) else raw).flatten()
        top_idx = int(np.argmax(np.abs(sv)))
        return {
            "shap_values": sv,
            "top_feature": self.columns[top_idx],
            "top_value": float(X.iloc[0, top_idx]),
            "top_shap": float(sv[top_idx]),
            "feature_names": list(self.columns),
        }
