import pandas as pd
import xgboost as xgb
import shap
import pickle
import numpy as np

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
        df = df.copy()
        for col in self.columns:
            if col not in df.columns:
                df[col] = 0.0
        return df[self.columns]

    def predict(self, df: pd.DataFrame) -> float:
        X = self._align(df)
        prob = self.model.predict_proba(X)[:, 1]
        return float(prob[0])

    def explain(self, client_row: pd.DataFrame) -> dict:
        X = self._align(client_row)
        raw = self._explainer.shap_values(X)
        sv = np.array(raw[1] if isinstance(raw, list) else raw).flatten()
        
        # Recuperer le TOP 3
        top_indices = np.argsort(np.abs(sv))[-3:][::-1]
        
        return {
            "shap_values": sv,
            "feature_names": list(self.columns),
            "top_3_features": [self.columns[i] for i in top_indices],
            "top_3_values": [float(X.iloc[0, i]) for i in top_indices],
            "top_3_shaps": [float(sv[i]) for i in top_indices],
        }