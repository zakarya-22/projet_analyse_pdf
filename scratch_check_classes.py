import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from src.credit_scorer import CreditScorer

scorer = CreditScorer(r'C:/Users/boume/OneDrive/Desktop/Projets_data_ai/projet_analyse_pdf/models/natixis_bankruptcy_xgb_model.pkl', r'C:/Users/boume/OneDrive/Desktop/Projets_data_ai/projet_analyse_pdf/models/colonnes_requises.pkl').load()
df = pd.read_csv(r'C:/Users/boume/OneDrive/Desktop/Projets_data_ai/projet_analyse_pdf/data/clients_test_db.csv', index_col=0)

X = scorer._align(df)
probs = scorer.model.predict_proba(X)

print(f'Mean proba col 0: {probs[:, 0].mean():.4f}')
print(f'Mean proba col 1: {probs[:, 1].mean():.4f}')
print(f'Number of records: {len(df)}')
print(f'Records > 0.0567 in col 0: {(probs[:, 0] > 0.0567).sum()}')
print(f'Records > 0.0567 in col 1: {(probs[:, 1] > 0.0567).sum()}')
try:
    print(f'Classes: {scorer.model.classes_}')
except:
    pass
