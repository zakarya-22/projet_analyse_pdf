import pandas as pd
import numpy as np
from src.credit_scorer import CreditScorer

scorer = CreditScorer(r'C:/Users/boume/OneDrive/Desktop/Projets_data_ai/projet_analyse_pdf/models/natixis_bankruptcy_xgb_model.pkl', r'C:/Users/boume/OneDrive/Desktop/Projets_data_ai/projet_analyse_pdf/models/colonnes_requises.pkl').load()
df = pd.read_csv(r'C:/Users/boume/OneDrive/Desktop/Projets_data_ai/projet_analyse_pdf/data/clients_test_db.csv', index_col=0)

X = scorer._align(df)
probs = scorer.model.predict_proba(X)

print(f'Records > 0.5 in col 1: {(probs[:, 1] > 0.5).sum()}')
print(f'Records > 0.5 in col 0: {(probs[:, 0] > 0.5).sum()}')
