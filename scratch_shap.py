import pandas as pd
import shap
import numpy as np
from src.credit_scorer import CreditScorer

scorer = CreditScorer('C:\Users\boume\OneDrive\Desktop\Projets_data_ai\projet_analyse_pdf/models/natixis_bankruptcy_xgb_model.pkl', 'C:\Users\boume\OneDrive\Desktop\Projets_data_ai\projet_analyse_pdf/models/colonnes_requises.pkl').load()
df = pd.read_csv('C:\Users\boume\OneDrive\Desktop\Projets_data_ai\projet_analyse_pdf/data/clients_test_db.csv', index_col=0)

print('Testing first 3 clients:')
for i in range(3):
    client = df.iloc[[i]]
    exp = scorer.explain(client)
    print(f'Client {i}: Prob={scorer.predict(client):.4f}, Top Feature={exp["top_feature"]}, Value={exp["top_value"]}, SHAP={exp["top_shap"]}')
