import pandas as pd
import pickle
from src.credit_scorer import CreditScorer

scorer = CreditScorer(r'C:/Users/boume/OneDrive/Desktop/Projets_data_ai/projet_analyse_pdf/models/natixis_bankruptcy_xgb_model.pkl', r'C:/Users/boume/OneDrive/Desktop/Projets_data_ai/projet_analyse_pdf/models/colonnes_requises.pkl').load()
df = pd.read_csv(r'C:/Users/boume/OneDrive/Desktop/Projets_data_ai/projet_analyse_pdf/data/clients_test_db.csv', index_col=0)

print('--- COLUMNS IN MODEL ---')
print(scorer.columns)

print('\n--- COLUMNS IN CSV ---')
print(list(df.columns))

print('\n--- MISMATCHES ---')
missing_in_df = [c for c in scorer.columns if c not in df.columns]
print(f'Columns in model but NOT in CSV: {missing_in_df}')

missing_in_model = [c for c in df.columns if c not in scorer.columns]
print(f'Columns in CSV but NOT in model: {missing_in_model}')
