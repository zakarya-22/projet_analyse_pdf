import pandas as pd
from src.credit_scorer import CreditScorer

scorer = CreditScorer(r'C:/Users/boume/OneDrive/Desktop/Projets_data_ai/projet_analyse_pdf/models/natixis_bankruptcy_xgb_model.pkl', r'C:/Users/boume/OneDrive/Desktop/Projets_data_ai/projet_analyse_pdf/models/colonnes_requises.pkl').load()
df = pd.read_csv(r'C:/Users/boume/OneDrive/Desktop/Projets_data_ai/projet_analyse_pdf/data/clients_test_db.csv')

print(f'Shape of new CSV: {df.shape}')

missing = [c for c in scorer.columns if c not in df.columns]
if not missing:
    print('SUCCESS: All 15 required features are perfectly present in the new CSV!')
else:
    print(f'ERROR: Still missing columns: {missing}')
