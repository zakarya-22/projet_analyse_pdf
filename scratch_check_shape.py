import pandas as pd
df = pd.read_csv(r'C:/Users/boume/OneDrive/Desktop/Projets_data_ai/projet_analyse_pdf/data/clients_test_db.csv', index_col=0)
print(f'Shape of clients_test_db.csv currently on disk: {df.shape}')
