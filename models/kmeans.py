import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib

df = pd.read_excel('/content/drive/MyDrive/Colab Notebooks/Players.xlsx')

df_treino = df[df['Segment Name'] == 'Whole Session'].copy()

colunas_vitais = [
    'Distance (m)', 'Workload', 'High Intensity Running (m)', 
    'Sprint Distance (m)', 'Top Speed (kph)', 'Accelerations', 
    'Decelerations', 'No. of Sprints'
]

df_atletas = df_treino.groupby('Athlete ID')[colunas_vitais].mean().reset_index()

X = df_atletas[colunas_vitais].fillna(0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df_atletas['Cluster_ID'] = kmeans.fit_predict(X_scaled)

print("=== MÉDIA DOS ATLETAS EM CADA CLUSTER ===")
medias_clusters = df_atletas.groupby('Cluster_ID')[colunas_vitais].mean()
print(medias_clusters.round(1))

print("\n=== QUANTIDADE DE ATLETAS POR CLUSTER ===")
print(df_atletas['Cluster_ID'].value_counts().sort_index())

joblib.dump(kmeans, 'modelo_kmeans.pkl')
joblib.dump(scaler, 'scaler_kmeans.pkl')

print("\nArquivos atualizados com sucesso!")