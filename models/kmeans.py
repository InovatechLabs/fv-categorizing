import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib

# 1. Leitura dos Dados
df = pd.read_excel('Players.xlsx')

# 2. Filtros Iniciais (Isolando Sessões Completas e evitando outliers extremos de duração)
df_treino = df[df['Segment Name'] == 'Whole Session'].copy()

# Opcional: Filtrar jogadores de linha (removendo goleiros se existirem no dataset)
# df_treino = df_treino[df_treino['Athlete Position'] != 'Goalkeeper']

# 3. Definição das agregações customizadas
colunas_agregacao = {
    'Distance (m)': 'mean', 
    'Workload': 'mean', 
    'High Intensity Running (m)': 'mean', 
    'Sprint Distance (m)': 'mean', 
    'Top Speed (kph)': 'max',  # Alterado para MAX para capturar o pico de velocidade
    'Accelerations': 'mean', 
    'Decelerations': 'mean', 
    'No. of Sprints': 'mean'
}

# 4. Agrupamento por atleta
df_atletas = df_treino.groupby('Athlete ID').agg(colunas_agregacao).reset_index()

# 5. Tratamento de Nulos (Imputação pela Mediana ao invés de 0)
X = df_atletas[list(colunas_agregacao.keys())]

imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

# 6. Padronização
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# 7. (Opcional, porém recomendado) Validar a quantidade de clusters via Silhueta
print("=== AVALIAÇÃO DE CLUSTERS ===")
for k in range(3, 7):
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans_temp.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    print(f"Silhouette Score para k={k}: {score:.3f}")

# 8. Modelo Final (Assumindo que k=5 foi validado como o ideal)
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df_atletas['Cluster_ID'] = kmeans.fit_predict(X_scaled)

print("\n=== MÉDIA DOS ATLETAS EM CADA CLUSTER ===")
medias_clusters = df_atletas.groupby('Cluster_ID')[list(colunas_agregacao.keys())].mean()
print(medias_clusters.round(1))

print("\n=== QUANTIDADE DE ATLETAS POR CLUSTER ===")
print(df_atletas['Cluster_ID'].value_counts().sort_index())

# 9. Exportando o Pipeline para integração
joblib.dump(kmeans, 'modelo_kmeans.pkl')
joblib.dump(scaler, 'scaler_kmeans.pkl')
joblib.dump(imputer, 'imputer_kmeans.pkl') # Importante salvar o imputer para a predição futura!

print("\nArquivos atualizados com sucesso!")