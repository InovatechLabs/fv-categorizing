from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from typing import List

app = FastAPI(title="API de Mapeamento de Perfis e Anomalias (Sports Analytics)")

# ==========================================
# CARREGAMENTO DO MODELO K-MEANS
# ==========================================
kmeans = joblib.load('modelo_kmeans.pkl')
scaler = joblib.load('scaler_kmeans.pkl')
imputer = joblib.load('imputer_kmeans.pkl')

PROFILE_MAP = {
    0: "Balanced",        
    1: "Explosive",       
    2: "Low Intensity",    
    3: "High Impact Load", 
    4: "High Endurance"
}

# ==========================================
# MODELOS DE DADOS (PYDANTIC)
# ==========================================
class AthleteMetrics(BaseModel):
    distance_m: float
    workload: float
    high_intensity_running_m: float
    sprint_distance_m: float
    top_speed_kph: float
    accelerations: float
    decelerations: float
    no_of_sprints: float

class AnomalyRequest(BaseModel):
    history: List[float]
    current: float

# ==========================================
# ROTAS DA API
# ==========================================

@app.post("/predict")
def predict_profile(metrics: AthleteMetrics):
    """
    K-MEANS: Classifica o perfil do jogador com base em suas métricas gerais.
    """
    df_input = pd.DataFrame([{
        'Distance (m)': metrics.distance_m,
        'Workload': metrics.workload,
        'High Intensity Running (m)': metrics.high_intensity_running_m,
        'Sprint Distance (m)': metrics.sprint_distance_m,
        'Top Speed (kph)': metrics.top_speed_kph,
        'Accelerations': metrics.accelerations,
        'Decelerations': metrics.decelerations,
        'No. of Sprints': metrics.no_of_sprints
    }])
    
    X_imputed = imputer.transform(df_input)
    X_scaled = scaler.transform(X_imputed)
    cluster_id = int(kmeans.predict(X_scaled)[0])
    
    profile_name = PROFILE_MAP.get(cluster_id, "Balanced")
    
    return {
        "cluster_id": cluster_id,
        "profile": profile_name
    }

@app.post("/detect-anomaly")
def detect_performance_drop(req: AnomalyRequest):
    """
    ISOLATION FOREST: Detecta quedas atípicas de desempenho comparando o jogo atual com o histórico isolado.
    """
    # Se não tiver histórico suficiente, não tem como analisar
    if len(req.history) < 3:
        return {"is_anomaly": False, "message": "Histórico insuficiente para análise."}
    
    # Prepara os dados para o scikit-learn
    X_train = np.array(req.history).reshape(-1, 1)
    X_test = np.array([req.current]).reshape(-1, 1)
    
    # Inicializa o Isolation Forest (contamination = 10% de chance de anomalia)
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(X_train)
    
    # Faz a predição: retorna -1 para anomalia (queda/ponto fora da curva) e 1 para normal
    prediction = model.predict(X_test)
    
    return {
        "is_anomaly": bool(prediction[0] == -1),
        "message": "Anomalia detectada pelo Isolation Forest!" if prediction[0] == -1 else "Desempenho normal."
    }

@app.get("/")
def health_check():
    return {"status": "IA Online e Operante! (K-Means + Isolation Forest)"}