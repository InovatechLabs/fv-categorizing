from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from typing import List, Optional

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

# Novos modelos para suportar a requisição em lote (Batch)
class AnomalyBatchItem(BaseModel):
    athlete_id: str
    history: List[float]
    current: float

class AnomalyBatchRequest(BaseModel):
    athletes: List[AnomalyBatchItem]

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
    ISOLATION FOREST: Detecta quedas atípicas de desempenho (Individual).
    """
    if len(req.history) < 3:
        return {"is_anomaly": False, "message": "Histórico insuficiente para análise."}
    
    X_train = np.array(req.history).reshape(-1, 1)
    X_test = np.array([req.current]).reshape(-1, 1)
    
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(X_train)
    prediction = model.predict(X_test)
    
    return {
        "is_anomaly": bool(prediction[0] == -1),
        "message": "Anomalia detectada pelo Isolation Forest!" if prediction[0] == -1 else "Desempenho normal."
    }

@app.post("/detect-anomaly/batch")
def detect_performance_drop_batch(req: AnomalyBatchRequest):
    """
    ISOLATION FOREST: Processa um lote de requisições de uma só vez para evitar gargalo de rede HTTP.
    """
    results = []
    
    for item in req.athletes:
        if len(item.history) < 3:
            results.append({
                "athlete_id": item.athlete_id, 
                "is_anomaly": False, 
                "message": "Histórico insuficiente."
            })
            continue
            
        X_train = np.array(item.history).reshape(-1, 1)
        X_test = np.array([item.current]).reshape(-1, 1)
        
        model = IsolationForest(contamination=0.1, random_state=42)
        model.fit(X_train)
        prediction = model.predict(X_test)
        
        results.append({
            "athlete_id": item.athlete_id,
            "is_anomaly": bool(prediction[0] == -1),
            "message": "Anomalia detectada pelo Isolation Forest!" if prediction[0] == -1 else "Desempenho normal."
        })
        
    return {"results": results}

@app.get("/")
def health_check():
    return {"status": "IA Online e Operante! (K-Means + Isolation Forest)"}