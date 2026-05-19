from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

app = FastAPI(title="API de Mapeamento de Perfis (Sports Analytics)")

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


class AthleteMetrics(BaseModel):
    distance_m: float
    workload: float
    high_intensity_running_m: float
    sprint_distance_m: float
    top_speed_kph: float
    accelerations: float
    decelerations: float
    no_of_sprints: float

@app.post("/predict")
def predict_profile(metrics: AthleteMetrics):
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

@app.get("/")
def health_check():
    return {"status": "IA Online e Operante!"}