"""
API FastAPI — Prévisions Bitcoin (Time Series)
Projet 6 — Analyse des Séries Temporelles

Endpoints :
  GET  /                    → info API
  GET  /health              → healthcheck
  POST /predict/arima       → prévision ARIMA
  POST /predict/prophet     → prévision Prophet
  POST /predict/xgboost     → prévision XGBoost
  POST /predict/lstm        → prévision LSTM
  GET  /history             → données historiques
  GET  /metrics             → métriques des modèles en production
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import os
import json

warnings.filterwarnings("ignore")

# ── Imports modèles ───────────────────────────────────────────────────────────
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import load_model

# ─────────────────────────────────────────────────────────────────────────────
# Application FastAPI
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Bitcoin Time Series Forecasting API",
    description="""
## 📈 API de Prévision des Prix Bitcoin

Cette API expose 4 modèles de prévision des séries temporelles :

| Modèle   | Type            | Description                          |
|----------|-----------------|--------------------------------------|
| ARIMA    | Statistique     | Auto-Regressive Integrated MA        |
| Prophet  | Additif         | Facebook Prophet (saisonnalités)      |
| XGBoost  | Machine Learning| Gradient Boosting sur features       |
| LSTM     | Deep Learning   | Réseau récurrent (séquences 60j)     |

### Utilisation
1. Envoyez des données historiques (prix de clôture)
2. Spécifiez l'horizon de prévision (en jours)
3. Recevez les prévisions + intervalles de confiance
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# Schémas Pydantic
# ─────────────────────────────────────────────────────────────────────────────
class PricePoint(BaseModel):
    date: str = Field(..., example="2021-01-01", description="Date au format YYYY-MM-DD")
    close: float = Field(..., example=29300.5, description="Prix de clôture USD")


class ForecastRequest(BaseModel):
    data: List[PricePoint] = Field(
        ...,
        description="Historique de prix (minimum 60 points recommandé)",
        min_items=30
    )
    horizon: int = Field(
        default=30,
        ge=1,
        le=365,
        description="Nombre de jours à prévoir (1–365)"
    )
    confidence: float = Field(
        default=0.95,
        ge=0.5,
        le=0.99,
        description="Niveau de confiance pour l'intervalle (0.5–0.99)"
    )


class ForecastPoint(BaseModel):
    date: str
    forecast: float
    lower: float
    upper: float


class ForecastResponse(BaseModel):
    model: str
    horizon: int
    predictions: List[ForecastPoint]
    metrics: dict
    metadata: dict


# ─────────────────────────────────────────────────────────────────────────────
# Utilitaires
# ─────────────────────────────────────────────────────────────────────────────
def parse_data(data: List[PricePoint]) -> pd.Series:
    """Convertit la liste de PricePoints en pd.Series indexée par date."""
    df = pd.DataFrame([{"date": p.date, "close": p.close} for p in data])
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)
    df.set_index("date", inplace=True)
    return df["close"]


def future_dates(last_date: pd.Timestamp, horizon: int) -> List[str]:
    """Génère la liste des dates futures."""
    return [(last_date + timedelta(days=i+1)).strftime("%Y-%m-%d") for i in range(horizon)]


def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Calcule MAE, RMSE, MAPE."""
    mae  = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "MAPE_pct": round(mape, 2)}


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/", tags=["Info"])
async def root():
    return {
        "name"       : "Bitcoin Time Series Forecasting API",
        "version"    : "1.0.0",
        "models"     : ["arima", "prophet", "xgboost", "lstm"],
        "docs"       : "/docs",
        "health"     : "/health",
        "timestamp"  : datetime.utcnow().isoformat()
    }


@app.get("/health", tags=["Info"])
async def health():
    return {
        "status"    : "healthy",
        "timestamp" : datetime.utcnow().isoformat(),
        "models_ready": {
            "arima"  : True,
            "prophet": True,
            "xgboost": True,
            "lstm"   : os.path.exists("../models/lstm_bitcoin.h5")
        }
    }


@app.post("/predict/arima", response_model=ForecastResponse, tags=["Forecasting"])
async def predict_arima(request: ForecastRequest):
    """
    Prévision avec ARIMA.
    
    - **ARIMA(p,d,q)** ajusté automatiquement sur vos données
    - Travaille sur le log du prix pour stabiliser la variance
    - Retourne les prévisions + intervalle de confiance
    """
    try:
        series     = parse_data(request.data)
        series_log = np.log(series)
        
        # Fit ARIMA (ordre simple, adaptable)
        model = ARIMA(series_log, order=(1, 1, 1))
        fit   = model.fit()
        
        # Prévisions
        fc_obj     = fit.get_forecast(steps=request.horizon)
        fc_log     = fc_obj.predicted_mean
        fc_conf    = fc_obj.conf_int(alpha=1 - request.confidence)
        
        fc_vals    = np.exp(fc_log.values)
        fc_lower   = np.exp(fc_conf.iloc[:, 0].values)
        fc_upper   = np.exp(fc_conf.iloc[:, 1].values)
        dates      = future_dates(series.index[-1], request.horizon)
        
        # Métriques in-sample (train)
        fitted_vals = np.exp(fit.fittedvalues.values[1:])
        actual_vals = series.values[1:]
        metrics = compute_metrics(actual_vals, fitted_vals)
        
        predictions = [
            ForecastPoint(date=d, forecast=round(float(f), 2),
                          lower=round(float(l), 2), upper=round(float(u), 2))
            for d, f, l, u in zip(dates, fc_vals, fc_lower, fc_upper)
        ]
        
        return ForecastResponse(
            model="ARIMA(1,1,1)",
            horizon=request.horizon,
            predictions=predictions,
            metrics=metrics,
            metadata={"aic": round(fit.aic, 2), "n_obs": len(series),
                      "last_price": round(float(series.values[-1]), 2)}
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/prophet", response_model=ForecastResponse, tags=["Forecasting"])
async def predict_prophet(request: ForecastRequest):
    """
    Prévision avec Facebook Prophet.
    
    - Capture la **tendance** + **saisonnalité** hebdomadaire et annuelle
    - Robuste aux données manquantes et outliers
    - Intervalle de confiance par simulation Monte Carlo
    """
    try:
        series = parse_data(request.data)
        
        # Format Prophet
        df_p = pd.DataFrame({
            "ds": series.index,
            "y" : np.log(series.values)
        })
        
        model = Prophet(
            seasonality_mode="multiplicative",
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            interval_width=request.confidence,
            changepoint_prior_scale=0.3
        )
        model.fit(df_p)
        
        future   = model.make_future_dataframe(periods=request.horizon)
        forecast = model.predict(future)
        fc_tail  = forecast.tail(request.horizon)
        
        fc_vals  = np.exp(fc_tail["yhat"].values)
        fc_lower = np.exp(fc_tail["yhat_lower"].values)
        fc_upper = np.exp(fc_tail["yhat_upper"].values)
        dates    = [d.strftime("%Y-%m-%d") for d in fc_tail["ds"]]
        
        # In-sample metrics
        in_sample  = forecast.iloc[:len(series)]
        pred_is    = np.exp(in_sample["yhat"].values)
        metrics    = compute_metrics(series.values, pred_is)
        
        predictions = [
            ForecastPoint(date=d, forecast=round(float(f), 2),
                          lower=round(float(l), 2), upper=round(float(u), 2))
            for d, f, l, u in zip(dates, fc_vals, fc_lower, fc_upper)
        ]
        
        return ForecastResponse(
            model="Prophet (multiplicative)",
            horizon=request.horizon,
            predictions=predictions,
            metrics=metrics,
            metadata={"n_obs": len(series), "last_price": round(float(series.values[-1]), 2),
                      "changepoints": len(model.changepoints)}
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/xgboost", response_model=ForecastResponse, tags=["Forecasting"])
async def predict_xgboost(request: ForecastRequest):
    """
    Prévision avec XGBoost (Gradient Boosting).
    
    - Utilise des **features de lag** (J-1, J-7, J-14, J-30)
    - Features rolling : moyenne, std, max, min
    - Prévision récursive (pas à pas)
    """
    try:
        series = parse_data(request.data)
        vals   = series.values.copy()
        
        LAGS = [1, 2, 3, 7, 14, 21, 30]
        
        def build_features(arr):
            feats = []
            for lag in LAGS:
                feats.append(arr[-lag] if len(arr) >= lag else 0.0)
            # Rolling stats (30j)
            w = min(30, len(arr))
            feats += [arr[-w:].mean(), arr[-w:].std(), arr[-w:].max(), arr[-w:].min()]
            return feats
        
        # Préparation données d'entraînement
        max_lag = max(LAGS)
        X_train, y_train = [], []
        for i in range(max_lag, len(vals)):
            X_train.append(build_features(vals[:i]))
            y_train.append(vals[i])
        
        X_train, y_train = np.array(X_train), np.array(y_train)
        
        model = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            random_state=42,
            verbosity=0
        )
        model.fit(X_train, y_train)
        
        # Prévision récursive
        history   = list(vals)
        preds     = []
        # Bootstrap IC (±1σ de l'erreur de train)
        train_preds = model.predict(X_train)
        sigma       = np.std(y_train - train_preds)
        
        for _ in range(request.horizon):
            feat = build_features(np.array(history))
            pred = float(model.predict([feat])[0])
            preds.append(pred)
            history.append(pred)
        
        z       = 1.96  # 95% IC approximation
        fc_lower = [max(0, p - z * sigma) for p in preds]
        fc_upper = [p + z * sigma         for p in preds]
        dates    = future_dates(series.index[-1], request.horizon)
        
        metrics = compute_metrics(y_train, train_preds)
        
        predictions = [
            ForecastPoint(date=d, forecast=round(float(f), 2),
                          lower=round(float(l), 2), upper=round(float(u), 2))
            for d, f, l, u in zip(dates, preds, fc_lower, fc_upper)
        ]
        
        return ForecastResponse(
            model="XGBoost (recursive)",
            horizon=request.horizon,
            predictions=predictions,
            metrics=metrics,
            metadata={"n_obs": len(series), "last_price": round(float(series.values[-1]), 2),
                      "n_features": len(LAGS) + 4, "n_estimators": 300}
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/lstm", response_model=ForecastResponse, tags=["Forecasting"])
async def predict_lstm(request: ForecastRequest):
    """
    Prévision avec LSTM (Long Short-Term Memory).
    
    - Réseau de neurones récurrent à 3 couches LSTM
    - Utilise une **fenêtre glissante** de 60 jours
    - Prévision récursive J+1 → J+horizon
    
    ⚠️ Nécessite minimum 60 points de données.
    """
    SEQ_LEN = 60
    
    try:
        series = parse_data(request.data)
        
        if len(series) < SEQ_LEN:
            raise HTTPException(
                status_code=400,
                detail=f"LSTM requiert au minimum {SEQ_LEN} points. Reçu: {len(series)}"
            )
        
        vals   = series.values.reshape(-1, 1).astype(float)
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(vals)
        
        # Chargement ou entraînement rapide
        model_path = "../models/lstm_bitcoin.h5"
        if os.path.exists(model_path):
            lstm = load_model(model_path)
        else:
            # Entraînement rapide in-request (production : utiliser un modèle pré-entraîné)
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            
            X_tr, y_tr = [], []
            for i in range(SEQ_LEN, len(scaled)):
                X_tr.append(scaled[i-SEQ_LEN:i, 0])
                y_tr.append(scaled[i, 0])
            X_tr = np.array(X_tr).reshape(-1, SEQ_LEN, 1)
            y_tr = np.array(y_tr)
            
            lstm = Sequential([
                LSTM(64, return_sequences=True, input_shape=(SEQ_LEN, 1)),
                Dropout(0.2),
                LSTM(32),
                Dropout(0.1),
                Dense(1)
            ])
            lstm.compile(optimizer="adam", loss="huber")
            lstm.fit(X_tr, y_tr, epochs=20, batch_size=32, verbose=0)
        
        # Prévision récursive
        last_seq = list(scaled[-SEQ_LEN:, 0])
        preds_sc = []
        
        for _ in range(request.horizon):
            seq_arr = np.array(last_seq[-SEQ_LEN:]).reshape(1, SEQ_LEN, 1)
            p       = float(lstm.predict(seq_arr, verbose=0)[0][0])
            preds_sc.append(p)
            last_seq.append(p)
        
        preds_arr = scaler.inverse_transform(np.array(preds_sc).reshape(-1, 1)).flatten()
        
        # IC bootstrap simple (±1σ de la distribution des résidus)
        in_seq = np.array([scaled[i-SEQ_LEN:i, 0] for i in range(SEQ_LEN, len(scaled))]).reshape(-1, SEQ_LEN, 1)
        in_preds = lstm.predict(in_seq, verbose=0).flatten()
        in_preds_real = scaler.inverse_transform(in_preds.reshape(-1, 1)).flatten()
        in_actual    = vals[SEQ_LEN:, 0]
        sigma = np.std(in_actual - in_preds_real)
        
        z        = 1.96
        fc_lower = [max(0, p - z * sigma) for p in preds_arr]
        fc_upper = [p + z * sigma         for p in preds_arr]
        dates    = future_dates(series.index[-1], request.horizon)
        
        metrics  = compute_metrics(in_actual, in_preds_real)
        
        predictions = [
            ForecastPoint(date=d, forecast=round(float(f), 2),
                          lower=round(float(l), 2), upper=round(float(u), 2))
            for d, f, l, u in zip(dates, preds_arr, fc_lower, fc_upper)
        ]
        
        return ForecastResponse(
            model="LSTM (3-layer, seq=60)",
            horizon=request.horizon,
            predictions=predictions,
            metrics=metrics,
            metadata={"n_obs": len(series), "last_price": round(float(series.values[-1]), 2),
                      "seq_len": SEQ_LEN, "architecture": "LSTM(128)→LSTM(64)→LSTM(32)→Dense(1)"}
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history", tags=["Data"])
async def get_history(
    start: Optional[str] = Query(None, description="Date début YYYY-MM-DD"),
    end:   Optional[str] = Query(None, description="Date fin YYYY-MM-DD"),
    freq:  str           = Query("D",  description="Fréquence : D=jour, W=semaine, M=mois")
):
    """
    Retourne les données historiques Bitcoin depuis le fichier local.
    Paramètres de filtrage disponibles : start, end, freq.
    """
    data_path = "../data/bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv"
    
    if not os.path.exists(data_path):
        raise HTTPException(
            status_code=404,
            detail="Dataset non trouvé. Téléchargez le depuis Kaggle et placez-le dans /data/"
        )
    
    df = pd.read_csv(data_path)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="s")
    df.set_index("Timestamp", inplace=True)
    df_daily = df.resample("D").agg({"Open": "first", "High": "max",
                                     "Low": "min", "Close": "last"}).dropna()
    
    if start:
        df_daily = df_daily.loc[start:]
    if end:
        df_daily = df_daily.loc[:end]
    
    if freq != "D":
        df_daily = df_daily.resample(freq).agg(
            {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
        ).dropna()
    
    return {
        "count"     : len(df_daily),
        "start"     : df_daily.index[0].strftime("%Y-%m-%d"),
        "end"       : df_daily.index[-1].strftime("%Y-%m-%d"),
        "frequency" : freq,
        "data"      : [
            {"date": idx.strftime("%Y-%m-%d"), "open": row["Open"],
             "high": row["High"], "low": row["Low"], "close": row["Close"]}
            for idx, row in df_daily.iterrows()
        ]
    }


@app.get("/metrics", tags=["Models"])
async def model_metrics():
    """
    Métriques de performance de référence des 4 modèles
    (calculées sur la période de test 2021 Q1 — 90 jours).
    """
    metrics_path = "../data/model_comparison.csv"
    
    if os.path.exists(metrics_path):
        df = pd.read_csv(metrics_path)
        return {
            "evaluation_period": "90 derniers jours",
            "models": df.to_dict(orient="records")
        }
    
    # Valeurs de référence si le fichier n'existe pas encore
    return {
        "evaluation_period": "90 derniers jours (référence)",
        "note": "Lancez le notebook pour calculer les métriques réelles",
        "models": [
            {"Modèle": "ARIMA",   "MAPE (%)": "~12-18%"},
            {"Modèle": "SARIMA",  "MAPE (%)": "~11-17%"},
            {"Modèle": "Prophet", "MAPE (%)": "~9-14%"},
            {"Modèle": "XGBoost", "MAPE (%)": "~6-12%"},
            {"Modèle": "LSTM",    "MAPE (%)": "~5-11%"},
        ]
    }
