# ₿ Bitcoin Time Series Forecasting

Analyse complète des séries temporelles sur les données historiques du Bitcoin.
Prévision multi-modèles (ARIMA, SARIMA, Prophet, XGBoost, LSTM) avec API FastAPI et dashboard Streamlit.

---

##  Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app)
[![FastAPI Docs](https://img.shields.io/badge/FastAPI-Docs-009688?logo=fastapi)](http://localhost:8000/docs)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)](./docker)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-Manifests-326CE5?logo=kubernetes)](./k8s)

---

##  Structure du Projet

```
bitcoin_timeseries/
│
├── notebooks/
│   └── Bitcoin_TimeSeries_Analysis.ipynb   # Analyse complète pas à pas
│
├── api/
│   └── main.py                              # API FastAPI (4 modèles)
│
├── streamlit_app/
│   └── app.py                               # Dashboard interactif
│
├── docker/
│   ├── Dockerfile                           # Image API
│   ├── Dockerfile.streamlit                 # Image Dashboard
│   └── docker-compose.yml                   # Stack complète
│
├── k8s/
│   └── manifests.yaml                       # Namespace, Deployment, Service, HPA, Ingress
│
├── data/
│   └── bitstampUSD_1-min_data_*.csv         # Dataset Kaggle (à télécharger)
│
├── models/
│   └── lstm_bitcoin.h5                      # Modèle LSTM sauvegardé
│
├── plots/
│   ├── 01_historique_complet.png
│   ├── 02_distribution_rendements.png
│   ├── 03_saisonnalite.png
│   ├── 04_acf_pacf.png
│   ├── 05_decomposition.png
│   ├── 06_arima_forecast.png
│   ├── 07_prophet_components.png
│   ├── 08_xgboost_feature_importance.png
│   ├── 09_lstm_results.png
│   └── 10_model_comparison.png
│
├── requirements.txt
└── README.md
```

---

##  Notebook — `Bitcoin_TimeSeries_Analysis.ipynb`

Le notebook documente **toute la démarche analytique** étape par étape.

| Section | Contenu |
|---|---|
| **0. Imports** | Configuration, palettes, répertoires |
| **1. Chargement** | Lecture CSV, audit qualité, rééchantillonnage journalier |
| **2. Nettoyage & FE** | Interpolation, 12+ nouvelles features (MA, RSI, Bollinger, logs) |
| **3. EDA** | Prix historique, volumes, rendements, heatmaps annuelles |
| **4. Stationnarité** | Tests ADF & KPSS, ACF/PACF, décomposition multiplicative |
| **5. ARIMA** | Auto-ARIMA, prévision 90j, intervalles de confiance |
| **6. SARIMA** | Saisonnalité hebdomadaire (m=7), comparaison avec ARIMA |
| **7. Prophet** | Tendance + saisonnalités multiples + changepoints |
| **8. XGBoost** | Features de lag + rolling + indicateurs techniques |
| **9. LSTM** | Architecture 3 couches, EarlyStopping, prévision récursive |
| **10. Comparaison** | MAE/RMSE/MAPE, scatter réel vs prédit, recommandations |

### Lancer le notebook

```bash
jupyter lab notebooks/Bitcoin_TimeSeries_Analysis.ipynb
```

---

##  API FastAPI — `api/main.py`

API REST avec 4 endpoints de prévision + historique + métriques.

### Endpoints

| Méthode | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Informations API |
| `GET` | `/health` | Healthcheck |
| `POST` | `/predict/arima` | Prévision ARIMA(1,1,1) |
| `POST` | `/predict/prophet` | Prévision Prophet |
| `POST` | `/predict/xgboost` | Prévision XGBoost (récursif) |
| `POST` | `/predict/lstm` | Prévision LSTM (seq=60) |
| `GET` | `/history` | Données historiques filtrées |
| `GET` | `/metrics` | Métriques de performance |

### Exemple d'appel

```python
import requests

payload = {
    "data": [
        {"date": "2021-01-01", "close": 29300},
        {"date": "2021-01-02", "close": 31700},
        # ... (minimum 30 points recommandé)
    ],
    "horizon": 30,
    "confidence": 0.95
}

response = requests.post("http://localhost:8000/predict/prophet", json=payload)
result   = response.json()

print(result["model"])          # "Prophet (multiplicative)"
print(result["metrics"])        # {"MAE": 1842, "RMSE": 2640, "MAPE_pct": 9.2}
for pred in result["predictions"][:3]:
    print(pred)                 # {"date": "2021-02-01", "forecast": 33200, "lower": ..., "upper": ...}
```

### Réponse type

```json
{
  "model": "Prophet (multiplicative)",
  "horizon": 30,
  "predictions": [
    {"date": "2021-02-01", "forecast": 33241.5, "lower": 28100.0, "upper": 38950.0},
    {"date": "2021-02-02", "forecast": 34102.3, "lower": 28800.0, "upper": 40050.0}
  ],
  "metrics": {"MAE": 1842.1, "RMSE": 2640.5, "MAPE_pct": 9.2},
  "metadata": {"n_obs": 2980, "last_price": 31700.0, "changepoints": 25}
}
```

---

##  Dashboard Streamlit — `streamlit_app/app.py`

Interface interactive avec thème sombre Bitcoin.

| Onglet | Contenu |
|---|---|
| **Historique** | Candlestick OHLC + MA + Bollinger + Volume |
| **EDA** | Distribution rendements, heatmap mensuelle, volatilité, DoW |
| **Indicateurs Techniques** | RSI, Bollinger, volatilité rolling (1 an) |
| **Prévisions** | Exécution multi-modèles, graphique interactif, export CSV |
| **Comparaison Modèles** | MAPE comparatif, radar chart, tableau récapitulatif |
| **API & Déploiement** | Instructions Docker, Kubernetes, exemples d'appels |

---

##  Dataset

**Source :** [Kaggle — Bitcoin Historical Data](https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data)

```bash
# Téléchargement via Kaggle CLI
kaggle datasets download mczielinski/bitcoin-historical-data
unzip bitcoin-historical-data.zip -d data/
```

| Variable | Description |
|---|---|
| `Timestamp` | Unix timestamp (données 1-minute) |
| `Open` | Prix d'ouverture |
| `High` | Prix le plus haut |
| `Low` | Prix le plus bas |
| `Close` | Prix de clôture |
| `Volume_(BTC)` | Volume en BTC |
| `Volume_(Currency)` | Volume en USD |
| `Weighted_Price` | Prix pondéré par volume |

Période : **2012-01-01 → 2021-03-31** (données à la minute, rééchantillonnées en journalier)

---

##  Features Créées (Feature Engineering)

| Feature | Formule |
|---|---|
| `Returns` | `Close.pct_change()` |
| `Log_Returns` | `log(Close / Close.shift(1))` |
| `MA_7/30/90` | Moyenne mobile 7, 30, 90 jours |
| `Volatility_30` | `std(Returns, 30) × √365` |
| `BB_Upper/Lower` | `MA_20 ± 2σ(Close, 20)` |
| `RSI` | Relative Strength Index (14j) |
| `lag_1/2/3/7/14/21/30` | Prix décalé (pour XGBoost) |
| `rolling_mean/std/max/min` | Stats glissantes (7, 14, 30j) |

---

##  Comparaison des Modèles

| Modèle | MAE ($) | RMSE ($) | MAPE (%) | Temps |
|---|---|---|---|---|
| ARIMA | ~2840 | ~3900 | ~14.2% | 2s |
| SARIMA | ~2710 | ~3720 | ~13.1% | 8s |
| Prophet | ~1950 | ~2800 | ~9.8% | 15s |
| XGBoost | ~1420 | ~2100 | ~7.2% | 5s |
| **LSTM** | **~1180** | **~1750** | **~5.9%** | **120s** |

>  Métriques indicatives sur la période de test (90 derniers jours). Les résultats varient selon la période choisie et le niveau de volatilité du marché.
---

##  Déploiement Docker

```bash
# 1. Build + Run stack complète
docker-compose -f docker/docker-compose.yml up --build

# API disponible sur      : http://localhost:8000/docs
# Dashboard disponible sur: http://localhost:8501
```

---

##  Déploiement Kubernetes

```bash
# 1. Build et push de l'image
docker build -t your-registry/bitcoin-ts-api:1.0.0 -f docker/Dockerfile .
docker push your-registry/bitcoin-ts-api:1.0.0

# 2. Déploiement
kubectl apply -f k8s/manifests.yaml

# 3. Vérification
kubectl get all -n bitcoin-ts

# 4. Autoscaling (CPU > 70% → scale up automatique jusqu'à 10 pods)
kubectl get hpa -n bitcoin-ts

# 5. Logs
kubectl logs -f deployment/bitcoin-ts-api -n bitcoin-ts
```

### Architecture Kubernetes

```
                    ┌─────────────────────────────────────────┐
                    │              Kubernetes Cluster          │
                    │  Namespace: bitcoin-ts                   │
                    │                                         │
  Internet ──── Ingress ──── Service (LB) ──── HPA           │
               (Nginx)       port 80           min:2          │
                                │              max:10         │
                         ┌──────┴──────┐                      │
                         │   Pod 1     │   Pod 2 ... Pod N    │
                         │  API:8000   │   (autoscale)        │
                         │  2Gi RAM    │                      │
                         │  1 vCPU     │                      │
                         └──────┬──────┘                      │
                                │                             │
                         ┌──────┴──────┐                      │
                         │    PVC      │                      │
                         │  /data (RO) │                      │
                         │  /models    │                      │
                         └─────────────┘                      │
                    └─────────────────────────────────────────┘
```

---

##  Installation Locale

```bash
# 1. Cloner le dépôt
git clone https://github.com/votre-username/bitcoin-timeseries.git
cd bitcoin-timeseries

# 2. Environnement virtuel
python -m venv venv
source venv/bin/activate        # Linux / Mac
venv\Scripts\activate           # Windows

# 3. Dépendances
pip install -r requirements.txt

# 4. Télécharger le dataset Kaggle
mkdir -p data
kaggle datasets download mczielinski/bitcoin-historical-data -p data/ --unzip

# 5. Lancer le notebook
jupyter lab notebooks/Bitcoin_TimeSeries_Analysis.ipynb

# 6. Lancer l'API (dans un terminal séparé)
uvicorn api.main:app --reload --port 8000
# → http://localhost:8000/docs

# 7. Lancer le dashboard (dans un terminal séparé)
streamlit run streamlit_app/app.py
# → http://localhost:8501
```

---

##  Compétences Acquises

| Compétence | Détail |
|---|---|
| **Stationnarité** | Tests ADF, KPSS, différenciation, log-transformation |
| **Modèles ARIMA/SARIMA** | Auto-paramétrage, prévision avec IC, évaluation |
| **Prophet** | Tendance + saisonnalités multiples + changepoints |
| **XGBoost sur séries temp.** | Feature engineering (lags, rolling), prévision récursive |
| **LSTM** | Séquences glissantes, architecture deep learning, EarlyStopping |
| **Évaluation** | MAE, RMSE, MAPE, comparaison multi-modèles |
| **API REST** | FastAPI, Pydantic, Swagger/ReDoc, gestion d'erreurs |
| **Docker** | Multi-stage build, Docker Compose, healthcheck |
| **Kubernetes** | Deployment, Service, HPA (autoscaling), Ingress, PVC |

---

##  Technologies

| Outil | Version | Usage |
|---|---|---|
| Python | 3.11 | Langage principal |
| Statsmodels | 0.14 | ARIMA, SARIMA, décomposition |
| pmdarima | 2.0 | Auto-ARIMA |
| Prophet | 1.1 | Prévision tendances/saisonnalités |
| XGBoost | 2.0 | Gradient Boosting |
| TensorFlow/Keras | 2.16 | LSTM |
| FastAPI | 0.111 | API REST |
| Uvicorn | 0.30 | Serveur ASGI |
| Streamlit | 1.35 | Dashboard |
| Plotly | 5.22 | Visualisations interactives |
| Docker | - | Containerisation |
| Kubernetes | - | Orchestration |

---

##  Licence

MIT — Libre d'utilisation pour tout projet éducatif ou professionnel.

---

##  Contribution

Les pull requests sont les bienvenues. Pour des changements majeurs, ouvrez d'abord une issue.

```bash
git checkout -b feature/ma-feature
git commit -m "feat: description de la feature"
git push origin feature/ma-feature
```
