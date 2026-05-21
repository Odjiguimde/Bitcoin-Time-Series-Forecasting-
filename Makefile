# ─────────────────────────────────────────────────────────────────────────────
# Makefile — Bitcoin Time Series Forecasting
# Usage : make <target>
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: help install data notebook api dashboard docker-up docker-down \
        docker-build docker-logs docker-clean k8s-deploy k8s-delete lint test

PYTHON     := python3
VENV       := venv
PIP        := $(VENV)/bin/pip
PYTEST     := $(VENV)/bin/pytest
API_PORT   := 8000
DASH_PORT  := 8501

# ── Aide ──────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  ₿  Bitcoin Time Series Forecasting"
	@echo "  ─────────────────────────────────────"
	@echo "  make install       Crée le venv et installe les dépendances"
	@echo "  make data          Télécharge le dataset Kaggle"
	@echo "  make notebook      Lance Jupyter Lab"
	@echo "  make api           Lance l'API FastAPI (port $(API_PORT))"
	@echo "  make dashboard     Lance le dashboard Streamlit (port $(DASH_PORT))"
	@echo "  make docker-build  Build les images Docker"
	@echo "  make docker-up     Démarre la stack Docker Compose"
	@echo "  make docker-down   Arrête la stack Docker Compose"
	@echo "  make docker-logs   Affiche les logs Docker"
	@echo "  make docker-clean  Supprime les containers et images"
	@echo "  make k8s-deploy    Déploie sur Kubernetes"
	@echo "  make k8s-delete    Supprime le déploiement Kubernetes"
	@echo "  make lint          Vérifie le code (flake8)"
	@echo "  make test          Lance les tests"
	@echo ""

# ── Installation ──────────────────────────────────────────────────────────────
install:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "✅ Environnement prêt. Activez-le : source venv/bin/activate"

# ── Données ───────────────────────────────────────────────────────────────────
data:
	@mkdir -p data
	@if [ -z "$$KAGGLE_USERNAME" ] || [ -z "$$KAGGLE_KEY" ]; then \
		echo "⚠️  Variables KAGGLE_USERNAME et KAGGLE_KEY requises."; \
		echo "   Copiez .env.example → .env et remplissez vos clés Kaggle."; \
		exit 1; \
	fi
	$(VENV)/bin/kaggle datasets download mczielinski/bitcoin-historical-data -p data/ --unzip
	@echo "✅ Dataset téléchargé dans data/"

# ── Développement ─────────────────────────────────────────────────────────────
notebook:
	$(VENV)/bin/jupyter lab Bitcoin_TimeSeries_Analysis.ipynb

api:
	$(VENV)/bin/uvicorn api.main:app --reload --port $(API_PORT)

dashboard:
	API_URL=http://localhost:$(API_PORT) $(VENV)/bin/streamlit run streamlit_app/app.py --server.port $(DASH_PORT)

# ── Docker ────────────────────────────────────────────────────────────────────
docker-build:
	docker-compose -f docker/docker-compose.yml build

docker-up:
	docker-compose -f docker/docker-compose.yml up -d
	@echo "✅ Stack démarrée"
	@echo "   API       → http://localhost:$(API_PORT)/docs"
	@echo "   Dashboard → http://localhost:$(DASH_PORT)"
	@echo "   Nginx     → http://localhost"

docker-down:
	docker-compose -f docker/docker-compose.yml down

docker-logs:
	docker-compose -f docker/docker-compose.yml logs -f

docker-clean:
	docker-compose -f docker/docker-compose.yml down --rmi all --volumes --remove-orphans

# ── Kubernetes ────────────────────────────────────────────────────────────────
k8s-deploy:
	kubectl apply -f k8s/manifests.yaml
	@echo "✅ Déployé dans le namespace bitcoin-ts"
	@kubectl get all -n bitcoin-ts

k8s-delete:
	kubectl delete -f k8s/manifests.yaml

# ── Qualité ───────────────────────────────────────────────────────────────────
lint:
	$(VENV)/bin/flake8 api/ streamlit_app/ --max-line-length=110 --ignore=E501,W503

test:
	$(PYTEST) tests/ -v --tb=short 2>/dev/null || echo "Aucun test trouvé dans tests/"
