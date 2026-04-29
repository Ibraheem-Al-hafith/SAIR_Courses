# Module 6 — MLOps: From Experiments to Production

> You can build the model. Now make it run reliably in the real world.

This module bridges the gap between a working Jupyter notebook and a production ML system. You will learn the tools, patterns, and discipline that separate a weekend project from a deployed product.

---

## Why MLOps?

A model that only runs on your laptop is not a product. MLOps is the engineering layer that turns ML work into software that ships, monitors, and improves itself over time.

| Without MLOps | With MLOps |
|---------------|------------|
| "It works on my machine" | Reproducible across any environment |
| Manual retraining | Automated pipelines triggered by data drift |
| No idea if the model degraded | Monitoring dashboards with alerts |
| Experiments scattered in notebooks | Tracked, versioned, comparable runs |
| Deploy once and hope | Staged rollouts with rollback capability |

---

## Module Structure

| # | Notebook | Topic | Time |
|---|----------|-------|------|
| 1 | `1_Experiment_Tracking.ipynb` | MLflow deep dive — logging, comparing, and registering runs | 3–4 hrs |
| 2 | `2_Packaging_and_Reproducibility.ipynb` | Docker, `uv`, pyproject.toml, environment pinning | 3–4 hrs |
| 3 | `3_Model_Serving.ipynb` | FastAPI inference server, REST endpoints, request validation | 4–5 hrs |
| 4 | `4_CI_CD_for_ML.ipynb` | GitHub Actions — lint, test, train, and deploy on every push | 4–5 hrs |
| 5 | `5_Monitoring_and_Drift.ipynb` | Data drift detection, model performance monitoring, alerting | 4–5 hrs |
| 6 | `6_Capstone_Pipeline.ipynb` | End-to-end MLOps pipeline from raw data to monitored deployment | 6–8 hrs |

**Total estimated time:** 24–31 hours (3–4 weeks alongside community participation)

---

## Learning Outcomes

By the end of this module you will be able to:

- Track, compare, and register ML experiments with **MLflow**
- Package any Python ML project into a reproducible **Docker** image
- Serve a trained model as a **REST API** with FastAPI
- Write a **GitHub Actions** workflow that trains and validates a model on every PR
- Detect **data drift** and set up model performance alerts
- Build a complete **end-to-end pipeline**: ingest → train → evaluate → serve → monitor

---

## Tools Covered

| Tool | Purpose |
|------|---------|
| MLflow | Experiment tracking, model registry |
| Docker | Reproducible environments |
| FastAPI | Model serving |
| GitHub Actions | CI/CD for ML pipelines |
| Evidently AI | Data drift and model monitoring |
| Prometheus + Grafana | Metrics and alerting |
| `uv` | Fast, deterministic Python packaging |

---

## Prerequisites

Complete all of the following before starting Module 6:

- [ ] Modules 0–5 complete
- [ ] Comfortable with `git`, `docker` basics, and REST APIs
- [ ] Missing Semester lectures: Shell, Git Internals, Metaprogramming (Make/CI)
- [ ] Read *Designing Machine Learning Systems* (Chip Huyen) — especially chapters on deployment and monitoring

---

## Notebook 1 — Experiment Tracking with MLflow

**File:** `1_Experiment_Tracking.ipynb`

### What you will build
A full MLflow tracking setup for a classification model — logging parameters, metrics, artifacts, and registering the best model to the Model Registry.

### Topics
- MLflow concepts: runs, experiments, artifacts, the Model Registry
- Auto-logging vs. manual logging — when to use each
- Comparing runs in the MLflow UI
- Promoting a model from "Staging" to "Production"
- Connecting MLflow to an existing pipeline (Module 2 or 3 pipeline)

### Key code patterns
```python
import mlflow

mlflow.set_experiment("sair-classification")

with mlflow.start_run(run_name="random_forest_v1"):
    mlflow.log_params({"n_estimators": 100, "max_depth": 5})
    mlflow.log_metric("val_accuracy", 0.94)
    mlflow.sklearn.log_model(model, "model")
```

---

## Notebook 2 — Packaging and Reproducibility

**File:** `2_Packaging_and_Reproducibility.ipynb`

### What you will build
A Dockerfile that reproducibly builds your ML project, plus a `pyproject.toml` that pins all dependencies.

### Topics
- Why `requirements.txt` breaks in 6 months and how to fix it
- `uv` deep dive: lock files, workspaces, editable installs
- Writing a Dockerfile for an ML project (CUDA base images, layer caching)
- Multi-stage builds: training image vs. serving image
- `docker-compose` for local dev with MLflow + model server

### Key Dockerfile pattern
```dockerfile
FROM python:3.12-slim AS base
RUN pip install uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

FROM base AS serve
COPY . .
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Notebook 3 — Model Serving with FastAPI

**File:** `3_Model_Serving.ipynb`

### What you will build
A production-grade FastAPI inference server that loads a trained model and serves predictions via a REST endpoint.

### Topics
- FastAPI fundamentals: routes, Pydantic models, async handlers
- Loading and caching a model at startup (avoid reload on every request)
- Input validation and error handling
- Health check and `/metrics` endpoints
- Testing the API with `httpx` and `pytest`

### Key code pattern
```python
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()
model = joblib.load("model.pkl")   # loaded once at startup

class PredictRequest(BaseModel):
    features: list[float]

@app.post("/predict")
def predict(req: PredictRequest):
    pred = model.predict([req.features])[0]
    return {"prediction": int(pred)}
```

---

## Notebook 4 — CI/CD for ML with GitHub Actions

**File:** `4_CI_CD_for_ML.ipynb`

### What you will build
A GitHub Actions workflow that runs linting, tests, and a fast training validation on every pull request.

### Topics
- GitHub Actions concepts: workflows, jobs, steps, runners
- Caching Python dependencies for fast CI runs
- Running `pytest` and `ruff` in CI
- Training a small model in CI and asserting accuracy thresholds
- Deploying to a staging environment on merge to `main`

### Key workflow pattern
```yaml
name: ML CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - run: uv sync
      - run: uv run pytest
      - run: uv run python train.py --fast-mode   # smoke test
```

---

## Notebook 5 — Monitoring and Drift Detection

**File:** `5_Monitoring_and_Drift.ipynb`

### What you will build
A monitoring pipeline that detects when incoming data has drifted from the training distribution, and alerts when model accuracy degrades.

### Topics
- Why models fail silently in production
- Data drift vs. concept drift — what each looks like
- Evidently AI: generating drift reports
- Setting up a Prometheus metrics endpoint in FastAPI
- Creating a Grafana dashboard for model health
- Alerting: when to retrain vs. when to roll back

---

## Notebook 6 — Capstone: End-to-End MLOps Pipeline

**File:** `6_Capstone_Pipeline.ipynb`

### What you will build
A complete, self-contained ML system that:
1. Loads and preprocesses data
2. Trains and evaluates a model (logged to MLflow)
3. Packages the best model into a Docker container
4. Serves predictions via FastAPI
5. Monitors for data drift and logs metrics to Prometheus

This is your Module 6 capstone submission.

### Submission Requirements
- Full pipeline runs end-to-end with a single command
- MLflow experiment with at least 3 tracked runs
- FastAPI server passing all provided integration tests
- Dockerfile that builds cleanly
- Evidently drift report on a held-out test set
- Written explanation (markdown cell) of every architectural decision

---

## Mandatory Reading

Read these chapters from *Designing Machine Learning Systems* (Chip Huyen) alongside this module:

| Chapter | When to Read |
|---------|-------------|
| Ch. 7 — Model Deployment and Prediction Service | Before Notebook 3 |
| Ch. 8 — Data Distribution Shifts and Monitoring | Before Notebook 5 |
| Ch. 9 — Continual Learning and Test in Production | Before Notebook 6 |

---

## Setup

All notebooks run locally with `uv`. Some exercises optionally use Docker.

```bash
# Install dependencies (includes fastapi, mlflow, evidently)
uv sync

# Start MLflow UI (run in a separate terminal)
mlflow ui --backend-store-uri mlruns

# Run the capstone pipeline
uv run python 6_capstone/run_pipeline.py
```

For Docker exercises: install [Docker Desktop](https://www.docker.com/products/docker-desktop/) or Docker Engine.

---

## Getting Unstuck

1. **MLflow UI not loading** — check `mlflow ui` is running and visit `http://127.0.0.1:5000`
2. **Docker build fails** — read the error from the top, not the bottom; layer errors are cumulative
3. **FastAPI 422 Unprocessable Entity** — your request body doesn't match the Pydantic schema; print the raw request
4. **CI test flaky** — add `--timeout` to pytest and check for resource limits on the runner
5. **Drift detected immediately** — verify train and test sets come from the same source; check for preprocessing leakage

Still stuck? Post in the [SAIR Telegram community](https://t.me/+jPPlO6ZFDbtlYzU0) with your error and a description of what you tried.

---

*SAIR MLOps Module — Building Sudan's AI engineers to production standard.*
