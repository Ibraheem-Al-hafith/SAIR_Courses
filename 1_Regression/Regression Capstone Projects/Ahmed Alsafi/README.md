# 🏥 Medical Insurance Cost Prediction

This project builds a machine learning model to predict medical insurance charges based on personal and health-related attributes. It demonstrates a complete ML workflow including preprocessing, training, evaluation, and model saving.

---

## 📌 Overview

The objective is to predict **insurance charges (`charges`)** using the following features:

- Age
- Sex
- BMI
- Number of children
- Smoking status
- Region

The project uses a **Random Forest Regressor** to model the relationship between these variables and insurance costs.

---

## 📂 Dataset

The dataset used is:

- `insurance.csv`

It contains structured tabular data with both numerical and categorical features commonly used in regression problems.

---

## ⚙️ Data Preprocessing

The preprocessing pipeline includes:

- Encoding categorical variables:
  - `sex`: male → 1, female → 0
  - `smoker`: yes → 1, no → 0

- One-hot encoding:
  - `region`

- Ensuring all features are numeric and ready for model input

---

## 🧠 Model

The model used in this project:

- `RandomForestRegressor` (from scikit-learn)

### Training Details:

- Train/Test Split: 80/20
- Random State: 42

---

## 📊 Evaluation

The model performance is evaluated using:

- **Mean Squared Error (MSE)**
- **R² Score**

These metrics help measure prediction accuracy and model fit.

---

## 📈 Experiment Tracking

Experiment tracking is handled using:

- `mlflow`

Tracked elements include:

- Model parameters
- Evaluation metrics (MSE, R²)
- Model artifacts

---

## 💾 Model Saving

The trained model is saved using:

```python
joblib.dump(model, "model.pkl")


pip install pandas numpy matplotlib seaborn scikit-learn mlflow joblib