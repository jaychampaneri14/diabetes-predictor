# Diabetes Predictor

Predicts diabetes onset using clinical features (glucose, BMI, age, insulin, etc.) with a soft-voting ensemble classifier.

## Features
- Pima Indians Diabetes-inspired synthetic dataset
- Zero-value imputation with column medians
- Feature engineering (GlucoseBMI, BMI_Age, risk indicators)
- Ensemble: Logistic Regression + Random Forest + SVM
- Cross-validated AUC evaluation

## Setup

```bash
pip install -r requirements.txt
python main.py
```

## Output
- `feature_distributions.png` — histograms by diabetes status
- `roc_curve.png` — ROC curve for ensemble
- `diabetes_model.pkl` — saved model + scaler

## Clinical Features
Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
