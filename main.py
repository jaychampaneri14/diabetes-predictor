"""
Diabetes Predictor
Uses Pima Indians Diabetes Dataset features to predict diabetes onset.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve)
import joblib
import warnings
warnings.filterwarnings('ignore')


def generate_pima_like_dataset(n=768, seed=42):
    """Generate Pima Indians Diabetes-like dataset."""
    np.random.seed(seed)
    # Approximate Pima dataset distributions
    n_pos = int(n * 0.349)
    n_neg = n - n_pos

    def gen_class(n_samples, diabetic=False):
        mult = 1.3 if diabetic else 1.0
        pregnancies  = np.random.poisson(3.8 * mult, n_samples).clip(0, 17)
        glucose      = np.random.normal(141 * mult if diabetic else 110, 30, n_samples).clip(0, 200)
        blood_press  = np.random.normal(70, 12, n_samples).clip(0, 122)
        skin_thick   = np.random.normal(33 * mult if diabetic else 26, 15, n_samples).clip(0, 99)
        insulin      = np.random.exponential(100 * mult if diabetic else 70, n_samples).clip(0, 846)
        bmi          = np.random.normal(35 * mult if diabetic else 30, 7, n_samples).clip(0, 67)
        dpf          = np.random.exponential(0.55 * mult if diabetic else 0.4, n_samples).clip(0.07, 2.42)
        age          = np.random.normal(37 * mult if diabetic else 30, 12, n_samples).clip(21, 81).astype(int)
        outcome      = np.ones(n_samples, dtype=int) if diabetic else np.zeros(n_samples, dtype=int)
        return np.column_stack([pregnancies, glucose, blood_press, skin_thick,
                                insulin, bmi, dpf, age, outcome])

    pos = gen_class(n_pos, diabetic=True)
    neg = gen_class(n_neg, diabetic=False)
    data = np.vstack([pos, neg])
    np.random.shuffle(data)
    cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    return pd.DataFrame(data, columns=cols)


def handle_zeros(df):
    """Replace physiologically impossible zeros with median."""
    df = df.copy()
    zero_replace_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in zero_replace_cols:
        median = df[col][df[col] != 0].median()
        df[col] = df[col].replace(0, median)
    return df


def feature_engineering(df):
    """Add derived features."""
    df = df.copy()
    df['GlucoseBMI']       = df['Glucose'] * df['BMI']
    df['AgePregnancies']   = df['Age'] * df['Pregnancies']
    df['InsulinGlucose']   = df['Insulin'] / df['Glucose'].replace(0, 1)
    df['BMI_Age']          = df['BMI'] / df['Age']
    df['GlucoseHighRisk']  = (df['Glucose'] > 140).astype(int)
    df['BMIObese']         = (df['BMI'] > 30).astype(int)
    df['AgeRisk']          = (df['Age'] > 45).astype(int)
    return df


def plot_feature_distributions(df, save_path='feature_distributions.png'):
    features = ['Glucose', 'BMI', 'Age', 'Insulin', 'BloodPressure', 'Pregnancies']
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, feat in zip(axes.ravel(), features):
        df[df['Outcome'] == 0][feat].hist(ax=ax, alpha=0.5, color='green', label='No Diabetes', bins=20)
        df[df['Outcome'] == 1][feat].hist(ax=ax, alpha=0.5, color='red',   label='Diabetes',    bins=20)
        ax.set_title(feat)
        ax.legend(fontsize=7)
    plt.suptitle('Feature Distributions by Diabetes Status')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def train_ensemble(X_train, X_test, y_train, y_test):
    """Train an ensemble voting classifier."""
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)

    lr  = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
    rf  = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    svc = SVC(probability=True, C=1.0, kernel='rbf', random_state=42)

    # Individual scores
    for name, clf in [('LogReg', lr), ('RF', rf), ('SVM', svc)]:
        clf.fit(X_tr_s, y_train)
        score = roc_auc_score(y_test, clf.predict_proba(X_te_s)[:, 1])
        print(f"  {name}: AUC = {score:.4f}")

    # Ensemble
    ensemble = VotingClassifier(
        estimators=[('lr', LogisticRegression(C=0.1, max_iter=1000, random_state=42)),
                    ('rf', RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)),
                    ('svc', SVC(probability=True, C=1.0, kernel='rbf', random_state=42))],
        voting='soft'
    )
    ensemble.fit(X_tr_s, y_train)
    y_prob = ensemble.predict_proba(X_te_s)[:, 1]
    y_pred = (y_prob >= 0.45).astype(int)
    auc = roc_auc_score(y_test, y_prob)
    print(f"\n  Ensemble: AUC = {auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=['No Diabetes', 'Diabetes']))
    return ensemble, scaler, y_pred, y_prob


def plot_roc(y_test, y_prob, save_path='roc_curve.png'):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, lw=2, color='darkorange', label=f'Ensemble (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Diabetes Predictor')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    print("=" * 60)
    print("DIABETES PREDICTOR")
    print("=" * 60)

    df = generate_pima_like_dataset(768)
    df = handle_zeros(df)
    df = feature_engineering(df)
    print(f"Dataset: {len(df)} patients, {df['Outcome'].mean():.1%} diabetic")

    plot_feature_distributions(df)

    feat_cols = [c for c in df.columns if c != 'Outcome']
    X = df[feat_cols].values
    y = df['Outcome'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print("\n--- Training Ensemble ---")
    model, scaler, y_pred, y_prob = train_ensemble(X_train, X_test, y_train, y_test)
    plot_roc(y_test, y_prob)

    # CV
    from sklearn.preprocessing import StandardScaler as SS
    sc = SS(); Xs = sc.fit_transform(X)
    cv_scores = cross_val_score(model, Xs, y, cv=StratifiedKFold(5), scoring='roc_auc')
    print(f"\nCross-val AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    joblib.dump({'model': model, 'scaler': scaler, 'features': feat_cols}, 'diabetes_model.pkl')
    print("Model saved to diabetes_model.pkl")

    # Example prediction
    patient = [6, 148, 72, 35, 0, 33.6, 0.627, 50,
               148*33.6, 6*50, 0/148, 33.6/50, 1, 1, 1]
    patient_scaled = scaler.transform([patient[:len(feat_cols)]])
    risk = model.predict_proba(patient_scaled)[0, 1]
    print(f"\nExample patient diabetes risk: {risk:.1%}")
    print("\n✓ Diabetes Predictor complete!")


if __name__ == '__main__':
    main()
