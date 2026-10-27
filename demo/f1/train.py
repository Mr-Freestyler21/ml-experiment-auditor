"""F1 pit-stop prediction: initial model with a random 80/20 split.

Adapted from f1.ipynb (cells 1, 21, 23, 25) in
github.com/Mr-Freestyler21/f1-pit-stop-prediction. The modelling code is unchanged,
including the random split, so the auditor has the original evaluation to examine.
Only the output changed: metrics are written to metrics.json in AUDIT_OUTPUT_DIR.
"""

import json
import os
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

HERE = Path(__file__).resolve().parent

# Cell 1
df = pd.read_csv(HERE / "data" / "laps.csv")

# Cell 21
features = ['Compound', 'TyreLife', 'Position', 'LapTime (s)',
            'RaceProgress', 'Stint', 'LapTime_Delta', 'Cumulative_Degradation']

target = 'PitNextLap'

ml_data = df[df['LapTime (s)'] <= 200][features + [target]].dropna()

le = LabelEncoder()
ml_data = ml_data.copy()
ml_data['Compound'] = le.fit_transform(ml_data['Compound'])

# Cell 23
X = ml_data[features]
y = ml_data[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model_v1 = RandomForestClassifier(n_estimators=100, random_state=42)
model_v1.fit(X_train, y_train)

# Cell 25
y_pred_v1 = model_v1.predict(X_test)

accuracy_v1 = accuracy_score(y_test, y_pred_v1)
print(f'Accuracy: {accuracy_v1:.2f}')
print()
print(classification_report(y_test, y_pred_v1))

# Added for the auditor: write metrics where the sandbox can collect them.
y_proba_v1 = model_v1.predict_proba(X_test)[:, 1]
positive_rate = float(y_test.mean())
metrics = {
    "accuracy": float(accuracy_v1),
    "precision": float(precision_score(y_test, y_pred_v1)),
    "recall": float(recall_score(y_test, y_pred_v1)),
    "f1": float(f1_score(y_test, y_pred_v1)),
    "roc_auc": float(roc_auc_score(y_test, y_proba_v1)),
    "majority_class_accuracy": max(positive_rate, 1 - positive_rate),
    "test_positive_rate": positive_rate,
    "n_train": int(len(X_train)),
    "n_test": int(len(X_test)),
}

out_dir = Path(os.environ.get("AUDIT_OUTPUT_DIR", HERE))
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
print(json.dumps(metrics, indent=2))
