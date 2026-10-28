import json
import os
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# Predict failure_next_day for the machines data.
df = pd.read_csv('data.csv')
features = ['temperature', 'vibration', 'load_pct', 'hours_since_service', 'machine_age', 'ambient_temp', 'vibration_avg_3d']
X = df[features + ['line']]
X = pd.get_dummies(X, columns=['line'], drop_first=True, dtype=float)
y = df['failure_next_day']
cutoff = df['day'].quantile(0.75)
is_train = df['day'] <= cutoff
X_train, X_test = X[is_train], X[~is_train]
y_train, y_test = y[is_train], y[~is_train]
model = RandomForestClassifier(n_estimators=150, random_state=0)
model.fit(X_train, y_train)
pred = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]
rate = float(y_test.mean())
metrics = {
    'accuracy': float(accuracy_score(y_test, pred)),
    'precision': float(precision_score(y_test, pred, zero_division=0)),
    'recall': float(recall_score(y_test, pred)),
    'roc_auc': float(roc_auc_score(y_test, proba)),
    'majority_class_accuracy': max(rate, 1 - rate),
}

out = os.environ.get('AUDIT_OUTPUT_DIR', '.')
with open(os.path.join(out, 'metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=2)
print(json.dumps(metrics))
