import json
import os
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

# Predict failure_next_day for the machines data.
df = pd.read_csv('data.csv')
features = ['temperature', 'vibration', 'load_pct', 'hours_since_service', 'machine_age', 'ambient_temp']
X = df[features + ['line']]
X = pd.get_dummies(X, columns=['line'], drop_first=True, dtype=float)
y = df['failure_next_day']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
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
