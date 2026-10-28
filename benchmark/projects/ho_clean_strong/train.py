import json
import os
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# Predict high_usage for the energy data.
df = pd.read_csv('data.csv')
features = ['temperature_avg', 'occupancy', 'floor_area', 'usage_last_month', 'insulation_index']
X = df[features + ['building_type']]
X = pd.get_dummies(X, columns=['building_type'], drop_first=True, dtype=float)
y = df['high_usage']
cutoff = df['month'].quantile(0.75)
is_train = df['month'] <= cutoff
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
