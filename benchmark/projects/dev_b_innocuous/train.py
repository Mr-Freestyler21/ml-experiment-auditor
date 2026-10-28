import json
import os
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

# Predict readmitted_30d for the hospital data.
df = pd.read_csv('data.csv')
features = ['age', 'num_prior_visits', 'length_of_stay', 'lab_glucose', 'home_distance_km', 'code_7b']
X = df[features + ['ward']]
X = pd.get_dummies(X, columns=['ward'], drop_first=True, dtype=float)
y = df['readmitted_30d']
splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0)
train_idx, test_idx = next(splitter.split(X, y, groups=df['patient_id']))
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
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
