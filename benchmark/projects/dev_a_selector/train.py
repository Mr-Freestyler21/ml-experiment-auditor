import json
import os
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# Predict churned for the churn data.
df = pd.read_csv('data.csv')
features = ['tenure_weeks', 'spend_last_4w', 'support_tickets', 'discount_rate', 'region_code']
X = df[features + ['plan']]
X = pd.get_dummies(X, columns=['plan'], drop_first=True, dtype=float)
y = df['churned']
X = pd.DataFrame(SelectKBest(f_classif, k=5).fit_transform(X, y), index=X.index)
cutoff = df['week'].quantile(0.75)
is_train = df['week'] <= cutoff
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
