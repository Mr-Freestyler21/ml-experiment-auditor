import json
import os
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Predict readmitted_30d for the hospital data.
df = pd.read_csv('data.csv')
features = ['age', 'num_prior_visits', 'length_of_stay', 'lab_glucose', 'home_distance_km']
X = df[features + ['ward']]
X = pd.get_dummies(X, columns=['ward'], drop_first=True, dtype=float)
y = df['readmitted_30d']
pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
scores = cross_val_score(pipeline, X, y, groups=df['patient_id'], cv=GroupKFold(n_splits=5), scoring='accuracy')
rate = float(y.mean())
metrics = {'accuracy': float(scores.mean()), 'fold_scores': [float(s) for s in scores],
           'majority_class_accuracy': max(rate, 1 - rate)}

out = os.environ.get('AUDIT_OUTPUT_DIR', '.')
with open(os.path.join(out, 'metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=2)
print(json.dumps(metrics))
