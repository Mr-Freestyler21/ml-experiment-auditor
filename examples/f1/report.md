# Audit report: f1

## 1. Verdict: **CONFIRMED**

Reported accuracy does not hold for new races. Grouped Year+Race evaluation gives much lower accuracy near its baseline, confirming race-group leakage.

## 2. Reproduced baseline

| | |
|---|---|
| Reported accuracy | 0.9500 |
| Measured accuracy | 0.9520 (reproduced, difference +0.0020) |
| Seed | 42 |
| Environment | Python 3.14.3; numpy 2.4.3, pandas 3.0.1, scikit-learn 1.8.0 |
| Run log | [../20261028T194159Z-713da9](../20261028T194159Z-713da9) |

## 3. Context metrics

| Metric | Value |
|---|---|
| accuracy | 0.9520 |
| precision | 0.9209 |
| recall | 0.8189 |
| f1 | 0.8669 |
| roc_auc | 0.9850 |
| majority_class_accuracy | 0.8090 |
| test_positive_rate (auditor) | 0.1910 |
| majority_class_accuracy (auditor) | 0.8090 |

Accuracy should be read against the majority-class baseline: always predicting the most common class scores that much without learning anything.

## 4. Findings

### 4.1 [C] Random split leaks race groups; performance collapses on new races

*Same entities or later periods in train and test*

Random split shared every Year+Race between train and test. On the same 2024-only data, random-split accuracy stays high while group-split accuracy is much lower with similar target rates, isolating race-group leakage. Grouped evaluation clears the group overlap.

Offending code (`47`, `52`, `55`):

```python
  47  X_train, X_test, y_train, y_test = train_test_split(    # rows assigned to train and test
  52  model_v1.fit(X_train, y_train)    # model fitted on the training rows
  55  y_pred_v1 = model_v1.predict(X_test)    # model evaluated on the test rows
```

Evidence: **E7** run_check: flagged: 48 of 48 test groups by Year + Race also appear in training (100.0% of test rows); 47.2% of test rows have Year earlier than the latest training data (2024), so the model was trained on their future.; **E11** run_with_split: accuracy 0.9346 (change -0.0173); **E12** run_with_split: accuracy 0.7212 (change -0.2308); **E14** run_patched: accuracy 0.7054 (change -0.2466), checks after patch {'A': 'clear', 'C_groups': 'clear', 'C_time': 'flagged'}

## 5. Patch

**P1** on `train.py` (11 lines). Replace random row split with GroupShuffleSplit on Year+Race so no race appears in both train and test, matching real use on new races. Model, features and settings unchanged.

```diff
--- a/train.py
+++ b/train.py
@@ -20,7 +20,7 @@
     recall_score,
     roc_auc_score,
 )
-from sklearn.model_selection import train_test_split
+from sklearn.model_selection import GroupShuffleSplit
 from sklearn.preprocessing import LabelEncoder
 
 HERE = Path(__file__).resolve().parent
@@ -43,10 +43,11 @@
 # Cell 23
 X = ml_data[features]
 y = ml_data[target]
-
-X_train, X_test, y_train, y_test = train_test_split(
-    X, y, test_size=0.2, random_state=42
-)
+race_groups = df.loc[ml_data.index, 'Year'].astype(str) + '|' + df.loc[ml_data.index, 'Race']
+gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
+train_idx, test_idx = next(gss.split(X, y, groups=race_groups))
+X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
+y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
 
 model_v1 = RandomForestClassifier(n_estimators=100, random_state=42)
 model_v1.fit(X_train, y_train)
```

The original file is unchanged; the patch was applied to a copy (`evidence/patch_P1.diff`).

## 6. Corrected evaluation

| Evaluation | accuracy | Change vs baseline | Majority baseline (test) | Target rate train / test | Shared groups | Checks after patch | Evidence |
|---|---|---|---|---|---|---|---|
| Original evaluation (rerun) | 0.9520 | +0.0000 | 0.8090 |  |  |  | E3 |
| Group split by Year + Race | 0.7054 | -0.2466 | 0.6513 | 0.1514 / 0.3487 | 0 |  | E9 |
| Temporal split on Year | 0.6218 | -0.3302 | 0.6645 | 0.0308 / 0.3355 | 0 |  | E10 |
| Random split, only Year = 2024 | 0.9346 | -0.0173 | 0.6647 | 0.3355 / 0.3353 | 25 |  | E11 |
| Group split by Year + Race, only Year = 2024 | 0.7212 | -0.2308 | 0.6338 | 0.3277 / 0.3662 | 0 |  | E12 |
| Patched script (P1) | 0.7054 | -0.2466 | 0.6513 | 0.1514 / 0.3487 | 0 | A: clear, C_groups: clear, C_time: flagged | E14 |

## 7. Limits and caveats

- Temporal evaluation is confounded by label shift with very different train and test target rates, so it does not isolate leakage. (E10)
- Patched grouped split still mixes years in train and test, so residual forward-looking time overlap remains. (E14)

## 8. Cost and time

Model `muse-spark-1.3`: 8 calls, 57006 tokens (cap 200000), estimated $0.0515. 8 agent steps in 157.1 s (stopped: finalized).

## Evidence index

| Id | Tool | Arguments | Result |
|---|---|---|---|
| E1 | read_file | `{"path": "train.py"}` | read train.py |
| E2 | list_files | `{}` | 4 files |
| E3 | run_baseline | `{}` | reproduced, measured 0.9520 |
| E4 | read_file | `{"path": "data/laps.csv"}` | read data/laps.csv |
| E5 | read_file | `{"path": "README.md"}` | read README.md |
| E6 | run_check | `{"check": "A"}` | clear: No preprocessing that learns from data saw test rows before the evaluation. Note: LabelEncoder at line 41 was fitted on rows including the test set, but… |
| E7 | run_check | `{"check": "C"}` | flagged: 48 of 48 test groups by Year + Race also appear in training (100.0% of test rows); 47.2% of test rows have Year earlier than the latest training data … |
| E8 | run_check | `{"check": "B", "suspects": [{"column": "LapTime_Delta", "reason": "If computed …` | clear: Suspected column(s) 'RaceProgress', 'Cumulative_Degradation', 'LapTime_Delta' were measured and none shows the strong dependence a leak would cause. Shu… |
| E9 | run_with_split | `{"group_columns": ["Year", "Race"], "strategy": "group", "test_size": 0.2}` | accuracy 0.7054 (change -0.2466) |
| E10 | run_with_split | `{"strategy": "temporal", "test_size": 0.2, "time_column": "Year"}` | accuracy 0.6218 (change -0.3302) |
| E11 | run_with_split | `{"restrict_to": {"column": "Year", "value": 2024}, "strategy": "random", "test_…` | accuracy 0.9346 (change -0.0173) |
| E12 | run_with_split | `{"group_columns": ["Year", "Race"], "restrict_to": {"column": "Year", "value": …` | accuracy 0.7212 (change -0.2308) |
| E13 | propose_patch | `{"edits": [{"new": "from sklearn.model_selection import GroupShuffleSplit", "ol…` | accepted as P1 |
| E14 | run_patched | `{"patch_id": "P1"}` | accuracy 0.7054 (change -0.2466), checks after patch {'A': 'clear', 'C_groups': 'clear', 'C_time': 'flagged'} |

Full evidence, the transcript and the verdict are in `evidence/`.
