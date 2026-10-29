# Audit report: clean

## 1. Verdict: **NOT_FOUND**

No evaluation flaw found. Baseline accuracy 0.765625 vs majority 0.68125 with roc_auc 0.7953318942255801 (E5). Preprocessing clear (E6) and grouped split has no patient overlap (E7). B dependence on num_prior_readmissions is strong but legitimate prior history known at discharge.

## 2. Reproduced baseline

| | |
|---|---|
| Reported accuracy | 0.7700 |
| Measured accuracy | 0.7656 (reproduced, difference -0.0044) |
| Seed | 0 |
| Environment | Python 3.14.3; numpy 2.4.3, pandas 3.0.1, scikit-learn 1.8.0 |
| Run log | [../20261029T190453Z-2d68e6](../20261029T190453Z-2d68e6) |

## 3. Context metrics

| Metric | Value |
|---|---|
| accuracy | 0.7656 |
| precision | 0.6517 |
| recall | 0.5686 |
| roc_auc | 0.7953 |
| majority_class_accuracy | 0.6813 |
| test_positive_rate (auditor) | 0.3187 |
| majority_class_accuracy (auditor) | 0.6813 |

Accuracy should be read against the majority-class baseline: always predicting the most common class scores that much without learning anything.

## 4. Findings

No evaluation flaw was confirmed.

## 5. Patch

No patch was applied.

## 6. Corrected evaluation

| Evaluation | accuracy | Change vs baseline | Majority baseline (test) | Target rate train / test | Shared groups | Checks after patch | Evidence |
|---|---|---|---|---|---|---|---|
| Original evaluation (rerun) | 0.7656 | +0.0000 | 0.6813 |  |  |  | E5 |

## 7. Limits and caveats

- Check B flagged dependence on num_prior_readmissions, but this column is prior history known at discharge per prediction-time definition, not future information, so not a leak (E8, E2)

## 8. Cost and time

Model `muse-spark-1.3`: 5 calls, 20210 tokens (cap 200000), estimated $0.0201. 5 agent steps in 48.1 s (stopped: finalized).

## Evidence index

| Id | Tool | Arguments | Result |
|---|---|---|---|
| E1 | list_files | `{}` | 4 files |
| E2 | read_file | `{"path": "train.py"}` | read train.py |
| E3 | read_file | `{"path": "README.md"}` | read README.md |
| E4 | read_file | `{"path": "data.csv"}` | read data.csv |
| E5 | run_baseline | `{}` | reproduced, measured 0.7656 |
| E6 | run_check | `{"check": "A"}` | clear: No preprocessing was fitted outside the evaluated models (e.g. all of it sits inside a pipeline). |
| E7 | run_check | `{"check": "C"}` | clear: 0 of 40 test groups by patient_id also appear in training (0.0% of test rows). |
| E8 | run_check | `{"check": "B", "suspects": [{"column": "num_prior_readmissions", "reason": "Str…` | flagged: Shuffling 'num_prior_readmissions' drops accuracy from 0.7656 to 0.6438 (+0.1219 lost), and it is suspected of not being known at prediction time (Str… |

Full evidence, the transcript and the verdict are in `evidence/`.
