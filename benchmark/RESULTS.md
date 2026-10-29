# Benchmark results

20 projects. A system flags a family only when it commits to it: a flagged check (rules), a confirmed finding (Muse + tools) or a listed flaw (Muse only). "Corrected" counts flawed projects whose patched evaluation landed within 0.02 of the answer key's correct metric.

## All projects

| System | Recall A | Recall B | Recall C | Precision | False alarms (clean) | Exact match | Corrected | Mean abs error of corrected metric | Failures | Median time | Mean cost |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Rules only | 5/5 | 3/4 | 4/4 | 12/12 | 0/7 | 19/20 | n/a |  | 0 | 9.3 s | $0.0000 |
| Muse + tools (auditor) | 5/5 | 3/4 | 4/4 | 12/12 | 0/7 | 19/20 | 13/13 | 0.0009 | 0 | 68.9 s | $0.0346 |
| Muse only (no tools) | 5/5 | 4/4 | 3/4 | 12/13 | 1/7 | 18/20 | n/a | 0.0678 | 1 | 13.4 s | $0.0106 |

## Development set

| System | Recall A | Recall B | Recall C | Precision | False alarms (clean) | Exact match | Corrected | Mean abs error of corrected metric | Failures | Median time | Mean cost |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Rules only | 4/4 | 2/3 | 2/2 | 8/8 | 0/5 | 13/14 | n/a |  | 0 | 8.8 s | $0.0000 |
| Muse + tools (auditor) | 4/4 | 2/3 | 2/2 | 8/8 | 0/5 | 13/14 | 9/9 | 0.0013 | 0 | 67.2 s | $0.0330 |
| Muse only (no tools) | 4/4 | 3/3 | 2/2 | 9/10 | 1/5 | 13/14 | n/a | 0.0709 | 0 | 14.4 s | $0.0104 |

## Held-out set

| System | Recall A | Recall B | Recall C | Precision | False alarms (clean) | Exact match | Corrected | Mean abs error of corrected metric | Failures | Median time | Mean cost |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Rules only | 1/1 | 1/1 | 2/2 | 4/4 | 0/2 | 6/6 | n/a |  | 0 | 17.0 s | $0.0000 |
| Muse + tools (auditor) | 1/1 | 1/1 | 2/2 | 4/4 | 0/2 | 6/6 | 4/4 | 0.0000 | 0 | 81.0 s | $0.0383 |
| Muse only (no tools) | 1/1 | 1/1 | 1/2 | 3/3 | 0/2 | 5/6 | n/a | 0.0585 | 1 | 8.9 s | $0.0111 |

## Per project

| Project | Split | Planted flaw (taxonomy) | Correct metric | Rules | Muse + tools | Patched metric | Muse only | Muse-only estimate |
|---|---|---|---|---|---|---|---|---|
| dev_a_imputer | dev | A (L1.2) | 0.7156 | A | confirmed A | 0.7156 | confirmed A | 0.7100 |
| dev_a_pandas_scaler | dev | A (L1.2) | 0.7633 | A | confirmed A | 0.7633 | confirmed A | 0.7600 |
| dev_a_scaler | dev | A (L1.2) | 0.7063 | A | confirmed A | 0.7063 | confirmed A | 0.7100 |
| dev_a_selector | dev | A (L1.3) | 0.7530 | A | confirmed A | 0.7530 | confirmed A | 0.7300 |
| dev_b_future_window | dev | B (L3.1) | 0.7400 | - | inconclusive - | 0.7517 | confirmed B | 0.6000 |
| dev_b_innocuous | dev | B (L2) | 0.7250 | B | confirmed B | 0.7250 | confirmed B | 0.6000 |
| dev_b_obvious | dev | B (L2) | 0.7864 | B | confirmed B | 0.7864 | confirmed B | 0.7300 |
| dev_c_groups | dev | C (L3.2) | 0.6813 | C | confirmed C | 0.6813 | confirmed C | 0.5000 |
| dev_c_time | dev | C (L3.1) | 0.7200 | C | confirmed C | 0.7200 | confirmed C | 0.6200 |
| dev_clean_group_split | dev | clean | 0.7250 | - | not_found - |  | not_found - | 0.7200 |
| dev_clean_label_encoder | dev | clean | 0.7955 | - | not_found - |  | confirmed A | 0.8000 |
| dev_clean_pipeline_cv | dev | clean | 0.7109 | - | not_found - |  | not_found - | 0.7100 |
| dev_clean_strong_feature | dev | clean | 0.7844 | - | inconclusive - |  | not_found - | 0.7800 |
| dev_clean_temporal | dev | clean | 0.7591 | - | not_found - |  | not_found - | 0.7600 |
| ho_a_selector | heldout | A (L1.3) | 0.5578 | A | confirmed A | 0.5578 | confirmed A | 0.6700 |
| ho_b_innocuous | heldout | B (L2) | 0.7333 | B | confirmed B | 0.7333 | confirmed B | 0.7500 |
| ho_c_groups | heldout | C (L3.2) | 0.6667 | C | confirmed C | 0.6667 | confirmed C | 0.6200 |
| ho_c_time | heldout | C (L3.1) | 0.7300 | C | confirmed C | 0.7300 | error |  |
| ho_clean_cv | heldout | clean | 0.6672 | - | inconclusive - |  | not_found - | 0.6700 |
| ho_clean_strong | heldout | clean | 0.7567 | - | not_found - |  | not_found - | 0.7600 |

## Consistency of repeated audits

| Project | Runs | Outcomes | Same every time |
|---|---|---|---|
| dev_b_innocuous | 3 | confirmed B; confirmed B; confirmed B | yes |
| dev_c_groups | 3 | confirmed C; confirmed C; confirmed C | yes |
| dev_clean_strong_feature | 3 | inconclusive -; inconclusive -; inconclusive - | yes |
| ho_c_time | 3 | confirmed C; confirmed C; confirmed C | yes |

## Notes

- Answer keys: benchmark/answers/. Held-out keys were committed with HELDOUT_LOCK.txt before any audit ran.
- Taxonomy codes follow Kapoor & Narayanan (2023), Leakage and the reproducibility crisis in machine-learning-based science.
- dev_a_pandas_scaler is included as a known limit: the tracer only sees scikit-learn.
- Some planted preprocessing flaws have almost no effect on the score (their correct metric equals the reported one); they are still flaws in procedure.
- How the auditor was developed against this benchmark (all on the development set; the held-out set was run once, at commit 96ab701, after the last change):
- Dev run 1 exposed: pandas-only preprocessing invisible to the tracer (check A now also reads the script); no patch proposed for leaky features (prompt: the fix is removing the feature); runs lost to network drops (longer retries); prediction_time wording that contradicted the data (reworded; data and answer keys unchanged).
- Dev run 2 exposed: a procedural flaw with no score effect left unconfirmed (prompt: confirm A and C flaws even when the effect is small); synthetic hospital data that was implausible, which Muse pointed out (regenerated); expired reasoning items in the kept history (left out).
- Dev run 3: check B confirmations calibrated to 'clearly not known at prediction time', otherwise inconclusive. This removed a false alarm and turned the 3-day window leak into an honest inconclusive.
- Runs that stopped on network outages were rerun; the table counts only completed audits.
