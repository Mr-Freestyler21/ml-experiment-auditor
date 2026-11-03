# Benchmark results

20 projects. A system flags a family only when it commits to it: a flagged check (rules), a confirmed finding (Muse + tools) or a listed flaw (Muse only). "Corrected" counts flawed projects whose patched evaluation landed within 0.02 of the answer key's correct metric.

## All projects

| System | Recall A | Recall B | Recall C | Precision | False alarms (clean) | Exact match | Corrected | Mean abs error of corrected metric | Failures | Median time | Mean cost |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Rules only | 5/5 | 3/4 | 4/4 | 12/12 | 0/7 | 19/20 | n/a |  | 0 | 9.9 s | $0.0000 |
| Muse + tools (auditor) | 5/5 | 4/4 | 4/4 | 13/13 | 0/7 | 20/20 | 13/13 | 0.0009 | 0 | 181.3 s | $0.0736 |
| Muse only (no tools) | 5/5 | 4/4 | 4/4 | 13/13 | 0/7 | 20/20 | n/a | 0.0549 | 0 | 19.8 s | $0.0106 |

## Development set

| System | Recall A | Recall B | Recall C | Precision | False alarms (clean) | Exact match | Corrected | Mean abs error of corrected metric | Failures | Median time | Mean cost |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Rules only | 4/4 | 2/3 | 2/2 | 8/8 | 0/5 | 13/14 | n/a |  | 0 | 9.9 s | $0.0000 |
| Muse + tools (auditor) | 4/4 | 3/3 | 2/2 | 9/9 | 0/5 | 14/14 | 9/9 | 0.0013 | 0 | 193.6 s | $0.0797 |
| Muse only (no tools) | 4/4 | 3/3 | 2/2 | 9/9 | 0/5 | 14/14 | n/a | 0.0509 | 0 | 19.8 s | $0.0107 |

## Held-out set

| System | Recall A | Recall B | Recall C | Precision | False alarms (clean) | Exact match | Corrected | Mean abs error of corrected metric | Failures | Median time | Mean cost |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Rules only | 1/1 | 1/1 | 2/2 | 4/4 | 0/2 | 6/6 | n/a |  | 0 | 8.8 s | $0.0000 |
| Muse + tools (auditor) | 1/1 | 1/1 | 2/2 | 4/4 | 0/2 | 6/6 | 4/4 | 0.0000 | 0 | 147.7 s | $0.0593 |
| Muse only (no tools) | 1/1 | 1/1 | 2/2 | 4/4 | 0/2 | 6/6 | n/a | 0.0639 | 0 | 20.3 s | $0.0102 |

## Per project

| Project | Split | Planted flaw (taxonomy) | Correct metric | Rules | Muse + tools | Patched metric | Muse only | Muse-only estimate |
|---|---|---|---|---|---|---|---|---|
| dev_a_imputer | dev | A (L1.2) | 0.7156 | A | confirmed A | 0.7156 | confirmed A | 0.7100 |
| dev_a_pandas_scaler | dev | A (L1.2) | 0.7633 | A | confirmed A | 0.7633 | confirmed A | 0.7600 |
| dev_a_scaler | dev | A (L1.2) | 0.7063 | A | confirmed A | 0.7063 | confirmed A | 0.7100 |
| dev_a_selector | dev | A (L1.3) | 0.7530 | A | confirmed A | 0.7530 | confirmed A | 0.7400 |
| dev_b_future_window | dev | B (L3.1) | 0.7400 | - | confirmed B | 0.7517 | confirmed B | 0.6200 |
| dev_b_innocuous | dev | B (L2) | 0.7250 | B | confirmed B | 0.7250 | confirmed B | 0.6500 |
| dev_b_obvious | dev | B (L2) | 0.7864 | B | confirmed B | 0.7864 | confirmed B | 0.7800 |
| dev_c_groups | dev | C (L3.2) | 0.6813 | C | confirmed C | 0.6813 | confirmed C | 0.5500 |
| dev_c_time | dev | C (L3.1) | 0.7200 | C | confirmed C | 0.7200 | confirmed C | 0.6200 |
| dev_clean_group_split | dev | clean | 0.7250 | - | not_found - |  | not_found - | 0.7200 |
| dev_clean_label_encoder | dev | clean | 0.7955 | - | not_found - |  | not_found - | 0.8000 |
| dev_clean_pipeline_cv | dev | clean | 0.7109 | - | not_found - |  | not_found - | 0.7100 |
| dev_clean_strong_feature | dev | clean | 0.7844 | - | inconclusive - |  | not_found - | 0.7800 |
| dev_clean_temporal | dev | clean | 0.7591 | - | not_found - |  | not_found - | 0.7600 |
| ho_a_selector | heldout | A (L1.3) | 0.5578 | A | confirmed A | 0.5578 | confirmed A | 0.6700 |
| ho_b_innocuous | heldout | B (L2) | 0.7333 | B | confirmed B | 0.7333 | confirmed B | 0.7800 |
| ho_c_groups | heldout | C (L3.2) | 0.6667 | C | confirmed C | 0.6667 | confirmed C | 0.6200 |
| ho_c_time | heldout | C (L3.1) | 0.7300 | C | confirmed C | 0.7300 | confirmed C | 0.6800 |
| ho_clean_cv | heldout | clean | 0.6672 | - | not_found - |  | not_found - | 0.6700 |
| ho_clean_strong | heldout | clean | 0.7567 | - | not_found - |  | not_found - | 0.7600 |

## Consistency of repeated audits

| Project | Runs | Outcomes | Same every time |
|---|---|---|---|
| dev_b_innocuous | 3 | confirmed B; confirmed B; confirmed B | yes |
| dev_c_groups | 3 | confirmed C; confirmed C; confirmed C | yes |
| dev_clean_strong_feature | 3 | inconclusive -; not_found -; confirmed B | no |
| ho_c_time | 3 | confirmed C; confirmed C; confirmed C | yes |

## Notes

- Answer keys: benchmark/answers/. Held-out keys were committed with HELDOUT_LOCK.txt before any audit ran.
- Taxonomy codes follow Kapoor & Narayanan (2023), Leakage and the reproducibility crisis in machine-learning-based science.
- dev_a_pandas_scaler is included as a known limit: the tracer only sees scikit-learn.
- Some planted preprocessing flaws have almost no effect on the score (their correct metric equals the reported one); they are still flaws in procedure.
- How the auditor was developed against this benchmark. Version 1 (checks A to C) was tuned on the development set only and run once on the held-out set at commit 96ab701, scoring 6/6. Version 2 added the deep review (code review of every script, checks D and E, whole-pipeline fixes, structured reports), developed on a separate pump-failure project. Its first full run (dev_v2, heldout_v2) had false alarms on two clean projects (one development, one held-out): Muse confirmed strong legitimate features as leaks on hedged reasons. Check B confirmations were recalibrated on the development set only (dev_v3, dev_v4: hedged reasons become suspicions; a strong feature with a legitimate reading is not one), a timed-out traced run is now retried once, and the held-out set was then run once more on the final version. The table shows the final version; the earlier held-out run is reported here, not hidden.
- In dev_v4, dev_b_innocuous lost every check to one traced run that timed out on a busy machine (Muse still named the leak, as a suspicion). After the retry fix it was rerun alone (dev_v5_rerun); the table uses that run. Rules only and Muse only are unchanged since dev_v2 and heldout_v2 and come from those runs.
- Repeats show the remaining weakness: on dev_clean_strong_feature (a strong, legitimate feature placed as bait) three audits gave inconclusive, not found, and once a wrong confirmed B. Version 1 gave inconclusive three times. Deciding whether a strong feature is known at prediction time, without code that shows how it was built, is judgement, and the judgement still varies.
- Version 1 history:
- Dev run 1 exposed: pandas-only preprocessing invisible to the tracer (check A now also reads the script); no patch proposed for leaky features (prompt: the fix is removing the feature); runs lost to network drops (longer retries); prediction_time wording that contradicted the data (reworded; data and answer keys unchanged).
- Dev run 2 exposed: a procedural flaw with no score effect left unconfirmed (prompt: confirm A and C flaws even when the effect is small); synthetic hospital data that was implausible, which Muse pointed out (regenerated); expired reasoning items in the kept history (left out).
- Dev run 3: check B confirmations calibrated to 'clearly not known at prediction time', otherwise inconclusive. This removed a false alarm and turned the 3-day window leak into an honest inconclusive.
- Runs that stopped on network outages were rerun; the table counts only completed audits.
