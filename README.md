# ML Experiment Auditor

Audits how a tabular scikit-learn experiment was evaluated: reruns it in a sandbox, runs
deterministic leakage checks, lets a Meta model investigate through tool calling, and
writes an evidence report. The model reasons, plain code measures.

Built for the Meta Global AI Developer Hackathon (Oct 26 to Nov 4, 2026).

## Status

- Phase 0: repo skeleton and one tool-calling request to the Meta Model API.
- Phase 1: sandbox runner (`auditor/sandbox`) and baseline reproduction. The F1 demo reproduces
  its reported 95% accuracy inside the sandbox.
- Phase 2: call tracing and the three deterministic checks: A (preprocessing fitted on test
  rows), B (future information in features, with Meta model suspects) and C (group and time
  overlap).
- Phase 3: the agent loop. The Meta model investigates with the checks as tools, measures
  corrected splits, and submits a verdict that code validates.

## Setup

```bash
py -3.14 -m venv .venv            # Python 3.14; pandas 3.0.1 has no 3.15 wheels yet
.venv\Scripts\activate          # Windows; use `source .venv/bin/activate` elsewhere
pip install -r requirements.txt
copy .env.example .env          # then put your key in .env
```

## Phase 0 smoke test

```bash
python scripts/phase0_tool_call.py --dry-run   # prints the request, no API call
python scripts/phase0_tool_call.py             # one real tool-calling round trip
```

## Reproduce a baseline

```bash
python -m auditor.cli baseline demo/f1
```

The project is copied into a temporary folder and run there with a scrubbed environment
(no API keys), its own Python environment (`.sandbox-venv`, built on first use from
`sandbox-requirements.txt`) and a timeout. Logs, outputs and the result are saved under
`runs/<run id>/`.

## Run the leakage checks

```bash
python -m auditor.cli check demo/f1
```

The script runs under a tracer that records which rows every scikit-learn `fit` and
`predict` call saw, and from which line. Check C then measures how many test rows come
from groups (`group_columns` in `audit.yaml`) that also appear in training, and whether
the model was trained on data later than some of its test rows (`time_column`). Check A
flags scalers, imputers, selectors and similar that were fitted on test rows before the
evaluation; encoders that only learn category names are noted, not flagged. Check B asks the
Meta model which features would not be known at prediction time (plus name and power rules),
then reruns the script with each suspect column shuffled to measure whether the result depends
on it. Use `--no-llm` for the rules-only baseline.

**Data note:** with Meta Model API free credits, content sent to the model may be used by Meta
for product improvement. The auditor sends column names, three example values per column and
the `prediction_time` sentence. Only audit projects you are comfortable sharing.

## Run a full audit

```bash
python -m auditor.cli audit demo/f1 --max-tokens 100000
```

The Meta model (Muse) reads the project, runs the checks, and re-evaluates the unchanged
script with corrected splits (`run_with_split` swaps its `train_test_split` for a grouped,
temporal or single-period split). It then submits a verdict: `confirmed`, `not_found` or
`inconclusive`. Code validates the verdict before accepting it: every claim must cite
evidence ids, `confirmed` needs a flagged check, and every number in the text must appear
in a tool result. The transcript, evidence and verdict are saved under `runs/audit-<time>/`.

## Tests

```bash
python -m pytest -q
```
