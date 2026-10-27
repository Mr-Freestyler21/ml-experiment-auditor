# ML Experiment Auditor

Audits how a tabular scikit-learn experiment was evaluated: reruns it in a sandbox, runs
deterministic leakage checks, lets a Meta model investigate through tool calling, and
writes an evidence report. The model reasons, plain code measures.

Built for the Meta Global AI Developer Hackathon (Oct 26 to Nov 4, 2026).

## Status

Phase 0: repo skeleton and one tool-calling request to the Meta Model API.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; use `source .venv/bin/activate` elsewhere
pip install -r requirements.txt
copy .env.example .env          # then put your key in .env
```

## Phase 0 smoke test

```bash
python scripts/phase0_tool_call.py --dry-run   # prints the request, no API call
python scripts/phase0_tool_call.py             # one real tool-calling round trip
```
