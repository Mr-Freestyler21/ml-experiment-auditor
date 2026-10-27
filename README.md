# ML Experiment Auditor

Audits how a tabular scikit-learn experiment was evaluated: reruns it in a sandbox, runs
deterministic leakage checks, lets a Meta model investigate through tool calling, and
writes an evidence report. The model reasons, plain code measures.

Built for the Meta Global AI Developer Hackathon (Oct 26 to Nov 4, 2026).

## Status

- Phase 0: repo skeleton and one tool-calling request to the Meta Model API.
- Phase 1: sandbox runner (`auditor/sandbox`) and baseline reproduction. The F1 demo reproduces
  its reported 95% accuracy inside the sandbox.

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

## Tests

```bash
python -m pytest -q
```
