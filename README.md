# ML Experiment Auditor

**An auditor for machine learning results.** It reruns a scikit-learn experiment in a sandbox,
finds evaluation flaws that inflate the score, fixes the evaluation, reruns it, and writes an
evidence report. A Meta model (Muse) leads the investigation through tool calling; plain code
does every measurement. *The model reasons, code measures.*

![The F1 audit report: confirmed, with the original 0.952 next to every corrected evaluation](docs/report_f1.png)

## Why

My F1 model predicted pit stops with **95% accuracy**. Trained on 2023 and tested on 2024, it
scored 62%. That drop alone proves nothing: pit laps were 3.1% of 2023 and 33.5% of 2024, so
the label mix changed too. Something was wrong, but what, and how much did it matter?

Leakage like this is common in published work, not only in side projects (Kapoor & Narayanan,
2023, *Leakage and the reproducibility crisis in machine-learning-based science*). The auditor
answers the question with evidence instead of suspicion. On the F1 project it found that every
race in the test set was also in the training set, separated that from the label shift (on 2024
alone: 0.935 with a random split, 0.721 with races kept apart), wrote a fix, and measured the
corrected score: **0.705**.

## Try it

```bash
python -m auditor.cli demo --replay   # no API key: replays two recorded audits, opens the reports
python -m auditor.cli demo            # live: Muse audits both projects (~4 min, ~$0.07)
```

Or use the local web page: pick a demo or upload your own project as a .zip, watch Muse's
investigation step by step, and read the report when it finishes. It runs only on your computer
(127.0.0.1), one audit at a time.

```bash
python -m auditor.cli serve           # opens http://127.0.0.1:8000
```

On Windows you can also double-click `start_web.bat`, or in VS Code use
**Terminal > Run Task > ML Auditor: open web page**.

Or read the results directly: [F1 report](examples/f1/report.md) (confirmed) and
[clean project report](examples/clean/report.md) (not found). Each folder also has a
`report.html`: download it and open it in a browser.

## How it works

```
 your project ──> sandbox rerun ──> tracer ──> checks A to E + R ┐
 (script, CSV,   (copy, no keys,   (which rows     (plain code)     │  tool results,
  audit.yaml)     timeout)          each fit and                    │  each an evidence id
                                    predict saw)                    v
                                                  Muse decides what to look at next,
                                                  re-evaluates with corrected splits,
                                                  proposes a fix ──> code validates it,
                                                  runs it, rechecks ──> verdict ──> code
                                                  validates the verdict ──> report
```

- **Sandbox.** Your project runs in a temporary copy with a scrubbed environment (no API keys),
  its own Python environment and a timeout. Original files are never modified.
- **Tracer.** A launcher wraps scikit-learn's `fit`, `predict`, `transform` and
  `train_test_split` and records which rows each call saw and from which line of your script.
  It never changes the result: every check also reruns the script untraced and compares.
- **Checks (deterministic).**
  **R** a code review of every `.py` file, data preparation included: centred windows, shifts
  into the future, resampling before the split, statistics over all rows, whole-period
  aggregates, unchecked joins, metrics computed inside a selection loop. Each is a lead with
  file and line, verified by the measuring checks.
  **A** preprocessing fitted on rows that include the test set (scikit-learn objects through the
  trace; plain-pandas statistics by reading the script).
  **B** features not known at prediction time: Muse proposes suspects from their meaning and
  the `prediction_time` sentence, code measures each with a shuffle rerun under an honest split
  (or on top of accepted fixes, so one leak cannot hide another).
  **C** the same entities (patients, races) or later periods in both train and test.
  **D** the same rows on both sides: copies made by oversampling before the split, duplicated
  records, duplicated keys.
  **E** choices made on the test rows: several candidate models or thresholds scored on them.
- **Agent.** Muse calls tools (`read_file`, `run_baseline`, `run_check`, `run_ablation`,
  `run_with_split`, `propose_patch`, `run_patched`, `finalize`). `run_with_split` reruns your
  unchanged script with its split swapped for a grouped, temporal or single-period one. Fixes
  can go into the data preparation script too: the patched pipeline is rerun from the raw data.
- **Guardrails.** Every claim must cite evidence ids; `confirmed` needs a flagged check; every
  number in Muse's text must appear in a tool result (an invented number is rejected and sent
  back). A fix may change the evaluation only: code rejects any change to how the model is
  built. As the step or token budget runs low, Muse is told to wrap up and finalize; a cap that
  is still reached ends the audit as `inconclusive`, never as a guess.
- **Report.** `report.md`, `report.json` and a self-contained `report.html`, built by code from
  the evidence, plus the full transcript. Findings are ranked by severity with their measured
  impact, followed by suspicions that are not proven, how far to trust the corrected score
  (an upper bound while known flaws remain), the code review table and concrete recommendations.

## Results

A benchmark of 20 generated projects (five themes, flaws labelled by type in the Kapoor &
Narayanan taxonomy, plus clean controls), comparing three systems. Six projects are held out:
their answer keys were committed with a fingerprint ([HELDOUT_LOCK.txt](benchmark/HELDOUT_LOCK.txt))
before any audit ran on them, and they were run once. Full table: [benchmark/RESULTS.md](benchmark/RESULTS.md).

| | Rules only | **Muse + tools (this auditor)** | Muse only (no tools) |
|---|---|---|---|
| Exact answers | 19/20 | **19/20** | 18/20 |
| False alarms on 7 clean projects | 0 | **0** | 1 |
| Fixes that reached the correct score | n/a | **13/13** | n/a |
| Mean error of the corrected score | n/a | **0.0009** | 0.068 (estimated) |
| Held-out set, exact answers | 6/6 | **6/6** | 5/6 |
| Same verdict in 3 repeated audits (4 projects) | n/a | **12/12** | n/a |
| Cost per audit | free | $0.035 | $0.011 |

What this shows, plainly: on these projects, fixed rules detect flaws as well as the auditor.
What only the auditor does is measure the truth: it writes the fix, reruns it and reports the
corrected score (off by 0.0009 on average), with every claim tied to evidence. Muse without
tools spots most flaws, but its estimates of the real score are off by about 7 points and it
accused one honest project.

## Limitations

- Tabular projects in Python with pandas and scikit-learn, one CSV, a run of a few minutes.
- Row identity comes from pandas indexes; scripts that only use plain arrays get weaker evidence
  (row counts) or `not_applicable`.
- Check B measures dependence, not meaning. A future-derived feature whose effect is small, or
  hidden by other leaks, stays "suspected" unless the code review shows how it was built.
- A project with many interacting flaws can use the whole budget before every fix is written.
  On the development project used for this (a pump-failure pipeline with a dozen planted
  flaws) the audit confirmed one flaw in each of the five families, listed three more as
  suspected, and reported the temporal score as an upper bound instead of a corrected score.
- The benchmark projects are synthetic. They test the method, not every real-world codebase.
- Data note: with Meta Model API free credits, content sent to the model may be used by Meta for
  product improvement. The auditor sends the script, column names, a few example values and the
  `prediction_time` sentence. Only audit projects you are comfortable sharing.

## Setup

Python 3.14 (pandas 3.0.1 has no wheels for 3.15 yet).

```bash
git clone https://github.com/Mr-Freestyler21/ml-experiment-auditor.git
cd ml-experiment-auditor
py -3.14 -m venv .venv          # macOS/Linux: python3.14 -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m auditor.cli demo --replay
```

For live audits, copy `.env.example` to `.env` and set `META_MODEL_API_KEY` (a Meta Model API
key) and `AUDITOR_MAX_TOKENS` (a per-audit cap; 200000 is plenty).

## Commands

```bash
python -m auditor.cli baseline demo/f1    # rerun in the sandbox, compare with the reported score
python -m auditor.cli check demo/f1       # the deterministic checks (add --no-llm for rules only)
python -m auditor.cli audit demo/f1       # full audit; writes runs/audit-<time>/report.{md,json,html}
python -m auditor.cli demo [--replay]     # the demo
python -m auditor.cli serve               # local web page: pick or upload a project, watch it live
python -m pytest -q                       # tests (no API calls; the model is scripted)
```

To audit your own project, add an `audit.yaml` next to the script (see
[demo/f1/audit.yaml](demo/f1/audit.yaml)): the entry script, data file, target, the reported
metric and value, a `prediction_time` sentence, and optionally `group_columns` (entities that
will be new in real use) and `time_column`. The script must write `metrics.json` to the folder
in `AUDIT_OUTPUT_DIR`.

Or let the converter do it, from a notebook, a script, a folder or a zip:

```bash
python -m auditor.cli convert path/to/analysis.ipynb     # writes path/to/analysis-audit/
```

Plain code, no model: it keeps the notebook's first complete experiment (up to the first cell
that scores a fitted model; `--cells N` to change), removes plots and display-only lines, copies
the CSV in and fixes its path, detects the target, adds a marked block that writes
`metrics.json`, and runs the result once in the sandbox to measure `reported_value`. In a folder
or zip it searches for what the script does not name directly: it picks the experiment among
several scripts (not a test, runs at the top level, named like a training script), reads
constants such as `TARGET` from the project files the script imports, finds the CSV when the
path is built at run time, and puts the metrics block inside `main()` when the scoring happens
there. When something cannot be found, it says exactly what is missing. It never
switches on `prediction_time`, `group_columns` or `time_column`: those are written as commented
suggestions to confirm, because a wrong group column causes false alarms. Your files are never
changed, and `CONVERSION.md` in the output lists everything the converter did. Converting the
original F1 notebook gives the same rules-only check results as the hand-made `demo/f1`.

The web page does the same for a zip without an `audit.yaml`: it converts the upload, then shows
a form with the detected target, metric and measured value, and the suggested group and time
columns (none ticked). The project appears in the list only after you confirm, and the settings
are checked first (real columns, a measured metric, a numeric value).

## Benchmark

```bash
python -m benchmark.generate --verify-lock    # held-out answer keys unchanged since the lock
python -m benchmark.run --split dev           # rules, Muse + tools, Muse only
python -m benchmark.run --split heldout
python -m benchmark.combine --main <dev run> <heldout run> --repeats <repeats run>
```

Answer keys live in `benchmark/answers/`, outside the folders the auditor reads, and each
project is audited from an anonymous copy. The notes in [RESULTS.md](benchmark/RESULTS.md) list
every change made while developing against the development set.

## Repository

```
auditor/       sandbox, tracer, checks, agent (client, tools, loop, prompts, verdict), patching, reports,
               CLI, demo, web (local page)
benchmark/     project generator, answer keys, runner, scoring, RESULTS.md
demo/          f1 (the origin story) and clean (an honest project)
examples/      recorded audits with reports and evidence
tests/         unit and end-to-end tests
```

Built for the Meta Global AI Developer Hackathon (October 26 to November 4, 2026). The F1
project predates the hackathon and is used only as an audit target: see
[demo/f1/README.md](demo/f1/README.md).
