# Example audits

Real outputs of `python -m auditor.cli audit`, copied from `runs/` so they can be read
without running anything.

| Folder | Project | Verdict | What it shows |
|---|---|---|---|
| [f1/](f1/report.md) | `demo/f1`, the F1 pit-stop model | **confirmed** | The 95% came from the same races in train and test. Muse isolated the leak from a label shift (2024 only: 0.935 random split vs 0.721 race-grouped), patched the split, and the patched script scored 0.705. |
| [clean/](clean/report.md) | `demo/clean`, an honestly evaluated readmission model | **not found** | A strong feature (`num_prior_readmissions`) is not accused: it is prior history, known at discharge. |

Each folder has `report.md` (human-readable), `report.json`, and `evidence/` with the full
transcript between the code and Muse, every tool result (E1, E2, ...), the accepted verdict
and, for f1, the patch diff and the patched script. Links in the reports to sandbox run logs
point into the local `runs/` folder, which is not part of the repository.
