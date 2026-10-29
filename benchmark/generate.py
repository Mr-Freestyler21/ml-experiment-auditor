"""Generate the benchmark projects and their answer keys.

    python -m benchmark.generate

Each project folder gets data.csv, train.py and audit.yaml; its answer key goes to
benchmark/answers/<id>.json, outside the folder the auditor reads. The answer key
records the planted flaw (if any), its type in the leakage taxonomy of Kapoor &
Narayanan (2023, "Leakage and the reproducibility crisis in machine-learning-based
science"), and the correct metric: the score of the same script with the flaw fixed,
measured by running that fixed version in the sandbox.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import yaml

from auditor.sandbox import collect_files, default_sandbox
from benchmark.datasets import THEMES, Theme, make_data

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
PROJECTS = ROOT / "projects"
# Answer keys live outside the project folders, so the auditor can never read them.
ANSWERS = ROOT / "answers"

TAXONOMY = {
    "L1.2": "Pre-processing on training and test set",
    "L1.3": "Feature selection on training and test set",
    "L2": "Model uses features that are not legitimate",
    "L3.1": "Temporal leakage",
    "L3.2": "Non-independence between train and test samples",
}


@dataclass(frozen=True)
class Spec:
    id: str
    split: str  # "dev" or "heldout"
    theme: str
    seed: int
    evaluation: str = "auto"  # random | group | temporal | cv_group | auto (correct for the theme)
    preprocess: str = "none"  # scaler/imputer/selector/pandas + _before or _after
    leak: str = "none"  # obvious | innocuous | future_window | trailing_window
    encoding: str = "dummies"  # dummies | label_encoder_all
    model: str = "rf"  # rf | lr
    strong: bool = False
    missing: bool = False
    flaws: list[dict] = field(default_factory=list)
    note: str = ""


def flaw(family: str, taxonomy: str, description: str) -> dict:
    return {"family": family, "taxonomy": taxonomy, "taxonomy_name": TAXONOMY[taxonomy], "description": description}


SPECS = [
    # --- development set: looked at while building the auditor ---
    Spec("dev_a_scaler", "dev", "hospital", 101, preprocess="scaler_before", model="lr",
         flaws=[flaw("A", "L1.2", "StandardScaler fitted on all rows before the split")]),
    Spec("dev_a_selector", "dev", "churn", 102, preprocess="selector_before",
         flaws=[flaw("A", "L1.3", "SelectKBest fitted with labels on all rows before the split")]),
    Spec("dev_a_imputer", "dev", "hospital", 103, preprocess="imputer_before", missing=True, model="lr",
         flaws=[flaw("A", "L1.2", "SimpleImputer fitted on all rows before the split")]),
    Spec("dev_a_pandas_scaler", "dev", "machines", 104, preprocess="pandas_before", model="lr",
         flaws=[flaw("A", "L1.2", "Standardised with pandas using all rows' mean and std before the split")],
         note="Expected limit: the tracer only sees scikit-learn, so pandas-only preprocessing is invisible to "
              "check A. An honest result here is inconclusive or a missed flaw."),
    Spec("dev_b_obvious", "dev", "churn", 105, leak="obvious",
         flaws=[flaw("B", "L2", "cancellation_request_logged is recorded after the churn it predicts")]),
    Spec("dev_b_innocuous", "dev", "hospital", 106, leak="innocuous",
         flaws=[flaw("B", "L2", "code_7b is a near-copy of the label under a meaningless name")]),
    Spec("dev_b_future_window", "dev", "machines", 107, leak="future_window",
         flaws=[flaw("B", "L3.1", "vibration_avg_3d is a centred window that includes the next day")]),
    Spec("dev_c_groups", "dev", "hospital", 108, evaluation="random",
         flaws=[flaw("C", "L3.2", "random split puts the same patients in train and test")]),
    Spec("dev_c_time", "dev", "machines", 109, evaluation="random",
         flaws=[flaw("C", "L3.1", "random split trains on days after the test days")]),
    Spec("dev_clean_pipeline_cv", "dev", "hospital", 110, evaluation="cv_group", model="lr"),
    Spec("dev_clean_group_split", "dev", "hospital", 111),
    Spec("dev_clean_temporal", "dev", "churn", 112),
    Spec("dev_clean_strong_feature", "dev", "hospital", 113, strong=True,
         note="Clean, but one legitimate feature predicts the label strongly: bait for false alarms."),
    Spec("dev_clean_label_encoder", "dev", "churn", 114, encoding="label_encoder_all",
         note="Clean: LabelEncoder fitted on all rows only learns category names."),
    # --- held-out set: generated with the rest, run once at the end, never tuned on ---
    Spec("ho_a_selector", "heldout", "students", 9001, preprocess="selector_before",
         flaws=[flaw("A", "L1.3", "SelectKBest fitted with labels on all rows before the split")]),
    Spec("ho_b_innocuous", "heldout", "energy", 9002, leak="innocuous",
         flaws=[flaw("B", "L2", "reading_q is a near-copy of the label under a meaningless name")]),
    Spec("ho_c_groups", "heldout", "students", 9003, evaluation="random",
         flaws=[flaw("C", "L3.2", "random split puts students of the same school in train and test")]),
    Spec("ho_c_time", "heldout", "energy", 9004, evaluation="random",
         flaws=[flaw("C", "L3.1", "random split trains on months after the test months")]),
    Spec("ho_clean_cv", "heldout", "students", 9005, evaluation="cv_group", model="lr"),
    Spec("ho_clean_strong", "heldout", "energy", 9006, strong=True,
         note="Clean, but usage_last_month legitimately predicts the label strongly."),
]


def correct_version(spec: Spec, theme: Theme) -> Spec:
    """The same project with its flaw fixed."""
    return replace(
        spec,
        evaluation="auto" if spec.evaluation == "random" else spec.evaluation,
        preprocess=spec.preprocess.replace("_before", "_after"),
        leak={"obvious": "none", "innocuous": "none", "future_window": "trailing_window"}.get(spec.leak, spec.leak),
    )


def evaluation_of(spec: Spec, theme: Theme) -> str:
    if spec.evaluation != "auto":
        return spec.evaluation
    return "group" if theme.entities_new_in_use else "temporal"


# --- the train.py writer ----------------------------------------------------------------


def build_script(spec: Spec, theme: Theme) -> str:
    evaluation = evaluation_of(spec, theme)
    features = list(theme.features)
    if spec.strong and theme.strong_feature and theme.strong_feature not in features:
        features.append(theme.strong_feature)
    if spec.leak in ("obvious", "innocuous"):
        features.append(theme.leak_names[spec.leak])
    if spec.leak in ("future_window", "trailing_window"):
        features.append(theme.leak_names["future_window"])

    imports = {"import json", "import os", "import pandas as pd"}
    sk: dict[str, set[str]] = {}

    def need(module: str, *names: str) -> None:
        sk.setdefault(module, set()).update(names)

    model = "LogisticRegression(max_iter=2000)" if spec.model == "lr" else \
        "RandomForestClassifier(n_estimators=150, random_state=0)"
    need("sklearn.linear_model" if spec.model == "lr" else "sklearn.ensemble",
         "LogisticRegression" if spec.model == "lr" else "RandomForestClassifier")

    body = [
        f"# Predict {theme.target} for the {theme.name} data.",
        "df = pd.read_csv('data.csv')",
        f"features = {features!r}",
        f"X = df[features + [{theme.categorical!r}]]",
    ]
    if spec.encoding == "label_encoder_all":
        need("sklearn.preprocessing", "LabelEncoder")
        body += ["X = X.copy()", f"X[{theme.categorical!r}] = LabelEncoder().fit_transform(X[{theme.categorical!r}])"]
    else:
        body.append(f"X = pd.get_dummies(X, columns=[{theme.categorical!r}], drop_first=True, dtype=float)")
    body.append(f"y = df[{theme.target!r}]")

    kind, _, when = spec.preprocess.partition("_")
    transformer = {
        "scaler": ("sklearn.preprocessing", "StandardScaler", "StandardScaler()"),
        "imputer": ("sklearn.impute", "SimpleImputer", "SimpleImputer(strategy='mean')"),
        "selector": ("sklearn.feature_selection", "SelectKBest", "SelectKBest(f_classif, k=5)"),
    }.get(kind)
    if transformer:
        need(transformer[0], transformer[1])
        if kind == "selector":
            need("sklearn.feature_selection", "f_classif")

    if evaluation == "cv_group":
        need("sklearn.model_selection", "GroupKFold", "cross_val_score")
        need("sklearn.pipeline", "make_pipeline")
        need("sklearn.preprocessing", "StandardScaler")
        body += [
            f"pipeline = make_pipeline(StandardScaler(), {model})",
            f"scores = cross_val_score(pipeline, X, y, groups=df[{theme.entity!r}], cv=GroupKFold(n_splits=5),"
            " scoring='accuracy')",
            "rate = float(y.mean())",
            "metrics = {'accuracy': float(scores.mean()), 'fold_scores': [float(s) for s in scores],",
            "           'majority_class_accuracy': max(rate, 1 - rate)}",
        ]
        return _assemble(imports, sk, body)

    if when == "before":
        if kind == "selector":
            body.append(f"X = pd.DataFrame({transformer[2]}.fit_transform(X, y), index=X.index)")
        elif kind == "pandas":
            body.append("X = (X - X.mean()) / X.std()")
        else:
            body.append(f"X = pd.DataFrame({transformer[2]}.fit_transform(X), columns=X.columns, index=X.index)")

    if evaluation == "random":
        need("sklearn.model_selection", "train_test_split")
        body.append("X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)")
    elif evaluation == "group":
        need("sklearn.model_selection", "GroupShuffleSplit")
        body += [
            "splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0)",
            f"train_idx, test_idx = next(splitter.split(X, y, groups=df[{theme.entity!r}]))",
            "X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]",
            "y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]",
        ]
    else:  # temporal
        body += [
            f"cutoff = df[{theme.time!r}].quantile(0.75)",
            f"is_train = df[{theme.time!r}] <= cutoff",
            "X_train, X_test = X[is_train], X[~is_train]",
            "y_train, y_test = y[is_train], y[~is_train]",
        ]

    if when == "after":
        if kind == "selector":
            body += [
                f"selector = {transformer[2]}.fit(X_train, y_train)",
                "X_train = pd.DataFrame(selector.transform(X_train), index=X_train.index)",
                "X_test = pd.DataFrame(selector.transform(X_test), index=X_test.index)",
            ]
        elif kind == "pandas":
            body += [
                "mean, std = X_train.mean(), X_train.std()",
                "X_train, X_test = (X_train - mean) / std, (X_test - mean) / std",
            ]
        else:
            body += [
                f"fitted = {transformer[2]}.fit(X_train)",
                "X_train = pd.DataFrame(fitted.transform(X_train), columns=X.columns, index=X_train.index)",
                "X_test = pd.DataFrame(fitted.transform(X_test), columns=X.columns, index=X_test.index)",
            ]

    need("sklearn.metrics", "accuracy_score", "precision_score", "recall_score", "roc_auc_score")
    body += [
        f"model = {model}",
        "model.fit(X_train, y_train)",
        "pred = model.predict(X_test)",
        "proba = model.predict_proba(X_test)[:, 1]",
        "rate = float(y_test.mean())",
        "metrics = {",
        "    'accuracy': float(accuracy_score(y_test, pred)),",
        "    'precision': float(precision_score(y_test, pred, zero_division=0)),",
        "    'recall': float(recall_score(y_test, pred)),",
        "    'roc_auc': float(roc_auc_score(y_test, proba)),",
        "    'majority_class_accuracy': max(rate, 1 - rate),",
        "}",
    ]
    return _assemble(imports, sk, body)


def _assemble(imports: set[str], sk: dict[str, set[str]], body: list[str]) -> str:
    lines = sorted(imports) + [""]
    lines += [f"from {m} import {', '.join(sorted(names))}" for m, names in sorted(sk.items())]
    lines += [""] + body + [
        "",
        "out = os.environ.get('AUDIT_OUTPUT_DIR', '.')",
        "with open(os.path.join(out, 'metrics.json'), 'w') as f:",
        "    json.dump(metrics, f, indent=2)",
        "print(json.dumps(metrics))",
    ]
    return "\n".join(lines) + "\n"


def audit_yaml(spec: Spec, theme: Theme, reported: float) -> dict:
    config = {
        "entry_script": "train.py",
        "data_path": "data.csv",
        "target": theme.target,
        "reported_metric": "accuracy",
        "reported_value": round(reported, 2),
        "prediction_time": theme.prediction_time,
    }
    if theme.entities_new_in_use:
        config["group_columns"] = [theme.entity]
    else:
        config["time_column"] = theme.time
    config.update({"random_seed": 0, "timeout_seconds": 300})
    return config


# --- generation -------------------------------------------------------------------------


def write_project(folder: Path, spec: Spec, data_spec: Spec, reported: float | None = None) -> None:
    theme = THEMES[spec.theme]
    folder.mkdir(parents=True, exist_ok=True)
    make_data(theme, data_spec.seed, leak=data_spec.leak, strong=data_spec.strong, missing=data_spec.missing) \
        .to_csv(folder / "data.csv", index=False)
    (folder / "train.py").write_text(build_script(spec, theme), encoding="utf-8")
    (folder / "audit.yaml").write_text(
        yaml.safe_dump(audit_yaml(spec, theme, reported or 0.0), sort_keys=False, width=100), encoding="utf-8"
    )


def run_metrics(sandbox, folder: Path) -> dict:
    run = sandbox.run(["python", "train.py"], files=collect_files(folder), timeout=300)
    if not run.ok:
        raise RuntimeError(f"{folder.name} failed:\n{run.stderr[-2000:]}")
    return json.loads(run.artifacts["metrics.json"].read_text())


def generate(only: set[str] | None = None) -> list[dict]:
    sandbox = default_sandbox(REPO)
    summaries = []
    with tempfile.TemporaryDirectory() as tmp:
        sandbox.runs_dir = Path(tmp) / "runs"
        for spec in SPECS:
            if only and spec.id not in only:
                continue
            theme = THEMES[spec.theme]
            folder = PROJECTS / spec.id
            if folder.exists():
                shutil.rmtree(folder)
            write_project(folder, spec, spec)
            flawed = run_metrics(sandbox, folder)

            fixed_spec = correct_version(spec, theme)
            fixed_dir = Path(tmp) / f"{spec.id}_fixed"
            write_project(fixed_dir, fixed_spec, fixed_spec)
            fixed = run_metrics(sandbox, fixed_dir) if spec.flaws else flawed

            write_project(folder, spec, spec, reported=flawed["accuracy"])
            labels = {
                "id": spec.id,
                "split": spec.split,
                "theme": spec.theme,
                "clean": not spec.flaws,
                "flaws": spec.flaws,
                "families": sorted({f["family"] for f in spec.flaws}),
                "metric": "accuracy",
                "reported_metric_exact": flawed["accuracy"],
                "correct_metric": fixed["accuracy"],
                "correct_script": build_script(fixed_spec, theme) if spec.flaws else None,
                "note": spec.note,
                "spec": asdict(spec),
            }
            ANSWERS.mkdir(exist_ok=True)
            (ANSWERS / f"{spec.id}.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
            summaries.append(labels)
            print(f"{spec.id:28s} reported {flawed['accuracy']:.4f}  correct {fixed['accuracy']:.4f}  "
                  f"gap {flawed['accuracy'] - fixed['accuracy']:+.4f}")
    write_lock()
    return summaries


def write_lock() -> str:
    """A fingerprint of the held-out answer keys, committed before any held-out audit runs."""
    digest = hashlib.sha256()
    for path in sorted(ANSWERS.glob("ho_*.json")):
        digest.update(path.name.encode() + path.read_bytes())
    value = digest.hexdigest()
    (ROOT / "HELDOUT_LOCK.txt").write_text(
        "SHA-256 of the held-out answer keys (benchmark/answers/ho_*.json), recorded before the auditor\n"
        f"was run on them. Recompute with: python -m benchmark.generate --verify-lock\n\n{value}\n",
        encoding="utf-8",
    )
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", help="regenerate only these project ids")
    parser.add_argument("--verify-lock", action="store_true", help="check the held-out answer keys are unchanged")
    args = parser.parse_args()
    if args.verify_lock:
        recorded = re.search(r"\b[0-9a-f]{64}\b", (ROOT / "HELDOUT_LOCK.txt").read_text()).group(0)
        digest = hashlib.sha256()
        for path in sorted(ANSWERS.glob("ho_*.json")):
            digest.update(path.name.encode() + path.read_bytes())
        print("held-out answer keys unchanged" if digest.hexdigest() == recorded else "MISMATCH")
        return
    generate(set(args.only) if args.only else None)


if __name__ == "__main__":
    main()
