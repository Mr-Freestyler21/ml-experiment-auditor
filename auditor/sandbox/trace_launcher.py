"""Runs inside the sandbox as `_audit_trace.py <script> [args]`.

Wraps scikit-learn's fit / predict / transform methods and train_test_split, then runs
the audited script unchanged. Every call is logged to $AUDIT_OUTPUT_DIR/trace.jsonl with
the row labels it saw (the pandas index, i.e. the original CSV row numbers) and the
script line it came from. Self-contained: it may only import the stdlib and sklearn
(and joblib, which sklearn depends on).

Parallel work (n_jobs=-1) normally runs in worker processes the wrappers cannot see, so
joblib is forced into sequential mode: same results, all in this process. Work that
still leaves the process (a hard-coded backend) is logged as a "parallel_escape" event.
"""

import functools
import json
import os
import runpy
import sys

METHODS = ("fit", "fit_transform", "transform", "predict", "predict_proba", "decision_function", "score")

_state = {"depth": 0, "seq": 0}
_script = os.path.abspath(sys.argv[1])
_trace = open(os.path.join(os.environ["AUDIT_OUTPUT_DIR"], "trace.jsonl"), "w", encoding="utf-8")


def _log(event):
    _state["seq"] += 1
    event["seq"] = _state["seq"]
    _trace.write(json.dumps(event, default=str) + "\n")
    _trace.flush()


def _script_line():
    """Line of the audited script that led to this call (the innermost script frame)."""
    frame = sys._getframe(2)
    while frame is not None:
        if frame.f_code.co_filename == _script:
            return frame.f_lineno
        frame = frame.f_back
    return None


def _rows(data):
    """Row labels if `data` is a pandas object, else None (numpy arrays carry no row identity)."""
    index = getattr(data, "index", None)
    if index is None or not hasattr(data, "shape"):
        return None
    return index.tolist()


def _size(data):
    try:
        return len(data)
    except TypeError:
        return None


def _wrap_method(cls, name):
    original = cls.__dict__[name]

    @functools.wraps(original)
    def wrapper(self, *args, **kwargs):
        X = args[0] if args else kwargs.get("X")
        has_y = (len(args) > 1 and args[1] is not None) or kwargs.get("y") is not None
        depth = _state["depth"]
        line = _script_line()
        from_script = sys._getframe(1).f_code.co_filename == _script
        _state["depth"] += 1
        try:
            result = original(self, *args, **kwargs)
        finally:
            _state["depth"] -= 1
        _log({
            "event": "call",
            "class": type(self).__name__,
            "method": name,
            "obj": id(self),
            "depth": depth,
            "line": line,
            # False when sklearn itself made the call (metrics, cross-validation helpers).
            "from_script": from_script,
            "has_y": bool(has_y),
            "n_rows": _size(X),
            # Nested calls (trees inside a forest, steps inside a pipeline) see the same
            # rows as their parent, so only top-level calls store the full row list.
            "rows": _rows(X) if depth == 0 else None,
        })
        return result

    setattr(cls, name, wrapper)


def _wrap_split(module):
    original = module.train_test_split

    @functools.wraps(original)
    def train_test_split(*arrays, **options):
        outputs = original(*arrays, **options)
        _log({
            "event": "split",
            "function": "train_test_split",
            "line": _script_line(),
            "options": {k: v for k, v in options.items() if isinstance(v, (int, float, str, bool, type(None)))},
            "outputs": [{"n_rows": _size(o), "rows": _rows(o)} for o in outputs],
        })
        return outputs

    module.train_test_split = train_test_split
    return train_test_split


def _wrap_parallel():
    from joblib.parallel import Parallel

    original = Parallel.__call__

    @functools.wraps(original)
    def __call__(self, iterable):
        result = original(self, iterable)
        backend = getattr(self, "_backend", None)
        try:
            workers = backend.effective_n_jobs(self.n_jobs)
        except Exception:
            workers = self.n_jobs
        if type(backend).__name__ != "SequentialBackend" and workers not in (1, None):
            _log({
                "event": "parallel_escape",
                "backend": type(backend).__name__,
                "n_jobs": workers,
                "line": _script_line(),
            })
        return result

    Parallel.__call__ = __call__


def _install():
    import warnings

    import sklearn.base
    import sklearn.model_selection
    import sklearn.model_selection._split
    from sklearn.utils import all_estimators

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        classes = [cls for _, cls in all_estimators()]
    # Methods are often defined on private parents (RandomForestClassifier.fit lives on
    # BaseForest), so wrap each method on the sklearn class that defines it, once.
    defining = set()
    for cls in classes:
        for parent in cls.__mro__:
            if parent.__module__.startswith("sklearn"):
                defining.add(parent)
    for cls in defining:
        for name in METHODS:
            if name in cls.__dict__ and callable(cls.__dict__[name]):
                _wrap_method(cls, name)

    patched = _wrap_split(sklearn.model_selection._split)
    sklearn.model_selection.train_test_split = patched
    _wrap_parallel()


def main():
    import joblib

    _install()
    _log({"event": "tracer", "joblib_backend": "sequential (forced)", "joblib_version": joblib.__version__})
    script_dir = os.path.dirname(_script)
    sys.argv = sys.argv[1:]
    sys.path.insert(0, script_dir)
    try:
        with joblib.parallel_config(backend="sequential"):
            runpy.run_path(_script, run_name="__main__")
    finally:
        _trace.close()


if __name__ == "__main__":
    main()
