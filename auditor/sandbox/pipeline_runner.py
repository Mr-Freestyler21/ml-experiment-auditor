"""Runs inside the sandbox as `_audit_pipeline.py <script> [<script> ...]`.

Runs each script in order, as if started with `python <script>`: preparation scripts first
(they rebuild the data file), the entry script last. When $AUDIT_DATA_PATH is set, the data
file the preparation produced is copied to $AUDIT_OUTPUT_DIR/prepared_data.csv so the
auditor can check the rows the model actually saw. Self-contained: stdlib only.
"""

import os
import runpy
import shutil
import sys


def run_scripts(scripts):
    for script in scripts:
        path = os.path.abspath(script)
        sys.argv = [path]
        sys.path.insert(0, os.path.dirname(path))
        runpy.run_path(path, run_name="__main__")


def copy_prepared_data():
    data = os.environ.get("AUDIT_DATA_PATH")
    out = os.environ.get("AUDIT_OUTPUT_DIR")
    if data and out and os.path.isfile(data) and os.path.getsize(data) < 200 * 1024 * 1024:
        shutil.copyfile(data, os.path.join(out, "prepared_data.csv"))


if __name__ == "__main__":
    scripts = sys.argv[1:]
    run_scripts(scripts[:-1])
    copy_prepared_data()
    run_scripts(scripts[-1:])
