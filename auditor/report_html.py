"""Render report.json as one self-contained report.html (no server, no network, no libraries).

    python -m auditor.report_html runs/audit-<time>/report.json

The page is built from the same data as report.md, entirely by code. All text comes from
audited code or model output, so every string is HTML-escaped before it is placed in the
page: nothing in a report can inject markup or script.
"""

from __future__ import annotations

import json
import sys
from html import escape
from pathlib import Path
from typing import Any

VERDICT_STYLE = {
    "confirmed": ("critical", "!", "Confirmed: an evaluation flaw inflates the reported result"),
    "not_found": ("good", "✓", "Not found: no evaluation flaw detected"),
    "inconclusive": ("warning", "?", "Inconclusive: the evidence does not settle it"),
}

CSS = """
.report {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10); --series: #2a78d6;
  --good: #0ca30c; --warning: #fab219; --critical: #d03b3b; --good-ink: #006300;
  --del-bg: #fbe4e4; --add-bg: #e3f4e3; --code-bg: #f3f2ee; --mark-bg: #fff3c4;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) .report {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10); --series: #3987e5;
    --good-ink: #0ca30c; --del-bg: #3a1d1d; --add-bg: #17331a; --code-bg: #232321; --mark-bg: #3d3416;
  }
}
:root[data-theme="dark"] .report {
  color-scheme: dark;
  --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10); --series: #3987e5;
  --good-ink: #0ca30c; --del-bg: #3a1d1d; --add-bg: #17331a; --code-bg: #232321; --mark-bg: #3d3416;
}
html, body { margin: 0; }
body { background: #f9f9f7; }
@media (prefers-color-scheme: dark) { :root:where(:not([data-theme="light"])) body { background: #0d0d0d; } }
:root[data-theme="dark"] body { background: #0d0d0d; }
.report { background: var(--page); color: var(--ink); min-height: 100vh;
  font: 15px/1.55 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
.wrap { max-width: 980px; margin: 0 auto; padding: 32px 16px 64px; }
h1 { font-size: 26px; margin: 0 0 4px; }
h2 { font-size: 18px; margin: 36px 0 12px; }
h3 { font-size: 16px; margin: 0 0 6px; }
.sub { color: var(--ink-2); margin: 0; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 18px 20px;
  margin: 12px 0; }
.verdict { display: flex; flex-wrap: wrap; gap: 12px 16px; align-items: flex-start; }
.verdict > div { flex: 1 1 320px; min-width: 0; }
.badge { display: inline-flex; align-items: center; gap: 8px; font-weight: 700; letter-spacing: .04em;
  padding: 6px 12px; border-radius: 999px; border: 2px solid var(--tone); color: var(--ink); white-space: nowrap; }
.badge .icon { display: inline-grid; place-items: center; width: 22px; height: 22px; border-radius: 50%;
  background: var(--tone); color: #fff; font-size: 13px; }
.tone-critical { --tone: var(--critical); } .tone-good { --tone: var(--good); } .tone-warning { --tone: var(--warning); }
.tone-warning .icon { color: #0b0b0b; }
.meta { display: flex; flex-wrap: wrap; gap: 6px 20px; color: var(--ink-2); font-size: 13px; margin-top: 10px; }
.meta b { color: var(--ink); font-weight: 600; }
.chip { display: inline-block; font: 12px/1 ui-monospace, SFMono-Regular, Consolas, monospace; padding: 4px 7px;
  border-radius: 6px; border: 1px solid var(--border); color: var(--ink); text-decoration: none; margin: 2px 4px 2px 0; }
.chip:hover { border-color: var(--series); }
.family { font-weight: 700; color: var(--ink-2); font-size: 13px; text-transform: uppercase; letter-spacing: .05em; }
pre, code { font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: 13px; }
pre { background: var(--code-bg); border-radius: 8px; padding: 12px 14px; overflow-x: auto; margin: 10px 0; }
.codeline { display: block; background: var(--mark-bg); }
.ln { color: var(--muted); user-select: none; display: inline-block; min-width: 3.2em; }
.role { color: var(--ink-2); }
.diff .del { display: block; background: var(--del-bg); }
.diff .add { display: block; background: var(--add-bg); }
.diff .hunk { display: block; color: var(--muted); }
.diff .ctx { display: block; }
table { border-collapse: collapse; width: 100%; font-size: 14px; }
th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--grid); vertical-align: top; }
th { color: var(--ink-2); font-weight: 600; font-size: 13px; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
.scroll { overflow-x: auto; }
ul.plain { padding-left: 18px; margin: 6px 0; }
.chart svg { width: 100%; height: auto; display: block; }
.chart .label { fill: var(--ink); font-size: 13px; }
.chart .value { fill: var(--ink); font-size: 13px; font-weight: 600; }
.chart .tick { fill: var(--muted); font-size: 12px; }
.chart .gridline { stroke: var(--grid); stroke-width: 1; }
.chart .baseline { stroke: var(--axis); stroke-width: 1; }
.chart .bar { fill: var(--series); }
.chart .majority { stroke: var(--ink); stroke-width: 2; }
.chart .hit { fill: transparent; cursor: default; }
.chart .row:hover .bar { opacity: .85; }
.legend { display: flex; gap: 18px; color: var(--ink-2); font-size: 13px; margin: 2px 0 8px; flex-wrap: wrap; }
.legend .sw { display: inline-block; width: 14px; height: 10px; border-radius: 2px; background: var(--series);
  vertical-align: middle; margin-right: 6px; }
.legend .tk { display: inline-block; width: 2px; height: 14px; background: var(--ink); vertical-align: middle;
  margin: 0 6px 0 4px; }
.tooltip { position: fixed; pointer-events: none; background: var(--surface); color: var(--ink);
  border: 1px solid var(--border); border-radius: 8px; padding: 8px 10px; font-size: 13px;
  box-shadow: 0 4px 16px rgba(0,0,0,.18); max-width: 300px; display: none; z-index: 10; }
.foot { color: var(--muted); font-size: 13px; margin-top: 40px; }
:target { outline: 2px solid var(--series); outline-offset: 2px; }
"""

TOOLTIP_JS = """
(function () {
  var tip = document.getElementById('tip');
  document.querySelectorAll('.chart .row').forEach(function (row) {
    row.addEventListener('mousemove', function (e) {
      tip.textContent = '';
      row.getAttribute('data-tip').split('\\n').forEach(function (line, i) {
        var div = document.createElement('div');
        div.textContent = line;
        if (i === 0) div.style.fontWeight = '600';
        tip.appendChild(div);
      });
      tip.style.display = 'block';
      var x = Math.min(e.clientX + 14, window.innerWidth - tip.offsetWidth - 8);
      tip.style.left = x + 'px';
      tip.style.top = (e.clientY + 14) + 'px';
    });
    row.addEventListener('mouseleave', function () { tip.style.display = 'none'; });
  });
})();
"""


def render_html(r: dict[str, Any]) -> str:
    metric = r["baseline"]["metric"] if r.get("baseline") else "metric"
    tone, icon, verdict_text = VERDICT_STYLE.get(r["verdict"], ("warning", "?", r["verdict"]))
    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>Audit report: {_e(r['project'])}</title><style>{CSS}</style></head>",
        "<body><div class='report'><div class='wrap'>",
        f"<p class='sub'>ML Experiment Auditor &middot; audit of <code>{_e(r['project'])}</code></p>",
        "<div class='card verdict'>",
        f"<span class='badge tone-{tone}'><span class='icon' aria-hidden='true'>{icon}</span>"
        f"{_e(r['verdict'].replace('_', ' ').upper())}</span>",
        f"<div><h1>{_e(verdict_text)}</h1><p>{_e(r['summary'])}</p>{_meta(r)}</div></div>",
        _chart_section(r, metric),
        _findings(r),
        _patch(r),
        _evaluations_table(r, metric),
        _baseline(r, metric),
        _limits(r),
        _evidence(r),
        "<p class='foot'>Every number on this page comes from a tool result recorded as evidence. The model's own "
        "text (verdict, findings, caveats) was accepted only after code checked that each claim cites evidence "
        "and that every number in it appears in a tool result.</p>",
        "</div><div id='tip' class='tooltip' role='status'></div></div>",
        f"<script>{TOOLTIP_JS}</script></body></html>",
    ]
    return "\n".join(parts)


# --- sections ---------------------------------------------------------------------------


def _meta(r: dict[str, Any]) -> str:
    c = r.get("cost") or {}
    b = r.get("baseline") or {}
    items = []
    if b:
        items.append(f"Reported {_e(b['metric'])} <b>{_f(b['reported'])}</b>, reproduced <b>{_f(b['measured'])}</b>")
    if c:
        items.append(f"Model <b>{_e(c.get('model'))}</b>")
        items.append(f"<b>{_e(c.get('steps'))}</b> steps, <b>{_e(c.get('duration_s'))} s</b>")
        items.append(f"<b>{_e(c.get('tokens'))}</b> tokens, about <b>${float(c.get('est_cost_usd') or 0):.3f}</b>")
    return "<div class='meta'>" + "".join(f"<span>{i}</span>" for i in items) + "</div>"


def _chart_rows(r: dict[str, Any], metric: str) -> list[dict[str, Any]]:
    context = r.get("context_metrics") or {}
    fallback = context.get("majority_class_accuracy") or context.get("majority_class_accuracy (auditor)")
    rows = []
    for e in r.get("evaluations") or []:
        value = e.get(metric)
        if not isinstance(value, (int, float)):
            continue
        majority = e.get("majority_class_accuracy_test")
        if majority is None and e["label"].startswith("Original"):
            majority = fallback
        rows.append({**e, "value": float(value), "majority": majority})
    return rows


def _chart_section(r: dict[str, Any], metric: str) -> str:
    rows = _chart_rows(r, metric)
    if not rows:
        return ""
    has_majority = any(isinstance(x["majority"], (int, float)) for x in rows)
    legend = "<div class='legend'><span><span class='sw'></span>" + _e(metric) + "</span>"
    if has_majority:
        legend += "<span><span class='tk'></span>majority-class baseline (always guessing the most common class)</span>"
    legend += "</div>"
    return (
        f"<h2>The numbers</h2><div class='card chart'>"
        f"<p class='sub'>The original evaluation next to every corrected evaluation the auditor measured. "
        f"Hover a bar for details.</p>{legend}{_svg(rows, metric)}</div>"
    )


def _svg(rows: list[dict[str, Any]], metric: str) -> str:
    label_w, bar_w, value_w, row_h, top, bottom = 340, 400, 70, 36, 8, 30
    width = label_w + bar_w + value_w
    height = top + row_h * len(rows) + bottom
    top_value = max(1.0, max(x["value"] for x in rows))

    def x(v: float) -> float:
        return label_w + bar_w * v / top_value

    out = [f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='{_e(metric)} per evaluation'>"]
    for t in (0, 0.25, 0.5, 0.75, 1.0):
        if t <= top_value:
            out.append(f"<line class='gridline' x1='{x(t):.1f}' x2='{x(t):.1f}' y1='{top}' "
                       f"y2='{height - bottom}'/>")
            out.append(f"<text class='tick' x='{x(t):.1f}' y='{height - 10}' text-anchor='middle'>{t:g}</text>")
    out.append(f"<line class='baseline' x1='{label_w}' x2='{label_w}' y1='{top}' y2='{height - bottom}'/>")
    for i, row in enumerate(rows):
        y = top + i * row_h
        bar_y, bar_h = y + 9, row_h - 18
        end = x(row["value"])
        tip = [row["label"], f"{metric}: {_f(row['value'])}"]
        if isinstance(row.get("change_vs_baseline"), (int, float)) and not row["label"].startswith("Original"):
            tip.append(f"change vs original: {row['change_vs_baseline']:+.4f}")
        if isinstance(row["majority"], (int, float)):
            tip.append(f"majority-class baseline: {_f(row['majority'])}")
        if row.get("target_rate_test") is not None:
            tip.append(f"target rate train / test: {_f(row.get('target_rate_train'))} / {_f(row['target_rate_test'])}")
        if row.get("shared_groups") is not None:
            tip.append(f"groups in both train and test: {row['shared_groups']}")
        if row.get("checks_after_patch"):
            tip.append("checks after patch: " + ", ".join(f"{k} {v}" for k, v in row["checks_after_patch"].items()))
        tip.append(f"evidence {row['evidence_id']}")
        out.append(f"<g class='row' data-tip='{_e(chr(10).join(tip))}'>")
        out.append(f"<rect class='hit' x='0' y='{y}' width='{width}' height='{row_h}'/>")
        out.append(f"<text class='label' x='{label_w - 10}' y='{y + row_h / 2 + 4:.1f}' text-anchor='end'>"
                   f"{_e(_short(row['label'], 48))}</text>")
        out.append(_bar_path(label_w, bar_y, end - label_w, bar_h))
        label_x = end
        if isinstance(row["majority"], (int, float)):
            m = x(row["majority"])
            out.append(f"<line class='majority' x1='{m:.1f}' x2='{m:.1f}' y1='{bar_y - 4}' y2='{bar_y + bar_h + 4}'/>")
            label_x = max(end, m)  # the value label sits after whichever ends later
        out.append(f"<text class='value' x='{label_x + 8:.1f}' y='{y + row_h / 2 + 4:.1f}'>{_f(row['value'])}</text>")
        out.append("</g>")
    out.append("</svg>")
    return "".join(out)


def _bar_path(x0: float, y: float, w: float, h: float) -> str:
    """A bar anchored square to the axis with 4px rounded data end."""
    w = max(w, 0.0)
    r = min(4.0, w, h / 2)
    return (f"<path class='bar' d='M{x0:.1f},{y:.1f} H{x0 + w - r:.1f} Q{x0 + w:.1f},{y:.1f} {x0 + w:.1f},{y + r:.1f} "
            f"V{y + h - r:.1f} Q{x0 + w:.1f},{y + h:.1f} {x0 + w - r:.1f},{y + h:.1f} H{x0:.1f} Z'/>")


def _findings(r: dict[str, Any]) -> str:
    findings = r.get("findings") or []
    if not findings:
        return "<h2>Findings</h2><div class='card'><p>No evaluation flaw was confirmed.</p></div>"
    cards = []
    for f in findings:
        code = ""
        if f.get("code"):
            lines = "".join(
                f"<span class='codeline'><span class='ln'>{_e(c['line'])}</span>{_e(c['code'])}"
                f"  <span class='role'># {_e(c['role'])}</span></span>" for c in f["code"])
            code = f"<p class='sub'>Offending code</p><pre>{lines}</pre>"
        chips = "".join(f"<a class='chip' href='#{_e(e['id'])}' title='{_e(e['brief'])}'>{_e(e['id'])} "
                        f"{_e(e['tool'])}</a>" for e in f.get("evidence") or [])
        cards.append(
            f"<div class='card'><div class='family'>Family {_e(f['family'])} &middot; {_e(f.get('family_name'))}</div>"
            f"<h3>{_e(f['title'])}</h3><p>{_e(f['explanation'])}</p>{code}<div>{chips}</div></div>"
        )
    return "<h2>Findings</h2>" + "".join(cards)


def _patch(r: dict[str, Any]) -> str:
    p = r.get("patch")
    if not p:
        return ""
    lines = []
    for line in p["diff"].splitlines():
        if line.startswith(("+++", "---")):
            continue
        kind = "hunk" if line.startswith("@@") else "add" if line.startswith("+") else \
            "del" if line.startswith("-") else "ctx"
        lines.append(f"<span class='{kind}'>{_e(line) or '&nbsp;'}</span>")
    return (
        f"<h2>The fix</h2><div class='card'><p><b>{_e(p['id'])}</b> on <code>{_e(p['path'])}</code> "
        f"({_e(p['lines_changed'])} lines changed). {_e(p['rationale'])}</p>"
        f"<pre class='diff'>{''.join(lines)}</pre><p class='sub'>Applied to a copy; the original file is "
        "unchanged. Code checked that the patch compiles and leaves the model's construction untouched.</p></div>"
    )


def _evaluations_table(r: dict[str, Any], metric: str) -> str:
    rows = r.get("evaluations") or []
    if not rows:
        return ""
    body = []
    for e in rows:
        rates = (f"{_f(e.get('target_rate_train'))} / {_f(e.get('target_rate_test'))}"
                 if e.get("target_rate_test") is not None else "")
        checks = ", ".join(f"{k}: {v}" for k, v in (e.get("checks_after_patch") or {}).items())
        change = e.get("change_vs_baseline")
        body.append(
            f"<tr><td>{_e(e['label'])}</td><td class='num'>{_f(e.get(metric))}</td>"
            f"<td class='num'>{'' if change is None else f'{change:+.4f}'}</td>"
            f"<td class='num'>{_f(e.get('majority_class_accuracy_test'))}</td><td class='num'>{rates}</td>"
            f"<td class='num'>{_e('' if e.get('shared_groups') is None else e['shared_groups'])}</td>"
            f"<td>{_e(checks)}</td><td><a class='chip' href='#{_e(e['evidence_id'])}'>{_e(e['evidence_id'])}</a></td></tr>"
        )
    return (
        "<h2>Corrected evaluation</h2><div class='card scroll'><table><thead><tr><th>Evaluation</th>"
        f"<th class='num'>{_e(metric)}</th><th class='num'>Change</th><th class='num'>Majority baseline</th>"
        "<th class='num'>Target rate train / test</th><th class='num'>Shared groups</th><th>Checks after patch</th>"
        "<th>Evidence</th></tr></thead><tbody>" + "".join(body) + "</tbody></table></div>"
    )


def _baseline(r: dict[str, Any], metric: str) -> str:
    b = r.get("baseline")
    if not b:
        return ""
    env = b.get("environment") or {}
    packages = ", ".join(f"{k} {v}" for k, v in (env.get("packages") or {}).items())
    context = "".join(f"<tr><td>{_e(k)}</td><td class='num'>{_f(v)}</td></tr>"
                      for k, v in (r.get("context_metrics") or {}).items())
    return (
        "<h2>Reproduced baseline and context</h2><div class='card scroll'><table><tbody>"
        f"<tr><td>Reported {_e(metric)}</td><td class='num'>{_f(b['reported'])}</td></tr>"
        f"<tr><td>Measured {_e(metric)} ({_e(b['status'])})</td><td class='num'>{_f(b['measured'])}</td></tr>"
        f"<tr><td>Seed</td><td class='num'>{_e(b.get('random_seed'))}</td></tr>"
        f"<tr><td>Environment</td><td class='num'>Python {_e(env.get('python'))}; {_e(packages)}</td></tr>"
        f"{context}</tbody></table></div>"
    )


def _limits(r: dict[str, Any]) -> str:
    items = [f"<li>{_e(c['text'])} " + "".join(f"<a class='chip' href='#{_e(i)}'>{_e(i)}</a>"
                                             for i in c.get("evidence_ids", [])) + "</li>"
             for c in r.get("caveats") or []]
    items += [f"<li>{_e(line)}</li>" for line in r.get("limits") or []]
    if not items:
        return ""
    return f"<h2>Caveats and limits</h2><div class='card'><ul class='plain'>{''.join(items)}</ul></div>"


def _evidence(r: dict[str, Any]) -> str:
    rows = "".join(
        f"<tr id='{_e(e['id'])}'><td><b>{_e(e['id'])}</b></td><td>{_e(e['tool'])}</td>"
        f"<td><code>{_e(_short(json.dumps(e.get('arguments') or {}), 90))}</code></td><td>{_e(e['brief'])}</td></tr>"
        for e in r.get("evidence_index") or []
    )
    return ("<h2>Evidence index</h2><div class='card scroll'><table><thead><tr><th>Id</th><th>Tool</th>"
            f"<th>Arguments</th><th>Result</th></tr></thead><tbody>{rows}</tbody></table></div>")


# --- helpers ----------------------------------------------------------------------------


def _e(value: Any) -> str:
    return escape("" if value is None else str(value), quote=True)


def _f(value: Any) -> str:
    return f"{value:.4f}" if isinstance(value, float) else _e(value)


def _short(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def write_html(report_json: Path) -> Path:
    data = json.loads(Path(report_json).read_text(encoding="utf-8"))
    out = Path(report_json).with_name("report.html")
    out.write_text(render_html(data), encoding="utf-8")
    return out


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        print(write_html(Path(arg)))
