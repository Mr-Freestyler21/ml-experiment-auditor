"""Synthetic tabular datasets with known structure, one per theme.

Each theme has entities observed over time, a hidden per-entity effect, and an
entity-constant observable column. That combination is what makes leakage measurable:
a model can memorise an entity through its constant column when the same entity is in
train and test (group leakage), and neighbouring time steps help when periods are mixed
(temporal leakage). Leak columns are added on request.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Theme:
    name: str
    target: str
    features: list[str]
    categorical: str
    prediction_time: str
    entity: str
    time: str
    # Entities that are new in real use (group leakage applies); otherwise time does.
    entities_new_in_use: bool
    leak_names: dict[str, str] = field(default_factory=dict)  # kind -> column name
    strong_feature: str | None = None


def _z(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-9)


def _label(logit: np.ndarray, rate: float, rng: np.random.Generator) -> np.ndarray:
    noisy = logit + rng.logistic(size=len(logit)) * 0.8
    return (noisy > np.quantile(noisy, 1 - rate)).astype(int)


def _flip(y: np.ndarray, share: float, rng: np.random.Generator) -> np.ndarray:
    flip = rng.random(len(y)) < share
    return np.where(flip, 1 - y, y)


HOSPITAL = Theme(
    name="hospital",
    target="readmitted_30d",
    features=["age", "num_prior_visits", "length_of_stay", "lab_glucose", "home_distance_km"],
    categorical="ward",
    prediction_time="At discharge, predict whether the patient is readmitted within 30 days, using only "
                    "information known at discharge. Patients in real use are new patients.",
    entity="patient_id",
    time="admission_month",
    entities_new_in_use=True,
    leak_names={"obvious": "readmission_followup_flag", "innocuous": "code_7b"},
    strong_feature="num_prior_readmissions",
)

CHURN = Theme(
    name="churn",
    target="churned",
    features=["tenure_weeks", "spend_last_4w", "support_tickets", "discount_rate", "region_code"],
    categorical="plan",
    prediction_time="At the start of each week, predict whether the customer churns that week. Each row's "
                    "features (spend, tickets, discount, plan, region) summarise activity up to the start "
                    "of that week. The same customers are scored every week.",
    entity="customer_id",
    time="week",
    entities_new_in_use=False,
    leak_names={"obvious": "cancellation_request_logged", "innocuous": "flag_q"},
)

MACHINES = Theme(
    name="machines",
    target="failure_next_day",
    features=["temperature", "vibration", "load_pct", "hours_since_service", "machine_age", "ambient_temp"],
    categorical="line",
    prediction_time="At the end of each day, predict whether the machine fails the next day, using only "
                    "readings up to and including that day. The same machines are monitored every day.",
    entity="machine_id",
    time="day",
    entities_new_in_use=False,
    leak_names={"future_window": "vibration_avg_3d"},
)

STUDENTS = Theme(
    name="students",
    target="passed_exam",
    features=["attendance", "homework_score", "prior_grade", "hours_study", "school_funding_index"],
    categorical="track",
    prediction_time="Before the final exam, predict whether a student passes, using coursework so far. "
                    "The model will be used at schools it has not seen.",
    entity="school_id",
    time="term",
    entities_new_in_use=True,
    leak_names={"innocuous": "grade_band"},
)

ENERGY = Theme(
    name="energy",
    target="high_usage",
    features=["temperature_avg", "occupancy", "floor_area", "usage_last_month", "insulation_index"],
    categorical="building_type",
    prediction_time="At the start of each month, predict whether the building's usage that month is high. "
                    "temperature_avg and occupancy are the month's forecast and booking figures, known at "
                    "the start of the month; usage_last_month is the previous month's reading. The same "
                    "buildings are scored every month.",
    entity="building_id",
    time="month",
    entities_new_in_use=False,
    leak_names={"innocuous": "reading_q"},
    strong_feature="usage_last_month",
)

THEMES = {t.name: t for t in (HOSPITAL, CHURN, MACHINES, STUDENTS, ENERGY)}


def make_data(theme: Theme, seed: int, leak: str = "none", strong: bool = False, missing: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    builder = {"hospital": _hospital, "churn": _churn, "machines": _machines,
               "students": _students, "energy": _energy}[theme.name]
    df = builder(rng, strong)
    y = df[theme.target].to_numpy()
    if leak in ("obvious", "innocuous"):
        df[theme.leak_names[leak]] = _flip(y, 0.05, rng)
    if theme.name == "machines":
        # The same 3-day vibration average, computed centred (sees the next day) or trailing.
        if leak in ("future_window", "trailing_window"):
            source = "_window_centred" if leak == "future_window" else "_window_trailing"
            df[theme.leak_names["future_window"]] = df[source]
        df = df.drop(columns=["_window_centred", "_window_trailing"])
    if missing:
        for col in theme.features[1:4]:
            df.loc[rng.random(len(df)) < 0.15, col] = np.nan
    return df.round(4)


def _panel(rng: np.random.Generator, n_entities: int, n_times: int):
    entity = np.repeat(np.arange(n_entities), n_times)
    t = np.tile(np.arange(n_times), n_entities)
    hidden = rng.normal(size=n_entities)[entity]
    constant = np.round(rng.uniform(1, 100, size=n_entities), 2)[entity]
    return entity, t, hidden, constant


def _hospital(rng: np.random.Generator, strong: bool) -> pd.DataFrame:
    e, t, v, c = _panel(rng, 160, 8)
    age = np.round(rng.uniform(25, 90, size=160))[e]
    stay = rng.gamma(2.0, 2.5, size=len(e))
    glucose = rng.normal(110, 25, size=len(e))
    prior = t + rng.poisson(1.0, size=len(e))
    logit = 0.7 * _z(stay) + 0.5 * _z(glucose) + 0.4 * _z(age) + 1.8 * v
    df = pd.DataFrame({
        "patient_id": [f"P{i:04d}" for i in e], "admission_month": t * 3 + rng.integers(0, 3, size=len(e)),
        "age": age, "num_prior_visits": prior, "length_of_stay": stay, "lab_glucose": glucose,
        "home_distance_km": c, "ward": rng.choice(["cardio", "general", "ortho"], size=len(e)),
    })
    if strong:
        r = rng.normal(size=len(e))
        df["num_prior_readmissions"] = np.clip(np.round(3 + 1.6 * r + rng.normal(0, 0.1, len(e))), 0, None)
        logit = logit * 0.2 + 5.0 * r
    df["readmitted_30d"] = _label(logit, 0.3, rng)
    return df


def _churn(rng: np.random.Generator, strong: bool) -> pd.DataFrame:
    e, t, v, c = _panel(rng, 220, 12)
    spend = np.clip(rng.normal(80, 30, size=len(e)) - 10 * v, 0, None)
    tickets = rng.poisson(np.exp(0.3 * v + 0.2), size=len(e))
    discount = np.round(rng.uniform(0, 0.3, size=len(e)), 3)
    logit = -0.8 * _z(spend) + 0.6 * _z(tickets) + 0.4 * _z(discount) + 0.08 * t + 1.4 * v
    df = pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in e], "week": t + 1,
        "tenure_weeks": rng.integers(1, 150, size=220)[e] + t, "spend_last_4w": spend,
        "support_tickets": tickets, "discount_rate": discount, "region_code": c,
        "plan": rng.choice(["basic", "plus", "pro"], size=len(e)),
    })
    df["churned"] = _label(logit, 0.25, rng)
    return df


def _machines(rng: np.random.Generator, strong: bool) -> pd.DataFrame:
    n_m, n_d = 40, 60
    # Mean-reverting stress: today says something about tomorrow, but not everything.
    stress = np.zeros((n_m, n_d + 1))
    stress[:, 0] = rng.normal(0, 1.4, size=n_m)
    for d in range(1, n_d + 1):
        stress[:, d] = 0.6 * stress[:, d - 1] + rng.normal(0, 1.0, size=n_m)
    # Plant-wide conditions on each day (unobserved) hit every machine; the day's ambient
    # reading is shared by all machines, so a random split can memorise days.
    shock = rng.normal(0, 0.9, size=n_d)
    ambient = np.round(rng.uniform(15, 35, size=n_d), 2)
    rows = []
    for m in range(n_m):
        age = round(float(rng.uniform(1, 20)), 2)
        line = rng.choice(["L1", "L2", "L3"])
        for d in range(n_d):
            rows.append({
                "machine_id": f"M{m:03d}", "day": d + 1,
                "temperature": 60 + 4 * stress[m, d] + rng.normal(0, 3),
                "vibration": stress[m, d] + rng.normal(0, 0.35),
                "load_pct": rng.uniform(40, 100), "hours_since_service": (d % 20) * 24 + rng.integers(0, 24),
                "machine_age": age, "ambient_temp": ambient[d], "line": line,
                "_tomorrow": stress[m, d + 1] + shock[d] + rng.normal(0, 0.3),
            })
    df = pd.DataFrame(rows)
    df["failure_next_day"] = (df["_tomorrow"] > df["_tomorrow"].quantile(0.75)).astype(int)
    by_machine = df.groupby("machine_id")["vibration"]
    df["_window_centred"] = by_machine.transform(lambda s: s.rolling(3, center=True, min_periods=1).mean())
    df["_window_trailing"] = by_machine.transform(lambda s: s.rolling(3, min_periods=1).mean())
    return df.drop(columns="_tomorrow")


def _students(rng: np.random.Generator, strong: bool) -> pd.DataFrame:
    e, t, v, c = _panel(rng, 60, 30)
    attendance = np.clip(rng.normal(0.85, 0.1, size=len(e)) + 0.03 * v, 0, 1)
    homework = rng.normal(65, 15, size=len(e))
    prior = rng.normal(60, 15, size=len(e))
    hours = rng.gamma(2, 3, size=len(e))
    logit = 0.6 * _z(attendance) + 0.6 * _z(homework) + 0.5 * _z(prior) + 0.3 * _z(hours) + 2.6 * v
    df = pd.DataFrame({
        "school_id": [f"S{i:03d}" for i in e], "term": rng.integers(1, 4, size=len(e)),
        "attendance": attendance, "homework_score": homework, "prior_grade": prior, "hours_study": hours,
        "school_funding_index": c, "track": rng.choice(["academic", "vocational"], size=len(e)),
    })
    df["passed_exam"] = _label(logit, 0.6, rng)
    return df


def _energy(rng: np.random.Generator, strong: bool) -> pd.DataFrame:
    n_b, n_m = 50, 24
    rows = []
    for b in range(n_b):
        area = round(float(rng.uniform(500, 5000)), 1)
        insulation = round(float(rng.uniform(1, 10)), 2)
        kind = rng.choice(["office", "retail", "school"])
        level, prev = rng.normal(), None
        for m in range(n_m):
            temp = 15 + 10 * np.sin(2 * np.pi * m / 12) + rng.normal(0, 2)
            occupancy = rng.uniform(0.2, 1.0)
            level = 0.75 * level + rng.normal(0, 0.6)
            usage = level + 0.04 * abs(temp - 18) + 0.8 * occupancy - 0.05 * insulation
            if strong:
                usage = level * 1.6 + 0.3 * occupancy
            rows.append({
                "building_id": f"B{b:03d}", "month": m + 1, "temperature_avg": temp, "occupancy": occupancy,
                "floor_area": area, "usage_last_month": prev if prev is not None else usage,
                "insulation_index": insulation, "building_type": kind, "_usage": usage,
            })
            prev = usage
    df = pd.DataFrame(rows)
    df["high_usage"] = (df["_usage"] > df["_usage"].median()).astype(int)
    return df.drop(columns="_usage")
