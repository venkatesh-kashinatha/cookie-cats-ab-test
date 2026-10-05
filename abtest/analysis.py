"""Load the Cookie Cats experiment and run the full analysis."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import stats as st

CONTROL, TREATMENT = "gate_30", "gate_40"
OUTLIER_ROUNDS = 10_000  # flag players with implausibly many rounds in 14 days


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = {"userid", "version", "sum_gamerounds", "retention_1", "retention_7"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    for col in ("retention_1", "retention_7"):
        # Kaggle's file uses TRUE/FALSE, other copies use True/False.
        df[col] = df[col].astype(str).str.strip().str.lower().map({"true": 1, "false": 0})
    df["version"] = df["version"].str.strip()
    return df


def quality_checks(df: pd.DataFrame) -> pd.DataFrame:
    checks = [
        ("Duplicate user IDs", int(df["userid"].duplicated().sum())),
        ("Unknown version labels", int((~df["version"].isin([CONTROL, TREATMENT])).sum())),
        ("Missing values", int(df.isna().sum().sum())),
        ("Negative game rounds", int((df["sum_gamerounds"] < 0).sum())),
        ("Retention flags not 0/1", int((~df[["retention_1", "retention_7"]].isin([0, 1])).sum().sum())),
    ]
    return pd.DataFrame(checks, columns=["check", "failing_rows"])


def analyze(df: pd.DataFrame, n_boot: int = 10_000, seed: int = 42) -> dict:
    a = df[df["version"] == CONTROL]
    b = df[df["version"] == TREATMENT]
    out = {"players": len(df), "control": CONTROL, "treatment": TREATMENT}

    out["srm"] = st.srm_check(len(a), len(b))

    boot = {}
    for metric in ("retention_1", "retention_7"):
        z = st.two_proportion_ztest(int(a[metric].sum()), len(a), int(b[metric].sum()), len(b))
        bs = st.bootstrap_diff(a[metric].to_numpy(), b[metric].to_numpy(), n_boot=n_boot, seed=seed)
        boot[metric] = bs
        z.update(boot_ci_low=bs["ci_low"], boot_ci_high=bs["ci_high"], prob_treatment_better=bs["prob_b_better"],
                 mde_80_power=st.minimum_detectable_effect(z["p_a"], len(a), len(b)))
        out[metric] = z

    # Engagement: rounds are very skewed, so compare medians and use a rank test.
    outliers = df[df["sum_gamerounds"] > OUTLIER_ROUNDS]
    clean = df[df["sum_gamerounds"] <= OUTLIER_ROUNDS]
    ca, cb = clean[clean["version"] == CONTROL], clean[clean["version"] == TREATMENT]
    out["rounds"] = {
        "outliers_removed": int(len(outliers)),
        "outlier_max": int(outliers["sum_gamerounds"].max()) if len(outliers) else None,
        "median_control": float(ca["sum_gamerounds"].median()),
        "median_treatment": float(cb["sum_gamerounds"].median()),
        "mean_control": float(ca["sum_gamerounds"].mean()),
        "mean_treatment": float(cb["sum_gamerounds"].mean()),
        "never_played_pct": float((df["sum_gamerounds"] == 0).mean()),
        **{f"mann_whitney_{k}": v for k, v in st.mann_whitney(ca["sum_gamerounds"], cb["sum_gamerounds"]).items()},
    }

    # Did the gate change behaviour for players who actually reached it? (30+ rounds as a proxy)
    reached = df[df["sum_gamerounds"] >= 30]
    ra, rb = reached[reached["version"] == CONTROL], reached[reached["version"] == TREATMENT]
    r7 = st.two_proportion_ztest(int(ra["retention_7"].sum()), len(ra), int(rb["retention_7"].sum()), len(rb))
    r7.update(players_control=len(ra), players_treatment=len(rb))
    out["retention_7_played_30plus"] = r7

    out["_boot"] = boot  # arrays for charts, not written to JSON
    return out


def retention_by_rounds(df: pd.DataFrame) -> pd.DataFrame:
    bins = [-1, 0, 9, 19, 29, 39, 49, 99, 199, 10**9]
    labels = ["0", "1-9", "10-19", "20-29", "30-39", "40-49", "50-99", "100-199", "200+"]
    d = df.assign(rounds_band=pd.cut(df["sum_gamerounds"], bins=bins, labels=labels))
    return (d.groupby(["rounds_band", "version"], observed=True)
              .agg(players=("userid", "size"), retention_1=("retention_1", "mean"),
                   retention_7=("retention_7", "mean"))
              .reset_index())


def summary_table(res: dict) -> pd.DataFrame:
    rows = []
    for metric, label in (("retention_1", "1-day retention"), ("retention_7", "7-day retention")):
        r = res[metric]
        rows.append({
            "metric": label,
            "gate_30": round(r["p_a"] * 100, 2),
            "gate_40": round(r["p_b"] * 100, 2),
            "diff_pts": round(r["diff"] * 100, 2),
            "relative_lift_pct": round(r["relative_lift"] * 100, 1),
            "z": round(r["z"], 2),
            "p_value": round(r["p_value"], 4),
            "boot_95ci_pts": f"{r['boot_ci_low']*100:.2f} to {r['boot_ci_high']*100:.2f}",
            "prob_gate_40_better": round(r["prob_treatment_better"], 3),
            "mde_pts_80pct_power": round(r["mde_80_power"] * 100, 2),
        })
    return pd.DataFrame(rows)


def save(res: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    clean = {k: v for k, v in res.items() if not k.startswith("_")}
    (out_dir / "results.json").write_text(json.dumps(clean, indent=2, default=float))
    summary_table(res).to_csv(out_dir / "summary_table.csv", index=False)
