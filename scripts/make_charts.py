"""Build the README charts in docs/.

    python scripts/make_charts.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from abtest.analysis import CONTROL, TREATMENT, analyze, load, retention_by_rounds  # noqa: E402

DOCS = ROOT / "docs"
A_COL, B_COL, INK = "#1F5FA8", "#E0A100", "#1F2933"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "axes.titlesize": 12, "figure.dpi": 130})


def retention_bars(res):
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    x = np.arange(2)
    w = 0.36
    for i, (key, label) in enumerate([("retention_1", "1-day"), ("retention_7", "7-day")]):
        r = res[key]
        for j, (p, n_label, col) in enumerate([(r["p_a"], "gate_30", A_COL), (r["p_b"], "gate_40", B_COL)]):
            n = res["srm"]["n_a"] if j == 0 else res["srm"]["n_b"]
            err = 1.96 * np.sqrt(p * (1 - p) / n) * 100
            ax.bar(i + (j - 0.5) * w, p * 100, w, color=col, yerr=err, capsize=4,
                   label=n_label if i == 0 else None)
            ax.text(i + (j - 0.5) * w, p * 100 + 1.2, f"{p:.2%}", ha="center", fontsize=9)
        ax.text(i, -6.5, f"diff {r['diff']*100:+.2f} pts, p = {r['p_value']:.4f}", ha="center", fontsize=8.5,
                color="#C0392B" if r["p_value"] < 0.05 else INK)
    ax.set_xticks(x, ["1-day retention", "7-day retention"])
    ax.set_ylim(0, 52)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax.legend(frameon=False)
    ax.set_title("Retention by version (95% CI)")
    fig.subplots_adjust(bottom=0.18)
    fig.savefig(DOCS / "retention_by_version.png", bbox_inches="tight")
    plt.close(fig)


def bootstrap_plot(res):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, key, label in zip(axes, ["retention_1", "retention_7"], ["1-day", "7-day"]):
        b = res["_boot"][key]
        ax.hist(b["boot_a"] * 100, bins=60, alpha=0.6, color=A_COL, label="gate_30")
        ax.hist(b["boot_b"] * 100, bins=60, alpha=0.6, color=B_COL, label="gate_40")
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
        ax.set_yticks([])
        ax.set_title(f"{label} retention: P(gate_40 better) = {b['prob_b_better']:.1%}", fontsize=11)
        ax.legend(frameon=False)
    fig.suptitle("Bootstrap distribution of the mean (10,000 resamples)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(DOCS / "bootstrap.png")
    plt.close(fig)


def by_rounds(df):
    t = retention_by_rounds(df)
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    bands = t["rounds_band"].astype(str).unique()
    x = np.arange(len(bands))
    for j, (v, col) in enumerate([(CONTROL, A_COL), (TREATMENT, B_COL)]):
        s = t[t["version"] == v].set_index(t[t["version"] == v]["rounds_band"].astype(str))
        ax.bar(x + (j - 0.5) * 0.38, s.loc[bands, "retention_7"] * 100, 0.38, color=col, label=v)
    ax.axvspan(3.5, 5.5, color="#9AA5B1", alpha=0.15)
    ax.text(4.5, 88, "gates at\nlevel 30 / 40", ha="center", fontsize=8.5)
    ax.set_xticks(x, bands)
    ax.set_xlabel("Game rounds played in the first 14 days")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax.set_ylim(0, 100)
    ax.set_title("7-day retention rises with engagement (descriptive, not causal)")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(DOCS / "retention_by_rounds.png")
    plt.close(fig)


def rounds_hist(df):
    fig, ax = plt.subplots(figsize=(9, 3.8))
    clean = df[df["sum_gamerounds"] <= 200]
    for v, col in [(CONTROL, A_COL), (TREATMENT, B_COL)]:
        ax.hist(clean.loc[clean["version"] == v, "sum_gamerounds"], bins=100, range=(0, 200),
                histtype="step", lw=1.6, color=col, label=v)
    zero = (df["sum_gamerounds"] == 0).mean()
    ax.set_title(f"Game rounds in the first 14 days (capped at 200 for display; {zero:.1%} never played)")
    ax.set_xlabel("Game rounds")
    ax.set_ylabel("Players")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(DOCS / "rounds_distribution.png")
    plt.close(fig)


def main():
    DOCS.mkdir(exist_ok=True)
    df = load(ROOT / "data" / "raw" / "cookie_cats.csv")
    res = analyze(df)
    retention_bars(res)
    bootstrap_plot(res)
    by_rounds(df)
    rounds_hist(df)
    print("Charts saved to", DOCS)


if __name__ == "__main__":
    main()
