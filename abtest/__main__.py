"""Command line: python -m abtest [--data PATH] [--out DIR]"""

import argparse
from pathlib import Path

from .analysis import analyze, load, quality_checks, retention_by_rounds, save, summary_table

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    p = argparse.ArgumentParser(prog="abtest", description="Cookie Cats gate 30 vs gate 40 A/B test")
    p.add_argument("--data", type=Path, default=ROOT / "data" / "raw" / "cookie_cats.csv")
    p.add_argument("--out", type=Path, default=ROOT / "outputs")
    p.add_argument("--n-boot", type=int, default=10_000)
    a = p.parse_args()

    if not a.data.exists():
        raise SystemExit(f"{a.data} not found. Download cookie_cats.csv from Kaggle (see README).")
    df = load(a.data)
    checks = quality_checks(df)
    a.out.mkdir(parents=True, exist_ok=True)
    checks.to_csv(a.out / "quality_checks.csv", index=False)
    print(checks.to_string(index=False))
    if checks["failing_rows"].sum():
        raise SystemExit("Data quality checks failed")

    res = analyze(df, n_boot=a.n_boot)
    save(res, a.out)
    retention_by_rounds(df).to_csv(a.out / "retention_by_rounds.csv", index=False)

    srm = res["srm"]
    print(f"\nPlayers: {res['players']:,}  (gate_30 {srm['n_a']:,} / gate_40 {srm['n_b']:,}, "
          f"SRM chi2 = {srm['chi2']:.2f}, p = {srm['p_value']:.4f})\n")
    print(summary_table(res).to_string(index=False))
    r = res["rounds"]
    print(f"\nRounds (excluding {r['outliers_removed']} outlier): median {r['median_control']:.0f} vs "
          f"{r['median_treatment']:.0f}, Mann-Whitney p = {r['mann_whitney_p_value']:.3f}")
    q = res["retention_7_played_30plus"]
    print(f"7-day retention, players with 30+ rounds: {q['p_a']:.2%} vs {q['p_b']:.2%} "
          f"(diff {q['diff']*100:+.2f} pts, p = {q['p_value']:.4f})")


if __name__ == "__main__":
    main()
