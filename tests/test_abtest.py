import math

import numpy as np
import pandas as pd
import pytest

from abtest import stats as st
from abtest.analysis import analyze, load, quality_checks, retention_by_rounds, summary_table


def test_srm_balanced_and_unbalanced():
    assert st.srm_check(5000, 5000)["p_value"] == pytest.approx(1.0)
    assert st.srm_check(5300, 4700)["p_value"] < 0.001


def test_ztest_matches_hand_calculation():
    r = st.two_proportion_ztest(8502, 44700, 8279, 45489)
    p = (8502 + 8279) / (44700 + 45489)
    z = (8279 / 45489 - 8502 / 44700) / math.sqrt(p * (1 - p) * (1 / 44700 + 1 / 45489))
    assert r["z"] == pytest.approx(z)
    assert r["ci_low"] < r["diff"] < r["ci_high"]


def test_ztest_no_difference():
    r = st.two_proportion_ztest(300, 1000, 300, 1000)
    assert r["z"] == pytest.approx(0)
    assert r["p_value"] == pytest.approx(1)


def test_bootstrap_binary_ci_covers_true_diff():
    rng = np.random.default_rng(1)
    a = (rng.random(20000) < 0.20).astype(int)
    b = (rng.random(20000) < 0.18).astype(int)
    res = st.bootstrap_diff(a, b, n_boot=4000, seed=3)
    assert res["ci_low"] < res["diff"] < res["ci_high"]
    assert res["ci_high"] < 0
    assert res["prob_b_better"] < 0.01


def test_bootstrap_numeric_median():
    a = np.arange(1, 101)
    b = np.arange(11, 111)
    res = st.bootstrap_diff(a, b, n_boot=500, seed=0, stat=np.median)
    assert res["diff"] == pytest.approx(10)
    assert res["ci_low"] <= 10 <= res["ci_high"]


def test_mde_shrinks_with_sample_size():
    assert st.minimum_detectable_effect(0.19, 1000, 1000) > st.minimum_detectable_effect(0.19, 40000, 40000)


def _toy(tmp_path, true_false=("True", "False")):
    t, f = true_false
    rows = []
    uid = 1
    for version, r7_rate in (("gate_30", 0.5), ("gate_40", 0.25)):
        for i in range(400):
            r7 = t if i < 400 * r7_rate else f
            rows.append((uid, version, i % 80, t if i % 2 else f, r7))
            uid += 1
    path = tmp_path / "cc.csv"
    pd.DataFrame(rows, columns=["userid", "version", "sum_gamerounds", "retention_1", "retention_7"]).to_csv(path, index=False)
    return path


def test_load_handles_both_boolean_spellings(tmp_path):
    a = load(_toy(tmp_path, ("True", "False")))
    b = load(_toy(tmp_path, ("TRUE", "FALSE")))
    assert a["retention_7"].sum() == b["retention_7"].sum() == 300
    assert set(a["retention_1"].unique()) == {0, 1}


def test_checks_pass_and_catch_duplicates(tmp_path):
    df = load(_toy(tmp_path))
    assert quality_checks(df)["failing_rows"].sum() == 0
    dup = pd.concat([df, df.head(3)])
    assert quality_checks(dup).set_index("check").loc["Duplicate user IDs", "failing_rows"] == 3


def test_analyze_detects_large_drop(tmp_path):
    res = analyze(load(_toy(tmp_path)), n_boot=500)
    r7 = res["retention_7"]
    assert r7["p_a"] == pytest.approx(0.5) and r7["p_b"] == pytest.approx(0.25)
    assert r7["p_value"] < 0.001
    assert res["srm"]["p_value"] == pytest.approx(1.0)
    table = summary_table(res)
    assert list(table["metric"]) == ["1-day retention", "7-day retention"]


def test_retention_by_rounds_bands(tmp_path):
    t = retention_by_rounds(load(_toy(tmp_path)))
    assert t["players"].sum() == 800
    assert set(t["version"]) == {"gate_30", "gate_40"}


def test_missing_column_raises(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"userid": [1], "version": ["gate_30"]}).to_csv(p, index=False)
    with pytest.raises(ValueError):
        load(p)
