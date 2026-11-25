"""Synthetic checks of the protocol-08 section 6 scoring (Codex 27 EXT-01..03). No episodes."""

import pandas as pd

from mrac_lab import ext_analysis as A


def _row(arm, label, task, swing=0.001, status="completed", dist="none", seed=0, tag="grid"):
    return dict(arm=arm, label=label, scenario="hover", dist=dist, seed=seed, tag=tag, d=0.0,
                status=status, task_rms_last60=task, swing_rms_10_30=swing)


def _df(rows):
    df = pd.DataFrame(rows)
    df["cell"] = df.label + "|" + df.scenario + "|" + df.dist + df.seed.astype(str) + "|" + df.tag + "|d" + df.d.astype(str)
    return df


def _base(label, soft=0.001, pid=0.01):
    return [_row("softer", label, 0.0, soft), _row("pidff", label, 0.0, pid)]


def test_unknown_g3_never_passes_and_failures_are_not_wins():
    rows = []
    rows += _base("c1") + [_row("a_online", "c1", 0.01), _row("v0_online", "c1", 0.02)]  # X scores
    rows += [_row("a_online", "c2", 0.01), _row("v0_online", "c2", 0.02)]  # no baselines: G3 unknown
    rows += _base("c3") + [_row("a_online", "c3", float("nan"), status="failed"), _row("v0_online", "c3", 0.02)]
    rows += _base("c4") + [_row("a_online", "c4", 0.01), _row("v0_online", "c4", float("nan"), status="failed")]
    res = A.compare(_df(rows))["a_online vs v0_online"]["D0 none"]
    assert res["n_cells"] == 4 and res["comparable"] == 2
    assert res["x_scores"] == 1 and res["g3_unknown_x"] == 1
    assert res["failed_x"] == 1 and res["failed_y"] == 1
    assert res["x_improves"] is False  # 1 / 2 < 75 %


def test_both_directions_use_their_own_g3():
    rows = []
    for i in range(4):
        lab = f"c{i}"
        rows += _base(lab) + [_row("a_online", lab, 0.05, swing=0.001),
                              _row("v0_online", lab, 0.01, swing=0.02)]  # Y lower but Y fails G3'
    res = A.compare(_df(rows))["a_online vs v0_online"]["D0 none"]
    assert res["y_lower"] == 4 and res["y_scores"] == 0 and res["y_improves"] is False
    assert res["x_scores"] == 0


def test_readout_sets_are_not_scored():
    rows = _base("c1") + [_row("a_online", "c1", 0.01, tag="long"), _row("v0_online", "c1", 0.02, tag="long")]
    assert A.compare(_df(rows))["a_online vs v0_online"] == {}
