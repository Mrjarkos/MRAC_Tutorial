"""Synthetic checks of the protocol-03 analysis rules (Codex 16 F6-F8, F11). No episodes."""

import json

import pytest

from mrac_lab import vanilla_analysis as A
from mrac_lab.vanilla_runs import design_tag

GOOD = dict(status="completed", sat_frac=0.0, f_c_min=100.0, f_c_max=180.0, ref_err_rms_last30=0.001,
            ref_err_rms_30_60=0.002, Kx_one_Fz_std_last30=0.01, swing_rms_10_30=0.001, task_rms_last30=0.01)
CELLS = [(m, s) for m in (2.0, 4.0, 6.0) for s in ("hover", "transfer")]


def _design(gs, cy=1.0):
    return {"gamma_x": [gs] * 6 + [500.0], "gamma_u": [1.0, 1.0], "gamma_cols": [cy, 1.0], "sigma": 0.0}


def _write(tmp_path, rows, status="completed", excluded=()):
    jobs = [{k: r[k] for k in ("id",)} for r in rows]
    (tmp_path / "summary.json").write_text(json.dumps(rows))
    (tmp_path / "jobs.json").write_text(json.dumps(jobs))
    (tmp_path / "receipt.json").write_text(json.dumps({"status": status, "excluded_designs": list(excluded),
                                                       "excluded_jobs": []}))
    return tmp_path


def _tune_rows(designs, override=None):
    rows = []
    for m, s in CELLS:
        base = dict(m_L=m, L=2.0, scenario=s)
        rows.append({**base, "arm": "softer", "id": f"b_s_{m}_{s}", "status": "completed", "swing_rms_10_30": 0.001})
        rows.append({**base, "arm": "pidff", "id": f"b_p_{m}_{s}", "status": "completed", "swing_rms_10_30": 0.01})
        for d, task in designs:
            r = {**base, **GOOD, "arm": "online", "design": d, "tag": design_tag(d),
                 "id": f"{design_tag(d)}_{m}_{s}", "task_rms_last30": task}
            if override:
                r.update(override(d, m, s))
            rows.append(r)
    return rows


def test_refuses_partial_runs_and_roster_mismatch(tmp_path):
    rows = _tune_rows([(_design(5.0), 0.01)])
    with pytest.raises(SystemExit, match="partial"):
        A.analyze_tune(_write(tmp_path, rows, status="wall_cap"))
    folder = _write(tmp_path, rows)
    (folder / "jobs.json").write_text(json.dumps([{"id": r["id"]} for r in rows] + [{"id": "missing"}]))
    with pytest.raises(SystemExit, match="roster"):
        A.analyze_tune(folder)


def test_selection_gates_and_tie_breaks(tmp_path):
    best, close, gated = _design(5.0), _design(20.0), _design(80.0)
    rows = _tune_rows([(best, 0.0100), (close, 0.0100), (gated, 0.001)],
                      override=lambda d, m, s: {"swing_rms_10_30": 0.02} if d == gated and m == 6.0 else {})
    res = A.analyze_tune(_write(tmp_path, rows))
    # gated design is best on task but fails G3 (B_S = 0.001 + 0.5 * 0.009 = 0.0055) in two cells
    assert res["n_pass"] == 2 and res["fail_counts"]["G3"] == 1
    # tie within 1 %: equal reference error, so the smaller elementwise gain sum wins
    assert res["selected"]["tag"] == design_tag(best)
    assert A.gain_sum(_design(5.0, cy=0.25)) < A.gain_sum(_design(5.0))


def test_heldout_counts_only_comparable_cells_and_lists_both_directions(tmp_path):
    rows = []
    for c, (m, L) in enumerate(((2.5, 2.0), (5.5, 2.0), (4.0, 1.5))):
        base = dict(m_L=m, L=L, m_q_scale=1.0, scenario="hover")
        cell = f"m{m:g}_L{L:g}_q1_hover"
        rows += [{**base, "arm": "softer", "id": f"base_softer_{cell}", "status": "completed", "swing_rms_10_30": 0.001,
                  "task_rms_last30": 0.0},
                 {**base, "arm": "pidff", "id": f"base_pidff_{cell}", "status": "completed", "swing_rms_10_30": 0.01}]
        rows.append({**base, **GOOD, "arm": "online", "id": f"tuned_online_{cell}", "task_rms_last30": 0.01})
        frozen = {**base, **GOOD, "arm": "frozen", "id": f"tuned_frozen_{cell}", "task_rms_last30": 0.3}
        if c == 1:  # failed FROZEN must not count as a TUNED win
            frozen.update(status="failed", task_rms_last30=float("nan"))
        rows.append(frozen)
        rows.append({**base, **GOOD, "arm": "online", "id": f"v0_online_{cell}", "task_rms_last30": 0.005})
    res = A.analyze_heldout(_write(tmp_path, rows))
    assert res["criterion2_adaptation_benefit"].startswith("1/1")
    assert "m5.5_L2_q1_hover" not in res["tuned_beats_frozen"]
    assert len(res["v0_beats_tuned"]) == 3 and res["tuned_beats_v0"] == []
    assert [r["cell"] for r in res["length_cells_unmatched_readout"]] == ["m4_L1.5_q1_hover"]


def test_g3_floor_admits_small_absolute_swing():
    base_cell = {"softer": {"swing_rms_10_30": 0.0001}, "pidff": {"swing_rms_10_30": 0.0005}}
    r = {**GOOD, "swing_rms_10_30": 0.004}  # B_S = 0.0003 rad, like 6 kg hover
    assert A.gates(r, base_cell, floor=0.0)["G3"] is False
    g = A.gates(r, base_cell, floor=0.005)
    assert g["G3"] is True and g["B_S"] == 0.005
    assert A.gates({**r, "swing_rms_10_30": 0.006}, base_cell, floor=0.005)["G3"] is False
