"""Protocol 03 analysis for the vanilla MRAC tuning (T1 selection, T2 scoring).

``python -m mrac_lab.vanilla_analysis tune <results/mrac_vanilla_tune_*>`` writes
``analysis.json`` into that folder (a new file; existing files are never touched).
``python -m mrac_lab.vanilla_analysis heldout <folder>`` scores T2. Rules are the frozen
protocol's (``.ai/work/2026-11-25-mrac-vanilla/03_tuning_protocol_vanilla.md``, sections
4 and 6, with amendments AM1-AM3). Analysis refuses partial or inconsistent runs
(Codex 16 F6).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

F_MAX = 201.036  # 2 T_max (N)
BASE_ARMS = ("softer", "pidff")


def _cell(r: dict) -> str:
    return f"m{r['m_L']:g}_L{r['L']:g}_q{r.get('m_q_scale', 1.0):g}_{r['scenario']}"


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def _load(folder: Path) -> tuple[list[dict], dict, list[dict]]:
    rows = json.loads((folder / "summary.json").read_text())
    receipt = json.loads((folder / "receipt.json").read_text())
    jobs = json.loads((folder / "jobs.json").read_text())
    if receipt.get("status") != "completed":
        raise SystemExit(f"receipt status {receipt.get('status')!r}: refusing to analyse a partial run")
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate result ids")
    expected = {j["id"] for j in jobs} - set(receipt.get("excluded_jobs", []))
    if set(ids) != expected:
        raise SystemExit(f"result roster mismatch: missing {sorted(expected - set(ids))[:5]}, "
                         f"extra {sorted(set(ids) - expected)[:5]}")
    return rows, receipt, jobs


def _baselines(rows, *, allow_failed: bool = False):
    """Baselines per cell. T1 requires all of them; T2 (AM5, post-data) allows a failed
    baseline, which makes G3 not evaluable in that cell."""
    base: dict[str, dict] = {}
    for r in rows:
        if r.get("arm") in BASE_ARMS:
            base.setdefault(_cell(r), {})[r["arm"]] = r
    for c, b in base.items():
        for arm in BASE_ARMS:
            if arm not in b:
                raise SystemExit(f"baseline {arm} missing in cell {c}")
            if b[arm].get("status") != "completed" or not _finite(b[arm].get("swing_rms_10_30")):
                if not allow_failed:
                    raise SystemExit(f"baseline {arm} failed in cell {c}")
                b["unevaluable"] = True
    return base


def swing_bound(base_cell: dict) -> tuple[float, bool]:
    soft = base_cell["softer"]["swing_rms_10_30"]
    pid = base_cell["pidff"]["swing_rms_10_30"]
    return soft + 0.5 * max(pid - soft, 0.0), pid > soft


G3_FLOOR = 0.0  # protocol 03; protocol 06 (T1'/T2') sets 0.005 rad via --g3-floor


def gates(r: dict, base_cell: dict, floor: float | None = None) -> dict:
    ok = r.get("status") == "completed"
    g1 = (ok and _finite(r.get("sat_frac")) and r["sat_frac"] <= 0.01
          and _finite(r.get("f_c_min")) and r["f_c_min"] >= 0.0
          and _finite(r.get("f_c_max")) and r["f_c_max"] <= F_MAX)
    e_last, e_mid = r.get("ref_err_rms_last30"), r.get("ref_err_rms_30_60")
    g2 = (ok and _finite(e_last) and _finite(e_mid) and e_last <= 1.1 * e_mid + 1e-12
          and _finite(r.get("Kx_one_Fz_std_last30")) and r["Kx_one_Fz_std_last30"] <= 1.0)
    if base_cell.get("unevaluable"):
        return {"G1": bool(g1), "G2": bool(g2), "G3": None, "B_S": float("nan"), "damper_benefit": None}
    bs, benefit = swing_bound(base_cell)
    bs = max(bs, G3_FLOOR if floor is None else floor)
    g3 = ok and _finite(r.get("swing_rms_10_30")) and r["swing_rms_10_30"] <= bs
    return {"G1": bool(g1), "G2": bool(g2), "G3": bool(g3), "B_S": bs, "damper_benefit": bool(benefit)}


def gain_sum(d: dict) -> float:
    """Sum of the elementwise gains, including the column multipliers (Codex 16 F8)."""
    cols = sum(d["gamma_cols"])
    return sum(d["gamma_x"]) * cols + sum(d["gamma_u"]) * cols


def analyze_tune(folder: Path) -> dict:
    rows, receipt, jobs = _load(folder)
    base = _baselines(rows)
    excluded = set(receipt.get("excluded_designs", []))
    by_tag: dict[str, list] = {}
    for r in rows:
        if r.get("arm") == "online":
            by_tag.setdefault(r["tag"], []).append(r)
    designs = []
    for tag, rs in by_tag.items():
        cells = {_cell(r) for r in rs}
        d = rs[0]["design"]
        dkey = json.dumps(d, sort_keys=True)
        g = [gates(r, base[_cell(r)]) for r in rs]
        complete = len(rs) == 6 and len(cells) == 6
        ok = complete and dkey not in excluded and all(x["G1"] and x["G2"] and x["G3"] for x in g)
        task = [r.get("task_rms_last30") for r in rs]
        ref = [r.get("ref_err_rms_last30") for r in rs]
        designs.append({
            "tag": tag, "design": d, "n_cells": len(rs), "pass": ok, "excluded_preflight": dkey in excluded,
            "G1_all": all(x["G1"] for x in g), "G2_all": all(x["G2"] for x in g), "G3_all": all(x["G3"] for x in g),
            "G3_cells_failed": [_cell(r) for r, x in zip(rs, g) if not x["G3"]],
            "mean_task_rms_last30": sum(task) / len(task) if all(map(_finite, task)) else math.inf,
            "mean_ref_err_rms_last30": sum(ref) / len(ref) if all(map(_finite, ref)) else math.inf,
            "gain_sum": gain_sum(d)})
    passing = sorted([d for d in designs if d["pass"]], key=lambda d: d["mean_task_rms_last30"])
    selected = None
    if passing:
        best = passing[0]["mean_task_rms_last30"]
        ties = [d for d in passing if d["mean_task_rms_last30"] <= 1.01 * best]
        ties.sort(key=lambda d: (d["mean_ref_err_rms_last30"], d["gain_sum"]))
        selected = ties[0]
    v0 = [d for d in designs if d["tag"] == "gs20_gc500_cy1_gu1_s0"]
    return {
        "n_designs": len(designs), "n_excluded_preflight": len(excluded), "n_pass": len(passing),
        "selected": selected, "outcome": "selected" if selected else "no design selected",
        "v0_equivalent": v0[0] if v0 else None, "top10": passing[:10],
        "fail_counts": {k: sum(not d[f"{k}_all"] for d in designs) for k in ("G1", "G2", "G3")},
        "baseline_swing_bounds": {c: dict(zip(("B_S", "damper_benefit"), swing_bound(b))) for c, b in base.items()},
        "all_designs": sorted(designs, key=lambda d: d["mean_task_rms_last30"]),
    }


def analyze_heldout(folder: Path) -> dict:
    rows, receipt, jobs = _load(folder)
    base = _baselines(rows, allow_failed=True)
    cells: dict[str, dict] = {}
    for r in rows:
        if r.get("arm") in BASE_ARMS:
            continue
        arm = r["id"].split("_m")[0]
        cells.setdefault(_cell(r), {})[arm] = r
    table = []
    for c, arms in sorted(cells.items()):
        t, f, v = arms["tuned_online"], arms["tuned_frozen"], arms["v0_online"]
        g = gates(t, base[c])
        matched = "_L2_" in c

        def done(x):
            return x.get("status") == "completed" and _finite(x.get("task_rms_last30"))

        both = done(t) and done(f)
        table.append({
            "cell": c, "matched": matched, **{k: g[k] for k in ("G1", "G2", "G3", "B_S", "damper_benefit")},
            "tuned_online_task": t.get("task_rms_last30"), "tuned_frozen_task": f.get("task_rms_last30"),
            "v0_online_task": v.get("task_rms_last30"), "softer_task": base[c]["softer"].get("task_rms_last30"),
            "tuned_swing_10_30": t.get("swing_rms_10_30"), "softer_swing_10_30": base[c]["softer"].get("swing_rms_10_30"),
            "tuned_status": t.get("status"), "frozen_status": f.get("status"), "v0_status": v.get("status"),
            "comparable_online_frozen": both,
            "tuned_beats_frozen": both and t["task_rms_last30"] < f["task_rms_last30"],
            "frozen_beats_tuned": both and f["task_rms_last30"] < t["task_rms_last30"],
            "tuned_beats_v0": done(t) and done(v) and t["task_rms_last30"] < v["task_rms_last30"],
            "v0_beats_tuned": done(t) and done(v) and v["task_rms_last30"] < t["task_rms_last30"],
            "regression_vs_v0": not done(t) and done(v),
        })
    n = len(table)
    m = [r for r in table if r["matched"] and r["comparable_online_frozen"]]
    wins = sum(r["tuned_beats_frozen"] for r in m)
    out = {
        "n_cells": n,
        "criterion1_G1G2_all": all(r["G1"] and r["G2"] for r in table),
        "criterion2_adaptation_benefit": f"{wins}/{len(m)} matched cells with both arms completed "
                                         f"({sum(r['matched'] for r in table)} matched cells in total)",
        "criterion2_pass": bool(m) and wins >= 0.75 * len(m),
        "criterion3_swing": f"{sum(r['G3'] is True for r in table)}/{sum(r['G3'] is not None for r in table)} evaluable "
                            f"({sum(r['G3'] is None for r in table)} not evaluable: a baseline failed, AM5)",
        "criterion3_pass": sum(r["G3"] is True for r in table) >= 0.75 * sum(r["G3"] is not None for r in table),
        "cells_all_controllers_failed": [r["cell"] for r in table if r["G3"] is None and r["tuned_status"] != "completed"],
        "criterion1_excluding_shared_domain_failures": all(r["G1"] and r["G2"] for r in table if r["G3"] is not None),
        "regressions_vs_v0": [r["cell"] for r in table if r["regression_vs_v0"]],
        "tuned_beats_frozen": [r["cell"] for r in table if r["tuned_beats_frozen"]],
        "frozen_beats_tuned": [r["cell"] for r in table if r["frozen_beats_tuned"]],
        "tuned_beats_v0": [r["cell"] for r in table if r["tuned_beats_v0"]],
        "v0_beats_tuned": [r["cell"] for r in table if r["v0_beats_tuned"]],
        "length_cells_unmatched_readout": [r for r in table if not r["matched"]],
        "cells": table,
    }
    out["good_enough"] = bool(out["criterion1_G1G2_all"] and out["criterion2_pass"] and out["criterion3_pass"]
                              and not out["regressions_vs_v0"])
    return out


def main() -> None:
    import argparse

    global G3_FLOOR
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["tune", "heldout"])
    ap.add_argument("folder", type=Path)
    ap.add_argument("--g3-floor", type=float, default=0.0, help="absolute swing floor (rad), protocol 06")
    ap.add_argument("--name", default="analysis", help="output file stem (never overwritten)")
    args = ap.parse_args()
    G3_FLOOR = args.g3_floor
    stage, folder = args.stage, args.folder
    target = folder / f"{args.name}.json"
    res = analyze_tune(folder) if stage == "tune" else analyze_heldout(folder)
    res["g3_floor_rad"] = G3_FLOOR
    with open(target, "x") as fh:  # never overwrite
        json.dump(res, fh, indent=1, default=str)
    skip = ("all_designs", "cells", "top10", "baseline_swing_bounds", "length_cells_unmatched_readout")
    print(json.dumps({k: v for k, v in res.items() if k not in skip}, indent=1, default=str))


if __name__ == "__main__":
    main()
