"""Orchestration of the offline vehicle-mismatch grid (notebook 08). No physics here.

Everything physical (plant, controllers, runner, metrics, the mismatch itself) is
imported from ``experiments.mrac.baselines`` and ``experiments.mrac.offline_metrics``.
This module only fixes the *experimental design* (cases, controllers, scenarios) and
runs the episodes in worker processes.

Scope (approved by Javier, 2026-11-24): the nominal load (4 kg, 2 m); the 15
one-at-a-time vehicle mismatch cases of ``experiments.mrac.baselines.MISMATCH_RANGES``
(nominal, ``m_q`` 4 values, ``J`` 4, ``a_p`` 4, ``delta`` 2) x 3 controllers x 2
scenarios = 90 deterministic episodes, one run each, cached as a CSV with a
provenance header. EXPLORATORY / offline / PROXY.

The controllers keep the NOMINAL vehicle values (PID mass and inertia, allocator arm,
``kappa = 1``, feedforward 4 kg); only the plant changes (``with_vehicle_mismatch``).

======== ============================================================ ==========
key      controller                                                   ``m_hat``
======== ============================================================ ==========
pid_ff   arm 1: PID+FF, mass-blind FF (= true mass at 4 kg)           4.0
softer   PID+FF + softer law G(-1,-0.5), FIXED form                   4.0
damper   PID+FF + damper G(-1,-2), FIXED form                         4.0
======== ============================================================ ==========

``G(kp, kd)`` is the FIXED form ``kd sqrt(L_nom/L)`` (S30 convention); at the nominal
length both forms coincide, and the oracle form is not run here.
"""

from __future__ import annotations

import datetime
import io
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, Iterable, Tuple

import pandas as pd

from . import grid as _grid

ARMS: Tuple[str, ...] = ("pid_ff", "softer", "damper")
M_L_NOM = 4.0
L_NOM = 2.0
SCENARIOS = _grid.SCENARIOS
HEADER_PREFIX = "# "


def case_table():
    """The 15 one-at-a-time cases as a list of (index, parameter, value, mismatch)."""
    from experiments.mrac.baselines import mismatch_cases

    return [(i, c.parameter, c.value, c.mismatch) for i, c in enumerate(mismatch_cases())]


def build_case(canon0, mismatch, arm: str, scenario: str):
    """(plant, controller, reference, initial state, n_ticks, canonical-with-mismatch)."""
    canon = canon0.with_vehicle_mismatch(
        m_q_scale=mismatch.m_q_scale,
        J_scale=mismatch.J_scale,
        a_p_scale=mismatch.a_p_scale,
        delta=mismatch.delta,
    )
    return _grid.build_case(canon, L_NOM, M_L_NOM, arm, scenario, fixed_form=True)


def run_case(task: Tuple[Any, ...]) -> Dict[str, Any]:
    """Run one episode. ``task = (thesis_root, case_index, arm, scenario)``; picklable worker."""
    thesis_root, index, arm, scenario = task
    from mrac_lab import provenance

    provenance.setup(thesis_root)
    import numpy as np

    from experiments.mrac.baselines import canonical_offline_params, mismatch_cases, run_episode
    from experiments.mrac.offline_metrics import episode_metrics

    case = mismatch_cases()[index]
    plant, ctrl, ref, ic, n, canon = build_case(canonical_offline_params(), case.mismatch, arm, scenario)
    result = run_episode(plant, ctrl, ref, canon.timing, n, ic)
    m = episode_metrics(result, canon.T_nom)
    k = result.n_completed
    row = dict(
        scenario=scenario, case=index, parameter=case.parameter, value=case.value,
        label=case.mismatch.label, m_q_scale=case.mismatch.m_q_scale, J_scale=case.mismatch.J_scale,
        a_p_scale=case.mismatch.a_p_scale, delta=case.mismatch.delta, arm=arm,
        pos_rms_m=m.pos_rms_m, swing_rms_deg=m.swing_rms_deg, peak_T_over_Tnom=m.peak_T_over_Tnom,
        final_pos_err_m=float(np.linalg.norm(result.p[k - 1] - result.ref_p[k - 1])) if k else float("nan"),
        peak_abs_theta_deg=float(np.degrees(np.max(np.abs(result.theta)))) if k else float("nan"),
        sat_fraction=float(np.mean(result.saturated)) if k else float("nan"),
        failed=m.failed, n_completed=m.n_completed, n_requested=m.n_requested,
        failure=None if result.failure is None else
        f"{result.failure.kind} at t={result.failure.time:.2f}s ({result.failure.source})",
    )
    return row


def run_grid(thesis_root, arms: Iterable[str] = ARMS, scenarios: Iterable[str] = tuple(SCENARIOS),
             workers: int = 6) -> pd.DataFrame:
    """All (case, arm, scenario) episodes plus the four ratio columns.

    ``c_T_off_vs_nom`` / ``c_S_off_vs_nom``: the same controller on the nominal vehicle is the
    denominator. ``c_T_off_vs_pidff`` / ``c_S_off_vs_pidff``: arm 1 (mass-blind PID+FF) at the
    SAME mismatch case is the denominator. NaN if either episode failed.
    """
    from experiments.mrac.offline_metrics import OffMetrics, c_S_off, c_T_off

    tasks = [(str(thesis_root), i, a, s) for s in scenarios for i, *_ in case_table() for a in arms]
    if workers <= 1:
        rows = [run_case(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(run_case, tasks, chunksize=3))
    df = pd.DataFrame(rows)

    def metrics(r) -> OffMetrics:
        return OffMetrics(r.pos_rms_m, r.swing_rms_deg, r.peak_T_over_Tnom, bool(r.failed),
                          int(r.n_completed), int(r.n_requested))

    nom = {(r.scenario, r.arm): metrics(r) for r in df.itertuples() if r.case == 0}
    blind = {(r.scenario, r.case): metrics(r) for r in df.itertuples() if r.arm == "pid_ff"}
    for name, fn in (("c_T_off", c_T_off), ("c_S_off", c_S_off)):
        df[name + "_vs_nom"] = [fn(metrics(r), nom[(r.scenario, r.arm)]) for r in df.itertuples()]
        df[name + "_vs_pidff"] = [fn(metrics(r), blind[(r.scenario, r.case)]) for r in df.itertuples()]
    return df


def save_csv(df: pd.DataFrame, path, prov, canon, workers: int, seconds: float) -> None:
    """Write ``df`` with commented provenance lines (thesis commit, parameters, ranges, date)."""
    from pathlib import Path

    from experiments.mrac.baselines import MISMATCH_RANGES

    lines = [
        "EXPLORATORY / offline / PROXY: offline vehicle-mismatch grid, one deterministic run per cell",
        f"created: {datetime.datetime.now().isoformat(timespec='seconds')}  workers={workers}  wall_s={seconds:.0f}",
        f"thesis_commit: {prov.commit}  branch: {prov.branch}  dirty: {prov.dirty} {list(prov.dirty_paths)}",
        f"thesis_root: {prov.root}",
        f"timing: T_s={canon.timing.T_s} N={canon.timing.N} sim_dt={canon.sim_dt}",
        f"nominal vehicle: m={canon.quad.m} J={canon.quad.J} a_p={canon.quad.a_p} T_max={canon.quad.T_max} tau_m={canon.quad.tau_m} d={canon.payload.d}; a_n={canon.allocator.a_n}",
        f"gains: {canon.gains}",
        f"beta_tau={canon.beta_tau} tau_max={canon.tau_max} L_nom={canon.L_nom} m_L_nom={canon.m_L_nom}; load m_L={M_L_NOM} L={L_NOM}",
        "mismatch ranges (fractions; delta is kappa_L = 1 - delta, kappa_R = 1 + delta): " + "; ".join(f"{k}: {v}" for k, v in MISMATCH_RANGES.items()),
        "controllers keep nominal values (PID m, J; allocator a_n, kappa = 1; FF m_hat = 4); only the plant changes; hover state = exact equilibrium of the TRUE vehicle",
        "arms: " + ", ".join(ARMS) + "  (softer = G(-1,-0.5), damper = G(-1,-2), FIXED form)",
        "scenarios: " + "; ".join(f"{k}: {v['text']}" for k, v in SCENARIOS.items()),
        "ratios *_vs_nom: same controller, nominal vehicle in the denominator; *_vs_pidff: arm 1 at the same case in the denominator; NaN if failed",
    ]
    buf = io.StringIO()
    for ln in lines:
        buf.write(HEADER_PREFIX + ln + "\n")
    df.to_csv(buf, index=False)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(buf.getvalue(), encoding="utf-8")


load_csv = _grid.load_csv
