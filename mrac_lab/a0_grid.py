"""Orchestration of the arm-3 A0 mismatch grid (notebook 07 section 10). No physics here.

Everything physical (plant, reference, law, runner, metrics) is imported from
``experiments.mrac`` (``baselines``, ``payload_law``, ``payload_reference``,
``offline_metrics``). This module only fixes the *experimental design* (conditions,
controllers, scenarios, gains) and runs the episodes in worker processes.

Scope (approved by Javier, 2026-11-24): 9 conditions (payload mass 2..6 kg at L = 2 m,
and cable length 1, 1.5, 2.5, 3 m at 4 kg = the H2 negative control) x 2 scenarios x
3 compared controllers = 54 deterministic episodes, plus 18 normaliser episodes (the
true-mass PID+FF at the same condition and scenario). One run per cell. EXPLORATORY /
offline / PROXY.

======== ====================================================================
key      controller
======== ====================================================================
ff_true  PID+FF with the TRUE payload mass (denominator of every ratio)
softer   arm 2: mass-blind PID+FF (m_hat = 4) + softer law G(-1,-0.5), FIXED form,
         acting on the PLANT swing, no MRAC
a0_online arm 3, A0: the payload-aware law over the nominal-payload reference
         (4 kg, 2 m, softer law inside the reference), estimates ADAPTING
a0_frozen same law, estimates FROZEN at the nominal ideals (incl. 4 kg / 2 m)
======== ====================================================================
"""

from __future__ import annotations

import dataclasses
import datetime
import io
import math
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, Iterable, Tuple

import pandas as pd

from . import grid as _grid

SCENARIOS = _grid.SCENARIOS
ARMS: Tuple[str, ...] = ("ff_true", "softer", "a0_online", "a0_frozen")
COMPARED: Tuple[str, ...] = ("softer", "a0_online", "a0_frozen")
#: (L, m_L, family): mass sweep at L = 2 m, length sweep at 4 kg (negative control).
CONDITIONS: Tuple[Tuple[float, float, str], ...] = (
    (2.0, 2.0, "mass"),
    (2.0, 3.0, "mass"),
    (2.0, 4.0, "nominal"),
    (2.0, 5.0, "mass"),
    (2.0, 6.0, "mass"),
    (1.0, 4.0, "length"),
    (1.5, 4.0, "length"),
    (2.5, 4.0, "length"),
    (3.0, 4.0, "length"),
)
HEADER_PREFIX = "# "

#: Gains of the payload-aware law: those of the PRELIMINARY run of notebook 07 section 5.2
#: (fixed once by a stability scan at 4 kg, not tuned for performance); ``d`` is filled in
#: from the canonical parameters.
LAW_GAINS: Dict[str, Any] = dict(
    T_s=0.02, T_adapt=0.02, lambda_p=1.5, lambda_theta=5.0, k_p=20.0, k_theta=25.0,
    filter_a=(125.0, 75.0, 15.0),
    gamma_p=(0.05, 3.0, 3.0), gamma_theta=(1e-2, 1e-2, 1e-2), gamma_alpha=1e-4,
    p_ranges=((5.0, 15.0), (0.0, 10.0), (0.0, 25.0)),
    theta_ranges=((0.5, 3.0), (0.0, 10.0), (0.0, 25.0)),
    alpha_range=(-0.5, 0.5),
)


def law_params(canon):
    from experiments.mrac.payload_law import PayloadLawParams

    return PayloadLawParams(d=canon.payload.d, **LAW_GAINS)


def scenario_setup(canon, scenario: str):
    """(trajectory, plant initial state, n_ticks) of a ``grid.SCENARIOS`` entry."""
    from experiments.mrac.baselines import MinJerkReference

    sc = SCENARIOS[scenario]
    ref = MinJerkReference(sc["start"], sc["end"], sc["duration"])
    return ref, canon.hover_state(position=sc["start"], phi=sc["phi0"]), sc["n_ticks"]


def run_a0(canon0, L: float, m_L: float, scenario: str, online: bool) -> Dict[str, Any]:
    """One arm-3 episode with the A0 reference; arrays are PRE-step rows like ``run_episode``.

    Returns the committed-tick arrays (``t, p, ref_p, phi, phi_r, ...``), the estimate
    history, projection hits, the filter telemetry, the exit record (or ``None``) and
    ``canon``/``system`` for the caller.
    """
    import numpy as np

    from experiments.mrac.baselines import swing_equilibrium_off
    from experiments.mrac.payload_reference import make_a0_system

    canon = canon0.with_condition(m_L=m_L, cable_length=L)
    traj, ic, n = scenario_setup(canon, scenario)
    system = make_a0_system(canon, traj, law_params(canon), ic, mode="copy_kinematics")
    records, exit_ = system.loop.run(system.start, n, online, online)
    dt, g = canon.timing.T_s, canon.quad.g
    k = len(records)
    t = np.arange(k) * dt
    t_end = np.array([r.plant_step.end_tension for r in records])
    t0 = canon.plant().accelerations(ic).tension
    tension = np.concatenate([[t0], t_end[:-1]]) if k else np.array([])
    ref = [traj(ti) for ti in t]
    prev = [traj(ti - dt) if i > 0 else traj(ti) for i, ti in enumerate(t)]
    phi_eq = np.array([swing_equilibrium_off(a, b, dt, g)[0] for a, b in zip(ref, prev)])
    est = np.array(
        [list(r.next_state.law.theta_p) + list(r.next_state.law.theta_theta) + [r.next_state.law.alpha_hat]
         for r in records]
    )
    boxes = LAW_GAINS["p_ranges"] + LAW_GAINS["theta_ranges"]
    hits = np.array(
        [[abs(e - b[0]) < 1e-12 or abs(e - b[1]) < 1e-12 for e, b in zip(row[:6], boxes)] for row in est]
    ) if k else np.zeros((0, 6), dtype=bool)
    out = dict(
        t=t,
        p=np.array([r.measurement.body.p for r in records]).reshape(-1, 2),
        ref_p=np.array([(a.y, a.z) for a in ref]).reshape(-1, 2),
        theta=np.array([r.measurement.body.theta for r in records]),
        phi=np.array([r.measurement.phi for r in records]),
        phi_dot=np.array([r.measurement.phi_dot for r in records]),
        phi_r=np.array([r.snapshot.phi for r in records]),
        phi_dot_r=np.array([r.snapshot.phi_dot for r in records]),
        T=tension, T_end=t_end, T_r=np.array([r.snapshot.tension for r in records]),
        phi_eq=phi_eq,
        f_req=np.array([r.command.f_req for r in records]),
        tau_req=np.array([r.command.tau_req for r in records]),
        f_alloc=np.array([sum(r.allocation.T_cmd) for r in records]),
        tau_alloc=np.array([canon.allocator.a_n * (r.allocation.T_cmd[1] - r.allocation.T_cmd[0]) for r in records]),
        sat=np.array([any(r.allocation.saturated_low) or any(r.allocation.saturated_high) for r in records]),
        est=est, hits=hits,
        v=np.array([r.command.v for r in records]),
        eta0_minus_v=np.array([r.command.eta_0_minus_v for r in records]),
        s_p=np.array([np.hypot(*r.command.s_p) for r in records]),
        s_theta=np.array([r.command.s_theta for r in records]),
        exit=exit_, canon=canon, system=system, n=k, n_requested=n, scenario=scenario,
        nominal_est=np.array(list(system.law.nominal_state().theta_p) + list(system.law.nominal_state().theta_theta)
                             + [system.law.nominal_state().alpha_hat]),
    )
    out["dphi"] = out["phi"] - out["phi_r"]
    out["dphi_dot"] = out["phi_dot"] - out["phi_dot_r"]
    return out


def metrics_of_a0(run: Dict[str, Any]):
    """``OffMetrics`` through ``experiments.mrac.offline_metrics.episode_metrics`` (template result)."""
    from experiments.mrac.baselines import EpisodeFailure, run_episode
    from experiments.mrac.offline_metrics import episode_metrics

    canon = run["canon"]
    traj, ic, n = scenario_setup(canon, run["scenario"])
    template = run_episode(canon.plant(), canon.controller(m_hat=4.0), traj, canon.timing, 0, ic)
    ex = run["exit"]
    fail = None if ex is None else EpisodeFailure(
        tick=ex.tick, time=ex.tick * canon.timing.T_s, kind="taut_exit", source=ex.source,
        message="taut exit (tension %.3g N)" % ex.tension, substep=ex.substep, stage=ex.stage,
        stage_time=ex.time, tension=ex.tension,
    )
    res = dataclasses.replace(
        template, t=run["t"], p=run["p"], ref_p=run["ref_p"], phi=run["phi"], phi_eq=run["phi_eq"],
        T=run["T"], T_end=run["T_end"], failure=fail, n_requested=n,
    )
    return episode_metrics(res, canon.T_nom)


def run_case(task: Tuple[Any, ...]) -> Dict[str, Any]:
    """One episode; ``task = (thesis_root, L, m_L, arm, scenario)``; picklable worker."""
    thesis_root, length, m_L, arm, scenario = task
    from mrac_lab import provenance

    provenance.setup(thesis_root)
    import numpy as np

    from experiments.mrac.baselines import canonical_offline_params, run_episode
    from experiments.mrac.offline_metrics import episode_metrics

    row: Dict[str, Any] = dict(scenario=scenario, L=length, m_L=m_L, arm=arm)
    nan = float("nan")
    tele = dict(dphi_rms_deg=nan, dphi_max_deg=nan, dphidot_rms=nan, mLhat_p_end=nan, mLhat_th_end=nan,
                muhat_p_end=nan, muhat_th_end=nan, est_dev_max=nan, hit_ticks=-1, v_abs_max=nan,
                eta_err_abs_max=nan, exit_source="")
    if arm in ("ff_true", "softer"):
        plant, ctrl, ref, ic, n, canon = _grid.build_case(canonical_offline_params(), length, m_L, arm, scenario, True)
        result = run_episode(plant, ctrl, ref, canon.timing, n, ic)
        m = episode_metrics(result, canon.T_nom)
        fail = None if result.failure is None else \
            f"{result.failure.kind} at t={result.failure.time:.2f}s ({result.failure.source})"
    else:
        run = run_a0(canonical_offline_params(), length, m_L, scenario, online=(arm == "a0_online"))
        m = metrics_of_a0(run)
        k = run["n"]
        ex = run["exit"]
        fail = None if ex is None else f"taut_exit at t={ex.tick * 0.02:.2f}s ({ex.source})"
        if k:
            est, nom = run["est"], run["nominal_est"]
            tele = dict(
                dphi_rms_deg=float(np.degrees(np.sqrt(np.mean(run["dphi"] ** 2)))),
                dphi_max_deg=float(np.degrees(np.max(np.abs(run["dphi"])))),
                dphidot_rms=float(np.sqrt(np.mean(run["dphi_dot"] ** 2))),
                mLhat_p_end=float(est[-1, 1]), mLhat_th_end=float(est[-1, 4]),
                muhat_p_end=float(est[-1, 2]), muhat_th_end=float(est[-1, 5]),
                est_dev_max=float(np.max(np.abs(est[:, :6] - nom[:6]))),
                hit_ticks=int(run["hits"].any(axis=1).sum()),
                v_abs_max=float(np.max(np.abs(run["v"]))), eta_err_abs_max=float(np.max(np.abs(run["eta0_minus_v"]))),
                exit_source="" if ex is None else ex.source,
            )
    row.update(
        pos_rms_m=m.pos_rms_m, swing_rms_deg=m.swing_rms_deg, peak_T_over_Tnom=m.peak_T_over_Tnom,
        failed=m.failed, n_completed=m.n_completed, n_requested=m.n_requested, failure=fail, **tele,
    )
    return row


def run_grid(thesis_root, workers: int = 6) -> pd.DataFrame:
    """All episodes plus ``c_*_off`` (vs the true-mass PID+FF) and ``c_*_vs_arm2`` columns."""
    from experiments.mrac.offline_metrics import OffMetrics, c_S_off, c_T_off

    tasks = [(str(thesis_root), L, m, a, s) for s in SCENARIOS for (L, m, _fam) in CONDITIONS for a in ARMS]
    if workers <= 1:
        rows = [run_case(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(run_case, tasks, chunksize=2))
    df = pd.DataFrame(rows)
    fam = {(L, m): f for (L, m, f) in CONDITIONS}
    df["family"] = [fam[(r.L, r.m_L)] for r in df.itertuples()]

    def metrics(r) -> OffMetrics:
        return OffMetrics(r.pos_rms_m, r.swing_rms_deg, r.peak_T_over_Tnom, bool(r.failed),
                          int(r.n_completed), int(r.n_requested))

    ref = {(r.scenario, r.L, r.m_L): metrics(r) for r in df.itertuples() if r.arm == "ff_true"}
    arm2 = {(r.scenario, r.L, r.m_L): metrics(r) for r in df.itertuples() if r.arm == "softer"}
    for name, fn in (("c_T_off", c_T_off), ("c_S_off", c_S_off)):
        df[name] = [fn(metrics(r), ref[(r.scenario, r.L, r.m_L)]) for r in df.itertuples()]
        df[name + "_vs_arm2"] = [fn(metrics(r), arm2[(r.scenario, r.L, r.m_L)]) for r in df.itertuples()]
    return df


def save_csv(df: pd.DataFrame, path, prov, canon, workers: int, seconds: float) -> None:
    """Write ``df`` with commented provenance lines (thesis commit, parameters, design, date)."""
    from .results import write_new_text

    lines = [
        "EXPLORATORY / offline / PROXY: arm 3 with the nominal-payload (A0) reference, one deterministic run per cell",
        f"created: {datetime.datetime.now().isoformat(timespec='seconds')}  workers={workers}  wall_s={seconds:.0f}",
        f"thesis_commit: {prov.commit}  branch: {prov.branch}  dirty: {prov.dirty} {list(prov.dirty_paths)}",
        f"thesis_source: {prov.source} (no machine path is recorded)",
        f"timing: T_s={canon.timing.T_s} N={canon.timing.N} sim_dt={canon.sim_dt}",
        f"vehicle (nominal, no vehicle mismatch): m={canon.quad.m} J={canon.quad.J} a_p={canon.quad.a_p} T_max={canon.quad.T_max} tau_m={canon.quad.tau_m} d={canon.payload.d}",
        f"PID gains: {canon.gains}",
        f"beta_tau={canon.beta_tau} tau_max={canon.tau_max} L_nom={canon.L_nom} m_L_nom={canon.m_L_nom}",
        f"law gains: {LAW_GAINS}",
        "arms: ff_true (true-mass PID+FF, normaliser); softer (arm 2: mass-blind m_hat=4 + softer law G(-1,-0.5) FIXED form on the plant swing, no MRAC); a0_online and a0_frozen (arm 3, A0: nominal-payload reference 4 kg / 2 m with the softer law FIXED form inside the reference only; frozen = estimates held at the nominal ideals)",
        "reference init: copy_kinematics at t=0 (motors at the reference hover command), never re-synchronised",
        "conditions (L, m_L, family): " + "; ".join(f"({L:g}, {m:g}, {f})" for L, m, f in CONDITIONS),
        "scenarios: " + "; ".join(f"{k}: {v['text']}" for k, v in SCENARIOS.items()),
        "c_T_off / c_S_off: ratio to the TRUE-MASS PID+FF at the same condition and scenario; *_vs_arm2: ratio to arm 2 at the same cell; NaN if either failed",
        "arrays are PRE-step rows (as run_episode); dphi = phi - phi_r (plant minus reference swing)",
    ]
    buf = io.StringIO()
    for ln in lines:
        buf.write(HEADER_PREFIX + ln + "\n")
    df.to_csv(buf, index=False)
    write_new_text(path, buf.getvalue())  # refuses an existing file: a cached grid is evidence


load_csv = _grid.load_csv
