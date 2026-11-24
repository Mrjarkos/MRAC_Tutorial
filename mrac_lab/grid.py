"""Orchestration of the offline S30-style grid (notebook 06 section d). No physics here.

Everything physical (plant, controllers, runner, metrics) is imported from
``experiments.mrac.baselines`` and ``experiments.mrac.offline_metrics``. This
module only fixes the *experimental design* (scenarios, arms) and runs the
episodes, optionally in worker processes.

Scope (approved by Javier, 2026-11-24): 25 S30 conditions (length 1-3 m by
payload 2-6 kg) x 4 controllers x 2 scenarios = 200 deterministic episodes, one
run each, results kept in memory. EXPLORATORY / offline / PROXY.

Arms (feedforward mass ``m_hat``; damper form):

======== ================================================== ===============
key      controller                                          ``m_hat``
======== ================================================== ===============
ff_true  PID+FF, true-mass FF (the denominator of every ratio) the episode ``m_L``
pid_ff   arm 1: PID+FF, mass-blind FF (S30 PIDFF-nom)          4.0
softer   arm 2: mass-blind PID+FF + softer law G(-1,-0.5)      4.0
damper   mass-blind PID+FF + damper G(-1,-2)                   4.0
s30match softer law G(-1,-0.5) with TRUE-mass FF (S30 match)   the episode ``m_L``
======== ================================================== ===============

``G(kp, kd)`` always means the FIXED form ``kd sqrt(L_nom/L)`` (S30 convention);
the oracle form (rate divided by the true ``omega_0``) is only run when a
function is called with ``fixed_form=False`` and is then not labelled ``G``.
"""

from __future__ import annotations

import datetime
import io
import math
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, Iterable, Optional, Tuple

import pandas as pd

ARMS: Tuple[str, ...] = ("ff_true", "pid_ff", "softer", "damper", "s30match")
M_HAT_BLIND = 4.0

#: Scenario design (fixed before any roll). Times in seconds, positions in metres (y, z).
SCENARIOS: Dict[str, Dict[str, Any]] = {
    "hover": dict(
        label="hover recovery",
        text="start at hover with the cable released at phi_0 = 0.35 rad (20 deg), reference held at (0, 5.5)",
        phi0=0.35, n_ticks=300, start=(0.0, 5.5), end=(0.0, 5.5), duration=1.0,
    ),
    "transfer": dict(
        label="min-jerk transfer",
        text="rest-to-rest minimum-jerk transfer of 4 m in y over 4 s from (0, 5.5), then hold (8 s total)",
        phi0=0.0, n_ticks=400, start=(0.0, 5.5), end=(4.0, 5.5), duration=4.0,
    ),
}


def build_case(
    canon0,
    cable_length: float,
    m_L: float,
    arm: str,
    scenario: str,
    fixed_form: bool = True,
    phi0: Optional[float] = None,
):
    """Assemble (plant, controller, reference, initial state, n_ticks, canonical-at-condition)."""
    from experiments.mrac.baselines import (
        MinJerkReference,
        damper_law,
        softer_law,
    )

    sc = SCENARIOS[scenario]
    canon = canon0.with_condition(m_L=m_L, cable_length=cable_length)   # keywords: s30_conditions() is (L, m_L)
    m_hat = m_L if arm in ("ff_true", "s30match") else M_HAT_BLIND
    damper = {"softer": softer_law(fixed_form), "s30match": softer_law(fixed_form),
              "damper": damper_law(fixed_form)}.get(arm)       # the spec carries its form (G(...) = fixed)
    ctrl = canon.controller(m_hat=m_hat, damper=damper)
    ref = MinJerkReference(sc["start"], sc["end"], sc["duration"])
    ic = canon.hover_state(position=sc["start"], phi=sc["phi0"] if phi0 is None else phi0)
    return canon.plant(), ctrl, ref, ic, sc["n_ticks"], canon


def run_case(task: Tuple[Any, ...]) -> Dict[str, Any]:
    """Run one episode. ``task = (thesis_root, L, m_L, arm, scenario, fixed_form)``; picklable worker."""
    thesis_root, length, m_L, arm, scenario, fixed_form = task
    from mrac_lab import provenance

    provenance.setup(thesis_root)
    from experiments.mrac.baselines import canonical_offline_params, run_episode
    from experiments.mrac.offline_metrics import episode_metrics

    plant, ctrl, ref, ic, n, canon = build_case(
        canonical_offline_params(), length, m_L, arm, scenario, fixed_form
    )
    result = run_episode(plant, ctrl, ref, canon.timing, n, ic)
    m = episode_metrics(result, canon.T_nom)
    row = dict(scenario=scenario, L=length, m_L=m_L, arm=arm, fixed_form=fixed_form,
               pos_rms_m=m.pos_rms_m, swing_rms_deg=m.swing_rms_deg,
               peak_T_over_Tnom=m.peak_T_over_Tnom, failed=m.failed,
               n_completed=m.n_completed, n_requested=m.n_requested,
               failure=None if result.failure is None else
               f"{result.failure.kind} at t={result.failure.time:.2f}s ({result.failure.source})")
    return row


def run_grid(
    thesis_root,
    conditions: Iterable[Tuple[float, float]],
    arms: Iterable[str] = ARMS,
    scenarios: Iterable[str] = tuple(SCENARIOS),
    fixed_form: bool = True,
    workers: int = 4,
) -> pd.DataFrame:
    """All (condition, arm, scenario) episodes; ratios to the true-mass PID+FF (``ff_true``) added as ``c_T_off`` / ``c_S_off``."""
    from experiments.mrac.offline_metrics import OffMetrics, c_S_off, c_T_off

    tasks = [
        (str(thesis_root), L, m, a, s, fixed_form)
        for s in scenarios for (L, m) in conditions for a in arms
    ]
    if workers <= 1:
        rows = [run_case(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(run_case, tasks, chunksize=4))
    df = pd.DataFrame(rows)

    def metrics(r) -> OffMetrics:
        return OffMetrics(r.pos_rms_m, r.swing_rms_deg, r.peak_T_over_Tnom, bool(r.failed),
                          int(r.n_completed), int(r.n_requested))

    ref = {(r.scenario, r.L, r.m_L): metrics(r) for r in df.itertuples() if r.arm == "ff_true"}
    df["c_T_off"] = [c_T_off(metrics(r), ref[(r.scenario, r.L, r.m_L)]) for r in df.itertuples()]
    df["c_S_off"] = [c_S_off(metrics(r), ref[(r.scenario, r.L, r.m_L)]) for r in df.itertuples()]
    return df


def to_matrix(df: pd.DataFrame, scenario: str, arm: str, column: str, lengths, masses):
    """``values[i_mass, j_length]`` for ``plots.heatmap_grid`` (NaN where missing)."""
    import numpy as np

    out = np.full((len(masses), len(lengths)), math.nan)
    sub = df[(df.scenario == scenario) & (df.arm == arm)]
    for r in sub.itertuples():
        out[list(masses).index(r.m_L), list(lengths).index(r.L)] = getattr(r, column)
    return out


# ---------------------------------------------------------------------------
# CSV cache with a provenance header (summary metrics only, no trajectories)
# ---------------------------------------------------------------------------
HEADER_PREFIX = "# "


def save_csv(df: pd.DataFrame, path, prov, canon, workers: int, seconds: float) -> None:
    """Write ``df`` with commented provenance lines (thesis commit, parameters, date)."""
    from pathlib import Path

    lines = [
        "EXPLORATORY / offline / PROXY: offline S30-style grid, one deterministic run per cell",
        f"created: {datetime.datetime.now().isoformat(timespec='seconds')}  workers={workers}  wall_s={seconds:.0f}",
        f"thesis_commit: {prov.commit}  branch: {prov.branch}  dirty: {prov.dirty}",
        f"thesis_root: {prov.root}",
        f"timing: T_s={canon.timing.T_s} N={canon.timing.N} sim_dt={canon.sim_dt}",
        f"vehicle: m={canon.quad.m} J={canon.quad.J} a_p={canon.quad.a_p} T_max={canon.quad.T_max} tau_m={canon.quad.tau_m} d={canon.payload.d}",
        f"gains: {canon.gains}",
        f"beta_tau={canon.beta_tau} tau_max={canon.tau_max} L_nom={canon.L_nom} m_L_nom={canon.m_L_nom}",
        "arms: " + ", ".join(ARMS) + f"  m_hat_blind={M_HAT_BLIND}",
        "scenarios: " + "; ".join(f"{k}: {v['text']}" for k, v in SCENARIOS.items()),
        "ratios c_T_off / c_S_off are relative to the TRUE-MASS PID+FF (ff_true) at the same condition and scenario (NaN if failed); G(...) = fixed form",
    ]
    buf = io.StringIO()
    for ln in lines:
        buf.write(HEADER_PREFIX + ln + "\n")
    df.to_csv(buf, index=False)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(buf.getvalue(), encoding="utf-8")


def load_csv(path) -> "tuple[pd.DataFrame, list[str]]":
    """Read a cached grid; returns the data frame and the provenance header lines."""
    from pathlib import Path

    text = Path(path).read_text(encoding="utf-8")
    header = [ln[len(HEADER_PREFIX):] for ln in text.splitlines() if ln.startswith(HEADER_PREFIX)]
    df = pd.read_csv(io.StringIO(text), comment="#")
    df["failed"] = df["failed"].astype(bool)
    return df, header
