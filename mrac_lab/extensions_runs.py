"""Protocol 08 runner: vanilla-MRAC extensions a / ab / abc vs TUNED' vanilla, PID+FF, softer.

``python -m mrac_lab.extensions_runs --stage a|ab|abc [--workers 7] [--wall-cap s]``

Hygiene as ``vanilla_runs``: exclusive dated folders, early and final receipts with
source hashes, hard deadline inside each episode, error-ratio and disk stops, per-tick
summaries (CSV keeps every 10th tick). Model scope (Javier, 2026-11-25): cable at the
vehicle CoG, ``d = 0``; the canonical ``d = 0.3795`` runs only as a robustness readout.
Stage ``a`` also runs the shared arms (TUNED', pid_ff, softer); ``ab`` and ``abc`` reuse
them (same recipe). EXPLORATORY, offline, PROXY.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import shutil
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np

from mrac_lab.vanilla_runs import _git, _init, _write, disturbance, scenario

ROOT = Path(__file__).resolve().parents[1]
T_S = 0.02
CSV_EVERY = 10
TUNED_PRIME = {"gamma_x": [5.0] * 6 + [1000.0], "gamma_u": [0.3, 0.3], "gamma_cols": [0.25, 1.0], "sigma": 0.0}
KICK_TICKS = (1500, 3000)  # t = 30 s and 60 s
KICKS = {"payload_kick": (0.0, 0.5), "vehicle_kick": (0.5, 0.0)}
VARIANTS = {
    "a": {},
    "ab": {"rbf": {"gamma_w": 50.0 / 13.6, "sigma_w": 0.0}},
    "abc": {"rbf": {"gamma_w": 50.0 / 13.6, "sigma_w": 0.1}, "sigma": 0.05},
}


# ---------------------------------------------------------------------------
def make_canon(job: dict):
    from experiments.mrac.baselines import canonical_offline_params
    from experiments.mrac.quad_load_pend import PayloadParams

    base = canonical_offline_params()
    sc = job.get("vehicle", {})
    if sc:
        base = base.with_vehicle_mismatch(**sc)
    canon = base.with_condition(m_L=job["m_L"], cable_length=job["L"])
    d = job.get("d", 0.0)
    return dataclasses.replace(canon, payload=PayloadParams(job["m_L"], job["L"], d))


def nl_params(variant: str):
    from experiments.mrac.vanilla_mrac import rbf_grid
    from experiments.mrac.vanilla_nl_mrac import NLParams

    v = dict(VARIANTS[variant])
    rbf = v.pop("rbf", None)
    if rbf:
        v["rbf"] = rbf_grid(gamma_w=rbf["gamma_w"], sigma_w=rbf["sigma_w"])
    return NLParams(**v)


def run_baseline(canon, arm, traj, n, ic, wind, dist, kicks, deadline, stop_reason):
    """PID+FF (+ softer damper) loop like ``baselines.run_episode`` with wind and kicks."""
    from experiments.mrac.baselines import softer_law
    from experiments.mrac.quad_load_pend import QuadLoadState, TautDomainExit
    from experiments.mrac.wind import WindQuadLoadPend

    ctrl = canon.controller(m_hat=4.0, damper=softer_law() if arm == "softer" else None)
    ctrl.reset()
    plant = WindQuadLoadPend(canon.quad, canon.payload)
    state, rows, failure = ic, [], None
    tension = plant.accelerations(ic).tension
    for k in range(n):
        t = k * T_S
        if k in kicks:
            dvy, dphid = kicks[k]
            state = QuadLoadState(dataclasses.replace(state.quad, v_y=state.quad.v_y + dvy),
                                  state.phi, state.phi_dot + dphid)
        ref = traj(t)
        prev = traj(t - T_S) if k > 0 else ref
        if wind is not None:
            plant.external = tuple(float(w) for w in wind[k])
        elif dist is not None:
            plant.external = tuple(float(w) for w in dist(state, t))
        try:
            rec = ctrl.command(state, ref, prev, tension, T_S, plant.payload.l)
            alloc = ctrl.allocator.allocate(rec.f_req, rec.tau_req)
            step = plant.step(state, alloc.u, canon.timing)
        except TautDomainExit as exc:
            failure = {"kind": "taut_exit", "time": t, "message": str(exc)}
            break
        except ValueError as exc:
            failure = {"kind": "value_error", "time": t, "message": str(exc)}
            break
        q = state.quad
        rows.append({"t": t, "y": q.p_y - ref.y, "z": q.p_z - ref.z, "vy": q.v_y, "vz": q.v_z,
                     "phi": state.phi, "phidot": state.phi_dot, "phi_eq": rec.phi_eq, "f_c": rec.f_req,
                     "sat": bool(any(alloc.saturated_low) or any(alloc.saturated_high)),
                     "kicked": k in kicks, "tension": float(step.end_tension)})
        state, tension = step.state, step.end_tension
        reason = "wall_cap" if time.time() > deadline else stop_reason(state, ref)
        if reason:
            failure = {"kind": reason, "time": t + T_S, "message": reason}
            break
    return rows, failure


def summarise(rows: list[dict], seconds: float, kick_ticks) -> dict:
    import pandas as pd

    out: dict = {"n_rows": len(rows)}
    if not rows:
        return out
    df = pd.DataFrame(rows)
    t = df.t.to_numpy()

    def rms(a):
        a = np.asarray(a)
        return float(np.sqrt(np.mean(a**2))) if len(a) else float("nan")

    tail = (t >= seconds - 60 - 1e-9) & (t < seconds - 1e-9)
    task = np.hypot(df.y, df.z).to_numpy()
    out.update({
        "t_end": float(t[-1]), "task_rms_last60": rms(task[tail]), "task_max": float(task.max()),
        "mean_y_last60": float(df.y[tail].mean()) if tail.any() else float("nan"),
        "mean_z_last60": float(df.z[tail].mean()) if tail.any() else float("nan"),
        "swing_rms_10_30": rms(df.phi[(t >= 10) & (t < 30)]), "swing_rms_last60": rms(df.phi[tail]),
        "sat_frac": float(df.sat.mean()), "f_c_min": float(df.f_c.min()), "f_c_max": float(df.f_c.max()),
    })
    if "norm_e" in df:
        e_pos = np.hypot(df.e_y, df.e_z).to_numpy()
        out.update({"ref_err_rms_last60": rms(e_pos[tail]), "ref_err_rms_30_60": rms(e_pos[(t >= 30) & (t < 60)]),
                    "V_e_last": float(df.V_e.iloc[-1]), "V_e_max": float(df.V_e.max())})
    kick_times = [kt * T_S for kt in kick_ticks]
    for i, t0 in enumerate(kick_times):
        t_next = kick_times[i + 1] if i + 1 < len(kick_times) else float("inf")
        w = (t >= t0) & (t < t0 + 5)
        if w.any():
            out[f"kick{i}_swing_rms_5s"] = rms(df.phi[w])
            out[f"kick{i}_task_max_5s"] = float(task[w].max())
            # settling: last time |phi| > 0.01 rad before the next kick
            after = (t >= t0) & (t < t_next) & (np.abs(df.phi) > 0.01)
            out[f"kick{i}_settle_s"] = float(t[after].max() - t0) if after.any() else 0.0
    for c in ("k_t", "k_u", "k_phi"):
        if c in df:
            out[f"{c}_last"] = float(df[c].iloc[-1])
            out[f"{c}_ideal"] = float(df[c + "_ideal"].iloc[0])
            for ts in (120, 300, 600):
                m = t < ts
                if m.any() and t[-1] >= ts - 1:
                    out[f"{c}_at_{ts}"] = float(df[c][m].iloc[-1])
    if "Kx_one_Fz" in df:
        out["Kx_one_Fz_last"] = float(df.Kx_one_Fz.iloc[-1])
    return out


def run_one(job: dict, out_dir: str, deadline: float = float("inf")) -> dict:
    from experiments.mrac.diagnostics import stop_reason
    from experiments.mrac.vanilla_mrac import VanillaDesign, run_vanilla
    from experiments.mrac.vanilla_nl_mrac import NLDesign, run_nl
    from mrac_lab.vanilla_runs import params_from

    t0 = time.time()
    res = dict(job)
    if t0 > deadline:
        res.update(status="cancelled_wall_cap", wall_s=0.0)
        return res
    canon = make_canon(job)
    n = int(round(job["seconds"] / T_S))
    traj, ic = scenario(job["scenario"], canon)
    dist_kind = job.get("dist", "none")
    kicks = {}
    if dist_kind in KICKS:
        kicks = {kt: KICKS[dist_kind] for kt in KICK_TICKS if kt < n}
        wind, dist = None, None
    else:
        wind, dist = disturbance(dist_kind, n, job.get("seed", 0))

    def stop(state, task):
        return "wall_cap" if time.time() > deadline else stop_reason(state, task)

    arm = job["arm"]
    if arm in ("pidff", "softer"):
        rows, failure = run_baseline(canon, arm, traj, n, ic, wind, dist, kicks, deadline, stop_reason)
    elif arm.startswith("v0"):
        design = VanillaDesign.build(canon, params_from(TUNED_PRIME))
        log = run_vanilla(canon, design, traj, n, ic, adapt=arm == "v0_online", wind=wind,
                          disturbance=dist, kicks=kicks, stop=stop, log_every=1)
        rows, failure = log.rows, log.failure
    else:
        variant, mode = arm.split("_")
        design = NLDesign.build(canon, nl_params(variant))
        log = run_nl(canon, design, traj, n, ic, adapt=mode == "online", wind=wind,
                     disturbance=dist, kicks=kicks, stop=stop, log_every=1)
        rows, failure = log.rows, log.failure
    res["stop"] = None if failure is None else failure["kind"]
    res["status"] = ("cancelled_wall_cap" if res["stop"] == "wall_cap" else
                     "failed" if failure else "completed")
    res.update(summarise(rows, job["seconds"], [k for k in KICK_TICKS if k in kicks]))
    import pandas as pd

    with open(Path(out_dir) / f"{job['id']}.csv", "x", newline="") as fh:
        pd.DataFrame(rows[::CSV_EVERY]).to_csv(fh, index=False)
    res["wall_s"] = time.time() - t0
    return res


# ---------------------------------------------------------------------------
def conditions() -> list[dict]:
    from experiments.mrac.baselines import mismatch_cases

    out = []
    for case in mismatch_cases():
        mm = case.mismatch
        out.append({"label": f"veh_{mm.label}", "m_L": 4.0, "L": 2.0,
                    "vehicle": {"m_q_scale": mm.m_q_scale, "J_scale": mm.J_scale,
                                "a_p_scale": mm.a_p_scale, "delta": mm.delta}})
    for m in (2.0, 3.0, 5.0, 6.0):
        out.append({"label": f"mb{m:g}", "m_L": m, "L": 2.0, "vehicle": {}})
    for L in (1.5, 2.5):
        out.append({"label": f"l{L:g}", "m_L": 4.0, "L": L, "vehicle": {}})
    return out


REPRESENTATIVE = ("veh_nominal", "mb6", "mb2", "l2.5")  # + m_q x1.2, delta 0.10 resolved below
DIST_CLASSES = [("const", 0), ("ou", 1), ("ou", 2), ("ou", 3), ("drag", 0), ("gust", 0),
                ("payload_kick", 0), ("vehicle_kick", 0)]


def jobs_for(stage: str) -> list[dict]:
    conds = conditions()
    rep = [c for c in conds if c["label"] in REPRESENTATIVE]
    rep += [c for c in conds if c["vehicle"].get("m_q_scale", 1.0) == 1.2 and c["vehicle"].get("delta", 0) == 0
            and c["vehicle"].get("J_scale", 1) == 1 and c["vehicle"].get("a_p_scale", 1) == 1][:1]
    rep += [c for c in conds if abs(c["vehicle"].get("delta", 0) - 0.10) < 1e-9][:1]
    arms = [f"{stage}_online", f"{stage}_frozen"] + (["v0_online", "pidff", "softer"] if stage == "a" else [])
    jobs = []

    def add(c, sc, arm, dist="none", seed=0, seconds=120.0, d=0.0, tag="grid"):
        jid = f"{tag}_{c['label']}_{sc}_{dist}{seed}_{arm}" + (f"_d{d:g}" if d else "")
        jobs.append({**c, "scenario": sc, "arm": arm, "dist": dist, "seed": seed, "seconds": seconds,
                     "d": d, "tag": tag, "id": jid})

    for c in conds:
        for sc in ("hover", "transfer"):
            for arm in arms:
                add(c, sc, arm)
    for c in rep:
        for dist, seed in DIST_CLASSES:
            for sc in ("hover", "transfer"):
                for arm in arms:
                    add(c, sc, arm, dist, seed, tag="dist")
    for c in [x for x in rep if x["label"] in ("veh_nominal", "mb6", "mb2")] + rep[-2:]:
        for arm in (f"{stage}_online", f"{stage}_frozen"):
            add(c, "transfer", arm, seconds=600.0, tag="long")
    for c in [x for x in rep if x["label"] in ("veh_nominal", "mb6")]:
        for sc in ("hover", "transfer"):
            for dist in ("none", "drag", "payload_kick"):
                for arm in [f"{stage}_online", f"{stage}_frozen"] + (["softer"] if stage == "a" else []):
                    add(c, sc, arm, dist, d=0.3795, tag="robust")
    return jobs


def sources(thesis_root: Path) -> dict:
    thesis = ["experiments/mrac/vanilla_nl_mrac.py", "experiments/mrac/vanilla_mrac.py",
              "experiments/mrac/baselines.py", "experiments/mrac/quad_load_pend.py", "experiments/mrac/quad.py",
              "experiments/mrac/wind.py", "experiments/mrac/diagnostics.py"]
    tut = ["mrac_lab/extensions_runs.py", "mrac_lab/vanilla_runs.py", "mrac_lab/provenance.py"]
    out = {f"thesis:{f}": hashlib.sha256((thesis_root / f).read_bytes()).hexdigest() for f in thesis}
    out.update({f"tutorial:{f}": hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in tut})
    return out


def preflight(stage: str) -> dict:
    from experiments.mrac.vanilla_nl_mrac import NLDesign, nl_hover_jacobian

    out = {}
    for c in conditions():
        canon = make_canon({**c, "d": 0.0})
        p = dataclasses.replace(nl_params(stage), sigma=0.0)
        r = nl_hover_jacobian(canon, NLDesign.build(canon, p))
        out[c["label"]] = {"rate_per_s": r["growth_rate_per_s"], "invariance": r["invariance_residual"],
                           "drift": r["equilibrium_drift"]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True, choices=list(VARIANTS))
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--wall-cap", type=float, default=4 * 3600.0)
    ap.add_argument("--thesis-root", default=None)
    ap.add_argument("--only", default=None, help="comma-separated job tags (grid,dist,long,robust)")
    ap.add_argument("--ids-file", default=None, help="JSON list of job ids to run (completion part)")
    args = ap.parse_args()

    from mrac_lab import provenance

    prov = provenance.setup(args.thesis_root, modules=("experiments.mrac.vanilla_nl_mrac",))
    jobs = jobs_for(args.stage)
    if args.only:
        jobs = [j for j in jobs if j["tag"] in args.only.split(",")]
    if args.ids_file:
        keep = set(json.loads(Path(args.ids_file).read_text()))
        jobs = [j for j in jobs if j["id"] in keep]
    ids = [j["id"] for j in jobs]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate job ids")
    stamp = time.strftime("%Y%m%dT%H%M%S")
    out = ROOT / "results" / f"mrac_ext_{args.stage}_{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    t0 = time.time()
    deadline = t0 + args.wall_cap
    receipt = {"stage": args.stage, "status": "started", "created": stamp, "protocol": "08 (approved 2026-11-25)",
               "thesis_commit": prov.commit, "dirty_paths": list(prov.dirty_paths),
               "dirty_sha256": {p: hashlib.sha256((prov.root / p).read_bytes()).hexdigest()
                                for p in prov.dirty_paths if (prov.root / p).is_file()},
               "tutorial_commit": _git(ROOT, "rev-parse", "HEAD"), "sources_sha256": sources(prov.root),
               "workers": args.workers, "wall_cap_s": args.wall_cap, "n_jobs": len(jobs),
               "variant_params": VARIANTS[args.stage], "tuned_prime": TUNED_PRIME,
               "label": "EXPLORATORY / offline / PROXY; d = 0 except robust jobs (d = 0.3795)"}
    with open(out / "jobs.json", "x") as fh:
        json.dump(jobs, fh, indent=1)
    _write(out / "receipt.json", receipt)
    results: list[dict] = []
    try:
        pre = preflight(args.stage)
        receipt["preflight"] = pre
        # protocol 08 AM1 threshold -0.05/s, invariance and drift tolerances (Codex 27 EXT-06);
        # sigma designs are certified through their sigma = 0 counterpart (AM4)
        bad = {k: v for k, v in pre.items()
               if not (math.isfinite(v["rate_per_s"]) and v["rate_per_s"] <= -0.05
                       and v["invariance"] < 1e-6 and v["drift"] < 1e-9)}
        receipt["preflight_failures"] = bad
        receipt["status"] = "running"
        _write(out / "receipt.json", receipt)
        if bad:
            raise SystemExit(f"preflight failed: {bad}")
        pool = ProcessPoolExecutor(args.workers, initializer=_init, initargs=(str(prov.root),))
        futs = {pool.submit(run_one, j, str(out), deadline): j for j in jobs}
        pending, stop_reason = set(futs), None
        while pending:
            done, pending = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
            for f in done:
                try:
                    results.append(f.result())
                except Exception as exc:
                    results.append({**futs[f], "status": "error", "error": repr(exc)})
                print(f"{len(results)}/{len(jobs)} {results[-1]['id']} {results[-1].get('status')} {time.time() - t0:.0f}s", flush=True)
            n_err = sum(r.get("status") == "error" for r in results)
            if len(results) >= 20 and n_err > 0.05 * len(results):
                stop_reason = "error_ratio"
            elif shutil.disk_usage(out).free < 3e9:
                stop_reason = "disk_below_3GB"
            elif time.time() > deadline:
                stop_reason = "wall_cap"
            if stop_reason:
                break
        if stop_reason:
            for f in pending:
                f.cancel()
            receipt["status"] = stop_reason
        pool.shutdown(wait=True, cancel_futures=True)
        for f in pending:
            if f.done() and not f.cancelled():
                try:
                    results.append(f.result())
                except Exception as exc:
                    results.append({**futs[f], "status": "error", "error": repr(exc)})
        done_ids = {r["id"] for r in results}
        receipt["not_run"] = [j["id"] for j in jobs if j["id"] not in done_ids]
        if receipt["status"] == "running":
            receipt["status"] = "completed"
    except BaseException as exc:
        receipt["status"] = f"aborted: {exc!r}"
        raise
    finally:
        _write(out / "summary.json", results)
        receipt["wall_s"] = time.time() - t0
        receipt["n_results"] = len(results)
        receipt["n_failed_or_error"] = sum(r.get("status") != "completed" for r in results)
        _write(out / "receipt.json", receipt)
        print(f"wrote {out}: {len(results)} results, {receipt['wall_s']:.0f} s, status {receipt['status']}")


if __name__ == "__main__":
    main()
