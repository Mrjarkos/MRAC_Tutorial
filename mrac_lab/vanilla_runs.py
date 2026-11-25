"""Vanilla MRAC (EQ-MR-22) validation and tuning runner: ``python -m mrac_lab.vanilla_runs``.

EXPLORATORY / offline / PROXY. Each stage writes a NEW dated folder under ``results/``
(never overwrites): one CSV per episode (every 5th tick), ``summary.json`` and
``receipt.json`` (source hashes, config, wall time, status). A preflight gate computes the
sampled hover Jacobian (``hover_tick_jacobian``) of every design on every plant in the
stage and refuses to run if any non-neutral mode grows (MQP 6.14.8).

Stages (Javier approved D1 on 2026-11-25: up to 4 CPU wall-hours in total):
  validation  linear theorem checks + nonlinear grid + sigma + disturbance/RBF
  tune        T1 tuning grid (protocol 03)
  heldout     T2 on the tuning-held-out set with the frozen design
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
T_S = 0.02
LOG_EVERY = 5
WIND_MU = (5.0, -3.0, 0.5)
WIND_SIGMA = (2.0, 1.0, 0.2)


# ---------------------------------------------------------------------------
# Designs, scenarios, disturbances
# ---------------------------------------------------------------------------
def params_from(spec: dict):
    from experiments.mrac.vanilla_mrac import VanillaParams, rbf_grid

    spec = dict(spec)
    rbf = spec.pop("rbf", None)
    if rbf:
        spec["rbf"] = rbf_grid(gamma_w=rbf["gamma_w"], sigma_w=rbf.get("sigma_w", 0.0))
    for key in ("gamma_x", "gamma_u", "gamma_cols", "q_diag"):
        if key in spec:
            spec[key] = tuple(spec[key])
    return VanillaParams(**spec)


V0 = {"gamma_x": [20.0] * 6 + [500.0], "gamma_u": [1.0, 1.0], "gamma_cols": [1.0, 1.0]}


def scenario(name: str, canon):
    from experiments.mrac.baselines import MinJerkReference

    if name == "hover":
        return MinJerkReference((0, 5.5), (0, 5.5), 4.0), canon.hover_state(position=(0, 5.5), phi=0.35)
    if name == "transfer":
        return MinJerkReference((0, 5.5), (4, 5.5), 4.0), canon.hover_state(position=(0, 5.5))
    if name == "transfer_fast":
        return MinJerkReference((0, 5.5), (4, 5.5), 2.5), canon.hover_state(position=(0, 5.5))
    raise ValueError(name)


def disturbance(kind: str, n: int, seed: int = 0):
    """Returns (wind_array, callable) with exactly one non-None, or (None, None)."""
    if kind == "none":
        return None, None
    if kind == "const":
        return np.tile(np.array(WIND_MU), (n, 1)), None
    if kind == "ou":
        from experiments.mrac.wind import WindSpec, wind_sequence

        return wind_sequence(WindSpec("ou", WIND_MU, WIND_SIGMA, 3.0, seed), n, T_S), None
    if kind == "gust":  # persistent periodic gust around a mean
        t = np.arange(n) * T_S
        w = np.stack([3 + 4 * np.sin(2 * np.pi * t / 8), -2 * np.sin(2 * np.pi * t / 5),
                      0.3 * np.sin(2 * np.pi * t / 6)], axis=1)
        return w, None
    if kind == "drag":  # quadratic drag in a steady 4 m/s lateral wind (state dependent)
        c, vw = 0.3, np.array([4.0, 0.0])

        def f(state, t):
            rel = vw - np.array([state.quad.v_y, state.quad.v_z])
            force = c * np.linalg.norm(rel) * rel
            return float(force[0]), float(force[1]), 0.0

        return None, f
    raise ValueError(kind)


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------
def _init(thesis_root: str) -> None:
    import sys

    if thesis_root not in sys.path:
        sys.path.insert(0, thesis_root)


def _summarise(rows: list[dict], seconds: float) -> dict:
    """Per-tick summary (Codex 16 F1): every control tick is in ``rows``.

    Windows: ``last30 = [T - 30, T)`` with ``T`` the requested horizon (empty, hence NaN,
    if the episode stopped early) and ``30_60 = [30, 60)``; disjoint for ``T >= 90``.
    """
    import pandas as pd

    df = pd.DataFrame(rows)
    out: dict = {"n_rows": len(df)}
    if df.empty:
        return out
    t = df.t.to_numpy()
    tail = (t >= seconds - 30.0 - 1e-9) & (t < seconds - 1e-9)

    def rms(a):
        return float(np.sqrt(np.mean(np.square(a)))) if len(a) else float("nan")

    vk = "V_KxKu" if "V_KxKu" in df else "V_K"  # gain part only (no RBF W term)
    task = np.hypot(df.y, df.z).to_numpy()
    e_pos = np.hypot(df.e_y, df.e_z).to_numpy()
    out.update({
        "t_end": float(t[-1]),
        "task_rms_last30": rms(task[tail]), "task_max": float(task.max()),
        "ref_err_rms_last30": rms(e_pos[tail]), "ref_err_max": float(e_pos.max()),
        "mean_ey_last30": float(df.e_y[tail].mean()), "mean_ez_last30": float(df.e_z[tail].mean()),
        "mean_y_last30": float(df.y[tail].mean()), "mean_z_last30": float(df.z[tail].mean()),
        "swing_rms_last30": rms(df.phi[tail]), "swing_err_rms_last30": rms(df.e_phi[tail]),
        "swing_rms_10_30": rms(df.phi[(t >= 10) & (t < 30)]),
        "ref_err_rms_30_60": rms(e_pos[(t >= 30) & (t < 60)]),
        "Kx_one_Fy_std_last30": float(df.Kx_one_Fy[tail].std()),
        "norm_e_last30": float(df.norm_e[tail].mean()),
        "V_e_max": float(df.V_e.max()), "V_e_last": float(df.V_e.iloc[-1]),
        "V_K_first": float(df[vk].iloc[0]), "V_K_last": float(df[vk].iloc[-1]),
        "Kx_tilde_first": float(df.norm_Kx_tilde.iloc[0]), "Kx_tilde_last": float(df.norm_Kx_tilde.iloc[-1]),
        "Ku_tilde_last": float(df.norm_Ku_tilde.iloc[-1]),
        "sat_frac": float(df.sat.mean()), "proj_hits": int(df.proj_kx.sum() + df.proj_ku.sum() + df.proj_w.sum()),
        "f_c_min": float(df.f_c.min()), "f_c_max": float(df.f_c.max()),
        "Kx_phi_Fy_last": float(df.Kx_phi_Fy.iloc[-1]), "Kx_one_Fz_last": float(df.Kx_one_Fz.iloc[-1]),
        "Kx_one_Fy_last": float(df.Kx_one_Fy.iloc[-1]),
        "Kx_one_Fz_std_last30": float(df.Kx_one_Fz[tail].std()),
    })
    xb = df[[c for c in ("y", "vy", "phi", "phidot", "z", "vz")]].to_numpy()
    reg = np.hstack([xb, np.ones((len(df), 1)), df[["xi_Fy", "xi_Fz"]].to_numpy() / 50.0])
    for name, mask in (("0_10", t < 10), ("last30", tail)):
        if mask.sum() > 5:
            G = (reg[mask].T @ reg[mask]) * T_S
            ev = np.linalg.eigvalsh(G)
            out[f"gram_min_{name}"] = float(ev[0]); out[f"gram_max_{name}"] = float(ev[-1])
    return out


def run_one(job: dict, out_dir: str, deadline: float = float("inf")) -> dict:
    from experiments.mrac.baselines import canonical_offline_params, run_episode, softer_law
    from experiments.mrac.diagnostics import stop_reason
    from experiments.mrac.vanilla_mrac import VanillaDesign, run_vanilla, simulate_linear

    t0 = time.time()
    res = dict(job)
    if t0 > deadline:  # F2: hard wall cap, checked before and during each episode
        res.update({"status": "cancelled_wall_cap", "wall_s": 0.0})
        return res
    base = canonical_offline_params()
    canon = base.with_condition(m_L=job["m_L"], cable_length=job["L"])
    if job.get("m_q_scale", 1.0) != 1.0:
        canon = canon.with_vehicle_mismatch(m_q_scale=job["m_q_scale"]).with_condition(m_L=job["m_L"], cable_length=job["L"])
    n = int(round(job["seconds"] / T_S))
    if job["kind"] == "linear":
        design = VanillaDesign.build(canon, params_from(job["design"]))
        cmd = (lambda t: (4.0 if t > 1 else 0.0, 0.0)) if job["scenario"] == "step" else (
            lambda t: (2 * math.sin(0.7 * t) + math.sin(1.9 * t), 0.8 * math.sin(1.1 * t) + 0.5 * math.sin(2.7 * t)))
        r = simulate_linear(design, m_q=canon.quad.m, m_L=job["m_L"], l=job["L"], T=job["seconds"],
                            command=cmd, sampled=job.get("sampled"), dt=1e-3, log_every=20)
        V = r["V_eKxKu"]
        kxt = np.linalg.norm(r["Kx"] - r["ideal_Kx"], axis=(1, 2))
        res.update({"status": "completed", "V0": float(V[0]), "V_end": float(V[-1]),
                    "V_max_increase": float(np.diff(V).max()), "norm_e_max": float(r["norm_e"].max()),
                    "norm_e_end": float(r["norm_e"][-1]), "Kx_tilde_0": float(kxt[0]), "Kx_tilde_end": float(kxt[-1]),
                    "Ku_tilde_end": float(np.linalg.norm(r["Ku"][-1] - r["ideal_Ku"])),
                    "matching_residual": r["matching_residual"]})
        name = f"lin_{job['id']}"
        np.savez_compressed(Path(out_dir) / f"{name}.npz", t=r["t"], V=V, V_e=r["V_e"], V_K=r["V_KxKu"],
                            norm_e=r["norm_e"], e=r["e"], Kx=r["Kx"], Ku=r["Ku"], ideal_Kx=r["ideal_Kx"], ideal_Ku=r["ideal_Ku"])
    elif job["arm"] in ("softer", "pidff"):
        traj, ic = scenario(job["scenario"], canon)
        ctrl = canon.controller(m_hat=4.0, damper=softer_law() if job["arm"] == "softer" else None)
        ep = run_episode(canon.plant(), ctrl, traj, canon.timing, n, ic)
        k = ep.n_completed
        tt = ep.t[:k]
        tail = (tt >= job["seconds"] - 30.0 - 1e-9) & (tt < job["seconds"] - 1e-9) if k else []
        task = np.hypot(ep.p[:k, 0] - ep.ref_p[:k, 0], ep.p[:k, 1] - ep.ref_p[:k, 1])
        res.update({"status": "failed" if ep.failed else "completed",
                    "stop": None if ep.failure is None else ep.failure.kind, "t_end": float(tt[-1]) if k else 0.0,
                    "task_rms_last30": float(np.sqrt(np.mean(task[tail] ** 2))) if k else float("nan"),
                    "swing_rms_last30": float(np.sqrt(np.mean(ep.phi[:k][tail] ** 2))) if k else float("nan"),
                    "swing_rms_10_30": float(np.sqrt(np.mean(ep.phi[:k][(tt >= 10) & (tt < 30)] ** 2))) if k else float("nan"),
                    "mean_y_last30": float(np.mean(ep.p[:k, 0][tail] - ep.ref_p[:k, 0][tail])) if k else float("nan"),
                    "mean_z_last30": float(np.mean(ep.p[:k, 1][tail] - ep.ref_p[:k, 1][tail])) if k else float("nan")})
    else:
        design = VanillaDesign.build(canon, params_from(job["design"]))
        traj, ic = scenario(job["scenario"], canon)
        wind, dist = disturbance(job.get("wind", "none"), n, job.get("seed", 0))
        def stop(state, task):
            return "wall_cap" if time.time() > deadline else stop_reason(state, task)

        log = run_vanilla(canon, design, traj, n, ic, adapt=job["arm"] == "online", wind=wind,
                          disturbance=dist, stop=stop, log_every=1)
        res["stop"] = None if log.failure is None else log.failure["kind"]
        res["status"] = ("cancelled_wall_cap" if res["stop"] == "wall_cap"
                         else "failed" if log.failure else "completed")
        res.update(_summarise(log.rows, job["seconds"]))
        import pandas as pd

        # Storage only: the CSV keeps every 5th tick; summaries above use every tick.
        with open(Path(out_dir) / f"{job['id']}.csv", "x", newline="") as fh:  # F12
            pd.DataFrame(log.rows[::LOG_EVERY]).to_csv(fh, index=False)
        _, kx, ku, residual = design.ideal(canon.quad.m, job["m_L"], job["L"])
        res["ideal_Kx_phi_Fy"] = float(kx[2, 0]); res["ideal_Kx_one_Fz"] = float(kx[6, 1])
        res["matching_residual"] = residual
    res["wall_s"] = time.time() - t0
    return res


# ---------------------------------------------------------------------------
# Stage definitions
# ---------------------------------------------------------------------------
PLANTS = [(2.0, 2.0), (4.0, 2.0), (6.0, 2.0), (4.0, 1.5), (4.0, 2.5), (6.0, 2.5), (2.0, 1.5)]


def validation_jobs() -> list[dict]:
    jobs = []
    for mL in (2.0, 4.0, 6.0):
        for sc in ("step", "pe"):
            for sampled in (None, 0.02):
                jobs.append({"kind": "linear", "m_L": mL, "L": 2.0, "scenario": sc, "sampled": sampled,
                             "seconds": 200.0, "design": V0, "arm": "online",
                             "id": f"lin_m{mL:g}_{sc}_{'cont' if sampled is None else 'zoh'}"})
    for mL, L in PLANTS:
        for sc in ("hover", "transfer"):
            for arm in ("online", "frozen", "softer"):
                jobs.append({"kind": "nonlinear", "m_L": mL, "L": L, "scenario": sc, "arm": arm, "seconds": 120.0,
                             "design": V0, "id": f"nl_m{mL:g}_L{L:g}_{sc}_{arm}"})
            jobs.append({"kind": "nonlinear", "m_L": mL, "L": L, "scenario": sc, "arm": "online", "seconds": 120.0,
                         "design": {**V0, "sigma": 0.1}, "id": f"sig_m{mL:g}_L{L:g}_{sc}"})
    for mL in (2.0, 6.0):
        jobs.append({"kind": "nonlinear", "m_L": mL, "L": 2.0, "scenario": "transfer", "arm": "online",
                     "seconds": 600.0, "design": V0, "id": f"long_m{mL:g}_transfer"})
    rbf = {"gamma_w": 50.0, "sigma_w": 0.0}
    rbf_s = {"gamma_w": 50.0, "sigma_w": 0.1}
    for wind, seeds in (("none", [0]), ("const", [0]), ("ou", [1, 2, 3]), ("gust", [0]), ("drag", [0])):
        for seed in seeds:
            for tag, design in (("V0", V0), ("rbf", {**V0, "rbf": rbf}), ("rbfs", {**V0, "rbf": rbf_s})):
                jobs.append({"kind": "nonlinear", "m_L": 6.0, "L": 2.0, "scenario": "transfer", "arm": "online",
                             "seconds": 120.0, "design": design, "wind": wind, "seed": seed,
                             "id": f"dist_{wind}{seed}_{tag}"})
            jobs.append({"kind": "nonlinear", "m_L": 6.0, "L": 2.0, "scenario": "transfer", "arm": "frozen",
                         "seconds": 120.0, "design": V0, "wind": wind, "seed": seed, "id": f"dist_{wind}{seed}_frozen"})
    return jobs


TUNE_PLANTS = [(2.0, 2.0), (4.0, 2.0), (6.0, 2.0)]


def tune_designs() -> list[dict]:
    """Protocol 03 section 4 grid after AM3 (54 designs, sigma = 0), fixed before T1 data."""
    out = []
    for gs in (5.0, 20.0, 80.0):
        for gc in (200.0, 500.0, 1000.0):
            for cy in (0.25, 1.0):
                for gu in (0.3, 1.0, 3.0):
                    out.append({"gamma_x": [gs] * 6 + [gc], "gamma_u": [gu, gu],
                                "gamma_cols": [cy, 1.0], "sigma": 0.0})
    return out


def design_tag(d: dict) -> str:
    return (f"gs{d['gamma_x'][0]:g}_gc{d['gamma_x'][6]:g}_cy{d['gamma_cols'][0]:g}"
            f"_gu{d['gamma_u'][0]:g}_s{d.get('sigma', 0.0):g}")


def tune_jobs() -> list[dict]:
    jobs = []
    for mL, L in TUNE_PLANTS:
        for sc in ("hover", "transfer"):
            for arm in ("softer", "pidff"):
                jobs.append({"kind": "nonlinear", "m_L": mL, "L": L, "scenario": sc, "arm": arm,
                             "seconds": 90.0, "design": None, "id": f"base_{arm}_m{mL:g}_L{L:g}_{sc}"})
            for d in tune_designs():
                jobs.append({"kind": "nonlinear", "m_L": mL, "L": L, "scenario": sc, "arm": "online",
                             "seconds": 90.0, "design": d, "tag": design_tag(d),
                             "id": f"{design_tag(d)}_m{mL:g}_L{L:g}_{sc}"})
    return jobs


HELDOUT_PLANTS = [(2.5, 2.0, 1.0), (5.5, 2.0, 1.0), (4.0, 1.5, 1.0), (4.0, 2.5, 1.0),
                  (6.0, 2.5, 1.0), (2.0, 1.5, 1.0), (4.0, 2.0, 1.2)]


# Protocol 06 (T2'): the T2 plants plus four never-run conditions.
UNSEEN_T2P = [(3.0, 2.0, 1.0), (5.0, 2.0, 1.0), (4.0, 1.75, 1.0), (4.0, 2.0, 0.85)]


def heldout_jobs(tuned: dict, plants=None) -> list[dict]:
    jobs = []
    for mL, L, sq in plants or HELDOUT_PLANTS:
        for sc in ("hover", "transfer", "transfer_fast"):
            base = {"kind": "nonlinear", "m_L": mL, "L": L, "m_q_scale": sq, "scenario": sc, "seconds": 120.0}
            cell = f"m{mL:g}_L{L:g}_q{sq:g}_{sc}"
            jobs.append({**base, "arm": "online", "design": tuned, "id": f"tuned_online_{cell}"})
            jobs.append({**base, "arm": "frozen", "design": tuned, "id": f"tuned_frozen_{cell}"})
            jobs.append({**base, "arm": "online", "design": V0, "id": f"v0_online_{cell}"})
            for arm in ("softer", "pidff"):
                jobs.append({**base, "arm": arm, "design": None, "id": f"base_{arm}_{cell}"})
    return jobs


def preflight(jobs: list[dict]) -> dict:
    from experiments.mrac.baselines import canonical_offline_params
    from experiments.mrac.vanilla_mrac import VanillaDesign, hover_tick_jacobian

    base = canonical_offline_params()
    seen, out = set(), {}
    for j in jobs:
        if j["kind"] != "nonlinear" or j["arm"] in ("softer", "pidff"):
            continue
        dkey = json.dumps(j["design"], sort_keys=True)
        key = (j["m_L"], j["L"], j.get("m_q_scale", 1.0), dkey)
        if key in seen:
            continue
        seen.add(key)
        canon = base.with_condition(m_L=j["m_L"], cable_length=j["L"])
        if j.get("m_q_scale", 1.0) != 1.0:
            canon = canon.with_vehicle_mismatch(m_q_scale=j["m_q_scale"]).with_condition(m_L=j["m_L"], cable_length=j["L"])
        p = dataclasses.replace(params_from(j["design"]), sigma=0.0)  # AM2
        r = hover_tick_jacobian(canon, VanillaDesign.build(canon, p), ideal=j["L"] == 2.0)
        rate = r["growth_rate_per_s"]
        certified = j["L"] == 2.0 and j.get("m_q_scale", 1.0) > 0
        out[f"m{j['m_L']:g}_L{j['L']:g}_q{j.get('m_q_scale', 1.0):g}_{hashlib.sha1(dkey.encode()).hexdigest()[:8]}"] = {
            "design": j["design"], "design_key": dkey, "rate_per_s": rate, "n_neutral": r["n_neutral"],
            "invariance_residual": r["invariance_residual"], "drift": r["equilibrium_drift"],
            "ideal_equilibrium": certified,
            "fails": bool(certified and (not math.isfinite(rate) or rate >= 0
                                         or r["invariance_residual"] > 1e-6 or r["equilibrium_drift"] > 1e-9))}
    return out


def sources(thesis_root: Path) -> dict:
    """SHA-256 of every runtime-relevant source, repository-relative keys (Codex 16 F4)."""
    thesis = ["experiments/mrac/vanilla_mrac.py", "experiments/mrac/baselines.py",
              "experiments/mrac/quad_load_pend.py", "experiments/mrac/quad.py", "experiments/mrac/wind.py",
              "experiments/mrac/diagnostics.py", "experiments/mrac/payload_law.py",
              "experiments/mrac/payload_reference.py", "experiments/mrac/law.py"]
    tutorial = ["mrac_lab/vanilla_runs.py", "mrac_lab/vanilla_analysis.py", "mrac_lab/provenance.py"]
    out = {f"thesis:{f}": hashlib.sha256((thesis_root / f).read_bytes()).hexdigest() for f in thesis}
    out.update({f"tutorial:{f}": hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in tutorial})
    return out


def _git(root: Path, *args: str) -> str:
    import subprocess

    r = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, timeout=30)
    return r.stdout.strip() if r.returncode == 0 else "UNKNOWN"


def _write(path: Path, obj) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True, choices=["validation", "tune", "heldout"])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--wall-cap", type=float, default=3600.0)
    ap.add_argument("--jobs-file", default=None, help="JSON list of jobs (tune / heldout stages)")
    ap.add_argument("--thesis-root", default=None)
    ap.add_argument("--make-jobs", default=None, help="write the stage's job list to this file and exit")
    ap.add_argument("--tuned", default=None, help="JSON design for the heldout stage")
    ap.add_argument("--t2prime", action="store_true", help="protocol 06 plant set (T2 + unseen)")
    args = ap.parse_args()
    if args.make_jobs:
        if args.stage == "tune":
            jobs = tune_jobs()
        elif args.stage == "heldout":
            plants = HELDOUT_PLANTS + UNSEEN_T2P if args.t2prime else None
            jobs = heldout_jobs(json.loads(args.tuned), plants)
        else:
            jobs = validation_jobs()
        with open(args.make_jobs, "x") as fh:  # F12: never overwrite a manifest
            json.dump(jobs, fh, indent=1)
        print(f"{len(jobs)} jobs -> {args.make_jobs}")
        return

    from mrac_lab import provenance
    import shutil
    from concurrent.futures import FIRST_COMPLETED, wait

    prov = provenance.setup(args.thesis_root, modules=("experiments.mrac.vanilla_mrac",))
    jobs = validation_jobs() if args.stage == "validation" else json.loads(Path(args.jobs_file).read_text())
    ids = [j["id"] for j in jobs]
    if len(ids) != len(set(ids)):  # F12
        raise SystemExit("duplicate job ids")
    stamp = time.strftime("%Y%m%dT%H%M%S")
    out = ROOT / "results" / f"mrac_vanilla_{args.stage}_{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    t0 = time.time()
    deadline = t0 + args.wall_cap
    receipt = {"stage": args.stage, "status": "started", "created": stamp,
               "thesis_commit": prov.commit, "branch": prov.branch, "dirty_paths": list(prov.dirty_paths),
               "dirty_sha256": {p: hashlib.sha256((prov.root / p).read_bytes()).hexdigest()
                                for p in prov.dirty_paths if (prov.root / p).is_file()},
               "tutorial_commit": _git(ROOT, "rev-parse", "HEAD"),
               "tutorial_dirty": _git(ROOT, "status", "--porcelain", "--", "mrac_lab"),
               "sources_sha256": sources(prov.root), "workers": args.workers, "wall_cap_s": args.wall_cap,
               "n_jobs": len(jobs), "label": "EXPLORATORY / offline / PROXY (development PC, reduced taut model)"}
    with open(out / "jobs.json", "x") as fh:
        json.dump(jobs, fh, indent=1)
    _write(out / "receipt.json", receipt)  # F5: initial receipt before any work
    results: list[dict] = []
    try:
        pre = preflight(jobs)
        receipt["preflight"] = pre
        receipt["preflight_wall_s"] = time.time() - t0
        # F3 / protocol 4 P4: exclude failing designs only (AM2: certified with sigma = 0).
        excluded = sorted({v["design_key"] for v in pre.values() if v["fails"]})
        receipt["excluded_designs"] = excluded
        run_jobs = [j for j in jobs if json.dumps(j["design"], sort_keys=True) not in set(excluded)
                    or j["arm"] in ("softer", "pidff")]
        receipt["excluded_jobs"] = [j["id"] for j in jobs if j not in run_jobs]
        receipt["status"] = "running"
        _write(out / "receipt.json", receipt)
        pool = ProcessPoolExecutor(args.workers, initializer=_init, initargs=(str(prov.root),))
        futs = {pool.submit(run_one, j, str(out), deadline): j for j in run_jobs}
        pending = set(futs)
        stop_reason = None
        while pending:
            done, pending = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
            for f in done:
                try:
                    results.append(f.result())
                except Exception as exc:  # preserved, never dropped
                    results.append({**futs[f], "status": "error", "error": repr(exc)})
                print(f"{len(results)}/{len(run_jobs)} {results[-1]['id']} {results[-1].get('status')} "
                      f"{time.time() - t0:.0f}s", flush=True)
            n_err = sum(r.get("status") == "error" for r in results)
            if len(results) >= 20 and n_err > 0.05 * len(results):
                stop_reason = "error_ratio"
            elif shutil.disk_usage(out).free < 5e9:
                stop_reason = "disk_below_5GB"
            elif time.time() > deadline:
                stop_reason = "wall_cap"
            if stop_reason:
                break
        if stop_reason:
            for f in pending:
                f.cancel()
            receipt["status"] = stop_reason
        pool.shutdown(wait=True, cancel_futures=True)  # running episodes stop at the deadline
        for f in pending:  # keep episodes that finished during shutdown (baselines ignore the deadline)
            if f.done() and not f.cancelled():
                try:
                    results.append(f.result())
                except Exception as exc:
                    results.append({**futs[f], "status": "error", "error": repr(exc)})
        done_ids = {r["id"] for r in results}
        receipt["not_run"] = [j["id"] for j in run_jobs if j["id"] not in done_ids]
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
