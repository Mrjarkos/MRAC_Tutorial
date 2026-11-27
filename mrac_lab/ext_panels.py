"""Interactive panels for NB11-NB13 (ipywidgets 8; inline matplotlib; no new packages).

* ``grid_panel(summary)``: heatmap of a metric over conditions x arms (arm checkboxes,
  metric / scenario / disturbance dropdowns).
* ``episode_panel(folders)``: time series of saved episodes; toggles for (i) temporal
  (task error, swing), (ii) adaptive gains with ideal lines, (iii) Lyapunov variables
  (``V_e``, ``||e||``, log scale).
* ``live_panel(variant)``: gain sliders + "run" button for ONE short episode of the
  nonlinear law (never runs automatically).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ARM_ORDER = ["a_online", "a_frozen", "ab_online", "ab_frozen", "abc_online", "abc_frozen",
             "v0_online", "softer", "pidff"]
METRICS = ["task_rms_last60", "swing_rms_10_30", "swing_rms_last60", "sat_frac", "kick0_swing_rms_5s",
           "kick0_task_max_5s", "kick0_settle_s", "ref_err_rms_last60"]


def load_summaries(folders) -> pd.DataFrame:
    frames = []
    for f in folders:
        f = Path(f)
        df = pd.DataFrame(json.loads((f / "summary.json").read_text()))
        df["folder"] = str(f)
        if "source_folder" in df:  # composite: CSVs live in the part folders
            df["folder"] = [str(f.parent / sf) if isinstance(sf, str) else str(f) for sf in df.source_folder]
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    # shared arms (TUNED', baselines) appear only in stage a; keep one copy per id
    return df.drop_duplicates("id", keep="first")


def grid_panel(summary: pd.DataFrame):
    import ipywidgets as W
    from IPython.display import display

    arms = [a for a in ARM_ORDER if a in set(summary.arm)]
    metric = W.Dropdown(options=[m for m in METRICS if m in summary], value="task_rms_last60", description="metric")
    scen = W.Dropdown(options=sorted(summary.scenario.unique()), description="scenario")
    dist = W.Dropdown(options=sorted(summary.dist.unique()), value="none", description="disturbance")
    tag = W.Dropdown(options=sorted(summary.tag.unique()), value="grid", description="set")
    boxes = [W.Checkbox(value=True, description=a, indent=False, layout=W.Layout(width="120px")) for a in arms]
    logc = W.Checkbox(value=True, description="log colour", indent=False)
    out = W.Output()

    def draw(*_):
        with out:
            out.clear_output(wait=True)
            sel = [b.description for b in boxes if b.value]
            d = summary[(summary.scenario == scen.value) & (summary.dist == dist.value) & (summary.tag == tag.value)
                        & summary.arm.isin(sel)]
            if d.empty:
                print("no data for this selection")
                return
            tab = d.pivot_table(index="label", columns="arm", values=metric.value, aggfunc="mean")
            tab = tab[[a for a in sel if a in tab.columns]]
            vals = tab.to_numpy(dtype=float)
            fig, ax = plt.subplots(figsize=(1.3 * len(tab.columns) + 3, 0.32 * len(tab) + 1.5))
            show = np.log10(np.clip(vals, 1e-7, None)) if logc.value else vals
            im = ax.imshow(show, aspect="auto", cmap="viridis_r")
            ax.set_xticks(range(len(tab.columns)), tab.columns, rotation=30, ha="right")
            ax.set_yticks(range(len(tab)), tab.index)
            for i in range(vals.shape[0]):
                for j in range(vals.shape[1]):
                    if np.isfinite(vals[i, j]):
                        ax.text(j, i, f"{vals[i, j]:.2g}", ha="center", va="center", fontsize=7, color="w")
            fig.colorbar(im, ax=ax, label=("log10 " if logc.value else "") + metric.value)
            fails = d[d.status != "completed"]
            ax.set_title(f"{metric.value} | {scen.value} | {dist.value} | failed episodes: {len(fails)}")
            plt.show()

    for w in [metric, scen, dist, tag, logc, *boxes]:
        w.observe(draw, names="value")
    display(W.VBox([W.HBox([metric, scen, dist, tag, logc]), W.HBox(boxes), out]))
    draw()


def episode_panel(summary: pd.DataFrame):
    import ipywidgets as W
    from IPython.display import display

    labels = sorted(summary.label.unique())
    cond = W.Dropdown(options=labels, value="mb6" if "mb6" in labels else labels[0], description="condition")
    scen = W.Dropdown(options=sorted(summary.scenario.unique()), value="transfer", description="scenario")
    dist = W.Dropdown(options=sorted(summary.dist.unique()), value="none", description="disturbance")
    tag = W.Dropdown(options=sorted(summary.tag.unique()), value="grid", description="set")
    arms = [a for a in ARM_ORDER if a in set(summary.arm)]
    boxes = [W.Checkbox(value=a in ("a_online", "softer", "v0_online", "ab_online", "abc_online"), description=a,
                        indent=False, layout=W.Layout(width="120px")) for a in arms]
    groups = {g: W.Checkbox(value=True, description=g, indent=False)
              for g in ("(i) temporal", "(ii) gains", "(iii) Lyapunov")}
    out = W.Output()

    def draw(*_):
        with out:
            out.clear_output(wait=True)
            sel = [b.description for b in boxes if b.value]
            d = summary[(summary.label == cond.value) & (summary.scenario == scen.value) & (summary.dist == dist.value)
                        & (summary.tag == tag.value) & summary.arm.isin(sel)]
            if d.empty:
                print("no episodes for this selection (disturbances run on 6 representative conditions)")
                return
            on = [g for g, w in groups.items() if w.value]
            if not on:
                print("tick at least one panel group (temporal, gains or Lyapunov)")
                return
            fig, axes = plt.subplots(len(on), 2, figsize=(14, 3.2 * len(on)), squeeze=False, sharex=True)
            for _, r in d.iterrows():
                df = pd.read_csv(Path(r.folder) / f"{r.id}.csv")
                lab = f"{r.arm} ({r.status})"
                row = 0
                if "(i) temporal" in on:
                    axes[row, 0].plot(df.t, np.hypot(df.y, df.z), label=lab)
                    axes[row, 1].plot(df.t, np.degrees(df.phi), label=lab)
                    axes[row, 0].set_ylabel("task error |r_q - r^d| [m]")
                    axes[row, 1].set_ylabel("swing φ_code [deg] (φ_tex = −φ_code)")
                    row += 1
                if "(ii) gains" in on:
                    if "k_u" in df:
                        for c, ls in (("k_t", "-"), ("k_u", "--")):
                            axes[row, 0].plot(df.t, df[c], ls, label=f"{lab} {c}")
                            axes[row, 0].axhline(df[c + "_ideal"].iloc[0], color="k", lw=0.6, ls=ls)
                        axes[row, 1].plot(df.t, df.k_phi, label=f"{lab} k_phi")
                        axes[row, 1].axhline(df.k_phi_ideal.iloc[0], color="k", lw=0.6)
                        if "W_norm_t" in df:
                            axes[row, 1].plot(df.t, df.W_norm_t, ":", label=f"{lab} |W_t|")
                            axes[row, 1].plot(df.t, df.W_norm_u, ":", label=f"{lab} |W_u|")
                    elif "Kx_one_Fz" in df:
                        axes[row, 0].plot(df.t, df.Kx_one_Fz / df.Kx_one_Fz.iloc[0], label=f"{lab} Kx[1,Fz]/init")
                        axes[row, 1].plot(df.t, df.Kx_phi_Fy, label=f"{lab} Kx[φ,Fy]")
                    axes[row, 0].set_ylabel("channel gains (black = ideal)")
                    axes[row, 1].set_ylabel("k_φ, |W| (black = ideal)")
                    row += 1
                if "(iii) Lyapunov" in on and "V_e" in df:
                    axes[row, 0].semilogy(df.t, df.V_e + 1e-16, label=lab)
                    axes[row, 1].semilogy(df.t, df.norm_e + 1e-16, label=lab)
                    axes[row, 0].set_ylabel("V_e = eᵀPe")
                    axes[row, 1].set_ylabel("||e|| (plant − reference)")
            for a in axes.flat:
                a.grid(alpha=0.3)
                a.legend(fontsize=6)
            for a in axes[-1]:
                a.set_xlabel("t [s]")
            kicks = d.dist.iloc[0] in ("payload_kick", "vehicle_kick")
            if kicks:
                for a in axes.flat:
                    for tk in (30, 60):
                        a.axvline(tk, color="crimson", lw=0.8, ls=":")
            fig.suptitle(f"{cond.value} | {scen.value} | {dist.value}" + (" (kicks at 30 s, 60 s)" if kicks else ""))
            plt.show()

    for w in [cond, scen, dist, tag, *boxes, *groups.values()]:
        w.observe(draw, names="value")
    display(W.VBox([W.HBox([cond, scen, dist, tag]), W.HBox(boxes), W.HBox(list(groups.values())), out]))
    draw()


def live_panel(variant: str = "a"):
    """One short live episode of the nonlinear law with user gains (button press only)."""
    import ipywidgets as W
    from IPython.display import display

    from mrac_lab.extensions_runs import KICKS, VARIANTS, make_canon
    from mrac_lab.vanilla_runs import disturbance, scenario

    v = VARIANTS[variant]
    s = {
        "gamma_t": W.FloatLogSlider(value=0.0078, base=10, min=-4, max=0, description="γ_t"),
        "gamma_u": W.FloatLogSlider(value=0.03, base=10, min=-4, max=-0.5, description="γ_u"),
        "gamma_phi": W.FloatLogSlider(value=0.37, base=10, min=-3, max=1, description="γ_φ"),
        "sigma": W.FloatSlider(value=v.get("sigma", 0.0), min=0, max=0.5, step=0.01, description="σ"),
        "sigma_w": W.FloatSlider(value=v.get("rbf", {}).get("sigma_w", 0.0), min=0, max=1, step=0.05, description="σ_W"),
    }
    use_rbf = W.Checkbox(value="rbf" in v, description="RBF weights (b)")
    m_b = W.FloatSlider(value=6.0, min=1.0, max=9.0, step=0.5, description="m_b [kg]")
    ell = W.FloatSlider(value=2.0, min=1.5, max=3.0, step=0.25, description="ℓ [m]")
    mq = W.FloatSlider(value=1.0, min=0.7, max=1.3, step=0.05, description="m_q scale")
    scen = W.Dropdown(options=["hover", "transfer", "transfer_fast"], value="transfer", description="scenario")
    dist = W.Dropdown(options=["none", "const", "ou", "drag", "gust", "payload_kick", "vehicle_kick"], description="disturbance")
    secs = W.IntSlider(value=60, min=20, max=200, step=10, description="seconds")
    btn = W.Button(description="run episode", button_style="primary")
    out = W.Output()

    def run(_):
        from experiments.mrac.baselines import canonical_offline_params  # noqa: F401
        from experiments.mrac.diagnostics import stop_reason
        from experiments.mrac.vanilla_mrac import rbf_grid
        from experiments.mrac.vanilla_nl_mrac import NLDesign, NLParams, run_nl

        with out:
            out.clear_output(wait=True)
            job = {"m_L": m_b.value, "L": ell.value, "vehicle": {"m_q_scale": mq.value} if mq.value != 1.0 else {}}
            canon = make_canon(job)
            rbf = rbf_grid(gamma_w=50.0 / 13.6, sigma_w=s["sigma_w"].value) if use_rbf.value else None
            p = NLParams(gamma_t=s["gamma_t"].value, gamma_u=s["gamma_u"].value, gamma_phi=s["gamma_phi"].value,
                         sigma=s["sigma"].value, rbf=rbf)
            n = int(secs.value / 0.02)
            traj, ic = scenario(scen.value, canon)
            kicks, wind, dfun, note = {}, None, None, ""
            if dist.value in KICKS:
                kicks = {kt: KICKS[dist.value] for kt in (1500, 3000) if kt < n}  # at 30 s and 60 s
                if len(kicks) < 2:
                    note = f"note: the 60 s kick needs more than 60 s of episode; only {len(kicks)} kick(s) applied"
            else:
                wind, dfun = disturbance(dist.value, n, 1)
            print("running ...")
            lg = run_nl(canon, NLDesign.build(canon, p), traj, n, ic, wind=wind, disturbance=dfun, kicks=kicks,
                        stop=stop_reason, log_every=5)
            df = pd.DataFrame(lg.rows)
            out.clear_output(wait=True)
            if note:
                print(note)  # printed after the clear, so it stays visible with the figure
            print("failure:", lg.failure)
            fig, ax = plt.subplots(1, 4, figsize=(16, 3.3))
            ax[0].plot(df.t, np.hypot(df.y, df.z)); ax[0].set_title("task error [m]")
            ax[1].plot(df.t, np.degrees(df.phi)); ax[1].set_title("swing [deg]")
            for c in ("k_t", "k_u"):
                ax[2].plot(df.t, df[c], label=c); ax[2].axhline(df[c + "_ideal"].iloc[0], color="k", lw=0.6)
            ax[2].legend(); ax[2].set_title("gains (black = ideal)")
            ax[3].semilogy(df.t, df.norm_e + 1e-16); ax[3].set_title("||e||")
            for a in ax:
                a.grid(alpha=0.3); a.set_xlabel("t [s]")
            plt.show()

    btn.on_click(run)
    display(W.VBox([W.HBox(list(s.values())[:3]), W.HBox(list(s.values())[3:] + [use_rbf]),
                    W.HBox([m_b, ell, mq]), W.HBox([scen, dist, secs, btn]), out]))


__all__ = ["episode_panel", "grid_panel", "live_panel", "load_summaries"]
