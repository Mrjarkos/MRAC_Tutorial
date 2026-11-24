"""Standard panels drawn from plain arrays. No physics, no simulation.

A *record* is a ``dict`` of equal-length 1-D/2-D numpy arrays. Keys used here
(any missing key is skipped, nothing is invented):

``t`` time [s]; ``p`` vehicle position ``(n, 2)`` [m] (y, z); ``ref_p`` reference
position ``(n, 2)``; ``theta`` [rad]; ``phi``, ``phi_dot`` cable angle/rate
[rad], [rad/s]; ``phi_eq`` equilibrium swing angle [rad]; ``T`` cable tension [N];
``f_req``, ``tau_req`` requested wrench; ``f_alloc``, ``tau_alloc`` allocated
wrench; ``sat`` boolean saturation flag per tick.

``runs`` is ``{arm_key: record}`` with ``arm_key`` from ``mrac_lab.style``.
Angles are plotted in degrees.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from . import style

Record = Mapping[str, np.ndarray]
Runs = Mapping[str, Record]

_DEG = 180.0 / np.pi


def _grid(ax) -> None:
    ax.grid(True, alpha=style.GRID_ALPHA)


def _ref_line(ax, t, y, label: str = "reference") -> None:
    ax.plot(t, y, **{**style.arm_style("ref"), "label": label})


def panel_trajectory(ax, runs: Runs, component: int = 0, ref: Optional[Record] = None) -> None:
    """Vehicle position component (0: y, 1: z) against the reference."""
    name = ("y", "z")[component]
    if ref is not None and "ref_p" in ref:
        _ref_line(ax, ref["t"], np.asarray(ref["ref_p"])[:, component])
    for key, rec in runs.items():
        ax.plot(rec["t"], np.asarray(rec["p"])[:, component], **style.arm_style(key))
    ax.set_xlabel("t [s]")
    ax.set_ylabel(f"vehicle ${name}$ [m]")
    ax.set_title(f"Vehicle {name}-position vs reference")
    ax.legend(loc="best", fontsize=7)
    _grid(ax)


def panel_swing(ax, runs: Runs, show_eq: bool = True, ref: Optional[Record] = None) -> None:
    """Cable angle ``phi`` and ``phi_eq`` in degrees."""
    for key, rec in runs.items():
        ax.plot(rec["t"], np.asarray(rec["phi"]) * _DEG, **style.arm_style(key))
    src = ref if ref is not None else next(iter(runs.values()), {})
    if show_eq and "phi_eq" in src:
        ax.plot(src["t"], np.asarray(src["phi_eq"]) * _DEG, **{**style.arm_style("ref"), "label": r"$\phi_{eq}$"})
    ax.axhline(0.0, color="0.7", lw=0.6)
    ax.set_xlabel("t [s]")
    ax.set_ylabel(r"swing $\phi$ [deg]")
    ax.set_title(r"Cable angle $\phi$ (0 = hanging straight down)")
    ax.legend(loc="best", fontsize=7)
    _grid(ax)


def panel_tension(ax, runs: Runs, T_nom: float) -> None:
    """``T / T_nom`` with the nominal line (1) and the slack line (0)."""
    for key, rec in runs.items():
        ax.plot(rec["t"], np.asarray(rec["T"]) / T_nom, **style.arm_style(key))
    ax.axhline(1.0, **{**style.arm_style("ref"), "label": r"$T_{nom}=m_L g$"})
    ax.axhline(0.0, color=style.SLACK_COLOR, lw=0.9, ls="-", label="slack ($T=0$)")
    ax.set_xlabel("t [s]")
    ax.set_ylabel(r"$T/T_{nom}$")
    ax.set_title("Cable tension (taut model requires $T>0$)")
    ax.legend(loc="best", fontsize=7)
    _grid(ax)


def panel_wrench(ax_f, ax_tau, runs: Runs) -> None:
    """Requested (thin) against allocated (thick) thrust and torque."""
    for key, rec in runs.items():
        st = style.arm_style(key)
        if "f_req" in rec:
            ax_f.plot(rec["t"], rec["f_req"], **{**st, "alpha": 0.45, "label": st["label"] + " (req)"})
        if "f_alloc" in rec:
            ax_f.plot(rec["t"], rec["f_alloc"], **{**st, "linewidth": st["linewidth"] + 0.5, "label": st["label"] + " (alloc)"})
        if "tau_req" in rec:
            ax_tau.plot(rec["t"], rec["tau_req"], **{**st, "alpha": 0.45, "label": st["label"] + " (req)"})
        if "tau_alloc" in rec:
            ax_tau.plot(rec["t"], rec["tau_alloc"], **{**st, "linewidth": st["linewidth"] + 0.5, "label": st["label"] + " (alloc)"})
        if "sat" in rec and np.any(rec["sat"]):
            idx = np.flatnonzero(np.asarray(rec["sat"], dtype=bool))
            ax_f.plot(np.asarray(rec["t"])[idx], np.zeros(idx.size), "|", color=style.SAT_COLOR, ms=6)
    ax_f.set_ylabel("collective $f$ [N]")
    ax_tau.set_ylabel(r"moment $\tau$ [N m]")
    for ax in (ax_f, ax_tau):
        ax.set_xlabel("t [s]")
        ax.legend(loc="best", fontsize=6)
        _grid(ax)
    ax_f.set_title("Wrench requested vs allocated (orange ticks: saturated)")


def panel_phase(ax, runs: Runs) -> None:
    """Phase portrait ``phi`` against ``phi_dot`` (degrees)."""
    for key, rec in runs.items():
        phi = np.asarray(rec["phi"]) * _DEG
        dphi = np.asarray(rec["phi_dot"]) * _DEG
        ax.plot(phi, dphi, **style.arm_style(key))
        ax.plot(phi[0], dphi[0], "o", color=style.arm_color(key), ms=4)
        ax.plot(phi[-1], dphi[-1], "*", color=style.arm_color(key), ms=8)
    ax.set_xlabel(r"$\phi$ [deg]")
    ax.set_ylabel(r"$\dot\phi$ [deg/s]")
    ax.set_title("Swing phase portrait (o start, * end)")
    ax.legend(loc="best", fontsize=7)
    _grid(ax)


def standard_figure(
    runs: Runs,
    T_nom: float,
    ref: Optional[Record] = None,
    title: str = "",
) -> plt.Figure:
    """The six standard panels: y, z, swing, tension, f/tau, phase portrait."""
    fig, axes = plt.subplots(3, 2, figsize=(11, 9))
    panel_trajectory(axes[0, 0], runs, 0, ref)
    panel_trajectory(axes[0, 1], runs, 1, ref)
    panel_swing(axes[1, 0], runs, ref=ref)
    panel_tension(axes[1, 1], runs, T_nom)
    panel_wrench(axes[2, 0], axes[2, 1], runs)
    fig.suptitle(title + "   [EXPLORATORY / offline / PROXY]", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def heatmap_grid(
    ax,
    lengths: Sequence[float],
    masses: Sequence[float],
    values: np.ndarray,
    title: str,
    nominal: Optional[tuple[float, float]] = None,
    center: Optional[float] = 1.0,
    cbar_label: str = "",
    fmt: str = "{:.2f}",
    failed: Optional[np.ndarray] = None,
):
    """Heatmap over (cable length, payload mass). ``values[i_mass, j_length]``.

    ``center`` draws a diverging scale around that value (1.0 for a ratio to
    arm 1: below 1 is better). ``failed`` marks conditions that exited with an
    ``x`` and no number. Cells with NaN are left blank.
    """
    vals = np.asarray(values, dtype=float)
    finite = vals[np.isfinite(vals)]
    if center is not None and finite.size:
        span = max(abs(finite.max() - center), abs(finite.min() - center), 1e-9)
        im = ax.imshow(vals, origin="lower", aspect="auto", cmap="RdBu_r", vmin=center - span, vmax=center + span)
    else:
        im = ax.imshow(vals, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(lengths)), [f"{x:g}" for x in lengths])
    ax.set_yticks(range(len(masses)), [f"{x:g}" for x in masses])
    ax.set_xlabel("cable length $l$ [m]")
    ax.set_ylabel("payload mass $m_L$ [kg]")
    ax.set_title(title, fontsize=8)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            if failed is not None and failed[i, j]:
                ax.text(j, i, "x", ha="center", va="center", color="k", fontsize=9, fontweight="bold")
            elif np.isfinite(vals[i, j]):
                ax.text(j, i, fmt.format(vals[i, j]), ha="center", va="center", fontsize=6.5, color="k")
    if nominal is not None:
        l0, m0 = nominal
        ax.plot(list(lengths).index(l0), list(masses).index(m0), "s", mfc="none", mec="k", ms=14, mew=1.4)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(cbar_label, fontsize=7)
    ax.grid(False)
    return im


def tabulate(rows: Sequence[Mapping[str, object]]):
    """Small pandas table for the 'Quantitative table' section (4 significant digits)."""
    import pandas as pd

    df = pd.DataFrame(rows)
    return df.style.format(precision=4, na_rep="n/a").hide(axis="index")
