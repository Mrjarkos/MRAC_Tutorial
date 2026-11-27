"""Signal-flow block diagrams for the MRAC tutorial notebooks (matplotlib patches only).

``draw_vanilla_extensions(variant)`` draws the reference loop, the error and command path,
the adaptive law, the cascade to the motors and the plant with disturbances, with every
arrow labelled by its residual.tex symbol and every summing junction by its signs.
Blocks of variants that are off are drawn dashed. Signal paths are orthogonal polylines;
a dot marks a branch point. No new packages.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path

COLORS = {"ref": "#E8F1FB", "law": "#FFF4E0", "adapt": "#EAF7EA", "plant": "#F3E8F8", "inner": "#F2F2F2"}
PAD = 0.02  # outward padding of the rounded block outline (data units)
R_SUM = 0.18  # radius of a summing junction
FS = 9.5  # block and signal font size (points)


class Diagram:
    def __init__(self, ax):
        self.ax = ax

    def block(self, x, y, w, h, text, kind="law", dashed=False, fs=FS, text_dy=0.0):
        p = FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle=f"round,pad={PAD},rounding_size=0.08",
                           fc=COLORS[kind], ec="0.25", lw=1.2, ls="--" if dashed else "-")
        self.ax.add_patch(p)
        self.ax.text(x, y + text_dy, text, ha="center", va="center", fontsize=fs, alpha=0.5 if dashed else 1.0)
        return (x, y, w, h)

    def sum(self, x, y, signs):
        """Summing junction; ``signs`` is a sequence of ``(sign, dx, dy)``, each beside its input."""
        self.ax.add_patch(Circle((x, y), R_SUM, fc="white", ec="0.25", lw=1.2))
        self.ax.text(x, y, "Σ", ha="center", va="center", fontsize=10)
        for sign, dx, dy in signs:
            self.ax.text(x + dx, y + dy, sign, ha="center", va="center", fontsize=12, color="crimson")
        return (x, y, 2 * R_SUM, 2 * R_SUM)

    def wire(self, points, label="", at=None, ha="center", va="bottom", ls="-"):
        """Arrow along the polyline ``points``; ``label`` is anchored at the point ``at``."""
        self.ax.add_patch(FancyArrowPatch(path=Path(points), arrowstyle="-|>", mutation_scale=11, lw=1.1,
                                          color="0.15", ls=ls))
        if label:
            if at is None:  # default anchor: midpoint of the first segment
                (x0, y0), (x1, y1) = points[0], points[1]
                at = ((x0 + x1) / 2, (y0 + y1) / 2)
            self.ax.text(*at, label, ha=ha, va=va, fontsize=FS, color="navy")

    def dot(self, x, y):
        self.ax.add_patch(Circle((x, y), 0.055, fc="0.15", ec="0.15", zorder=3))


def draw_vanilla_extensions(variant: str = "a", ax=None):
    """variant in {'vanilla', 'a', 'ab', 'abc'}: blocks of disabled features are dashed."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(13.4, 7.4))
        fig.subplots_adjust(left=0.005, right=0.995, bottom=0.005, top=0.945)
    ax.set_xlim(0, 17)
    ax.set_ylim(0, 9)
    ax.set_aspect("equal")
    ax.axis("off")
    D = Diagram(ax)
    nl = variant in ("a", "ab", "abc")
    b, c = variant in ("ab", "abc"), variant == "abc"
    # rows (top: reference loop and error; middle: law and adaptation; bottom: plant) and the command column
    top, mid, low = 7.6, 4.2, 1.5
    xc = 14.2
    # reference loop (top)
    D.block(0.95, top, 1.5, 0.7, "task $x^d(t)$\n(min-jerk)", "ref")
    D.block(3.7, top, 2.7, 0.75, "PID+FF ($\\hat m_b{=}4$)\n+ linear damper on $\\varphi_r{-}\\varphi_{eq}$", "ref")
    D.block(7.6, top, 3.3, 0.75, "nominal plant  $m_{q,r},\\ m_{b,r}{=}4,\\ \\ell_r{=}2$\n(reference model, Guha $x_r$)", "ref")
    D.wire([(1.72, top), (2.33, top)], "$x^d$", at=(2.02, top + 0.07))
    D.wire([(5.07, top), (5.93, top)], "$u_r$", at=(5.7, top - 0.09), va="top")
    D.wire([(9.27, top), (10.2 - R_SUM, top)], "$x_r$", at=(9.65, top + 0.07))
    D.wire([(7.6, top - 0.395), (7.6, 6.6), (3.7, 6.6), (3.7, top - 0.395)])
    # error and nominal force
    D.sum(10.2, top, [("-", -0.32, 0.27), ("+", 0.3, -0.3)])  # x_r from the left (-), x from below (+): e = x - x_r
    D.wire([(10.2 + R_SUM, top), (11.73, top)], "$e=x-x_r$", at=(11.05, top + 0.07))
    D.block(12.5, top, 1.5, 0.75, "$K_e^\\top$ (PD +\ndamper equiv.)", "law", fs=9)
    D.sum(xc, top, [("+", 0.3, 0.32), ("-", -0.32, -0.28)])  # u_r from above (+), K_e^T e from the left (-)
    D.wire([(13.27, top), (xc - R_SUM, top)], "$K_e^\\top e$", at=(13.64, top + 0.07))
    D.dot(5.45, top)
    # vanilla: xi = u_r - K_e^T e; nonlinear variants: F_n = u_r + M_r g e_2 - K_e^T e
    D.wire([(5.45, top), (5.45, 8.5), (xc, 8.5), (xc, top + R_SUM)], "$u_r + M_r g\\,e_2$" if nl else "$u_r$",
           at=(9.8, 8.56))
    # command path
    if nl:
        D.block(xc, 5.95, 3.2, 1.0, "cable frame $R(\\varphi)=[\\hat t\\ \\hat q]$\n$\\xi_t=\\hat t^\\top F_n$,  $\\xi_u=\\hat q^\\top F_n+m_{b,r}\\ell\\dot\\varphi^2$", "law", fs=9)
        D.wire([(xc, top - R_SUM), (xc, 6.47)], "$F_n$", at=(xc + 0.12, 6.94), ha="left", va="center")
        D.block(xc, mid, 3.2, 1.0, "$w_t=\\hat k_t\\xi_t$\n$w_u=\\hat k_u\\xi_u+\\hat k_\\varphi\\Phi_\\varphi$  ;  $F=R\\,w$", "law", fs=9)
        D.wire([(xc, 5.43), (xc, mid + 0.52)], "$\\xi,\\Phi_\\varphi$", at=(xc + 0.12, 5.07), ha="left", va="center")
    else:
        D.block(xc, mid, 3.2, 1.0, "$u=\\hat K_x^\\top\\bar x+\\hat K_u^\\top\\xi$\n$\\xi=u_r-K_e^\\top e$", "law")
        D.wire([(xc, top - R_SUM), (xc, mid + 0.52)], "$\\xi$", at=(xc + 0.12, 5.95), ha="left", va="center")
    D.block(16.05, 2.85, 1.35, 0.75, "$+\\hat W^\\top\\Phi_d(v)$\n(RBF, b)", "law", dashed=not b, fs=9)
    # into the law block: the code adds the RBF term to w before F = R w (to u in the vanilla variant)
    D.wire([(15.55, 2.85 + 0.395), (15.55, mid - 0.52)], ls="--" if not b else "-")
    # adaptation
    D.block(9.5, mid, 4.6, 1.5,
            ("$\\dot{\\hat k}=-\\gamma\\,\\mathrm{reg}\\,(e^\\top P B_e(\\varphi))_{ch}$" if nl else
             "$\\dot{\\hat K}_x=-\\Gamma_x\\odot(\\bar x e^\\top P B_r)$\n$\\dot{\\hat K}_u=-\\Gamma_u\\odot(\\xi e^\\top P B_r)$")
            + ("\n$-\\sigma(\\hat k-k_0)$,  $-\\sigma_W\\hat W$  (c)" if c else ""),  # leak terms exist only in (c)
            "adapt", text_dy=0.0 if c else 0.14)
    ax.text(9.5, 3.28, "projection on boxes; $PA_H+A_H^\\top P=-Q$", ha="center", va="top", fontsize=9)
    D.wire([(11.82, mid), (xc - 1.62, mid)], "gains", at=(12.2, mid + 0.07))
    D.dot(11.15, top)
    D.wire([(11.15, top), (11.15, mid + 0.77)], "$e$", at=(11.27, 6.0), ha="left", va="center")
    if not c:
        ax.text(9.5, 3.7 if nl else 3.62, "(σ terms off in this variant)", ha="center", va="center", fontsize=8,
                color="0.4")
    # cascade and plant (bottom)
    D.block(xc, low, 3.2, 1.0, "$f_q=\\|F\\|$,  $\\theta_d=\\mathrm{atan2}(-F_y,F_z)$\nattitude PD + $\\omega_d$ FF (+ ref. damper $\\tau$)", "inner", fs=9)
    D.wire([(xc, mid - 0.52), (xc, low + 0.52)], "$F$ (world force)", at=(xc - 0.12, 2.85), ha="right", va="center")
    D.block(9.9, low, 2.4, 0.85, "allocation $B^{-1}$\nmotors $\\tau_m$ (1st order)", "inner")
    D.wire([(xc - 1.62, low), (11.12, low)], "$u=[f_q,\\tau_q]$", at=(11.85, low + 0.07))
    # nonlinear variants run at d = 0 (protocol 08); the vanilla (NB10) episodes use the
    # canonical attachment offset d = 0.3795 m (lane C item C6)
    D.block(5.7, low, 3.2, 0.95,
            "real plant  $m_q, J_q, m_b, \\ell$\n" + ("(taut; $d=0$)" if nl else "(taut; $d=0.3795$ m)"), "plant")
    D.wire([(8.68, low), (7.32, low)], "$T_{1,2}$", at=(8.0, low + 0.07))
    D.block(5.7, 0.35, 4.4, 0.45, "$\\mathscr{D}=[d_{q,y}, d_{q,z}, d_{q,\\tau}, d_\\varphi]$: wind, drag, gust, kicks", "plant", fs=9)
    D.wire([(5.7, 0.595), (5.7, low - 0.495)])
    D.wire([(4.08, low), (0.45, low), (0.45, 5.9), (10.2, 5.9), (10.2, top - R_SUM)],
           "$x=[y_q, v_{q,y}, z_q, v_{q,z}, \\varphi, \\dot\\varphi]$", at=(0.6, 3.7), ha="left", va="center")
    ax.text(10.33, 6.6, "$x$", ha="left", va="center", fontsize=FS, color="navy")
    title = {"vanilla": "Vanilla MRAC (EQ-MR-22): linear outer model", "a": "(a) nonlinear-model MRAC (EQ-MR-23)",
             "ab": "(a)+(b) + disturbance weights", "abc": "(a)+(b)+(c): disturbance weights + σ-modification"}[variant]
    ax.set_title(title + "   (signs at Σ; dashed = off)", fontsize=12)
    return ax


__all__ = ["Diagram", "draw_vanilla_extensions"]
