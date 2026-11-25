"""Signal-flow block diagrams for the MRAC tutorial notebooks (matplotlib patches only).

``draw_vanilla_extensions(variant)`` draws the reference loop, the error and command path,
the adaptive law, the cascade to the motors and the plant with disturbances, with every
arrow labelled by its residual.tex symbol and every summing junction by its signs.
Blocks of variants that are off are drawn dashed. No new packages.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

COLORS = {"ref": "#E8F1FB", "law": "#FFF4E0", "adapt": "#EAF7EA", "plant": "#F3E8F8", "inner": "#F2F2F2"}


class Diagram:
    def __init__(self, ax):
        self.ax = ax

    def block(self, x, y, w, h, text, kind="law", dashed=False, fs=8.5):
        p = FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                           fc=COLORS[kind], ec="0.25", lw=1.2, ls="--" if dashed else "-")
        self.ax.add_patch(p)
        self.ax.text(x, y, text, ha="center", va="center", fontsize=fs, alpha=0.5 if dashed else 1.0)
        return (x, y, w, h)

    def sum(self, x, y, signs=("+", "-"), r=0.16):
        self.ax.add_patch(Circle((x, y), r, fc="white", ec="0.25", lw=1.2))
        self.ax.text(x, y, "Σ", ha="center", va="center", fontsize=9)
        offs = [(-0.22, 0.2), (0.12, -0.3), (-0.22, -0.2)]
        for s, (dx, dy) in zip(signs, offs):
            self.ax.text(x + dx, y + dy, s, fontsize=9, color="crimson")
        return (x, y, 2 * r, 2 * r)

    def arrow(self, a, b, label="", dy=0.12, dx=0.0, color="0.15", ls="-", rad=0.0):
        self.ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=11, lw=1.1, color=color,
                                          ls=ls, connectionstyle=f"arc3,rad={rad}"))
        if label:
            self.ax.text((a[0] + b[0]) / 2 + dx, (a[1] + b[1]) / 2 + dy, label, ha="center", va="bottom",
                         fontsize=8, color="navy")


def draw_vanilla_extensions(variant: str = "a", ax=None):
    """variant in {'vanilla', 'a', 'ab', 'abc'}: blocks of disabled features are dashed."""
    if ax is None:
        _, ax = plt.subplots(figsize=(15, 7.2))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 7.4)
    ax.axis("off")
    D = Diagram(ax)
    nl = variant in ("a", "ab", "abc")
    b, c = variant in ("ab", "abc"), variant == "abc"
    # reference loop (top)
    D.block(1.0, 6.6, 1.5, 0.6, "task $x^d(t)$\n(min-jerk)", "ref")
    D.block(3.6, 6.6, 2.4, 0.7, "PID+FF ($\\hat m_b{=}4$)\n+ linear damper on $\\varphi_r{-}\\varphi_{eq}$", "ref")
    D.block(7.2, 6.6, 3.0, 0.7, "nominal plant  $m_{q,r},\\ m_{b,r}{=}4,\\ \\ell_r{=}2$\n(reference model, Guha $x_r$)", "ref")
    D.arrow((1.75, 6.6), (2.4, 6.6), "$x^d$")
    D.arrow((4.8, 6.6), (5.7, 6.6), "$u_r$")
    D.arrow((8.7, 6.6), (9.8, 6.6), "$x_r$")
    D.arrow((7.2, 6.25), (3.6, 6.25), "", rad=0.25)
    # error and nominal force
    D.sum(10.1, 6.6, ("-", "+"))  # x_r enters from the left (-), x from below (+): e = x - x_r
    D.arrow((10.1, 5.4), (10.1, 6.44), "$x$", dx=0.25)
    D.arrow((10.26, 6.6), (11.2, 6.6), "$e=x-x_r$")
    D.block(12.2, 6.6, 1.6, 0.6, "$K_e^\\top$ (PD +\ndamper equiv.)", "law")
    D.sum(12.2, 5.6, ("+", "-"))
    D.arrow((12.2, 6.3), (12.2, 5.76), "$K_e^\\top e$", dx=0.55)
    D.arrow((5.4, 6.25), (12.04, 5.6), "$u_r + M_r g\\,e_2$", dy=-0.35, rad=-0.1)
    # command path
    if nl:
        D.block(12.2, 4.5, 2.6, 0.9, "cable frame $R(\\varphi)=[\\hat t\\ \\hat q]$\n$\\xi_t=\\hat t^\\top F_n$,  $\\xi_u=\\hat q^\\top F_n+m_{b,r}\\ell\\dot\\varphi^2$", "law", fs=8)
        D.arrow((12.2, 5.44), (12.2, 4.95), "$F_n$", dx=0.3)
        D.block(12.2, 3.25, 2.8, 0.9, "$w_t=\\hat k_t\\xi_t$\n$w_u=\\hat k_u\\xi_u+\\hat k_\\varphi\\Phi_\\varphi$  ;  $F=R\\,w$", "law", fs=8)
        D.arrow((12.2, 4.05), (12.2, 3.7), "$\\xi,\\Phi_\\varphi$", dx=0.4)
    else:
        D.block(12.2, 3.9, 2.8, 0.9, "$u=\\hat K_x^\\top\\bar x+\\hat K_u^\\top\\xi$\n$\\xi=u_r-K_e^\\top e$", "law")
        D.arrow((12.2, 5.44), (12.2, 4.35), "$\\xi$", dx=0.3)
    D.block(14.1, 3.25 if nl else 3.9, 1.5, 0.7, "$+\\hat W^\\top\\Phi_d(v)$\n(RBF, b)", "law", dashed=not b, fs=8)
    D.arrow((14.1, 2.9 if nl else 3.55), (13.0, 2.5), "", ls="--" if not b else "-")
    # adaptation
    D.block(8.6, 3.6, 4.3, 1.3,
            ("$\\dot{\\hat k}=-\\gamma\\,\\mathrm{reg}\\,(e^\\top P B_e(\\varphi))_{ch}$" if nl else
             "$\\dot{\\hat K}_x=-\\Gamma_x\\odot(\\bar x e^\\top P B_r)$\n$\\dot{\\hat K}_u=-\\Gamma_u\\odot(\\xi e^\\top P B_r)$")
            + ("\n$-\\sigma(\\hat k-k_0)$,  $-\\sigma_W\\hat W$  (c)" if True else ""), "adapt", fs=8.5)
    ax.text(8.6, 2.75, "projection on boxes; $PA_H+A_H^\\top P=-Q$", ha="center", fontsize=8)
    D.arrow((10.75, 3.6), (10.8 if not nl else 10.8, 3.4 if nl else 3.9), "gains", dy=0.05)
    D.arrow((10.1, 6.3), (9.6, 4.25), "$e$", dx=-0.2)
    if not c:
        ax.text(8.6, 3.0, "(σ terms off in this variant)", ha="center", fontsize=7.5, color="0.4")
    # cascade and plant (bottom)
    D.block(12.2, 1.6, 2.8, 0.8, "$f_q=\\|F\\|$,  $\\theta_d=\\mathrm{atan2}(-F_y,F_z)$\nattitude PD + $\\omega_d$ FF (+ ref. damper $\\tau$)", "inner", fs=8)
    D.arrow((12.2, 2.8 if nl else 3.45), (12.2, 2.0), "$F$ (world force)", dx=0.6)
    D.block(8.9, 1.6, 2.2, 0.7, "allocation $B^{-1}$\nmotors $\\tau_m$ (1st order)", "inner", fs=8)
    D.arrow((10.8, 1.6), (10.0, 1.6), "$u=[f_q,\\tau_q]$")
    D.block(5.4, 1.6, 3.2, 0.9, "real plant  $m_q, J_q, m_b, \\ell$\n(taut; $d=0$)", "plant")
    D.arrow((7.8, 1.6), (7.0, 1.6), "$T_{1,2}$")
    D.block(5.4, 0.35, 3.4, 0.45, "$\\mathscr{D}=[d_{q,y}, d_{q,z}, d_{q,\\tau}, d_\\varphi]$: wind, drag, gust, kicks", "plant", fs=8)
    D.arrow((5.4, 0.58), (5.4, 1.15), "")
    D.arrow((3.8, 1.6), (2.2, 1.6), "")
    D.arrow((2.2, 1.6), (2.2, 5.0), "$x=[y_q, v_{q,y}, z_q, v_{q,z}, \\varphi, \\dot\\varphi]$", dx=-1.1, dy=0.0)
    D.arrow((2.2, 5.0), (10.1, 5.4), "", rad=-0.05)
    title = {"vanilla": "Vanilla MRAC (EQ-MR-22): linear outer model", "a": "(a) nonlinear-model MRAC (EQ-MR-23)",
             "ab": "(a)+(b) + disturbance weights", "abc": "(a)+(b)+(c) + σ-modification"}[variant]
    ax.set_title(title + "   (signs at Σ; dashed = off)", fontsize=11)
    return ax


__all__ = ["Diagram", "draw_vanilla_extensions"]
