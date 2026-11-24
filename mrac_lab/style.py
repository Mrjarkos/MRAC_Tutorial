"""Colours and line styles for the MRAC notebooks, taken from the thesis style roles.

``experiments.notebook_utils.style`` is the single source (``PAPER_ROLES``,
``CONTROLLER_STYLE``); this module only maps the notebook's arms onto those
roles and applies the on-screen preset. Call ``provenance.setup`` first so the
thesis root is on ``sys.path``.

Role map (kept identical across notebooks 05-07):

====================  ========================  =====================
key                   meaning                   thesis role
====================  ========================  =====================
``ref``               reference / equilibrium   grey dotted
``ff_true``           PID+FF, true-mass FF      dark grey solid
``pid_ff``            PID+FF, mass-blind FF     black dashed (arm 1)
``softer``            softer law G(-1,-0.5)     blue dash-dot
``s30match``          softer law, true-mass FF  sky-blue solid
``damper``            linear damper G(-1,-2)    green dashed (arm 2)
``mrac``              MRAC (ONLINE)             vermillion solid
``frozen``            MRAC, nominal estimates   reddish-purple dotted
``frozen_learned``    MRAC, learned estimates   purple dashed
====================  ========================  =====================
"""

from __future__ import annotations

from typing import Any, Dict

import matplotlib.pyplot as plt

_ROLE_NAMES = {
    "ref": "reference",
    "pid_ff": "pid",
    "softer": "third",
    "damper": "fourth",
    "mrac": "residual",
    "frozen": "applied",
    "frozen_learned": "mixer_simple",
}

_LABELS = {
    "ref": "reference",
    "ff_true": "PID+FF, true-mass FF (denominator)",
    "pid_ff": "arm 1: PID+FF, mass-blind FF",
    "softer": "arm 2: softer law G(-1,-0.5), mass-blind FF",
    "damper": "damper G(-1,-2), mass-blind FF",
    "s30match": "softer law G(-1,-0.5), true-mass FF (S30 match)",
    "mrac": "MRAC ONLINE",
    "frozen": "MRAC FROZEN (nominal estimates)",
    "frozen_learned": "MRAC FROZEN-learned",
}

#: Labels replaced at run time (for example the oracle form must not be called ``G(...)``).
LABEL_OVERRIDE: Dict[str, str] = {}

#: ``G(kp, kd)`` means the FIXED form ``kd sqrt(L_nom/L)`` (S30 convention).
ORACLE_LABELS = {
    "softer": "arm 2: softer law, ORACLE form (-1,-0.5), mass-blind FF",
    "damper": "damper, ORACLE form (-1,-2), mass-blind FF",
    "s30match": "softer law, ORACLE form (-1,-0.5), true-mass FF",
}


def set_form(fixed: bool) -> None:
    """Choose the labels of the damper arms: fixed form is ``G(...)``, oracle form is not."""
    LABEL_OVERRIDE.clear()
    if not fixed:
        LABEL_OVERRIDE.update(ORACLE_LABELS)

SAT_COLOR = "#E69F00"
SLACK_COLOR = "#B22222"
GRID_ALPHA = 0.3


def init(scale: float = 1.0) -> None:
    """Apply the thesis ``screen`` preset (inline matplotlib, no ipympl)."""
    from experiments.notebook_utils.style import apply_style

    apply_style("screen", scale=scale)
    plt.rcParams.update({"figure.dpi": 100, "figure.max_open_warning": 0})


def arm_style(key: str, **overrides: Any) -> Dict[str, Any]:
    """``color``, ``linestyle``, ``linewidth``, ``label`` kwargs for an arm key."""
    from experiments.notebook_utils.style import PAPER_ROLES

    if key == "ff_true":
        base = {"color": "#555555", "linestyle": "-", "linewidth": 1.0}
    elif key == "s30match":
        base = {"color": "#56B4E9", "linestyle": "-", "linewidth": 1.0}
    else:
        role = PAPER_ROLES[_ROLE_NAMES[key]]
        base = {
            k: role[k]
            for k in ("color", "linestyle", "linewidth", "alpha", "zorder")
            if k in role
        }
    if key == "frozen":
        base["linestyle"] = ":"
        base["linewidth"] = 1.3
    if key == "pid_ff":
        base["linewidth"] = 1.1
    base["label"] = LABEL_OVERRIDE.get(key, _LABELS[key])
    base.update(overrides)
    return base


def arm_color(key: str) -> str:
    return arm_style(key)["color"]


ARM_KEYS = tuple(_LABELS)
